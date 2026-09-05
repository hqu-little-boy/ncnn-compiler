#include "ncnn-mlir/Transforms/VectorizeNCNN/VectorizeNCNN.hpp"

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VECTORIZENCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// 判定 body 内 op 可整体提升为向量语义（逐元素同型）。
bool isLiftableElementwise(Operation& operation) {
  if (isa<arith::ConstantOp>(operation)) {
    return true;
  }
  return operation.getName().getDialect()->getNamespace() == "arith" ||
         operation.getName().getDialect()->getNamespace() == "math";
}

// 行级向量化（分块形态）：静态、恒等映射的纯逐元素 generic 改写为
// scf.forall(shared_outs) 网格（除最内维外的全部输出维）+ 最内维按
// lanes 分块的 rank-1 vector.transfer_read / 向量化 body。每个迭代把
// 结果行写入私有行张量，再经 tensor.parallel_insert_slice 落回共享
// 输出——各迭代写不相交切片，bufferize 后即 OpenMP 安全的外层线程
// 并行形态，与内层 SIMD 正交叠加（外层 OpenMP × 内层 SIMD）。
// 行宽刻意不取最内维整行：末维极宽的模型（CTC logits 行 18385、特征
// 行 37249+）整行成向量后，math op 超出 libmvec ABI 等宽上限走不到
// LowerVectorMathNCNN，被标量 convert-math-to-libm 逐 lane 拆成
// extract/call/insert 风暴（vector<18385xf32> exp → 1.8 万组标量
// 调用），超宽向量本身也把 LLVM 合法化期推向小时级——即"编译爆炸
// 名单"的机制（P2 spike 定界，见 docs/ncnn-performance-parity-plan.md
// §3-P2）。分块预算按元素位宽分档：≤32 位元素取 4×lanes（仍处
// LowerVectorMathNCNN 可拆分上限内，math op 拆成 ≤4 段等宽向量库调用，
// 循环粒度靠近整行形态的调度质量；i8/i16 行不退化为 8 字节级 SIMD），
// 64 位元素取 lanes。行宽不超过分块预算时保持整行形态；不能整除的
// 余数（< 分块宽个元素）以标量 tensor.extract/insert 兜底。
// rank-1 连续 transfer 保持 VectorToLLVM 的可靠下降形态。rank-1 张量
// 没有可并行的外层维，保持串行直写；动态 shape 与非恒等映射实例保持
// 标量。
LogicalResult vectorizeElementwiseRows(MLIRContext* context,
                                       ModuleOp module,
                                       unsigned lanes) {
  SmallVector<linalg::GenericOp> candidates;
  module.walk([&](linalg::GenericOp generic) {
    const unsigned rankDims = generic.getNumLoops();
    const int64_t rank = generic.getNumLoops();
    if (rank == 0 || generic.getNumResults() != 1 ||
        generic.getNumDpsInits() != 1 ||
        llvm::any_of(generic.getIndexingMapsArray(), [rankDims](AffineMap map) {
          return !map.isIdentity() || map.getNumDims() != rankDims;
        })) {
      return;
    }
    auto resultType =
      dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getRank() != rank ||
        !llvm::all_of(resultType.getShape(),
                      [](int64_t extent) { return extent > 0; })) {
      return;
    }
    // 最内维宽度 1 的退化行（如 matmul N=1 输出的 clamp）没有向量化收
    // 益，且 vector<1> 触发上游 TransferWriteOp::build 的越界崩溃，直接
    // 跳过保持标量。
    if (resultType.getShape().back() == 1) {
      return;
    }
    auto& block = generic.getRegion().front();
    auto yieldValue = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yieldValue ||
        !llvm::all_of(block.without_terminator(), isLiftableElementwise)) {
      return;
    }
    for (Value input : generic.getInputs()) {
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || !inputType.hasStaticShape() ||
          inputType.getElementType() != resultType.getElementType()) {
        return;
      }
    }
    candidates.push_back(generic);
  });
  if (candidates.empty()) {
    return success();
  }

  IRRewriter rewriter(context);
  for (linalg::GenericOp generic : llvm::reverse(candidates)) {
    auto resultType = cast<RankedTensorType>(generic.getResult(0).getType());
    const ArrayRef<int64_t> shape = resultType.getShape();
    const int64_t rank = resultType.getRank();
    Type elementType = resultType.getElementType();
    Location location = generic.getLoc();
    const int64_t rowWidth = shape.back();
    const auto laneBudget = static_cast<int64_t>(lanes);
    // 分块预算：≤32 位元素取 4×lanes（寄存器预算 4×YMM，循环粒度靠近
    // 旧整行形态的调度质量；4×lanes 仍是 LowerVectorMathNCNN 可拆分
    // 上限——math op 拆成 ≤4 段等宽向量库调用，不走标量风暴）；64 位
    // 元素无向量数学库覆盖，保持 lanes 等宽（超限即标量 libcall）。
    const int64_t chunkBudget =
      elementType.getIntOrFloatBitWidth() <= 32 ? 4 * laneBudget : laneBudget;
    const int64_t chunkWidth =
      (laneBudget > 1 && rowWidth > chunkBudget) ? chunkBudget : rowWidth;
    const int64_t fullChunks = rowWidth / chunkWidth;
    const int64_t tailWidth = rowWidth % chunkWidth;
    const unsigned gridRank = rank - 1;

    rewriter.setInsertionPoint(generic);
    Value resultBuffer =
      rewriter.create<tensor::EmptyOp>(location, shape, elementType);
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    Value stepOneIndex = rewriter.create<arith::ConstantIndexOp>(location, 1);
    Value chunkWidthIndex =
      rewriter.create<arith::ConstantIndexOp>(location, chunkWidth);
    VectorType chunkType = VectorType::get({chunkWidth}, elementType);

    auto& genericBlock = generic.getRegion().front();
    linalg::YieldOp genericYield =
      cast<linalg::YieldOp>(genericBlock.getTerminator());

    // 最内层：生成一个分块的向量读写与向量化 body，返回该分块的结果
    // 向量。outerIndices 为除最内维外的索引（rank-1 时为空），
    // lastDimOffset 为最内维偏移。body 引用的外部标量值提升为分块宽
    // splat，并缓存复用。
    auto buildChunkVector = [&](OpBuilder& builder,
                                ValueRange outerIndices,
                                Value lastDimOffset) -> Value {
      SmallVector<Value> indices(outerIndices);
      indices.push_back(lastDimOffset);
      // in_bounds 长度 = 置换结果数（identity minor transfer 即向量秩），
      // 与源张量秩无关。
      SmallVector<bool> inBounds(1, true);
      IRMapping mapping;

      auto resolveOperand = [&](Value value) -> Value {
        if (mapping.contains(value)) {
          return mapping.lookup(value);
        }
        Value splat =
          builder.create<vector::SplatOp>(location, chunkType, value);
        mapping.map(value, splat);
        return splat;
      };

      unsigned inputIndex = 0;
      for (Value input : generic.getInputs()) {
        mapping.map(genericBlock.getArgument(inputIndex),
                    builder.create<vector::TransferReadOp>(
                      location,
                      chunkType,
                      input,
                      indices,
                      builder.create<ub::PoisonOp>(location, elementType),
                      inBounds));
        ++inputIndex;
      }
      mapping.map(genericBlock.getArgument(generic.getInputs().size()),
                  builder.create<vector::TransferReadOp>(
                    location,
                    chunkType,
                    generic.getDpsInitOperand(0)->get(),
                    indices,
                    builder.create<ub::PoisonOp>(location, elementType),
                    inBounds));

      Value current;
      for (Operation& operation : genericBlock.without_terminator()) {
        if (isa<arith::ConstantOp>(operation)) {
          auto constant = cast<arith::ConstantOp>(operation);
          Value splat = builder.create<arith::ConstantOp>(
            location,
            chunkType,
            DenseElementsAttr::get(chunkType, constant.getValue()));
          mapping.map(constant.getResult(), splat);
          continue;
        }
        SmallVector<Value> operands;
        operands.reserve(operation.getNumOperands());
        for (Value operand : operation.getOperands()) {
          operands.push_back(resolveOperand(operand));
        }
        OperationState state(location, operation.getName());
        state.addOperands(operands);
        state.addTypes(SmallVector<Type>(operation.getNumResults(), chunkType));
        state.addAttributes(SmallVector<NamedAttribute>(
          operation.getAttrs().begin(), operation.getAttrs().end()));
        Operation* lifted = builder.create(state);
        for (auto [oldResult, newResult] :
             llvm::zip(operation.getResults(), lifted->getResults())) {
          mapping.map(oldResult, newResult);
        }
        current = lifted->getResult(0);
      }
      return mapping.lookup(genericYield.getValues().front());
    };

    // 标量兜底一个尾元素：tensor.extract 提升为标量 body，外部标量
    // 经 clone 的 mapping 缺省映射原样复用。lastDimOffset 为动态值。
    auto buildScalarElement = [&](OpBuilder& builder,
                                  ValueRange outerIndices,
                                  Value lastDimOffset) -> Value {
      SmallVector<Value> indices(outerIndices);
      indices.push_back(lastDimOffset);
      IRMapping mapping;
      unsigned inputIndex = 0;
      for (Value input : generic.getInputs()) {
        mapping.map(
          genericBlock.getArgument(inputIndex),
          builder.create<tensor::ExtractOp>(location, input, indices));
        ++inputIndex;
      }
      mapping.map(genericBlock.getArgument(generic.getInputs().size()),
                  builder.create<tensor::ExtractOp>(
                    location, generic.getDpsInitOperand(0)->get(), indices));
      Value current;
      for (Operation& operation : genericBlock.without_terminator()) {
        Operation* lifted = builder.clone(operation, mapping);
        current = lifted->getResult(0);
      }
      return mapping.lookup(genericYield.getValues().front());
    };

    // 写满一行：fullChunks 个 lanes 宽分块的 scf.for 循环 + 标量尾循环
    // （余数 < lanes），块间经 iter_args 链传递目标行张量。outerIndices
    // 为除最内维外的索引；dest 为待写满的行张量（rank-1 时是结果张量
    // 本身，forall 内是私有行张量）。写入索引一律取全零外层维 + 最内维
    // 偏移：私有行张量形状是 [1,...,1,rowWidth]，外层网格位置只由
    // parallel_insert_slice 落回共享输出时使用——写入若复用网格索引会
    // 对 1×...×1 的行张量大规模越界（堆损坏）。调用方负责把插入点设到
    // 目标区域。
    auto buildRow =
      [&](OpBuilder& builder, ValueRange outerIndices, Value dest) -> Value {
      auto writeIndices = [&](Value lastDimOffset) {
        SmallVector<Value> indices(gridRank, zeroIndex);
        indices.push_back(lastDimOffset);
        return indices;
      };
      Value current = dest;
      if (fullChunks > 1) {
        Value fullBound =
          builder.create<arith::ConstantIndexOp>(location, fullChunks);
        // 步长 1 遍历分块下标，块内偏移由下标 × 分块宽给出。
        auto chunkLoop = builder.create<scf::ForOp>(
          location,
          zeroIndex,
          fullBound,
          stepOneIndex,
          ValueRange{current},
          [&](
            OpBuilder& nested, Location nestedLoc, Value iv, ValueRange args) {
            Value offset =
              nested.create<arith::MulIOp>(nestedLoc, iv, chunkWidthIndex);
            Value chunk = buildChunkVector(nested, outerIndices, offset);
            Value written =
              nested
                .create<vector::TransferWriteOp>(nestedLoc,
                                                 chunk,
                                                 args[0],
                                                 writeIndices(offset),
                                                 SmallVector<bool>(1, true))
                .getResult();
            nested.create<scf::YieldOp>(nestedLoc, ValueRange{written});
          });
        current = chunkLoop.getResult(0);
      } else {
        Value chunk = buildChunkVector(builder, outerIndices, zeroIndex);
        current = builder
                    .create<vector::TransferWriteOp>(location,
                                                     chunk,
                                                     current,
                                                     writeIndices(zeroIndex),
                                                     SmallVector<bool>(1, true))
                    .getResult();
      }
      if (tailWidth > 0) {
        Value tailBase = builder.create<arith::ConstantIndexOp>(
          location, fullChunks * chunkWidth);
        Value tailBound =
          builder.create<arith::ConstantIndexOp>(location, tailWidth);
        auto tailLoop = builder.create<scf::ForOp>(
          location,
          zeroIndex,
          tailBound,
          stepOneIndex,
          ValueRange{current},
          [&](
            OpBuilder& nested, Location nestedLoc, Value iv, ValueRange args) {
            Value offset =
              nested.create<arith::AddIOp>(nestedLoc, tailBase, iv);
            Value element = buildScalarElement(nested, outerIndices, offset);
            Value updated = nested.create<tensor::InsertOp>(
              nestedLoc, element, args[0], writeIndices(offset));
            nested.create<scf::YieldOp>(nestedLoc, ValueRange{updated});
          });
        current = tailLoop.getResult(0);
      }
      return current;
    };

    if (rank == 1) {
      // 无外层维可并行：串行直写（历史形态）。
      Value updated = buildRow(rewriter, ValueRange{}, resultBuffer);
      rewriter.replaceOp(generic, updated);
      continue;
    }

    // 除最内维外的全部输出维组成 forall 网格，每迭代分块写一行。
    SmallVector<OpFoldResult> upperBounds;
    for (unsigned dimension = 0; dimension < gridRank; ++dimension) {
      upperBounds.push_back(rewriter.getIndexAttr(shape[dimension]));
    }
    auto forall = rewriter.create<scf::ForallOp>(
      location, upperBounds, ValueRange{resultBuffer}, std::nullopt);

    Block* body = &forall.getRegion().front();
    Value sharedOut = body->getArgument(gridRank);
    rewriter.setInsertionPointToStart(body);

    SmallVector<Value> gridIndices;
    llvm::append_range(gridIndices, forall.getInductionVars());

    // 私有行张量 [1,...,1,D]：先经 buildRow 分块写满，再在 in_parallel
    // 终结符里以 parallel_insert_slice 落回共享输出（forall 要求共享写
    // 只出现在 in_parallel 内，保证各迭代写不相交切片）。行张量内部的
    // 写入索引除最内维偏移外全零——网格索引只用于 insert_slice 的外层
    // 定位。
    SmallVector<int64_t> tileShape(rank, 1);
    tileShape.back() = rowWidth;
    Value emptyTile =
      rewriter.create<tensor::EmptyOp>(location, tileShape, elementType);
    Value tile = buildRow(rewriter, gridIndices, emptyTile);

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (unsigned dimension = 0; dimension < gridRank; ++dimension) {
      offsets.push_back(OpFoldResult(forall.getInductionVars()[dimension]));
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    offsets.push_back(rewriter.getIndexAttr(0));
    sizes.push_back(rewriter.getIndexAttr(rowWidth));
    strides.push_back(rewriter.getIndexAttr(1));

    scf::InParallelOp inParallel = forall.getTerminator();
    rewriter.setInsertionPointToEnd(&inParallel.getRegion().front());
    rewriter.create<tensor::ParallelInsertSliceOp>(
      location, tile, sharedOut, offsets, sizes, strides);

    rewriter.replaceOp(generic, forall.getResults());
  }
  return success();
}

class VectorizeNCNNPass final
  : public impl::VectorizeNCNNPassBase<VectorizeNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    const unsigned lanes = this->lanes.getValue();
    if (lanes == 0) {
      return;
    }

    // 阶段一：静态逐元素 generic 行级向量化为 forall 网格 + lanes 分块
    // 的 rank-1 vector op。
    if (failed(vectorizeElementwiseRows(&getContext(), module, lanes))) {
      signalPassFailure();
      return;
    }

    // 阶段二：归约链提升为 contract 并完成向量级规范化。
    RewritePatternSet cleanupPatterns(&getContext());
    vector::populateVectorReductionToContractPatterns(cleanupPatterns);
    if (failed(
          applyPatternsAndFoldGreedily(module, std::move(cleanupPatterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
