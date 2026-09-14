#include "ncnn-mlir/Transforms/VectorizeNCNN/VectorizeNCNN.hpp"

#include <algorithm>
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
#include "ncnn-mlir/Support/KernelContract.hpp"

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
        generic->getParentOfType<scf::ForallOp>() != nullptr ||
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
        // 外部标量的类型跟随其自身元素类型（不一定等于输出元素类型）。
        Type scalarType = isa<ShapedType>(value.getType())
                            ? cast<ShapedType>(value.getType()).getElementType()
                            : value.getType();
        auto splatType = VectorType::get({chunkWidth}, scalarType);
        Value splat =
          builder.create<vector::SplatOp>(location, splatType, value);
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
          auto splatType = VectorType::get({chunkWidth}, constant.getType());
          Value splat = builder.create<arith::ConstantOp>(
            location,
            splatType,
            DenseElementsAttr::get(splatType, constant.getValue()));
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
        // 逐结果类型向量化：body 可含谓词 op（如 quantize 的 cmpf 产出
        // i1），统一赋行向量类型会产出非法 IR。
        for (Type resultType : operation.getResultTypes()) {
          state.addTypes(VectorType::get({chunkWidth}, resultType));
        }
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
    contract::annotateTile(forall.getOperation(),
                           "elementwise_simd",
                           "identity",
                           "identity",
                           "identity",
                           1,
                           rowWidth,
                           0,
                           "outer_tile+inner_simd",
                           tailWidth > 0 ? "scalar_tail" : "none");
    contract::annotatePacking(forall.getOperation(), "unpacked", 1, 0, 0);
    contract::setInteger(forall.getOperation(), contract::kSimdLanes, lanes);
    contract::setInteger(
      forall.getOperation(), contract::kSimdChunk, chunkWidth);
    contract::setString(
      forall.getOperation(), contract::kFma, "vector_elementwise");

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

// P5 depthwise 行向量化（docs/ncnn-performance-parity-plan.md §3-P5）：
// multiplier=1 的 linalg.depthwise_conv_2d_nhwc_hwcm 窗口 gather 不满足
// 逐元素行向量化的恒等映射前置条件，此前整层走标量下降（rec/attention
// 系每模型 27–28 层，仅 formula_encoder 即 27 层）。multiplier=1 时窗口
// 读沿 C 维连续：输入 [N,pH,pW,C] 的 x[n,ih,iw,c0..c0+VL]（ih/iw 由
// stride/dilation 给出，padding 已由 tensor.pad 物化进输入），权重 HWCM
// [KH,KW,C,1] 折叠为 [KH*KW,C] 后的 w[kh*KW+kw,c0..c0+VL]。改写为
// forall(n,oh,ow) 网格 + C 维分块的 rank-1 transfer：每分块以 init 行
// 作累加种子，逐窗口 vector.fma 累加后一次写回——累加链全程驻留寄存器
// （named op 逐元素展开时每个 MAC 都要读写内存）。语义对照 named op
// body（acc = addf(mulf(in,w), acc)，kh 外 kw 内归约序）：窗口循环序一
// 致，单舍入 FMA 的舍入差异由数值黄金预算吸收（P1 同款取舍）；标量尾
// 逐位复刻 mulf+addf 不做收缩。multi-channel multiplier 变体降级标量
// 留待 P8 评估；动态 shape、非 f32、分块宽非 2 的幂（vector<3>/24 非
// 2 幂宽度教训，同 MatmulKernelNCNN 通道门控）保持原状。
LogicalResult vectorizeDepthwiseConvRows(MLIRContext* context,
                                         ModuleOp module,
                                         unsigned lanes) {
  struct DepthwiseCandidate {
    linalg::DepthwiseConv2DNhwcHwcmOp op;
    int64_t chunkWidth;
  };
  SmallVector<DepthwiseCandidate> candidates;
  const auto isVectorWidth = [](int64_t width) {
    return width >= 2 && (width & (width - 1)) == 0;
  };
  module.walk([&](linalg::DepthwiseConv2DNhwcHwcmOp op) {
    auto inputType =
      dyn_cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType());
    auto weightType =
      dyn_cast<RankedTensorType>(op.getDpsInputOperand(1)->get().getType());
    auto initType =
      dyn_cast<RankedTensorType>(op.getDpsInitOperand(0)->get().getType());
    if (!inputType || !weightType || !initType || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape() || !initType.hasStaticShape() ||
        op->getParentOfType<scf::ForallOp>() != nullptr) {
      return;
    }
    if (!inputType.getElementType().isF32() || inputType.getRank() != 4 ||
        weightType.getRank() != 4 || initType.getRank() != 5) {
      return;
    }
    const int64_t channels = inputType.getShape()[3];
    // multiplier=1 前置：weight [KH,KW,C,1]、init [N,OH,OW,C,1]。
    if (channels <= 0 || weightType.getShape()[2] != channels ||
        weightType.getShape()[3] != 1 || initType.getShape()[3] != channels ||
        initType.getShape()[4] != 1) {
      return;
    }
    if (llvm::any_of(inputType.getShape(),
                     [](int64_t extent) { return extent <= 0; }) ||
        llvm::any_of(initType.getShape(),
                     [](int64_t extent) { return extent <= 0; }) ||
        weightType.getShape()[0] <= 0 || weightType.getShape()[1] <= 0) {
      return;
    }
    // 分块宽沿用逐元素路径的位宽预算（f32 → 4×lanes）；非 2 幂时退到
    // lanes，仍非 2 幂（窄于 lanes 的奇数宽通道）保持标量。
    const auto laneBudget = static_cast<int64_t>(lanes);
    int64_t chunkWidth = std::min(4 * laneBudget, channels);
    if (!isVectorWidth(chunkWidth)) {
      chunkWidth = std::min(laneBudget, channels);
    }
    if (!isVectorWidth(chunkWidth)) {
      return;
    }
    candidates.push_back(
      DepthwiseCandidate{.op = op, .chunkWidth = chunkWidth});
  });
  if (candidates.empty()) {
    return success();
  }

  IRRewriter rewriter(context);
  for (DepthwiseCandidate& candidate : llvm::reverse(candidates)) {
    linalg::DepthwiseConv2DNhwcHwcmOp op = candidate.op;
    auto inputType =
      cast<RankedTensorType>(op.getDpsInputOperand(0)->get().getType());
    auto weightType =
      cast<RankedTensorType>(op.getDpsInputOperand(1)->get().getType());
    auto initType =
      cast<RankedTensorType>(op.getDpsInitOperand(0)->get().getType());
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    const ArrayRef<int64_t> inputShape = inputType.getShape();
    const ArrayRef<int64_t> outputShape = initType.getShape();
    const int64_t batch = inputShape[0];
    const int64_t outputHeight = outputShape[1];
    const int64_t outputWidth = outputShape[2];
    const int64_t channels = inputShape[3];
    const int64_t kernelHeight = weightType.getShape()[0];
    const int64_t kernelWidth = weightType.getShape()[1];
    const int64_t strideHeight = op.getStrides().getValues<int64_t>()[0];
    const int64_t strideWidth = op.getStrides().getValues<int64_t>()[1];
    const int64_t dilationHeight = op.getDilations().getValues<int64_t>()[0];
    const int64_t dilationWidth = op.getDilations().getValues<int64_t>()[1];
    const int64_t chunkWidth = candidate.chunkWidth;
    const int64_t fullChunks = channels / chunkWidth;
    const int64_t tailWidth = channels % chunkWidth;
    Type elementType = inputType.getElementType();
    Location location = op.getLoc();

    rewriter.setInsertionPoint(op);
    Value input = op.getDpsInputOperand(0)->get();
    Value weight = op.getDpsInputOperand(1)->get();
    Value init = op.getDpsInitOperand(0)->get();

    // 权重/初始化的 C 行视图（无拷贝）：named op 的最内维是 m=1，行向
    // 量 transfer 需要沿 C 连续的形态。输出同样以折叠 4D 计算，替换时
    // expand 回 5D——下游既有 collapse_shape 消费者与该 expand 对消。
    auto weightRowsType = RankedTensorType::get(
      {kernelHeight * kernelWidth, channels}, elementType);
    Value weightRows = rewriter.create<tensor::CollapseShapeOp>(
      location,
      weightRowsType,
      weight,
      SmallVector<ReassociationIndices>{{0, 1}, {2, 3}});
    auto initCollapsedType = RankedTensorType::get(
      {batch, outputHeight, outputWidth, channels}, elementType);
    Value initCollapsed = rewriter.create<tensor::CollapseShapeOp>(
      location,
      initCollapsedType,
      init,
      SmallVector<ReassociationIndices>{{0}, {1}, {2}, {3, 4}});
    Value resultBuffer = rewriter.create<tensor::EmptyOp>(
      location,
      ArrayRef<int64_t>{batch, outputHeight, outputWidth, channels},
      elementType);
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    Value stepOneIndex = rewriter.create<arith::ConstantIndexOp>(location, 1);
    Value chunkWidthIndex =
      rewriter.create<arith::ConstantIndexOp>(location, chunkWidth);
    Value kernelWidthIndex =
      rewriter.create<arith::ConstantIndexOp>(location, kernelWidth);
    Value kernelHeightIndex =
      rewriter.create<arith::ConstantIndexOp>(location, kernelHeight);
    Value strideHeightIndex =
      rewriter.create<arith::ConstantIndexOp>(location, strideHeight);
    Value strideWidthIndex =
      rewriter.create<arith::ConstantIndexOp>(location, strideWidth);
    Value dilationHeightIndex =
      rewriter.create<arith::ConstantIndexOp>(location, dilationHeight);
    Value dilationWidthIndex =
      rewriter.create<arith::ConstantIndexOp>(location, dilationWidth);
    VectorType chunkType = VectorType::get({chunkWidth}, elementType);
    SmallVector<bool> inBounds(1, true);
    Value poison = rewriter.create<ub::PoisonOp>(location, elementType);

    // 一个分块：acc 以 init 行种子起链，kh 外 kw 内逐窗口 fma（与
    // named op 归约序一致）。gridIndices = {n, oh, ow}，ohStride/
    // owStride 已在行级预算。
    auto buildChunkVector = [&](OpBuilder& builder,
                                ValueRange gridIndices,
                                Value ohStride,
                                Value owStride,
                                Value offset) -> Value {
      Value accumulator = builder.create<vector::TransferReadOp>(
        location,
        chunkType,
        initCollapsed,
        ValueRange{gridIndices[0], gridIndices[1], gridIndices[2], offset},
        poison,
        inBounds);
      auto kernelHeightLoop = builder.create<scf::ForOp>(
        location,
        zeroIndex,
        kernelHeightIndex,
        stepOneIndex,
        ValueRange{accumulator},
        [&](OpBuilder& nested, Location nestedLoc, Value kh, ValueRange args) {
          Value ih = nested.create<arith::AddIOp>(
            nestedLoc,
            ohStride,
            nested.create<arith::MulIOp>(nestedLoc, kh, dilationHeightIndex));
          auto kernelWidthLoop = nested.create<scf::ForOp>(
            nestedLoc,
            zeroIndex,
            kernelWidthIndex,
            stepOneIndex,
            ValueRange{args[0]},
            [&](OpBuilder& inner,
                Location innerLoc,
                Value kw,
                ValueRange innerArgs) {
              Value iw = inner.create<arith::AddIOp>(
                innerLoc,
                owStride,
                inner.create<arith::MulIOp>(innerLoc, kw, dilationWidthIndex));
              Value weightRow = inner.create<arith::AddIOp>(
                innerLoc,
                inner.create<arith::MulIOp>(innerLoc, kh, kernelWidthIndex),
                kw);
              Value inputChunk = inner.create<vector::TransferReadOp>(
                innerLoc,
                chunkType,
                input,
                ValueRange{gridIndices[0], ih, iw, offset},
                poison,
                inBounds);
              Value weightChunk = inner.create<vector::TransferReadOp>(
                innerLoc,
                chunkType,
                weightRows,
                ValueRange{weightRow, offset},
                poison,
                inBounds);
              Value fused = inner.create<vector::FMAOp>(
                innerLoc, chunkType, inputChunk, weightChunk, innerArgs[0]);
              inner.create<scf::YieldOp>(innerLoc, ValueRange{fused});
            });
          nested.create<scf::YieldOp>(nestedLoc,
                                      ValueRange{kernelWidthLoop.getResult(0)});
        });
      return kernelHeightLoop.getResult(0);
    };

    // 标量尾逐元素：逐位复刻 named op body 的 mulf+addf（不经 FMA 收缩）。
    auto buildScalarElement = [&](OpBuilder& builder,
                                  ValueRange gridIndices,
                                  Value ohStride,
                                  Value owStride,
                                  Value offset) -> Value {
      Value accumulator = builder.create<tensor::ExtractOp>(
        location,
        initCollapsed,
        ValueRange{gridIndices[0], gridIndices[1], gridIndices[2], offset});
      auto kernelHeightLoop = builder.create<scf::ForOp>(
        location,
        zeroIndex,
        kernelHeightIndex,
        stepOneIndex,
        ValueRange{accumulator},
        [&](OpBuilder& nested, Location nestedLoc, Value kh, ValueRange args) {
          Value ih = nested.create<arith::AddIOp>(
            nestedLoc,
            ohStride,
            nested.create<arith::MulIOp>(nestedLoc, kh, dilationHeightIndex));
          auto kernelWidthLoop = nested.create<scf::ForOp>(
            nestedLoc,
            zeroIndex,
            kernelWidthIndex,
            stepOneIndex,
            ValueRange{args[0]},
            [&](OpBuilder& inner,
                Location innerLoc,
                Value kw,
                ValueRange innerArgs) {
              Value iw = inner.create<arith::AddIOp>(
                innerLoc,
                owStride,
                inner.create<arith::MulIOp>(innerLoc, kw, dilationWidthIndex));
              Value weightRow = inner.create<arith::AddIOp>(
                innerLoc,
                inner.create<arith::MulIOp>(innerLoc, kh, kernelWidthIndex),
                kw);
              Value inputElement = inner.create<tensor::ExtractOp>(
                innerLoc, input, ValueRange{gridIndices[0], ih, iw, offset});
              Value weightElement = inner.create<tensor::ExtractOp>(
                innerLoc, weightRows, ValueRange{weightRow, offset});
              Value product = inner.create<arith::MulFOp>(
                innerLoc, inputElement, weightElement);
              Value updated =
                inner.create<arith::AddFOp>(innerLoc, product, innerArgs[0]);
              inner.create<scf::YieldOp>(innerLoc, ValueRange{updated});
            });
          nested.create<scf::YieldOp>(nestedLoc,
                                      ValueRange{kernelWidthLoop.getResult(0)});
        });
      return kernelHeightLoop.getResult(0);
    };

    // 写满一行：C 维分块 scf.for（iter_args 链携带私有行张量）+ 标量尾。
    // 写入索引外层全零（私有行张量形状 [1,1,1,C]，网格索引只用于
    // parallel_insert_slice 定位，同逐元素路径的越界教训）。
    auto buildRow =
      [&](OpBuilder& builder, ValueRange gridIndices, Value dest) -> Value {
      Value ohStride = builder.create<arith::MulIOp>(
        location, gridIndices[1], strideHeightIndex);
      Value owStride = builder.create<arith::MulIOp>(
        location, gridIndices[2], strideWidthIndex);
      auto writeIndices = [&](Value offset) {
        return SmallVector<Value>{zeroIndex, zeroIndex, zeroIndex, offset};
      };
      Value current = dest;
      if (fullChunks > 1) {
        Value fullBound =
          builder.create<arith::ConstantIndexOp>(location, fullChunks);
        auto chunkLoop = builder.create<scf::ForOp>(
          location,
          zeroIndex,
          fullBound,
          stepOneIndex,
          ValueRange{current},
          [&](OpBuilder& nested,
              Location nestedLoc,
              Value chunk,
              ValueRange args) {
            Value offset =
              nested.create<arith::MulIOp>(nestedLoc, chunk, chunkWidthIndex);
            Value chunkResult =
              buildChunkVector(nested, gridIndices, ohStride, owStride, offset);
            Value written =
              nested
                .create<vector::TransferWriteOp>(nestedLoc,
                                                 chunkResult,
                                                 args[0],
                                                 writeIndices(offset),
                                                 SmallVector<bool>(1, true))
                .getResult();
            nested.create<scf::YieldOp>(nestedLoc, ValueRange{written});
          });
        current = chunkLoop.getResult(0);
      } else {
        Value chunkResult =
          buildChunkVector(builder, gridIndices, ohStride, owStride, zeroIndex);
        current = builder
                    .create<vector::TransferWriteOp>(location,
                                                     chunkResult,
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
          [&](OpBuilder& nested,
              Location nestedLoc,
              Value lane,
              ValueRange args) {
            Value offset =
              nested.create<arith::AddIOp>(nestedLoc, tailBase, lane);
            Value element = buildScalarElement(
              nested, gridIndices, ohStride, owStride, offset);
            Value updated = nested.create<tensor::InsertOp>(
              nestedLoc, element, args[0], writeIndices(offset));
            nested.create<scf::YieldOp>(nestedLoc, ValueRange{updated});
          });
        current = tailLoop.getResult(0);
      }
      return current;
    };

    SmallVector<OpFoldResult> upperBounds;
    upperBounds.push_back(rewriter.getIndexAttr(batch));
    upperBounds.push_back(rewriter.getIndexAttr(outputHeight));
    upperBounds.push_back(rewriter.getIndexAttr(outputWidth));
    auto forall = rewriter.create<scf::ForallOp>(
      location, upperBounds, ValueRange{resultBuffer}, std::nullopt);
    contract::annotateTile(forall.getOperation(),
                           "depthwise_simd",
                           "nhwc",
                           "hwcm",
                           "nhwc",
                           1,
                           channels,
                           kernelHeight * kernelWidth,
                           "outer_tile+inner_simd",
                           tailWidth > 0 ? "scalar_tail" : "none");
    contract::annotatePacking(forall.getOperation(), "unpacked", 1, 0, 0);
    contract::setInteger(forall.getOperation(), contract::kSimdLanes, lanes);
    contract::setInteger(
      forall.getOperation(), contract::kSimdChunk, chunkWidth);
    contract::setString(forall.getOperation(), contract::kFma, "vector.fma");

    Block* body = &forall.getRegion().front();
    Value sharedOut = body->getArgument(3);
    rewriter.setInsertionPointToStart(body);
    SmallVector<Value> gridIndices;
    llvm::append_range(gridIndices, forall.getInductionVars());

    SmallVector<int64_t> tileShape(4, 1);
    tileShape.back() = channels;
    Value emptyTile =
      rewriter.create<tensor::EmptyOp>(location, tileShape, elementType);
    Value tile = buildRow(rewriter, gridIndices, emptyTile);

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
      offsets.push_back(OpFoldResult(forall.getInductionVars()[dimension]));
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    offsets.push_back(rewriter.getIndexAttr(0));
    sizes.push_back(rewriter.getIndexAttr(channels));
    strides.push_back(rewriter.getIndexAttr(1));

    scf::InParallelOp inParallel = forall.getTerminator();
    rewriter.setInsertionPointToEnd(&inParallel.getRegion().front());
    rewriter.create<tensor::ParallelInsertSliceOp>(
      location, tile, sharedOut, offsets, sizes, strides);

    SmallVector<OpFoldResult> expandedShape;
    for (int64_t extent : resultType.getShape()) {
      expandedShape.push_back(rewriter.getIndexAttr(extent));
    }
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
      location,
      resultType,
      forall.getResult(0),
      SmallVector<ReassociationIndices>{{0}, {1}, {2}, {3, 4}},
      expandedShape);
    rewriter.replaceOp(op, expanded);
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
    if (this->scalable.getValue()) {
      // The implementation below intentionally emits fixed-width vectors.  Do
      // not silently claim scalable-vector support while producing fixed IR.
      module.walk([&](Operation* operation) {
        if (isa<linalg::GenericOp, linalg::DepthwiseConv2DNhwcHwcmOp>(
              operation)) {
          contract::annotateFallback(operation, "unsupported_scalable_vector");
        }
      });
      return;
    }
    if (lanes == 0) {
      return;
    }

    // 阶段一：静态逐元素 generic 行级向量化为 forall 网格 + lanes 分块
    // 的 rank-1 vector op；multiplier=1 的 depthwise named op 同形态改
    // 写（窗口 gather 不满足逐元素前置条件，见 vectorizeDepthwiseConvRows）。
    if (failed(vectorizeElementwiseRows(&getContext(), module, lanes)) ||
        failed(vectorizeDepthwiseConvRows(&getContext(), module, lanes))) {
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
