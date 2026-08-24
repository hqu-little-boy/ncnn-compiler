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

// 行级向量化：静态、恒等映射的纯逐元素 generic 改写为
// scf.forall(shared_outs) 网格（除最内维外的全部输出维）+ 最内维整行
// rank-1 vector.transfer_read / 向量化 body。每个迭代把结果行写入私有
// 行张量，再经 tensor.parallel_insert_slice 落回共享输出——各迭代写
// 不相交切片，bufferize 后即 OpenMP 安全的外层线程并行形态，与内层
// SIMD 正交叠加（外层 OpenMP × 内层 SIMD）。rank-1 连续 transfer 保持
// VectorToLLVM 的可靠下降形态；行宽由最内维 extent 决定，寄存器宽度由
// LLVM 合法化拆分。rank-1 张量没有可并行的外层维，保持串行直写；动态
// shape 与非恒等映射实例保持标量。
LogicalResult vectorizeElementwiseRows(MLIRContext* context, ModuleOp module) {
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

    rewriter.setInsertionPoint(generic);
    Value resultBuffer =
      rewriter.create<tensor::EmptyOp>(location, shape, elementType);

    auto& genericBlock = generic.getRegion().front();
    linalg::YieldOp genericYield =
      cast<linalg::YieldOp>(genericBlock.getTerminator());

    // 最内层：生成整行向量读写与向量化 body，返回该行的结果向量。
    // 调用方负责设定插入点。body 引用的外部标量值提升为整行 splat，
    // 并缓存复用。
    auto buildRowVector = [&](ValueRange indices) -> Value {
      VectorType rowType = VectorType::get(shape.back(), elementType);
      SmallVector<bool> inBounds(1, true);
      IRMapping mapping;

      auto resolveOperand = [&](Value value) -> Value {
        if (mapping.contains(value)) {
          return mapping.lookup(value);
        }
        Value splat =
          rewriter.create<vector::SplatOp>(location, rowType, value);
        mapping.map(value, splat);
        return splat;
      };

      unsigned inputIndex = 0;
      for (Value input : generic.getInputs()) {
        mapping.map(genericBlock.getArgument(inputIndex),
                    rewriter.create<vector::TransferReadOp>(
                      location,
                      rowType,
                      input,
                      indices,
                      rewriter.create<ub::PoisonOp>(location, elementType),
                      inBounds));
        ++inputIndex;
      }
      mapping.map(genericBlock.getArgument(generic.getInputs().size()),
                  rewriter.create<vector::TransferReadOp>(
                    location,
                    rowType,
                    generic.getDpsInitOperand(0)->get(),
                    indices,
                    rewriter.create<ub::PoisonOp>(location, elementType),
                    inBounds));

      Value current;
      for (Operation& operation : genericBlock.without_terminator()) {
        if (isa<arith::ConstantOp>(operation)) {
          auto constant = cast<arith::ConstantOp>(operation);
          Value splat = rewriter.create<arith::ConstantOp>(
            location,
            rowType,
            DenseElementsAttr::get(rowType, constant.getValue()));
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
        state.addTypes(SmallVector<Type>(operation.getNumResults(), rowType));
        state.addAttributes(SmallVector<NamedAttribute>(
          operation.getAttrs().begin(), operation.getAttrs().end()));
        Operation* lifted = rewriter.create(state);
        for (auto [oldResult, newResult] :
             llvm::zip(operation.getResults(), lifted->getResults())) {
          mapping.map(oldResult, newResult);
        }
        current = lifted->getResult(0);
      }
      return mapping.lookup(genericYield.getValues().front());
    };

    if (rank == 1) {
      // 无外层维可并行：串行直写（历史形态）。
      SmallVector<Value> indexZero{
        rewriter.create<arith::ConstantIndexOp>(location, 0)};
      Value rowVector = buildRowVector(indexZero);
      Value updated =
        rewriter
          .create<vector::TransferWriteOp>(location,
                                           rowVector,
                                           resultBuffer,
                                           indexZero,
                                           SmallVector<bool>(1, true))
          .getResult();
      rewriter.replaceOp(generic, updated);
      continue;
    }

    // 除最内维外的全部输出维组成 forall 网格，每迭代处理一行。
    SmallVector<OpFoldResult> upperBounds;
    for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
      upperBounds.push_back(rewriter.getIndexAttr(shape[dimension]));
    }
    auto forall = rewriter.create<scf::ForallOp>(
      location, upperBounds, ValueRange{resultBuffer}, std::nullopt);

    Block* body = &forall.getRegion().front();
    const unsigned gridRank = rank - 1;
    Value sharedOut = body->getArgument(gridRank);
    rewriter.setInsertionPointToStart(body);

    SmallVector<Value> gridIndices;
    llvm::append_range(gridIndices, forall.getInductionVars());
    gridIndices.push_back(rewriter.create<arith::ConstantIndexOp>(location, 0));
    Value rowVector = buildRowVector(gridIndices);

    // 私有行张量 [1,...,1,D]：先写满，再在 in_parallel 终结符里以
    // parallel_insert_slice 落回共享输出（forall 要求共享写只出现在
    // in_parallel 内，保证各迭代写不相交切片）。行张量内部的写入索
    // 引是全零——网格索引只用于 insert_slice 的外层定位。
    SmallVector<int64_t> tileShape(rank, 1);
    tileShape.back() = shape.back();
    Value emptyTile =
      rewriter.create<tensor::EmptyOp>(location, tileShape, elementType);
    Value tileZeroIndex = rewriter.create<arith::ConstantIndexOp>(location, 0);
    SmallVector<Value> tileIndices(rank, tileZeroIndex);
    Value rowTensor =
      rewriter
        .create<vector::TransferWriteOp>(location,
                                         rowVector,
                                         emptyTile,
                                         tileIndices,
                                         SmallVector<bool>(1, true))
        .getResult();

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (unsigned dimension = 0; dimension < gridRank; ++dimension) {
      offsets.push_back(OpFoldResult(forall.getInductionVars()[dimension]));
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    offsets.push_back(rewriter.getIndexAttr(0));
    sizes.push_back(rewriter.getIndexAttr(shape.back()));
    strides.push_back(rewriter.getIndexAttr(1));

    scf::InParallelOp inParallel = forall.getTerminator();
    rewriter.setInsertionPointToEnd(&inParallel.getRegion().front());
    rewriter.create<tensor::ParallelInsertSliceOp>(
      location, rowTensor, sharedOut, offsets, sizes, strides);

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

    // 阶段一：静态逐元素 generic 行级向量化为 forall 网格 + rank-1 vector op。
    if (failed(vectorizeElementwiseRows(&getContext(), module))) {
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
