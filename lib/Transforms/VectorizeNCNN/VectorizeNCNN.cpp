#include "ncnn-mlir/Transforms/VectorizeNCNN/VectorizeNCNN.hpp"

#include <cstdint>
#include <functional>

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
// 外层标量 scf.for 嵌套 + 最内维整行 rank-1 vector.transfer_read /
// 向量化 body / vector.transfer_write。rank-1 连续 transfer 是
// VectorToLLVM 的可靠下降形态；行宽由最内维 extent 决定，寄存器宽度
// 由 LLVM 合法化拆分。动态 shape 与非恒等映射实例保持标量。
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

    SmallVector<Value> indexZero;
    indexZero.reserve(rank);
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      indexZero.push_back(rewriter.create<arith::ConstantIndexOp>(location, 0));
    }

    auto& genericBlock = generic.getRegion().front();
    linalg::YieldOp genericYield =
      cast<linalg::YieldOp>(genericBlock.getTerminator());

    // 递归构造循环嵌套；最内层生成整行向量读写与向量化 body。
    // 返回该层级写出的最新 tensor（函数式语义下逐次串联）。
    std::function<Value(int64_t, SmallVector<Value>, Value)> buildLevel =
      [&](
        int64_t level, SmallVector<Value> indices, Value accumulator) -> Value {
      if (level == rank - 1) {
        VectorType rowType = VectorType::get(shape.back(), elementType);
        // in_bounds 与向量秩一致（rank-1 行读取）。
        SmallVector<bool> inBounds(1, true);
        IRMapping mapping;

        // body 引用的外部标量值提升为整行 splat，并缓存复用。
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
        current = mapping.lookup(genericYield.getValues().front());
        return rewriter
          .create<vector::TransferWriteOp>(
            location, current, accumulator, indices, inBounds)
          .getResult();
      }

      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(location, 0);
      Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(location, shape[level]);
      Value step = rewriter.create<arith::ConstantIndexOp>(location, 1);
      auto loop = rewriter.create<scf::ForOp>(
        location, lowerBound, upperBound, step, ValueRange{accumulator});
      rewriter.setInsertionPointToStart(loop.getBody());
      SmallVector<Value> nextIndices(indices);
      nextIndices[level] = loop.getInductionVar();
      Value updated =
        buildLevel(level + 1, nextIndices, loop.getRegionIterArg(0));
      rewriter.setInsertionPointAfter(loop);
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<scf::YieldOp>(location, updated);
      return loop.getResult(0);
    };

    Value accumulated = buildLevel(0, indexZero, resultBuffer);
    rewriter.replaceOp(generic, accumulated);
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

    // 阶段一：静态逐元素 generic 行级向量化为 rank-1 vector op。
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
