#include "ncnn-mlir/Transforms/FoldLinalgConstantTranspose/FoldLinalgConstantTranspose.hpp"

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/ConstantFold.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_FOLDLINALGCONSTANTTRANSPOSEPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class FoldLinalgConstantTransposePass final
  : public impl::FoldLinalgConstantTransposePassBase<
      FoldLinalgConstantTransposePass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::TransposeOp> transposes;
    module.walk([&](linalg::TransposeOp transpose) {
      if (auto constant =
            transpose.getInput().getDefiningOp<arith::ConstantOp>()) {
        if (isa<ElementsAttr>(constant.getValue())) {
          transposes.push_back(transpose);
        }
      }
    });
    if (transposes.empty()) {
      return;
    }
    IRRewriter rewriter(&getContext());
    for (linalg::TransposeOp transpose : transposes) {
      auto constant = transpose.getInput().getDefiningOp<arith::ConstantOp>();
      if (!constant) {
        continue;
      }
      auto elements = dyn_cast<ElementsAttr>(constant.getValue());
      auto sourceType =
        dyn_cast<RankedTensorType>(transpose.getInput().getType());
      auto resultType =
        dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
      if (!elements || !sourceType || !resultType ||
          !sourceType.hasStaticShape() || !resultType.hasStaticShape()) {
        continue;
      }
      SmallVector<int32_t> permutation;
      llvm::append_range(permutation, transpose.getPermutation());
      DenseElementsAttr folded = ncnn_mlir::transpose_dense_elements(
        elements, sourceType, permutation, resultType);
      if (!folded) {
        continue;
      }
      rewriter.setInsertionPoint(transpose);
      auto replacement = rewriter.create<arith::ConstantOp>(
        transpose.getLoc(), resultType, folded);
      rewriter.replaceOp(transpose, replacement.getResult());
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
