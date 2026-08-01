#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"

#include <cstdint>
#include <limits>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace mlir::ncnn {
namespace {

struct ExplicitPadding {
  int64_t before;
  int64_t after;
};

FailureOr<ExplicitPadding> computeSamePadding(int64_t input,
                                              int64_t kernel,
                                              int64_t stride,
                                              int64_t dilation,
                                              bool sameLower) {
  if (input <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0) {
    return failure();
  }
  if (kernel - 1 > (std::numeric_limits<int64_t>::max() - 1) / dilation) {
    return failure();
  }
  const int64_t extent = (dilation * (kernel - 1)) + 1;
  const int64_t output = 1 + ((input - 1) / stride);
  if (output - 1 > (std::numeric_limits<int64_t>::max() - extent) / stride) {
    return failure();
  }
  const int64_t required = ((output - 1) * stride) + extent;
  const int64_t total = required > input ? required - input : 0;
  const int64_t before = sameLower ? (total + 1) / 2 : total / 2;
  return ExplicitPadding{.before = before, .after = total - before};
}

class NormalizeNCNNPass final
  : public PassWrapper<NormalizeNCNNPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeNCNNPass)

  StringRef getArgument() const final { return "normalize-ncnn"; }
  StringRef getDescription() const final {
    return "Normalize target-independent ncnn semantics";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<ModelOp>().empty()) {
      module.emitError(
        "normalize-ncnn requires ncnn.model to be converted to "
        "func.func first");
      signalPassFailure();
      return;
    }

    OwningOpRef<ModuleOp> candidate = cast<ModuleOp>(module->clone());
    WalkResult validation = candidate->walk([&](Operation* operation) {
      if (operation->getDialect() == nullptr ||
          operation->getDialect()->getNamespace() != "ncnn") {
        return WalkResult::advance();
      }
      if (operation->getParentOfType<func::FuncOp>() == nullptr) {
        operation->emitOpError("requires a func.func boundary");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (validation.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    SmallVector<SplitOp> splits;
    WalkResult normalization = candidate->walk([&](Operation* operation) {
      if (auto convolution = dyn_cast<ConvolutionOp>(operation)) {
        return normalizeConvolution(convolution);
      }
      if (auto pooling = dyn_cast<PoolingOp>(operation)) {
        return normalizePooling(pooling);
      }
      if (auto concat = dyn_cast<ConcatOp>(operation)) {
        normalizeAxis(concat, concat.getInputs().front().getType());
      } else if (auto softmax = dyn_cast<SoftmaxOp>(operation)) {
        normalizeAxis(softmax, softmax.getInput().getType());
      } else if (auto relu = dyn_cast<ReluOp>(operation)) {
        setF32(
          relu, "negative_slope", relu.getNegativeSlope().convertToDouble());
      } else if (auto dropout = dyn_cast<DropoutOp>(operation)) {
        setF32(dropout, "scale", dropout.getScale().convertToDouble());
      } else if (auto split = dyn_cast<SplitOp>(operation)) {
        for (OpResult result : split.getResults()) {
          result.replaceAllUsesWith(split.getInput());
        }
        splits.push_back(split);
      }
      return WalkResult::advance();
    });
    if (normalization.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    for (SplitOp split : splits) {
      split.erase();
    }
    module.getBodyRegion().takeBody(candidate->getBodyRegion());
  }

 private:
  template <typename Op>
  static void normalizeAxis(Op operation, Type inputType) {
    auto ranked = dyn_cast<RankedTensorType>(inputType);
    int64_t axis = operation.getAxis();
    if (axis < 0) {
      axis += ranked.getRank();
    }
    operation->setAttr("axis",
                       Builder(operation.getContext()).getI64IntegerAttr(axis));
  }

  template <typename Op>
  static void setI64(Op operation, StringRef name, int64_t value) {
    operation->setAttr(
      name, Builder(operation.getContext()).getI64IntegerAttr(value));
  }

  template <typename Op>
  static void setF32(Op operation, StringRef name, double value) {
    operation->setAttr(name,
                       Builder(operation.getContext()).getF32FloatAttr(value));
  }

  static WalkResult normalizeConvolution(ConvolutionOp operation) {
    setI64(operation, "int8_scale_term", operation.getInt8ScaleTerm());
    const int64_t pad = operation.getPadTop();
    if (pad != -233 && pad != -234) {
      return WalkResult::advance();
    }

    auto input = cast<RankedTensorType>(operation.getInput().getType());
    const bool sameLower = pad == -234;
    FailureOr<ExplicitPadding> height =
      computeSamePadding(input.getShape()[1],
                         operation.getKernelH(),
                         operation.getStrideH(),
                         operation.getDilationH(),
                         sameLower);
    FailureOr<ExplicitPadding> width =
      computeSamePadding(input.getShape()[2],
                         operation.getKernelW(),
                         operation.getStrideW(),
                         operation.getDilationW(),
                         sameLower);
    if (failed(height) || failed(width)) {
      operation.emitOpError("cannot resolve static SAME padding");
      return WalkResult::interrupt();
    }
    setI64(operation, "pad_top", height->before);
    setI64(operation, "pad_bottom", height->after);
    setI64(operation, "pad_left", width->before);
    setI64(operation, "pad_right", width->after);
    return WalkResult::advance();
  }

  static WalkResult normalizePooling(PoolingOp operation) {
    if (operation.getMode() != static_cast<int64_t>(PoolMode::Regular) ||
        (operation.getPadMode() != 2 && operation.getPadMode() != 3)) {
      return WalkResult::advance();
    }

    auto input = cast<RankedTensorType>(operation.getInput().getType());
    const bool sameLower = operation.getPadMode() == 3;
    FailureOr<ExplicitPadding> height =
      computeSamePadding(input.getShape()[1],
                         operation.getKernelH(),
                         operation.getStrideH(),
                         1,
                         sameLower);
    FailureOr<ExplicitPadding> width =
      computeSamePadding(input.getShape()[2],
                         operation.getKernelW(),
                         operation.getStrideW(),
                         1,
                         sameLower);
    if (failed(height) || failed(width)) {
      operation.emitOpError("cannot resolve static SAME padding");
      return WalkResult::interrupt();
    }
    setI64(operation, "pad_top", height->before);
    setI64(operation, "pad_bottom", height->after);
    setI64(operation, "pad_left", width->before);
    setI64(operation, "pad_right", width->after);
    setI64(operation, "pad_mode", 1);
    return WalkResult::advance();
  }
};

}  // namespace

std::unique_ptr<Pass> createNormalizeNCNNPass() {
  return std::make_unique<NormalizeNCNNPass>();
}

void registerNormalizeNCNNPasses() {
  static PassRegistration<NormalizeNCNNPass> registration;
}

}  // namespace mlir::ncnn
