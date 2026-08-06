#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_NORMALIZENCNNPASS
#include "ncnn-mlir/Passes.h.inc"

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

using PaddingMap = DenseMap<Operation*, std::array<int64_t, 4>>;

template <typename Op>
void setI64(Op operation, StringRef name, int64_t value) {
  operation->setAttr(name,
                     Builder(operation.getContext()).getI64IntegerAttr(value));
}

bool hasI64(Operation* operation, StringRef name, int64_t value) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  return attribute && attribute.getInt() == value;
}

template <typename Op>
class NormalizeAxis final : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op operation,
                                PatternRewriter& rewriter) const final {
    auto ranked = cast<RankedTensorType>(operation->getOperand(0).getType());
    int64_t axis = operation.getAxis();
    if (axis >= 0) {
      return failure();
    }
    axis += ranked.getRank();
    rewriter.modifyOpInPlace(operation, [&] {
      operation->setAttr("axis", rewriter.getI64IntegerAttr(axis));
    });
    return success();
  }
};

template <typename Op>
class NormalizeF32Attribute final : public OpRewritePattern<Op> {
 public:
  NormalizeF32Attribute(MLIRContext* context,
                        StringRef attributeName,
                        double defaultValue)
    : OpRewritePattern<Op>(context),
      attributeName_(attributeName),
      defaultValue_(defaultValue) {}

  LogicalResult matchAndRewrite(Op operation,
                                PatternRewriter& rewriter) const final {
    auto attribute =
      operation->template getAttrOfType<FloatAttr>(attributeName_);
    const double value =
      attribute ? attribute.getValue().convertToDouble() : defaultValue_;
    auto normalized = rewriter.getF32FloatAttr(value);
    if (attribute == normalized) {
      return failure();
    }
    rewriter.modifyOpInPlace(
      operation, [&] { operation->setAttr(attributeName_, normalized); });
    return success();
  }

 private:
  StringRef attributeName_;
  double defaultValue_;
};

class NormalizeConvolution final : public OpRewritePattern<ConvolutionOp> {
 public:
  NormalizeConvolution(MLIRContext* context, const PaddingMap& explicitPadding)
    : OpRewritePattern(context), explicitPadding_(&explicitPadding) {}

  LogicalResult matchAndRewrite(ConvolutionOp operation,
                                PatternRewriter& rewriter) const final {
    auto padding = explicitPadding_->find(operation);
    const bool normalizeScale =
      operation->getAttr("int8_scale_term") == nullptr;
    const bool normalizePadding =
      padding != explicitPadding_->end() &&
      (!hasI64(operation, "pad_top", padding->second[0]) ||
       !hasI64(operation, "pad_bottom", padding->second[1]) ||
       !hasI64(operation, "pad_left", padding->second[2]) ||
       !hasI64(operation, "pad_right", padding->second[3]));
    if (!normalizePadding && !normalizeScale) {
      return failure();
    }
    rewriter.modifyOpInPlace(operation, [&] {
      setI64(operation, "int8_scale_term", operation.getInt8ScaleTerm());
      if (padding != explicitPadding_->end()) {
        setI64(operation, "pad_top", padding->second[0]);
        setI64(operation, "pad_bottom", padding->second[1]);
        setI64(operation, "pad_left", padding->second[2]);
        setI64(operation, "pad_right", padding->second[3]);
      }
    });
    return success();
  }

 private:
  const PaddingMap* explicitPadding_;
};

class NormalizePooling final : public OpRewritePattern<PoolingOp> {
 public:
  NormalizePooling(MLIRContext* context, const PaddingMap& explicitPadding)
    : OpRewritePattern(context), explicitPadding_(&explicitPadding) {}

  LogicalResult matchAndRewrite(PoolingOp operation,
                                PatternRewriter& rewriter) const final {
    auto padding = explicitPadding_->find(operation);
    if (padding == explicitPadding_->end()) {
      return failure();
    }
    if (hasI64(operation, "pad_top", padding->second[0]) &&
        hasI64(operation, "pad_bottom", padding->second[1]) &&
        hasI64(operation, "pad_left", padding->second[2]) &&
        hasI64(operation, "pad_right", padding->second[3]) &&
        hasI64(operation, "pad_mode", 1)) {
      return failure();
    }
    rewriter.modifyOpInPlace(operation, [&] {
      setI64(operation, "pad_top", padding->second[0]);
      setI64(operation, "pad_bottom", padding->second[1]);
      setI64(operation, "pad_left", padding->second[2]);
      setI64(operation, "pad_right", padding->second[3]);
      setI64(operation, "pad_mode", 1);
    });
    return success();
  }

 private:
  const PaddingMap* explicitPadding_;
};

class NormalizeNCNNPass final
  : public impl::NormalizeNCNNPassBase<NormalizeNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<ModelOp>().empty()) {
      module.emitError(
        "normalize-ncnn requires ncnn.model to be converted to "
        "func.func first");
      signalPassFailure();
      return;
    }

    PaddingMap explicitPadding;
    WalkResult validation = module.walk([&](Operation* operation) {
      if (operation->getDialect() == nullptr ||
          operation->getDialect()->getNamespace() != "ncnn") {
        return WalkResult::advance();
      }
      if (operation->getParentOfType<func::FuncOp>() == nullptr) {
        operation->emitOpError("requires a func.func boundary");
        return WalkResult::interrupt();
      }
      return TypeSwitch<Operation*, WalkResult>(operation)
        .Case<ConvolutionOp>([&](ConvolutionOp convolution) {
          return validateConvolution(convolution, explicitPadding);
        })
        .Case<PoolingOp>([&](PoolingOp pooling) {
          return validatePooling(pooling, explicitPadding);
        })
        .Default([](Operation*) { return WalkResult::advance(); });
    });
    if (validation.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    applyPattern(module, NormalizeConvolution(&getContext(), explicitPadding));
    applyPattern(module, NormalizePooling(&getContext(), explicitPadding));
    applyPattern(module, NormalizeAxis<ConcatOp>(&getContext()));
    applyPattern(module, NormalizeAxis<SoftmaxOp>(&getContext()));
    applyPattern(
      module,
      NormalizeF32Attribute<ReluOp>(&getContext(), "negative_slope", 0.0));
    applyPattern(module,
                 NormalizeF32Attribute<DropoutOp>(&getContext(), "scale", 1.0));
  }

 private:
  template <typename Op>
  static void applyPattern(ModuleOp module, OpRewritePattern<Op>&& pattern) {
    PatternRewriter rewriter(module.getContext());
    module.walk([&](Op operation) {
      rewriter.setInsertionPoint(operation);
      (void)pattern.matchAndRewrite(operation, rewriter);
    });
  }

  static WalkResult validateConvolution(ConvolutionOp operation,
                                        PaddingMap& explicitPadding) {
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
    explicitPadding[operation] = {
      height->before, height->after, width->before, width->after};
    return WalkResult::advance();
  }

  static WalkResult validatePooling(PoolingOp operation,
                                    PaddingMap& explicitPadding) {
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
    explicitPadding[operation] = {
      height->before, height->after, width->before, width->after};
    return WalkResult::advance();
  }
};

}  // namespace

}  // namespace mlir::ncnn
