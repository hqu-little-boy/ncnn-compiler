#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  const int64_t before = sameLower ? (total / 2) + (total % 2) : total / 2;
  return ExplicitPadding{.before = before, .after = total - before};
}

FailureOr<ExplicitPadding> resolveSamePadding(int64_t input,
                                              int64_t kernel,
                                              int64_t stride,
                                              int64_t dilation,
                                              bool sameLower) {
  if (!ShapedType::isDynamic(input)) {
    return computeSamePadding(input, kernel, stride, dilation, sameLower);
  }
  if (kernel <= 0 || stride != 1 || dilation <= 0 ||
      kernel - 1 > std::numeric_limits<int64_t>::max() / dilation) {
    return failure();
  }
  const int64_t total = dilation * (kernel - 1);
  const int64_t before = sameLower ? (total / 2) + (total % 2) : total / 2;
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

ElementsAttr getConstantElements(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant) {
    return {};
  }
  return dyn_cast<ElementsAttr>(constant.getValue());
}

DenseElementsAttr scaleConstantChannels(ElementsAttr elements,
                                        RankedTensorType type,
                                        ArrayRef<float> factors) {
  auto dense = dyn_cast<DenseElementsAttr>(elements);
  if (!dense || !type.hasStaticShape() || type.getRank() < 1 ||
      !type.getElementType().isF32() ||
      factors.size() != static_cast<size_t>(type.getShape()[0])) {
    return {};
  }
  const int64_t channels = type.getShape()[0];
  const int64_t count = type.getNumElements();
  const int64_t rowSize = count / channels;
  std::vector<APFloat> scaled;
  scaled.reserve(count);
  for (auto [index, value] : llvm::enumerate(dense.getValues<APFloat>())) {
    if (!value.isFinite()) {
      return {};
    }
    scaled.emplace_back(value.convertToFloat() * (factors[index / rowSize]));
  }
  return DenseElementsAttr::get(type, scaled);
}

template <typename Conv>
Operation* foldBatchNormIntoConvolution(PatternRewriter& rewriter,
                                        BatchNormOp batchNorm,
                                        Conv convolution) {
  auto outputType =
    dyn_cast<RankedTensorType>(convolution.getOutput().getType());
  auto weightType =
    dyn_cast<RankedTensorType>(convolution.getWeight().getType());
  if (!outputType || !weightType || convolution.getInt8ScaleTerm() != 0 ||
      outputType.getRank() != 3 || weightType.getRank() != 4 ||
      weightType.getElementType() != rewriter.getF32Type() ||
      !outputType.hasStaticShape() || !weightType.hasStaticShape() ||
      !convolution.getOutput().hasOneUse()) {
    return nullptr;
  }
  const int64_t channels = outputType.getShape()[0];
  if (channels <= 0 || weightType.getShape()[0] != channels ||
      batchNorm.getSlope().getType() != batchNorm.getMean().getType() ||
      batchNorm.getVariance().getType() != batchNorm.getBias().getType()) {
    return nullptr;
  }
  auto parameterType =
    dyn_cast<RankedTensorType>(batchNorm.getSlope().getType());
  if (!parameterType || !parameterType.hasStaticShape() ||
      parameterType.getRank() != 1 || parameterType.getShape()[0] != channels ||
      parameterType.getElementType() != rewriter.getF32Type()) {
    return nullptr;
  }
  ElementsAttr slope = getConstantElements(batchNorm.getSlope());
  ElementsAttr mean = getConstantElements(batchNorm.getMean());
  ElementsAttr variance = getConstantElements(batchNorm.getVariance());
  ElementsAttr bias = getConstantElements(batchNorm.getBias());
  ElementsAttr weight = getConstantElements(convolution.getWeight());
  if (!slope || !mean || !variance || !bias || !weight) {
    return nullptr;
  }
  auto slopeValues = dyn_cast<DenseElementsAttr>(slope);
  auto meanValues = dyn_cast<DenseElementsAttr>(mean);
  auto varianceValues = dyn_cast<DenseElementsAttr>(variance);
  auto biasValues = dyn_cast<DenseElementsAttr>(bias);
  if (!slopeValues || !meanValues || !varianceValues || !biasValues ||
      !isa<DenseElementsAttr>(weight)) {
    return nullptr;
  }
  const float epsilon = batchNorm.getEpsilon().convertToFloat();
  std::vector<float> factors(channels);
  std::vector<float> offsets(channels);
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip(slopeValues.getValues<APFloat>(),
                                 meanValues.getValues<APFloat>(),
                                 varianceValues.getValues<APFloat>(),
                                 biasValues.getValues<APFloat>()))) {
    const float slope = std::get<0>(pair).convertToFloat();
    const float channelMean = std::get<1>(pair).convertToFloat();
    const float variance = std::get<2>(pair).convertToFloat();
    const float channelBias = std::get<3>(pair).convertToFloat();
    const float varianceWithEpsilon = variance + epsilon;
    const float inverseStd = varianceWithEpsilon == 0.0F
                               ? 10000.0F
                               : std::pow(varianceWithEpsilon, -0.5F);
    if (!std::isfinite(inverseStd)) {
      return nullptr;
    }
    const float scale = slope * inverseStd;
    factors[index] = scale;
    offsets[index] = channelBias - (scale * channelMean);
  }
  DenseElementsAttr foldedWeight =
    scaleConstantChannels(weight, weightType, factors);
  if (!foldedWeight) {
    return nullptr;
  }
  ElementsAttr originalBias =
    convolution.getHasBias()
      ? getConstantElements(convolution.getBiasAndScales().front())
      : ElementsAttr();
  DenseElementsAttr foldedBias;
  auto biasType = RankedTensorType::get({channels}, rewriter.getF32Type());
  if (originalBias) {
    auto denseBias = dyn_cast<DenseElementsAttr>(originalBias);
    auto originalBiasType = dyn_cast<RankedTensorType>(
      convolution.getBiasAndScales().front().getType());
    if (!denseBias || !originalBiasType || !originalBiasType.hasStaticShape() ||
        originalBiasType.getNumElements() != channels ||
        !originalBiasType.getElementType().isF32()) {
      return nullptr;
    }
    SmallVector<APFloat> shifted;
    shifted.reserve(channels);
    for (auto [index, value] :
         llvm::enumerate(denseBias.getValues<APFloat>())) {
      if (!value.isFinite()) {
        return nullptr;
      }
      shifted.emplace_back((value.convertToFloat() * factors[index]) +
                           offsets[index]);
    }
    foldedBias = DenseElementsAttr::get(biasType, shifted);
  } else {
    SmallVector<APFloat> shifted;
    shifted.reserve(channels);
    llvm::transform(offsets, std::back_inserter(shifted), [](float offset) {
      return APFloat(offset);
    });
    foldedBias = DenseElementsAttr::get(biasType, shifted);
  }
  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(convolution);
  Value foldedWeightValue = rewriter.create<arith::ConstantOp>(
    convolution.getLoc(), foldedWeight.getType(), foldedWeight);
  Value foldedBiasValue = rewriter.create<arith::ConstantOp>(
    convolution.getLoc(), foldedBias.getType(), foldedBias);
  Operation* replacement = rewriter.clone(*convolution);
  replacement->setOperand(1, foldedWeightValue);
  if (convolution.getHasBias()) {
    replacement->setOperand(2, foldedBiasValue);
  } else {
    replacement->insertOperands(2, {foldedBiasValue});
    replacement->setAttr("has_bias", rewriter.getBoolAttr(true));
  }
  rewriter.replaceOp(batchNorm, replacement->getResult(0));
  rewriter.eraseOp(convolution);
  return replacement;
}

void foldBatchNorms(ModuleOp module) {
  SmallVector<BatchNormOp> batchNorms;
  module.walk([&](BatchNormOp batchNorm) { batchNorms.push_back(batchNorm); });
  if (batchNorms.empty()) {
    return;
  }
  PatternRewriter rewriter(module.getContext());
  for (BatchNormOp batchNorm : batchNorms) {
    Value input = batchNorm.getInput();
    if (auto convolution = input.getDefiningOp<ConvolutionOp>()) {
      foldBatchNormIntoConvolution(rewriter, batchNorm, convolution);
    } else if (auto depthwise = input.getDefiningOp<ConvolutionDepthWiseOp>()) {
      foldBatchNormIntoConvolution(rewriter, batchNorm, depthwise);
    }
  }
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

template <typename Op>
class NormalizeI64Attribute final : public OpRewritePattern<Op> {
 public:
  NormalizeI64Attribute(MLIRContext* context,
                        StringRef attributeName,
                        int64_t defaultValue)
    : OpRewritePattern<Op>(context),
      attributeName_(attributeName),
      defaultValue_(defaultValue) {}

  LogicalResult matchAndRewrite(Op operation,
                                PatternRewriter& rewriter) const final {
    if (operation->getAttr(attributeName_)) {
      return failure();
    }
    rewriter.modifyOpInPlace(operation, [&] {
      operation->setAttr(attributeName_,
                         rewriter.getI64IntegerAttr(defaultValue_));
    });
    return success();
  }

 private:
  StringRef attributeName_;
  int64_t defaultValue_;
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

class NormalizeDepthwiseConvolution final
  : public OpRewritePattern<ConvolutionDepthWiseOp> {
 public:
  NormalizeDepthwiseConvolution(MLIRContext* context,
                                const PaddingMap& explicitPadding)
    : OpRewritePattern(context), explicitPadding_(&explicitPadding) {}

  LogicalResult matchAndRewrite(ConvolutionDepthWiseOp operation,
                                PatternRewriter& rewriter) const final {
    auto padding = explicitPadding_->find(operation);
    const bool normalizeScale =
      operation->getAttr("int8_scale_term") == nullptr;
    if (padding == explicitPadding_->end() && !normalizeScale) {
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

    foldBatchNorms(module);

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
        .Case<ConvolutionDepthWiseOp>([&](ConvolutionDepthWiseOp convolution) {
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
    applyPattern(module,
                 NormalizeDepthwiseConvolution(&getContext(), explicitPadding));
    applyPattern(module, NormalizePooling(&getContext(), explicitPadding));
    applyPattern(module,
                 NormalizeI64Attribute<InnerProductOp>(
                   &getContext(), "int8_scale_term", 0));
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

  template <typename Op>
  static WalkResult validateConvolution(Op operation,
                                        PaddingMap& explicitPadding) {
    const int64_t pad = operation.getPadTop();
    if (pad != -233 && pad != -234) {
      return WalkResult::advance();
    }

    auto input = cast<RankedTensorType>(operation.getInput().getType());
    const bool sameLower = pad == -234;
    FailureOr<ExplicitPadding> height =
      resolveSamePadding(input.getShape()[1],
                         operation.getKernelH(),
                         operation.getStrideH(),
                         operation.getDilationH(),
                         sameLower);
    FailureOr<ExplicitPadding> width =
      resolveSamePadding(input.getShape()[2],
                         operation.getKernelW(),
                         operation.getStrideW(),
                         operation.getDilationW(),
                         sameLower);
    if (failed(height) || failed(width)) {
      operation.emitOpError(
        "cannot resolve SAME padding; dynamic dimensions require stride 1");
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
