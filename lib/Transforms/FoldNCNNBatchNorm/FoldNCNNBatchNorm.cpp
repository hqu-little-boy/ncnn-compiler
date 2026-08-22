#include "ncnn-mlir/Transforms/FoldNCNNBatchNorm/FoldNCNNBatchNorm.hpp"

#include <cmath>
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

#define GEN_PASS_DEF_FOLDNCNNBATCHNORMPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

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

class FoldNCNNBatchNormPass final
  : public impl::FoldNCNNBatchNormPassBase<FoldNCNNBatchNormPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<ModelOp>().empty()) {
      module.emitError(
        "fold-ncnn-batchnorm requires ncnn.model to be converted to "
        "func.func first");
      signalPassFailure();
      return;
    }
    SmallVector<BatchNormOp> batchNorms;
    module.walk(
      [&](BatchNormOp batchNorm) { batchNorms.push_back(batchNorm); });
    if (batchNorms.empty()) {
      return;
    }
    PatternRewriter rewriter(module.getContext());
    for (BatchNormOp batchNorm : batchNorms) {
      Value input = batchNorm.getInput();
      if (auto convolution = input.getDefiningOp<ConvolutionOp>()) {
        foldBatchNormIntoConvolution(rewriter, batchNorm, convolution);
      } else if (auto depthwise =
                   input.getDefiningOp<ConvolutionDepthWiseOp>()) {
        foldBatchNormIntoConvolution(rewriter, batchNorm, depthwise);
      }
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
