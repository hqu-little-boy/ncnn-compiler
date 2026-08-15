#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::ncnn {
namespace {

// 与原自定义 IR 的 InferSupport 一致：溢出安全的整数算术。失败返回 failure()，
// 由调用方附上语境错误。
FailureOr<int64_t> checkedAdd(int64_t left, int64_t right) {
  if ((right > 0 && left > std::numeric_limits<int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<int64_t>::min() - right)) {
    return failure();
  }
  return left + right;
}

FailureOr<int64_t> checkedMultiply(int64_t left, int64_t right) {
  if (left < 0 || right < 0) {
    return failure();
  }
  if (left != 0 && right > std::numeric_limits<int64_t>::max() / left) {
    return failure();
  }
  return left * right;
}

// Convolution 结果类型推断（含全部结构校验）。inferReturnTypeComponents 与
// verify 共用，避免两处重复。
FailureOr<RankedTensorType> computeConvResult(MLIRContext* context,
                                              std::optional<Location> location,
                                              RankedTensorType input,
                                              RankedTensorType weight,
                                              ValueRange tail,
                                              bool hasBias,
                                              int64_t term,
                                              int64_t kernelH,
                                              int64_t kernelW,
                                              int64_t strideH,
                                              int64_t strideW,
                                              int64_t dilationH,
                                              int64_t dilationW,
                                              int64_t padTop,
                                              int64_t padBottom,
                                              int64_t padLeft,
                                              int64_t padRight,
                                              int64_t weightScaleSize) {
  auto fail = [&](const Twine& msg) -> FailureOr<RankedTensorType> {
    return emitOptionalError(location, msg);
  };
  if (term != 0 && term != 1 && term != 2 && term != 101 && term != 102) {
    return fail("Convolution has unsupported int8_scale_term");
  }
  const bool quantized = term != 0;
  const bool requantized = term > 100;
  const std::size_t minimumOperands = hasBias ? 3 : 2;
  const std::size_t expectedOperands =
    minimumOperands + (quantized ? 2 : 0) + (requantized ? 1 : 0);
  if (2 + tail.size() != expectedOperands) {
    return fail("Convolution operands do not match bias and quantization mode");
  }
  Type inElem = input.getElementType();
  if (quantized) {
    if (!inElem.isF32() && !inElem.isInteger(8)) {
      return fail("quantized convolution input must be f32 or i8");
    }
  } else if (!inElem.isF32()) {
    return fail("non-quantized convolution input must be f32");
  }
  if (input.getRank() != 3) {
    return fail("convolution input must have [C,H,W] layout");
  }
  if (weight.getRank() != 4) {
    return fail("convolution weight must have [O,I,H,W] layout");
  }
  Type wElem = weight.getElementType();
  if (!quantized && wElem.isInteger(8)) {
    return fail("non-quantized convolution weight cannot be i8");
  }
  if (quantized && wElem.isF16()) {
    return fail("quantized convolution weight must be f32 or i8");
  }
  if (!wElem.isF32() && !wElem.isF16() && !wElem.isBF16() &&
      !wElem.isInteger(8)) {
    return fail("convolution weight has unsupported element type");
  }
  ArrayRef<int64_t> inShape = input.getShape();
  ArrayRef<int64_t> wShape = weight.getShape();
  for (int64_t dim : inShape) {
    if (!ShapedType::isDynamic(dim) && dim <= 0) {
      return fail("convolution input dimensions must be positive or dynamic");
    }
  }
  for (int64_t dim : wShape) {
    if (dim <= 0) {
      return fail("convolution weight dimensions must be positive");
    }
  }
  if (!ShapedType::isDynamic(inShape[0]) && wShape[1] != inShape[0]) {
    return fail("convolution input channels do not match weight channels");
  }
  if (wShape[2] != kernelH || wShape[3] != kernelW) {
    return fail("convolution kernel attributes do not match weight");
  }
  if (hasBias) {
    auto bias = llvm::dyn_cast<RankedTensorType>(tail[0].getType());
    if (bias == nullptr || bias.getRank() != 1 ||
        bias.getShape()[0] != wShape[0]) {
      return fail("convolution bias must have shape [O]");
    }
    if (!bias.getElementType().isF32()) {
      return fail("convolution bias must be f32");
    }
  }
  const std::size_t scaleCount = (2 + tail.size()) - minimumOperands;
  for (std::size_t scaleIndex = 0; scaleIndex < scaleCount; ++scaleIndex) {
    Value scaleValue = tail[minimumOperands - 2 + scaleIndex];
    auto scale = llvm::dyn_cast<RankedTensorType>(scaleValue.getType());
    const int64_t expectedSize = scaleIndex == 0 ? weightScaleSize : 1;
    if (scale == nullptr || scale.getRank() != 1 ||
        scale.getShape()[0] != expectedSize) {
      return fail("convolution scale has the wrong role or shape");
    }
    if (!scale.getElementType().isF32()) {
      return fail("convolution scale must be f32");
    }
  }
  const std::pair<int64_t, const char*> dims[] = {
    {kernelH, "convolution kernel height"},
    {kernelW, "convolution kernel width"},
    {strideH, "convolution stride height"},
    {strideW, "convolution stride width"},
    {dilationH, "convolution dilation height"},
    {dilationW, "convolution dilation width"}};
  for (const auto& [value, name] : dims) {
    if (value <= 0) {
      return fail(Twine(name) + " must be positive");
    }
  }
  const int64_t pads[] = {padTop, padBottom, padLeft, padRight};
  bool sameUpper = true;
  bool sameLower = true;
  for (int64_t pad : pads) {
    if (pad < 0 && pad != -233 && pad != -234) {
      return fail("convolution padding has unsupported value");
    }
    sameUpper = sameUpper && pad == -233;
    sameLower = sameLower && pad == -234;
  }
  if ((!sameUpper && !sameLower) &&
      (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0)) {
    return fail("convolution SAME padding must use one pad mode");
  }
  FailureOr<int64_t> extentHeight = checkedMultiply(dilationH, kernelH - 1);
  if (failed(extentHeight)) {
    return fail("convolution kernel height extent overflows");
  }
  extentHeight = checkedAdd(*extentHeight, 1);
  if (failed(extentHeight)) {
    return fail("convolution kernel height overflows");
  }
  FailureOr<int64_t> extentWidth = checkedMultiply(dilationW, kernelW - 1);
  if (failed(extentWidth)) {
    return fail("convolution kernel width extent overflows");
  }
  extentWidth = checkedAdd(*extentWidth, 1);
  if (failed(extentWidth)) {
    return fail("convolution kernel width overflows");
  }
  int64_t outputHeight = 0;
  int64_t outputWidth = 0;
  if (ShapedType::isDynamic(inShape[1])) {
    outputHeight = ShapedType::kDynamic;
  } else if (sameUpper || sameLower) {
    outputHeight = 1 + ((inShape[1] - 1) / strideH);
  } else {
    FailureOr<int64_t> height = checkedAdd(inShape[1], padTop);
    height = succeeded(height) ? checkedAdd(*height, padBottom) : height;
    if (failed(height) || *height < *extentHeight) {
      return fail("convolution kernel exceeds padded input height");
    }
    outputHeight = 1 + ((*height - *extentHeight) / strideH);
  }
  if (ShapedType::isDynamic(inShape[2])) {
    outputWidth = ShapedType::kDynamic;
  } else if (sameUpper || sameLower) {
    outputWidth = 1 + ((inShape[2] - 1) / strideW);
  } else {
    FailureOr<int64_t> width = checkedAdd(inShape[2], padLeft);
    width = succeeded(width) ? checkedAdd(*width, padRight) : width;
    if (failed(width) || *width < *extentWidth) {
      return fail("convolution kernel exceeds padded input width");
    }
    outputWidth = 1 + ((*width - *extentWidth) / strideW);
  }
  Type resultElem = requantized
                      ? static_cast<Type>(IntegerType::get(context, 8))
                      : static_cast<Type>(Float32Type::get(context));
  SmallVector<int64_t> shape = {wShape[0], outputHeight, outputWidth};
  return RankedTensorType::get(shape, resultElem);
}

FailureOr<RankedTensorType> computeDeconvResult(
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType weight,
  ValueRange bias,
  bool hasBias,
  int64_t kernelH,
  int64_t kernelW,
  int64_t strideH,
  int64_t strideW,
  int64_t dilationH,
  int64_t dilationW,
  int64_t padTop,
  int64_t padBottom,
  int64_t padLeft,
  int64_t padRight,
  int64_t outputPadBottom,
  int64_t outputPadRight) {
  auto fail = [&](const Twine& message) -> FailureOr<RankedTensorType> {
    return emitOptionalError(location, message);
  };
  if (!input || !weight || !input.getElementType().isF32() ||
      !isa<FloatType>(weight.getElementType()) || !weight.hasStaticShape() ||
      input.getRank() != 3 || weight.getRank() != 4) {
    return fail(
      "Deconvolution requires FP32 input [I,H,W] and static floating weight "
      "[O,I,K_h,K_w]");
  }
  if (ShapedType::isDynamic(input.getShape()[0]) ||
      weight.getShape()[1] != input.getShape()[0] ||
      weight.getShape()[2] != kernelH || weight.getShape()[3] != kernelW) {
    return fail(
      "Deconvolution input channels or kernel attributes do not "
      "match weight");
  }
  for (int64_t dimension : input.getShape()) {
    if (!ShapedType::isDynamic(dimension) && dimension <= 0) {
      return fail("Deconvolution input dimensions must be positive or dynamic");
    }
  }
  if (kernelH <= 0 || kernelW <= 0 || strideH <= 0 || strideW <= 0 ||
      dilationH <= 0 || dilationW <= 0 || padTop < 0 || padBottom < 0 ||
      padLeft < 0 || padRight < 0 || outputPadBottom < 0 ||
      outputPadRight < 0) {
    return fail(
      "Deconvolution kernel, stride, dilation, or padding is invalid");
  }
  if (hasBias) {
    if (bias.size() != 1) {
      return fail("Deconvolution bias is required");
    }
    auto type = dyn_cast<RankedTensorType>(bias.front().getType());
    if (!type || !type.getElementType().isF32() || type.getRank() != 1 ||
        type.getShape()[0] != weight.getShape()[0]) {
      return fail("Deconvolution bias must be FP32 [O]");
    }
  } else if (!bias.empty()) {
    return fail("Deconvolution has unexpected bias");
  }
  auto inferDimension = [&](int64_t inputSize,
                            int64_t kernel,
                            int64_t stride,
                            int64_t dilation,
                            int64_t padBefore,
                            int64_t padAfter,
                            int64_t outputPad) -> FailureOr<int64_t> {
    if (ShapedType::isDynamic(inputSize)) {
      return ShapedType::kDynamic;
    }
    auto result = checkedMultiply(inputSize - 1, stride);
    auto extent = checkedMultiply(kernel - 1, dilation);
    if (failed(result) || failed(extent)) {
      return failure();
    }
    result = checkedAdd(*result, *extent);
    if (succeeded(result)) {
      result = checkedAdd(*result, 1);
    }
    if (succeeded(result)) {
      result = checkedAdd(*result, outputPad);
    }
    if (succeeded(result)) {
      result = checkedAdd(*result, -padBefore);
    }
    if (succeeded(result)) {
      result = checkedAdd(*result, -padAfter);
    }
    if (failed(result) || *result <= 0) {
      return failure();
    }
    return result;
  };
  auto outputH = inferDimension(input.getShape()[1],
                                kernelH,
                                strideH,
                                dilationH,
                                padTop,
                                padBottom,
                                outputPadBottom);
  auto outputW = inferDimension(input.getShape()[2],
                                kernelW,
                                strideW,
                                dilationW,
                                padLeft,
                                padRight,
                                outputPadRight);
  if (failed(outputH) || failed(outputW)) {
    return fail("Deconvolution output dimension is invalid or overflows");
  }
  return RankedTensorType::get({weight.getShape()[0], *outputH, *outputW},
                               input.getElementType());
}

FailureOr<RankedTensorType> computePaddingResult(
  std::optional<Location> location,
  RankedTensorType input,
  int64_t top,
  int64_t bottom,
  int64_t left,
  int64_t right) {
  if (!input || !input.getElementType().isF32() || input.getRank() != 3) {
    return emitOptionalError(location,
                             "Padding input must be an FP32 CHW tensor");
  }
  if (top < 0 || bottom < 0 || left < 0 || right < 0) {
    return emitOptionalError(location, "Padding extents must be non-negative");
  }
  auto addPadding = [&](unsigned dimension,
                        int64_t before,
                        int64_t after) -> FailureOr<int64_t> {
    if (input.isDynamicDim(dimension)) {
      return ShapedType::kDynamic;
    }
    auto result = checkedAdd(input.getShape()[dimension], before);
    if (succeeded(result)) {
      result = checkedAdd(*result, after);
    }
    return result;
  };
  auto height = addPadding(1, top, bottom);
  auto width = addPadding(2, left, right);
  if (failed(height) || failed(width) ||
      (!ShapedType::isDynamic(*height) && *height <= 0) ||
      (!ShapedType::isDynamic(*width) && *width <= 0)) {
    return emitOptionalError(location, "Padding output dimension overflows");
  }
  return RankedTensorType::get({input.getShape()[0], *height, *width},
                               input.getElementType());
}

FailureOr<RankedTensorType> computeInterpResult(
  std::optional<Location> location,
  RankedTensorType input,
  int64_t heightScale,
  int64_t widthScale,
  int64_t outputH,
  int64_t outputW) {
  if (!input || !input.getElementType().isF32() || input.getRank() != 3 ||
      heightScale <= 0 || widthScale <= 0 || outputH < 0 || outputW < 0) {
    return emitOptionalError(
      location,
      "Interp requires an FP32 CHW input, positive scales, and non-negative "
      "output dimensions");
  }
  auto scaleDimension = [&](unsigned dimension,
                            int64_t scale) -> FailureOr<int64_t> {
    return input.isDynamicDim(dimension)
             ? FailureOr<int64_t>(ShapedType::kDynamic)
             : checkedMultiply(input.getShape()[dimension], scale);
  };
  FailureOr<int64_t> height =
    outputH == 0 ? scaleDimension(1, heightScale) : outputH;
  FailureOr<int64_t> width =
    outputW == 0 ? scaleDimension(2, widthScale) : outputW;
  if (failed(height) || failed(width)) {
    return emitOptionalError(location, "Interp output dimension overflows");
  }
  return RankedTensorType::get({input.getShape()[0], *height, *width},
                               input.getElementType());
}

FailureOr<RankedTensorType> computeDetectionOutputResult(
  std::optional<Location> location,
  RankedTensorType locationType,
  RankedTensorType confidenceType,
  RankedTensorType priorboxType,
  int64_t numClass,
  int64_t nmsTopK,
  int64_t keepTopK,
  double nmsThreshold,
  double confidenceThreshold,
  ArrayRef<double> variances) {
  auto fail = [&](const Twine& message) -> FailureOr<RankedTensorType> {
    return emitOptionalError(location, message);
  };
  for (RankedTensorType type : {locationType, confidenceType, priorboxType}) {
    if (!type || !type.getElementType().isF32() || !type.hasStaticShape()) {
      return fail("DetectionOutput requires static FP32 inputs");
    }
  }
  if (numClass <= 1 || nmsTopK <= 0 || keepTopK <= 0 ||
      !std::isfinite(nmsThreshold) || nmsThreshold < 0.0 ||
      !std::isfinite(confidenceThreshold) || confidenceThreshold < 0.0 ||
      confidenceThreshold > 1.0 || llvm::any_of(variances, [](double value) {
        return !std::isfinite(value) || value <= 0.0;
      })) {
    return fail("DetectionOutput parameters are invalid");
  }
  const int64_t locationElements = locationType.getNumElements();
  const int64_t confidenceElements = confidenceType.getNumElements();
  const int64_t priorboxElements = priorboxType.getNumElements();
  if (locationElements <= 0 || locationElements % 4 != 0) {
    return fail("DetectionOutput location element count must be 4 * num_prior");
  }
  const int64_t numPrior = locationElements / 4;
  const ArrayRef<int64_t> priorShape = priorboxType.getShape();
  if (confidenceElements != numPrior * numClass || priorboxType.getRank() < 2 ||
      priorShape[priorboxType.getRank() - 2] != 2 ||
      priorShape.back() != locationElements ||
      priorboxElements != locationElements * 2) {
    return fail("DetectionOutput input shapes are inconsistent");
  }
  auto perClass = checkedMultiply(numClass - 1, std::min(nmsTopK, numPrior));
  if (failed(perClass)) {
    return fail("DetectionOutput maximum detection count overflows");
  }
  const int64_t maximumDetections = std::min(keepTopK, *perClass);
  return RankedTensorType::get({maximumDetections, 6},
                               locationType.getElementType());
}

// Pooling 单个空间维的输出尺寸（regular 模式）。
FailureOr<int64_t> inferRegularDimension(int64_t input,
                                         int64_t kernel,
                                         int64_t stride,
                                         int64_t padBefore,
                                         int64_t padAfter,
                                         int64_t padMode,
                                         const Twine& name,
                                         std::optional<Location> location) {
  if (kernel <= 0) {
    return emitOptionalError(location, name, " kernel must be positive");
  }
  if (stride <= 0) {
    return emitOptionalError(location, name, " stride must be positive");
  }
  if (padBefore < 0 || padAfter < 0) {
    return emitOptionalError(location, name, " padding must be non-negative");
  }
  if (ShapedType::isDynamic(input)) {
    return ShapedType::kDynamic;
  }
  if (input <= 0) {
    return emitOptionalError(location, name, " input must be positive");
  }
  if (padMode == 2 || padMode == 3) {
    return 1 + ((input - 1) / stride);
  }
  FailureOr<int64_t> numerator = checkedAdd(input, padBefore);
  if (succeeded(numerator)) {
    numerator = checkedAdd(*numerator, padAfter);
  }
  if (failed(numerator)) {
    return emitOptionalError(location, name, " padded dimension overflows");
  }
  if (padMode == 0) {
    const int64_t difference = *numerator - kernel;
    const int64_t quotient = difference / stride;
    const int64_t remainder = difference % stride;
    return (remainder == 0 ? 1 : 2) + quotient;
  }
  if (*numerator < kernel) {
    return emitOptionalError(location, name, " kernel exceeds padded input");
  }
  return 1 + ((*numerator - kernel) / stride);
}

FailureOr<RankedTensorType> computePoolResult(std::optional<Location> location,
                                              RankedTensorType input,
                                              PoolKind kind,
                                              PoolMode mode,
                                              int64_t kernelH,
                                              int64_t kernelW,
                                              int64_t strideH,
                                              int64_t strideW,
                                              int64_t padTop,
                                              int64_t padBottom,
                                              int64_t padLeft,
                                              int64_t padRight,
                                              int64_t padMode,
                                              bool includePad) {
  (void)includePad;
  auto fail = [&](const Twine& msg) -> FailureOr<RankedTensorType> {
    return emitOptionalError(location, msg);
  };
  if (input.getRank() != 3) {
    return fail("pooling input must have [C,H,W] layout");
  }
  if (kind != PoolKind::Maximum && kind != PoolKind::Average) {
    return fail("pooling kind is invalid");
  }
  const bool signedI8 = input.getElementType().isSignlessInteger(8);
  if (!input.getElementType().isF32() && !signedI8) {
    return fail("pooling input must be f32 or signed i8");
  }
  if (signedI8 && kind != PoolKind::Maximum) {
    return fail("signed i8 pooling only supports maximum");
  }
  ArrayRef<int64_t> inShape = input.getShape();
  if (!ShapedType::isDynamic(inShape[0]) && inShape[0] <= 0) {
    return fail("pooling input channels must be positive or dynamic");
  }
  for (int64_t dim : inShape.drop_front()) {
    if (!ShapedType::isDynamic(dim) && dim <= 0) {
      return fail("pooling spatial dimensions must be positive or dynamic");
    }
  }
  Type elem = input.getElementType();
  if (mode == PoolMode::Global) {
    return RankedTensorType::get(SmallVector<int64_t>{inShape[0]}, elem);
  }
  if (mode == PoolMode::Adaptive) {
    const int64_t outputHeight = kernelH == -233 ? inShape[1] : kernelH;
    const int64_t outputWidth = kernelW == -233 ? inShape[2] : kernelW;
    if ((!ShapedType::isDynamic(outputHeight) && outputHeight <= 0) ||
        (!ShapedType::isDynamic(outputWidth) && outputWidth <= 0)) {
      return fail("adaptive pooling output dimensions must be positive");
    }
    return RankedTensorType::get(
      SmallVector<int64_t>{inShape[0], outputHeight, outputWidth}, elem);
  }
  if (mode != PoolMode::Regular || padMode < 0 || padMode > 3) {
    return fail("pooling mode or pad mode is invalid");
  }
  FailureOr<int64_t> outputHeight = inferRegularDimension(inShape[1],
                                                          kernelH,
                                                          strideH,
                                                          padTop,
                                                          padBottom,
                                                          padMode,
                                                          "pooling height",
                                                          location);
  if (failed(outputHeight)) {
    return failure();
  }
  FailureOr<int64_t> outputWidth = inferRegularDimension(inShape[2],
                                                         kernelW,
                                                         strideW,
                                                         padLeft,
                                                         padRight,
                                                         padMode,
                                                         "pooling width",
                                                         location);
  if (failed(outputWidth)) {
    return failure();
  }
  return RankedTensorType::get(
    SmallVector<int64_t>{inShape[0], *outputHeight, *outputWidth}, elem);
}

FailureOr<RankedTensorType> computeConcatResult(
  std::optional<Location> location, ValueRange inputs, int64_t axis) {
  auto fail = [&](const Twine& msg) -> FailureOr<RankedTensorType> {
    return emitOptionalError(location, msg);
  };
  if (inputs.size() < 2) {
    return fail("Concat requires at least two operands");
  }
  auto first = llvm::dyn_cast<RankedTensorType>(inputs[0].getType());
  if (first == nullptr) {
    return fail("Concat operands must be ranked tensors");
  }
  const int64_t rank = first.getRank();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return fail("Concat axis is outside operand rank");
  }
  SmallVector<int64_t> shape(first.getShape().begin(), first.getShape().end());
  for (std::size_t operandIndex = 1; operandIndex < inputs.size();
       ++operandIndex) {
    auto operand =
      llvm::dyn_cast<RankedTensorType>(inputs[operandIndex].getType());
    if (operand == nullptr) {
      return fail("Concat operands must be ranked tensors");
    }
    if (operand.getElementType() != first.getElementType() ||
        operand.getRank() != rank) {
      return fail("Concat operands must have matching types and rank");
    }
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      if (dimension == axis) {
        continue;
      }
      const int64_t current = shape[dimension];
      const int64_t candidate = operand.getShape()[dimension];
      if (!ShapedType::isDynamic(current) &&
          !ShapedType::isDynamic(candidate) && current != candidate) {
        return fail("Concat non-axis dimensions must match");
      }
      if (ShapedType::isDynamic(current) || ShapedType::isDynamic(candidate)) {
        shape[dimension] = ShapedType::kDynamic;
      }
    }
    if (ShapedType::isDynamic(shape[axis]) || operand.isDynamicDim(axis)) {
      shape[axis] = ShapedType::kDynamic;
      continue;
    }
    FailureOr<int64_t> sum =
      checkedAdd(shape[static_cast<std::size_t>(axis)],
                 operand.getShape()[static_cast<std::size_t>(axis)]);
    if (failed(sum)) {
      return fail("Concat axis dimension overflows");
    }
    shape[static_cast<std::size_t>(axis)] = *sum;
  }
  return RankedTensorType::get(shape, first.getElementType());
}

FailureOr<RankedTensorType> computeReshapeResult(
  std::optional<Location> location,
  RankedTensorType input,
  ArrayRef<int64_t> shapeRef) {
  if (!input || !input.getElementType().isF32()) {
    return emitOptionalError(location, "Reshape input must be f32");
  }
  int64_t inputCount = 1;
  bool dynamicInputCount = false;
  for (int64_t dim : input.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      dynamicInputCount = true;
    } else if (llvm::MulOverflow(inputCount, dim, inputCount)) {
      return emitOptionalError(location,
                               "Reshape input element count overflows");
    }
  }
  SmallVector<int64_t> shape(shapeRef);
  int unknown = -1;
  int64_t outputCount = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      if (unknown != -1) {
        return emitOptionalError(location,
                                 "Reshape has multiple -1 dimensions");
      }
      unknown = static_cast<int>(i);
      continue;
    }
    if (ShapedType::isDynamic(shape[i])) {
      continue;
    }
    if (shape[i] <= 0) {
      return emitOptionalError(location,
                               "Reshape dimensions must be positive or -1");
    }
    if (llvm::MulOverflow(outputCount, shape[i], outputCount)) {
      return emitOptionalError(location,
                               "Reshape output element count overflows");
    }
  }
  if (unknown != -1) {
    if (dynamicInputCount) {
      shape[unknown] = ShapedType::kDynamic;
    } else if (outputCount == 0 || inputCount % outputCount != 0) {
      return emitOptionalError(location,
                               "Reshape element count does not match input");
    } else {
      shape[unknown] = inputCount / outputCount;
    }
  } else if (!dynamicInputCount &&
             llvm::none_of(shape, ShapedType::isDynamic) &&
             inputCount != outputCount) {
    return emitOptionalError(location,
                             "Reshape element count does not match input");
  }
  return RankedTensorType::get(shape, input.getElementType());
}

FailureOr<RankedTensorType> computeReshapeSpecResult(
  std::optional<Location> location,
  RankedTensorType input,
  ArrayRef<int64_t> spec,
  ArrayRef<int64_t> zeroSources) {
  if (!input || !input.getElementType().isF32() || spec.empty() ||
      spec.size() != zeroSources.size()) {
    return emitOptionalError(location, "Reshape shape_spec is invalid");
  }
  int64_t inputCount = 1;
  bool dynamicInputCount = false;
  SmallVector<bool> consumedInputDimensions(input.getRank(), false);
  for (int64_t dimension : input.getShape()) {
    if (ShapedType::isDynamic(dimension)) {
      dynamicInputCount = true;
    } else if (llvm::MulOverflow(inputCount, dimension, inputCount)) {
      return emitOptionalError(location,
                               "Reshape input element count overflows");
    }
  }
  SmallVector<int64_t> outputShape;
  int64_t outputCount = 1;
  int64_t unknown = -1;
  bool dynamicOutputCount = false;
  for (size_t index = 0; index < spec.size(); ++index) {
    int64_t dimension = spec[index];
    if (dimension == -1) {
      if (unknown != -1) {
        return emitOptionalError(location,
                                 "Reshape has multiple -1 dimensions");
      }
      if (zeroSources[index] != -1) {
        return emitOptionalError(location,
                                 "Reshape -1 cannot have a zero source");
      }
      unknown = outputShape.size();
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }
    if (dimension == 0) {
      const int64_t source = zeroSources[index];
      if (source < 0 || source >= input.getRank()) {
        return emitOptionalError(
          location, "Reshape 0 dimension is outside the input rank");
      }
      dimension = input.getShape()[source];
      if (consumedInputDimensions[source]) {
        return emitOptionalError(location,
                                 "Reshape cannot prove exact element count");
      }
      consumedInputDimensions[source] = true;
    } else if (dimension < 0 || zeroSources[index] != -1) {
      return emitOptionalError(location,
                               "Reshape shape_spec dimension is invalid");
    }
    outputShape.push_back(dimension);
    if (ShapedType::isDynamic(dimension)) {
      dynamicOutputCount = true;
    } else if (llvm::MulOverflow(outputCount, dimension, outputCount)) {
      return emitOptionalError(location,
                               "Reshape output element count overflows");
    }
  }
  if (unknown != -1) {
    if (!dynamicInputCount && !dynamicOutputCount) {
      if (inputCount % outputCount != 0) {
        return emitOptionalError(location,
                                 "Reshape element count does not match input");
      }
      outputShape[unknown] = inputCount / outputCount;
    } else {
      int64_t uncancelledInputCount = 1;
      for (auto [index, dimension] : llvm::enumerate(input.getShape())) {
        if (!consumedInputDimensions[index] &&
            !ShapedType::isDynamic(dimension) &&
            llvm::MulOverflow(
              uncancelledInputCount, dimension, uncancelledInputCount)) {
          return emitOptionalError(location,
                                   "Reshape input element count overflows");
        }
      }
      int64_t uncancelledOutputCount = 1;
      for (size_t index = 0; index < spec.size(); ++index) {
        if (std::cmp_equal(index, unknown) || spec[index] == 0) {
          continue;
        }
        if (llvm::MulOverflow(
              uncancelledOutputCount, spec[index], uncancelledOutputCount)) {
          return emitOptionalError(location,
                                   "Reshape output element count overflows");
        }
      }
      if (uncancelledInputCount % uncancelledOutputCount != 0) {
        return emitOptionalError(
          location, "Reshape cannot prove exact inferred dimension");
      }
    }
  } else if (dynamicInputCount || dynamicOutputCount) {
    int64_t uncancelledInputCount = 1;
    for (auto [index, dimension] : llvm::enumerate(input.getShape())) {
      if (consumedInputDimensions[index]) {
        continue;
      }
      if (ShapedType::isDynamic(dimension)) {
        return emitOptionalError(location,
                                 "Reshape cannot prove exact element count");
      }
      if (llvm::MulOverflow(
            uncancelledInputCount, dimension, uncancelledInputCount)) {
        return emitOptionalError(location,
                                 "Reshape input element count overflows");
      }
    }
    int64_t uncancelledOutputCount = 1;
    for (int64_t dimension : spec) {
      if (dimension > 0 &&
          llvm::MulOverflow(
            uncancelledOutputCount, dimension, uncancelledOutputCount)) {
        return emitOptionalError(location,
                                 "Reshape output element count overflows");
      }
    }
    if (uncancelledInputCount != uncancelledOutputCount) {
      return emitOptionalError(location,
                               "Reshape cannot prove exact element count");
    }
  } else if (inputCount != outputCount) {
    return emitOptionalError(location,
                             "Reshape element count does not match input");
  }
  return RankedTensorType::get(outputShape, input.getElementType());
}

FailureOr<RankedTensorType> computeSqueezeResult(
  std::optional<Location> location,
  RankedTensorType input,
  ArrayRef<int64_t> requestedAxes) {
  if (!input || !input.getElementType().isF32()) {
    return emitOptionalError(location, "Squeeze input must be ranked f32");
  }
  std::set<int64_t> axes;
  for (int64_t axis : requestedAxes) {
    if (axis < 0) {
      axis += input.getRank();
    }
    if (axis < 0 || axis >= input.getRank() || !axes.insert(axis).second) {
      return emitOptionalError(location,
                               "Squeeze axes must be unique and in range");
    }
    if (input.getShape()[axis] != 1) {
      return emitOptionalError(location, "Squeeze axis must have unit extent");
    }
  }
  SmallVector<int64_t> shape;
  for (int64_t axis = 0; axis < input.getRank(); ++axis) {
    if (!axes.contains(axis)) {
      shape.push_back(input.getShape()[axis]);
    }
  }
  if (shape.empty()) {
    shape.push_back(1);
  }
  return RankedTensorType::get(shape, input.getElementType());
}

FailureOr<RankedTensorType> computeExpandDimsResult(
  std::optional<Location> location,
  RankedTensorType input,
  ArrayRef<int64_t> requestedAxes) {
  if (!input || !input.getElementType().isF32()) {
    return emitOptionalError(location, "ExpandDims input must be ranked f32");
  }
  const int64_t outputRank = input.getRank() + requestedAxes.size();
  std::set<int64_t> axes;
  for (int64_t axis : requestedAxes) {
    if (axis < 0) {
      axis += outputRank;
    }
    if (axis < 0 || axis >= outputRank || !axes.insert(axis).second) {
      return emitOptionalError(
        location, "ExpandDims axes must be unique and in output rank");
    }
  }
  SmallVector<int64_t> shape;
  const auto* inputDimension = input.getShape().begin();
  for (int64_t axis = 0; axis < outputRank; ++axis) {
    shape.push_back(axes.contains(axis) ? 1 : *inputDimension++);
  }
  return RankedTensorType::get(shape, input.getElementType());
}

FailureOr<RankedTensorType> computePermuteResult(
  std::optional<Location> location,
  RankedTensorType input,
  ArrayRef<int64_t> permutation) {
  if (!input || !input.getElementType().isF32() ||
      static_cast<int64_t>(permutation.size()) != input.getRank()) {
    return emitOptionalError(
      location, "Permute requires a ranked f32 input and one axis per rank");
  }
  std::set<int64_t> axes;
  SmallVector<int64_t> shape;
  for (int64_t axis : permutation) {
    if (axis < 0 || axis >= input.getRank() || !axes.insert(axis).second) {
      return emitOptionalError(location,
                               "Permute axes must form a permutation");
    }
    shape.push_back(input.getShape()[axis]);
  }
  return RankedTensorType::get(shape, input.getElementType());
}

FailureOr<RankedTensorType> computeLayerNormResult(
  std::optional<Location> location,
  RankedTensorType input,
  ValueRange affineParameters,
  int64_t affineSize,
  bool affine,
  double epsilon) {
  if (!input || !input.getElementType().isF32() || input.getRank() < 1 ||
      ShapedType::isDynamic(input.getShape().back()) ||
      input.getShape().back() <= 0) {
    return emitOptionalError(location,
                             "LayerNorm requires ranked f32 input with a "
                             "static positive last dimension");
  }
  if (affineSize != input.getShape().back()) {
    return emitOptionalError(
      location, "LayerNorm affine_size must equal the last input dimension");
  }
  if (!std::isfinite(epsilon) || epsilon < 0.0) {
    return emitOptionalError(
      location, "LayerNorm epsilon must be finite and non-negative");
  }
  if (affineParameters.size() != (affine ? 2U : 0U)) {
    return emitOptionalError(
      location, "LayerNorm affine mode requires exactly gamma and beta");
  }
  for (Value parameter : affineParameters) {
    auto type = dyn_cast<RankedTensorType>(parameter.getType());
    if (!type || !type.getElementType().isF32() || type.getRank() != 1 ||
        type.getShape()[0] != input.getShape().back()) {
      return emitOptionalError(
        location, "LayerNorm gamma and beta must match the last dimension");
    }
  }
  return input;
}

FailureOr<RankedTensorType> computeMultiHeadAttentionResult(
  std::optional<Location> location, MultiHeadAttentionOp::Adaptor adaptor) {
  auto input = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  const int64_t embedDim = adaptor.getEmbedDim();
  const int64_t numHeads = adaptor.getNumHeads();
  const int64_t qdim = adaptor.getQdim();
  const int64_t kdim = adaptor.getKdim();
  const int64_t vdim = adaptor.getVdim();
  if (!input || !input.getElementType().isF32() || input.getRank() != 2) {
    return emitOptionalError(
      location, "MultiHeadAttention input must be rank-2 f32 [sequence,qdim]");
  }
  if (!input.isDynamicDim(0) && input.getShape()[0] <= 0) {
    return emitOptionalError(
      location,
      "MultiHeadAttention sequence dimension must be positive when static");
  }
  if (input.isDynamicDim(1) || embedDim <= 0 || numHeads <= 0 || qdim <= 0 ||
      kdim != qdim || vdim != qdim || embedDim % numHeads != 0 ||
      input.getShape()[1] != qdim ||
      !std::isfinite(adaptor.getScale().convertToDouble())) {
    return emitOptionalError(
      location,
      "MultiHeadAttention dimensions must describe matching self attention and "
      "embed_dim must divide num_heads");
  }
  auto weightCount = checkedMultiply(embedDim, qdim);
  if (failed(weightCount) ||
      static_cast<uint64_t>(*weightCount) != adaptor.getWeightDataSize()) {
    return emitOptionalError(
      location,
      "MultiHeadAttention weight_data_size must equal embed_dim * qdim");
  }
  auto hasShape = [](Value value, ArrayRef<int64_t> shape) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getElementType().isF32() && type.hasStaticShape() &&
           type.getShape() == shape;
  };
  const int64_t projectionShape[] = {embedDim, qdim};
  const int64_t projectionBiasShape[] = {embedDim};
  const int64_t outputShape[] = {qdim, embedDim};
  const int64_t outputBiasShape[] = {qdim};
  if (!hasShape(adaptor.getQWeight(), projectionShape) ||
      !hasShape(adaptor.getQBias(), projectionBiasShape) ||
      !hasShape(adaptor.getKWeight(), projectionShape) ||
      !hasShape(adaptor.getKBias(), projectionBiasShape) ||
      !hasShape(adaptor.getVWeight(), projectionShape) ||
      !hasShape(adaptor.getVBias(), projectionBiasShape) ||
      !hasShape(adaptor.getOutWeight(), outputShape) ||
      !hasShape(adaptor.getOutBias(), outputBiasShape)) {
    return emitOptionalError(
      location,
      "MultiHeadAttention projection weights or biases have invalid shapes");
  }
  return input;
}

FailureOr<RankedTensorType> computeBinaryResult(
  std::optional<Location> location,
  ValueRange inputs,
  bool withScalar,
  llvm::APFloat scalar,
  int64_t opType) {
  (void)scalar;
  if (opType != 0 && opType != 2 && opType != 4) {
    return emitOptionalError(location,
                             "BinaryOp only supports add, multiply, and max");
  }
  if ((withScalar && inputs.size() != 1) ||
      (!withScalar && inputs.size() != 2)) {
    return emitOptionalError(location, "BinaryOp has invalid operand count");
  }
  auto first = llvm::dyn_cast<RankedTensorType>(inputs.front().getType());
  if (!first || (!first.getElementType().isF32() &&
                 !first.getElementType().isSignlessInteger(8))) {
    return emitOptionalError(
      location,
      "BinaryOp operands must be f32 or signed i8 ranked "
      "tensors");
  }
  if (first.getElementType().isSignlessInteger(8) &&
      (withScalar || opType != 4)) {
    return emitOptionalError(
      location, "signed i8 BinaryOp only supports two-input maximum");
  }
  if (withScalar) {
    return first;
  }
  auto second = llvm::dyn_cast<RankedTensorType>(inputs[1].getType());
  if (!second || second.getElementType() != first.getElementType() ||
      second.getRank() != first.getRank()) {
    return emitOptionalError(
      location, "BinaryOp inputs must have matching rank and element type");
  }
  SmallVector<int64_t> shape;
  shape.reserve(first.getRank());
  for (int64_t i = 0; i < first.getRank(); ++i) {
    const int64_t left = first.getShape()[i];
    const int64_t right = second.getShape()[i];
    if (left == 1) {
      shape.push_back(right);
    } else if (right == 1) {
      shape.push_back(left);
    } else if (ShapedType::isDynamic(left) != ShapedType::isDynamic(right)) {
      return emitOptionalError(
        location,
        "BinaryOp dynamic extent cannot be broadcast to a static extent");
    } else if (ShapedType::isDynamic(left)) {
      shape.push_back(ShapedType::kDynamic);
    } else if (left == right) {
      shape.push_back(left);
    } else {
      return emitOptionalError(location,
                               "BinaryOp input shapes are not broadcastable");
    }
  }
  return RankedTensorType::get(shape, first.getElementType());
}

FailureOr<RankedTensorType> computeInnerProductResult(
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType weight,
  ValueRange tail,
  bool hasBias,
  int64_t term) {
  if (term != 0 && term != 1 && term != 2) {
    return emitOptionalError(location,
                             "InnerProduct has unsupported int8_scale_term");
  }
  const bool quantized = term != 0;
  if (!input || !weight || !input.getElementType().isF32() ||
      (!weight.getElementType().isF32() &&
       !(quantized && weight.getElementType().isInteger(8)))) {
    return emitOptionalError(location,
                             "InnerProduct input must be f32 and weight must "
                             "be f32 or quantized i8");
  }
  if (!quantized && weight.getElementType().isInteger(8)) {
    return emitOptionalError(location,
                             "non-quantized InnerProduct weight cannot be i8");
  }
  const bool dynamicMatrix =
    input.getRank() == 2 && input.isDynamicDim(0) && !input.isDynamicDim(1);
  const int64_t inputElements =
    dynamicMatrix ? input.getShape()[1]
                  : (input.hasStaticShape() ? input.getNumElements() : -1);
  if (input.getRank() < 1 || weight.getRank() != 2 ||
      (!input.hasStaticShape() && !dynamicMatrix) || !weight.hasStaticShape() ||
      weight.getShape()[1] != inputElements) {
    return emitOptionalError(
      location, "InnerProduct input elements must match weight [O,I]");
  }
  const std::size_t expectedOperands = (hasBias ? 1 : 0) + (quantized ? 2 : 0);
  if (tail.size() != expectedOperands) {
    return emitOptionalError(
      location,
      "InnerProduct operands do not match bias and quantization mode");
  }
  if (hasBias) {
    auto type = llvm::dyn_cast<RankedTensorType>(tail[0].getType());
    if (!type || type.getRank() != 1 ||
        type.getShape()[0] != weight.getShape()[0] ||
        !type.getElementType().isF32()) {
      return emitOptionalError(location,
                               "InnerProduct bias must have shape [O]");
    }
  }
  const std::size_t scaleOffset = hasBias ? 1 : 0;
  for (std::size_t index = 0; index < (quantized ? 2U : 0U); ++index) {
    auto type =
      llvm::dyn_cast<RankedTensorType>(tail[scaleOffset + index].getType());
    const int64_t expectedSize = index == 0 ? weight.getShape()[0] : 1;
    if (!type || !type.getElementType().isF32() || type.getRank() != 1 ||
        type.getShape()[0] != expectedSize) {
      return emitOptionalError(
        location, "InnerProduct scale has the wrong dtype or shape");
    }
  }
  if (dynamicMatrix) {
    return RankedTensorType::get({input.getShape()[0], weight.getShape()[0]},
                                 input.getElementType());
  }
  return RankedTensorType::get({weight.getShape()[0]}, input.getElementType());
}

FailureOr<RankedTensorType> computeBatchNormResult(
  std::optional<Location> location,
  RankedTensorType input,
  ValueRange parameters) {
  if (!input || !input.getElementType().isF32() || input.getRank() < 1 ||
      ShapedType::isDynamic(input.getShape()[0]) || input.getShape()[0] <= 0 ||
      parameters.size() != 4) {
    return emitOptionalError(
      location,
      "BatchNorm requires a ranked f32 input with static positive channels");
  }
  const int64_t channels = input.getShape()[0];
  for (Value parameter : parameters) {
    auto type = dyn_cast<RankedTensorType>(parameter.getType());
    if (!type || !type.getElementType().isF32() || type.getRank() != 1 ||
        type.getShape()[0] != channels) {
      return emitOptionalError(
        location, "BatchNorm parameters must have input leading dimension");
    }
  }
  return input;
}

FailureOr<RankedTensorType> computeGemmResult(std::optional<Location> location,
                                              RankedTensorType input,
                                              RankedTensorType weight,
                                              RankedTensorType bias,
                                              ValueRange scales,
                                              int64_t int8ScaleTerm) {
  const bool quantized = int8ScaleTerm != 0;
  if (int8ScaleTerm != 0 && int8ScaleTerm != 1 && int8ScaleTerm != 2) {
    return emitOptionalError(location,
                             "Gemm int8 scale term must be 0, 1, or 2");
  }
  if (!input || !weight || !bias || !input.getElementType().isF32() ||
      (quantized ? !weight.getElementType().isSignlessInteger(8)
                 : !isa<FloatType>(weight.getElementType())) ||
      !bias.getElementType().isF32() || input.getRank() != 2 ||
      weight.getRank() != 2 || bias.getRank() != 1 ||
      ShapedType::isDynamic(input.getShape()[1]) ||
      ShapedType::isDynamic(weight.getShape()[0]) ||
      ShapedType::isDynamic(weight.getShape()[1]) ||
      ShapedType::isDynamic(bias.getShape()[0]) ||
      input.getShape()[1] != weight.getShape()[1] ||
      bias.getShape()[0] != weight.getShape()[0]) {
    return emitOptionalError(
      location,
      "Gemm expects input [M,K], weight [N,K], and bias [N] with static K/N");
  }
  if ((!quantized && !scales.empty()) || (quantized && scales.size() != 1)) {
    return emitOptionalError(
      location, "Gemm scale operands do not match quantization mode");
  }
  if (quantized) {
    auto scale = dyn_cast<RankedTensorType>(scales.front().getType());
    if (!scale || !scale.getElementType().isF32() || scale.getRank() != 1 ||
        !scale.hasStaticShape() || scale.getShape()[0] != 1) {
      return emitOptionalError(location,
                               "quantized Gemm B scale must be f32 [1]");
    }
  }
  return RankedTensorType::get({input.getShape()[0], weight.getShape()[0]},
                               input.getElementType());
}

LogicalResult validateScale(std::optional<Location> location,
                            RankedTensorType input,
                            RankedTensorType scale,
                            StringRef operation) {
  if (!scale || !scale.getElementType().isF32() || scale.getRank() != 1 ||
      !scale.hasStaticShape() || scale.getShape()[0] <= 0) {
    return emitOptionalError(
      location, operation, " scale must be static f32 [N]");
  }
  const int64_t count = scale.getShape()[0];
  if (count != 1 &&
      (input.getRank() == 0 || ShapedType::isDynamic(input.getShape()[0]) ||
       count != input.getShape()[0])) {
    return emitOptionalError(
      location, operation, " scale count must be one or match dimension zero");
  }
  return success();
}

LogicalResult validateOptionalBias(std::optional<Location> location,
                                   RankedTensorType input,
                                   ValueRange bias,
                                   StringRef operation) {
  if (bias.empty()) {
    return success();
  }
  if (bias.size() != 1) {
    return emitOptionalError(location, operation, " accepts at most one bias");
  }
  auto type = dyn_cast<RankedTensorType>(bias.front().getType());
  if (!type || !type.getElementType().isF32() || type.getRank() != 1 ||
      !type.hasStaticShape() || type.getShape()[0] <= 0 ||
      (type.getShape()[0] != 1 &&
       (input.getRank() == 0 || ShapedType::isDynamic(input.getShape()[0]) ||
        type.getShape()[0] != input.getShape()[0]))) {
    return emitOptionalError(
      location, operation, " bias count must be one or match dimension zero");
  }
  return success();
}

FailureOr<RankedTensorType> computeQuantizeResult(
  MLIRContext* context,
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType scale) {
  if (!input || !input.getElementType().isF32()) {
    return emitOptionalError(location, "Quantize input must be ranked f32");
  }
  if (failed(validateScale(location, input, scale, "Quantize"))) {
    return failure();
  }
  return input.clone(IntegerType::get(context, 8));
}

FailureOr<RankedTensorType> computeDequantizeResult(
  MLIRContext* context,
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType scale,
  ValueRange bias) {
  if (!input || !input.getElementType().isSignlessInteger(32)) {
    return emitOptionalError(location, "Dequantize input must be ranked i32");
  }
  if (failed(validateScale(location, input, scale, "Dequantize")) ||
      failed(validateOptionalBias(location, input, bias, "Dequantize"))) {
    return failure();
  }
  return input.clone(Float32Type::get(context));
}

FailureOr<RankedTensorType> computeRequantizeResult(
  MLIRContext* context,
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType inputScale,
  RankedTensorType outputScale,
  ValueRange bias,
  int64_t activationType) {
  if (!input || !input.getElementType().isSignlessInteger(32)) {
    return emitOptionalError(location, "Requantize input must be ranked i32");
  }
  if (activationType < 0 || activationType > 6) {
    return emitOptionalError(location, "Requantize activation type is invalid");
  }
  if (failed(validateScale(location, input, inputScale, "Requantize input")) ||
      failed(
        validateScale(location, input, outputScale, "Requantize output")) ||
      failed(validateOptionalBias(location, input, bias, "Requantize"))) {
    return failure();
  }
  return input.clone(IntegerType::get(context, 8));
}

FailureOr<RankedTensorType> computeCastResult(MLIRContext* context,
                                              std::optional<Location> location,
                                              RankedTensorType input,
                                              int64_t from,
                                              int64_t to) {
  auto typeForCode = [&](int64_t code) -> Type {
    switch (code) {
      case 1:
        return Float32Type::get(context);
      case 2:
        return Float16Type::get(context);
      case 3:
        return IntegerType::get(context, 8);
      case 4:
        return BFloat16Type::get(context);
      default:
        return {};
    }
  };
  Type source = typeForCode(from);
  Type target = typeForCode(to);
  if (!input || !source || !target || input.getElementType() != source) {
    return emitOptionalError(
      location, "Cast requires explicit valid source and destination types");
  }
  const bool supported =
    source == target ||
    (source.isF32() && (target.isF16() || target.isBF16())) ||
    (target.isF32() &&
     (source.isF16() || source.isBF16() || source.isSignlessInteger(8)));
  if (!supported) {
    return emitOptionalError(location,
                             "Cast conversion is not supported by ncnn");
  }
  return input.clone(target);
}

FailureOr<RankedTensorType> computeZeroPointCastResult(
  MLIRContext* context,
  std::optional<Location> location,
  RankedTensorType input,
  int64_t zeroPoint,
  bool toUnsigned) {
  if (!input || zeroPoint < 0 || zeroPoint > 255) {
    return emitOptionalError(location,
                             "ZeroPointCast requires an 8-bit zero point");
  }
  Type expected =
    IntegerType::get(context,
                     8,
                     toUnsigned ? IntegerType::SignednessSemantics::Signless
                                : IntegerType::SignednessSemantics::Unsigned);
  if (input.getElementType() != expected) {
    return emitOptionalError(
      location,
      "ZeroPointCast input signedness must be opposite the requested output");
  }
  Type result =
    IntegerType::get(context,
                     8,
                     toUnsigned ? IntegerType::SignednessSemantics::Unsigned
                                : IntegerType::SignednessSemantics::Signless);
  return input.clone(result);
}

LogicalResult inferSliceResults(
  std::optional<Location> location,
  RankedTensorType input,
  ArrayRef<int64_t> requestedSlices,
  int64_t axis,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (!input || !input.getElementType().isF32()) {
    return emitOptionalError(location, "Slice input must be ranked f32");
  }
  if (requestedSlices.size() < 2) {
    return emitOptionalError(location, "Slice requires at least two slices");
  }
  const int64_t rank = input.getRank();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return emitOptionalError(location, "Slice axis is outside input rank");
  }
  const int64_t extent = input.getShape()[axis];
  const bool dynamicExtent = ShapedType::isDynamic(extent);
  if (dynamicExtent && requestedSlices.back() != -233) {
    return emitOptionalError(
      location, "Slice dynamic axis requires a trailing -233 remainder slice");
  }
  int64_t consumed = 0;
  for (std::size_t index = 0; index < requestedSlices.size(); ++index) {
    int64_t size = requestedSlices[index];
    if (size == -233) {
      if (dynamicExtent) {
        SmallVector<int64_t> shape(input.getShape());
        shape[axis] = ShapedType::kDynamic;
        inferredReturnShapes.emplace_back(shape, input.getElementType());
        continue;
      }
      const auto remainingResults =
        static_cast<int64_t>(requestedSlices.size() - index);
      size = (extent - consumed) / remainingResults;
    }
    if (size <= 0 || (!dynamicExtent && size > extent - consumed)) {
      return emitOptionalError(location,
                               "Slice sizes must be positive and fit input");
    }
    SmallVector<int64_t> shape(input.getShape());
    shape[axis] = size;
    inferredReturnShapes.emplace_back(shape, input.getElementType());
    consumed += size;
  }
  if (!dynamicExtent && consumed != extent) {
    return emitOptionalError(location,
                             "Slice sizes must consume the entire axis");
  }
  return success();
}

FailureOr<RankedTensorType> computeReductionResult(
  std::optional<Location> location,
  RankedTensorType input,
  int64_t operation,
  bool reduceAll,
  ArrayRef<int64_t> axes,
  bool keepDims) {
  if (!input || !input.getElementType().isF32()) {
    return emitOptionalError(location, "Reduction input must be ranked f32");
  }
  if (operation != 3) {
    return emitOptionalError(location, "Reduction only supports mean");
  }
  const int64_t rank = input.getRank();
  std::set<int64_t> reducedAxes;
  if (reduceAll) {
    for (int64_t axis = 0; axis < rank; ++axis) {
      reducedAxes.insert(axis);
    }
  } else {
    if (axes.empty()) {
      return emitOptionalError(location,
                               "Reduction requires axes when reduce_all=false");
    }
    for (int64_t axis : axes) {
      if (axis < 0) {
        axis += rank;
      }
      if (axis < 0 || axis >= rank || !reducedAxes.insert(axis).second) {
        return emitOptionalError(location,
                                 "Reduction axes must be unique and in range");
      }
    }
  }
  for (int64_t axis : reducedAxes) {
    if (ShapedType::isDynamic(input.getShape()[axis])) {
      return emitOptionalError(location,
                               "Reduction axes must have static extents");
    }
  }
  SmallVector<int64_t> shape;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (!reducedAxes.contains(axis)) {
      shape.push_back(input.getShape()[axis]);
    } else if (keepDims) {
      shape.push_back(1);
    }
  }
  if (shape.empty()) {
    shape.push_back(1);
  }
  return RankedTensorType::get(shape, input.getElementType());
}

}  // namespace

//===----------------------------------------------------------------------===//
// Model boundary ops
//===----------------------------------------------------------------------===//

LogicalResult ModelOp::verifyRegions() {
  Block& body = getBody().front();
  llvm::StringSet<> inputBlobs;
  SmallVector<InputOp> inputs;
  bool hasOutput = false;
  for (Operation& operation : body) {
    if (operation.getDialect() != getOperation()->getDialect()) {
      return emitOpError("body can only contain ncnn dialect operations");
    }
    if (auto input = llvm::dyn_cast<InputOp>(operation)) {
      if (!inputBlobs.insert(input.getBlobName()).second) {
        return input.emitOpError("duplicates input blob '")
               << input.getBlobName() << "'";
      }
      inputs.push_back(input);
    }
    hasOutput = hasOutput || llvm::isa<OutputOp>(operation);
  }
  if (!hasOutput) {
    return emitOpError("requires at least one ncnn.output");
  }
  auto constraints =
    getOperation()->getAttrOfType<ArrayAttr>("ncnn.shape_constraints");
  if (!constraints) {
    return success();
  }
  std::set<std::pair<uint32_t, uint32_t>> constrainedDimensions;
  for (Attribute attribute : constraints) {
    auto constraint = dyn_cast<DimConstraintAttr>(attribute);
    if (!constraint) {
      return emitOpError("shape_constraints must contain dim constraints");
    }
    if (constraint.getInput() >= inputs.size()) {
      return emitOpError("shape constraint input is out of range");
    }
    auto type = cast<RankedTensorType>(
      inputs[constraint.getInput()].getOutput().getType());
    if (constraint.getDim() >= static_cast<uint32_t>(type.getRank())) {
      return emitOpError("shape constraint dimension is out of range");
    }
    if (!type.isDynamicDim(constraint.getDim())) {
      return emitOpError("shape constraint requires a dynamic dimension");
    }
    if (!constrainedDimensions
           .insert({constraint.getInput(), constraint.getDim()})
           .second) {
      return emitOpError("shape constraint dimension is duplicated");
    }
  }
  return success();
}

OpFoldResult ConstOp::fold(FoldAdaptor) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

LogicalResult ConvolutionOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  ConvolutionOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto weight = llvm::dyn_cast<RankedTensorType>(adaptor.getWeight().getType());
  if (input == nullptr || weight == nullptr) {
    return emitOptionalError(
      location, "convolution input and weight must be ranked tensors");
  }
  FailureOr<RankedTensorType> result =
    computeConvResult(context,
                      location,
                      input,
                      weight,
                      adaptor.getBiasAndScales(),
                      adaptor.getHasBias(),
                      adaptor.getInt8ScaleTerm(),
                      adaptor.getKernelH(),
                      adaptor.getKernelW(),
                      adaptor.getStrideH(),
                      adaptor.getStrideW(),
                      adaptor.getDilationH(),
                      adaptor.getDilationW(),
                      adaptor.getPadTop(),
                      adaptor.getPadBottom(),
                      adaptor.getPadLeft(),
                      adaptor.getPadRight(),
                      weight.getShape()[0]);
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

LogicalResult ConvolutionDepthWiseOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  ConvolutionDepthWiseOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto weight = llvm::dyn_cast<RankedTensorType>(adaptor.getWeight().getType());
  const bool quantized = adaptor.getInt8ScaleTerm() != 0;
  const bool supportedWeight =
    weight &&
    (weight.getElementType().isF32() || weight.getElementType().isF16() ||
     weight.getElementType().isBF16() || weight.getElementType().isInteger(8));
  if (!input || !weight || input.getRank() != 3 || weight.getRank() != 4 ||
      (!input.getElementType().isF32() &&
       !(quantized && input.getElementType().isInteger(8))) ||
      !supportedWeight || !weight.hasStaticShape() ||
      ShapedType::isDynamic(input.getShape()[0]) || weight.getShape()[1] != 1 ||
      weight.getShape()[0] != input.getShape()[0]) {
    return emitOptionalError(location,
                             "ConvolutionDepthWise requires pure "
                             "depthwise weights [C,1,H,W]");
  }
  auto convolutionWeight = RankedTensorType::get({weight.getShape()[0],
                                                  input.getShape()[0],
                                                  weight.getShape()[2],
                                                  weight.getShape()[3]},
                                                 weight.getElementType());
  FailureOr<RankedTensorType> result = computeConvResult(
    context,
    location,
    input,
    convolutionWeight,
    adaptor.getBiasAndScales(),
    adaptor.getHasBias(),
    adaptor.getInt8ScaleTerm(),
    adaptor.getKernelH(),
    adaptor.getKernelW(),
    adaptor.getStrideH(),
    adaptor.getStrideW(),
    adaptor.getDilationH(),
    adaptor.getDilationW(),
    adaptor.getPadTop(),
    adaptor.getPadBottom(),
    adaptor.getPadLeft(),
    adaptor.getPadRight(),
    adaptor.getInt8ScaleTerm() == 1 || adaptor.getInt8ScaleTerm() == 101
      ? input.getShape()[0]
      : 1);
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

LogicalResult DeconvolutionOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  DeconvolutionOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeDeconvResult(
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    dyn_cast<RankedTensorType>(adaptor.getWeight().getType()),
    adaptor.getBias(),
    adaptor.getHasBias(),
    adaptor.getKernelH(),
    adaptor.getKernelW(),
    adaptor.getStrideH(),
    adaptor.getStrideW(),
    adaptor.getDilationH(),
    adaptor.getDilationW(),
    adaptor.getPadTop(),
    adaptor.getPadBottom(),
    adaptor.getPadLeft(),
    adaptor.getPadRight(),
    adaptor.getOutputPadBottom(),
    adaptor.getOutputPadRight());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult PaddingOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  PaddingOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computePaddingResult(
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    adaptor.getTop(),
    adaptor.getBottom(),
    adaptor.getLeft(),
    adaptor.getRight());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult InterpOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  InterpOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeInterpResult(
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    adaptor.getHeightScale(),
    adaptor.getWidthScale(),
    adaptor.getOutputH(),
    adaptor.getOutputW());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult DetectionOutputOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  DetectionOutputOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<double> variances{adaptor.getVarianceX().convertToDouble(),
                                adaptor.getVarianceY().convertToDouble(),
                                adaptor.getVarianceW().convertToDouble(),
                                adaptor.getVarianceH().convertToDouble()};
  auto result = computeDetectionOutputResult(
    location,
    dyn_cast<RankedTensorType>(adaptor.getLocation().getType()),
    dyn_cast<RankedTensorType>(adaptor.getConfidence().getType()),
    dyn_cast<RankedTensorType>(adaptor.getPriorbox().getType()),
    adaptor.getNumClass(),
    adaptor.getNmsTopK(),
    adaptor.getKeepTopK(),
    adaptor.getNmsThreshold().convertToDouble(),
    adaptor.getConfidenceThreshold().convertToDouble(),
    variances);
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  inferredReturnShapes.emplace_back(ArrayRef<int64_t>{2},
                                    IntegerType::get(context, 64));
  return success();
}

LogicalResult SigmoidOp::verify() {
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  if (!input || !input.getElementType().isF32() || input.getRank() != 3) {
    return emitOpError("input must be an FP32 CHW tensor");
  }
  return success();
}

LogicalResult SwishOp::verify() {
  auto input = dyn_cast<RankedTensorType>(getInput().getType());
  if (!input || !input.getElementType().isF32()) {
    return emitOpError("input must be a ranked f32 tensor");
  }
  return success();
}

LogicalResult ReshapeOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  ReshapeOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (adaptor.getShapeExpression()) {
    if (adaptor.getShapeSpec() || adaptor.getShapeZeroSources()) {
      return emitOptionalError(
        location, "Reshape shape expression cannot have a shape_spec");
    }
    auto sources = adaptor.getShapeSources();
    if (!sources || sources->empty() || sources->size() % 2 != 0 ||
        sources->size() / 2 != adaptor.getShape().size()) {
      return emitOptionalError(
        location, "Reshape shape expression requires one source per dimension");
    }
    SmallVector<int64_t> expressionShape;
    for (std::size_t index = 0; index < sources->size(); index += 2) {
      int64_t inputIndex = (*sources)[index];
      int64_t dimension = (*sources)[index + 1];
      if (inputIndex < 0 ||
          static_cast<uint64_t>(inputIndex) >= adaptor.getOperands().size()) {
        return emitOptionalError(location,
                                 "Reshape shape source input is out of range");
      }
      auto source =
        dyn_cast<RankedTensorType>(adaptor.getOperands()[inputIndex].getType());
      if (!source || dimension < 0 || dimension >= source.getRank()) {
        return emitOptionalError(location,
                                 "Reshape shape source is inconsistent");
      }
      expressionShape.push_back(source.getShape()[dimension]);
    }
    if (adaptor.getShape() != ArrayRef<int64_t>(expressionShape)) {
      return emitOptionalError(
        location, "Reshape shape attribute does not match shape sources");
    }
    auto result = computeReshapeResult(location, input, expressionShape);
    if (failed(result)) {
      return failure();
    }
    inferredReturnShapes.emplace_back(result->getShape(),
                                      result->getElementType());
    return success();
  }
  if (!adaptor.getShapeReferences().empty() || adaptor.getShapeSources()) {
    return emitOptionalError(
      location, "Reshape shape references require a shape expression");
  }
  ArrayRef<int64_t> shape = adaptor.getShape();
  FailureOr<RankedTensorType> result;
  if (adaptor.getShapeSpec()) {
    ArrayRef<int64_t> spec = *adaptor.getShapeSpec();
    std::optional<ArrayRef<int64_t>> zeroSources =
      adaptor.getShapeZeroSources();
    if (spec.empty() || spec.size() != shape.size() || !zeroSources ||
        zeroSources->size() != spec.size()) {
      return emitOptionalError(
        location, "Reshape shape_spec must match the resolved shape");
    }
    SmallVector<int64_t> resolvedSpec(spec.begin(), spec.end());
    for (size_t index = 0; index < resolvedSpec.size(); ++index) {
      if (resolvedSpec[index] == 0) {
        const int64_t source = (*zeroSources)[index];
        if (source < 0 || source >= input.getRank()) {
          return emitOptionalError(
            location, "Reshape 0 dimension is outside the input rank");
        }
        resolvedSpec[index] = input.getShape()[source];
      } else if ((*zeroSources)[index] != -1) {
        return emitOptionalError(location,
                                 "Reshape zero source is inconsistent");
      }
    }
    if (resolvedSpec != shape) {
      return emitOptionalError(
        location, "Reshape shape does not match shape_spec semantics");
    }
    result = computeReshapeSpecResult(location, input, spec, *zeroSources);
  } else if (adaptor.getShapeZeroSources()) {
    return emitOptionalError(location,
                             "Reshape shape_zero_sources requires shape_spec");
  } else {
    result = computeReshapeResult(location, input, shape);
  }
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

LogicalResult SqueezeOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  SqueezeOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto result = computeSqueezeResult(location, input, adaptor.getAxes());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult ExpandDimsOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  ExpandDimsOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto result = computeExpandDimsResult(location, input, adaptor.getAxes());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult PermuteOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  PermuteOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto result = computePermuteResult(location, input, adaptor.getPermutation());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult LayerNormOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  LayerNormOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeLayerNormResult(
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    adaptor.getAffineParameters(),
    adaptor.getAffineSize(),
    adaptor.getAffine(),
    adaptor.getEpsilon().convertToDouble());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult MultiHeadAttentionOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  MultiHeadAttentionOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeMultiHeadAttentionResult(location, adaptor);
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult BinaryOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  BinaryOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeBinaryResult(location,
                                    adaptor.getInputs(),
                                    adaptor.getWithScalar(),
                                    adaptor.getScalar(),
                                    adaptor.getOpType());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

LogicalResult InnerProductOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  InnerProductOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto weight = llvm::dyn_cast<RankedTensorType>(adaptor.getWeight().getType());
  auto result = computeInnerProductResult(location,
                                          input,
                                          weight,
                                          adaptor.getBiasAndScales(),
                                          adaptor.getHasBias(),
                                          adaptor.getInt8ScaleTerm());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

LogicalResult BatchNormOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  BatchNormOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  SmallVector<Value> parameters{adaptor.getSlope(),
                                adaptor.getMean(),
                                adaptor.getVariance(),
                                adaptor.getBias()};
  auto result = computeBatchNormResult(location, input, parameters);
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult GemmOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  GemmOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result =
    computeGemmResult(location,
                      dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
                      dyn_cast<RankedTensorType>(adaptor.getWeight().getType()),
                      dyn_cast<RankedTensorType>(adaptor.getBias().getType()),
                      adaptor.getScales(),
                      adaptor.getInt8ScaleTerm());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult QuantizeOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  QuantizeOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeQuantizeResult(
    context,
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    dyn_cast<RankedTensorType>(adaptor.getScale().getType()));
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult DequantizeOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  DequantizeOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeDequantizeResult(
    context,
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    dyn_cast<RankedTensorType>(adaptor.getScale().getType()),
    adaptor.getBias());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult RequantizeOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  RequantizeOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeRequantizeResult(
    context,
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    dyn_cast<RankedTensorType>(adaptor.getInputScale().getType()),
    dyn_cast<RankedTensorType>(adaptor.getOutputScale().getType()),
    adaptor.getBias(),
    adaptor.getActivationType());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult CastOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  CastOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result =
    computeCastResult(context,
                      location,
                      dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
                      adaptor.getTypeFrom(),
                      adaptor.getTypeTo());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult ZeroPointCastOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  ZeroPointCastOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto result = computeZeroPointCastResult(
    context,
    location,
    dyn_cast<RankedTensorType>(adaptor.getInput().getType()),
    adaptor.getZeroPoint(),
    adaptor.getToUnsigned());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

LogicalResult ShuffleChannelOp::verify() {
  auto input = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (!input || !input.getElementType().isF32() || input.getRank() != 3 ||
      ShapedType::isDynamic(input.getShape()[0]) || input.getShape()[0] <= 0) {
    return emitOpError(
      "input must be a positive-channel CHW f32 tensor with static channels");
  }
  if (getGroup() <= 0 || input.getShape()[0] % getGroup() != 0) {
    return emitOpError("group must be positive and divide channel count");
  }
  return success();
}

LogicalResult SliceOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  SliceOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  return inferSliceResults(location,
                           input,
                           adaptor.getSlices(),
                           adaptor.getAxis(),
                           inferredReturnShapes);
}

FailureOr<std::optional<InputDimensionRequirement>>
inferInputDimensionRequirement(SliceOp operation) {
  auto inputType = dyn_cast<RankedTensorType>(operation.getInput().getType());
  if (!inputType) {
    return std::optional<InputDimensionRequirement>{};
  }
  int64_t axis = operation.getAxis();
  if (axis < 0) {
    axis += inputType.getRank();
  }
  if (axis < 0 || axis >= inputType.getRank() ||
      !inputType.isDynamicDim(axis)) {
    return std::optional<InputDimensionRequirement>{};
  }

  if (operation.getSlices().empty() || operation.getSlices().back() != -233 ||
      llvm::any_of(operation.getSlices(),
                   [](int64_t size) { return size <= 0 && size != -233; })) {
    return std::optional<InputDimensionRequirement>{};
  }

  int64_t requiredMinimum = 0;
  for (int64_t size : operation.getSlices()) {
    const int64_t contribution = size == -233 ? 1 : size;
    if (contribution <= 0 ||
        llvm::AddOverflow(requiredMinimum, contribution, requiredMinimum)) {
      operation.emitOpError("has invalid dynamic slice minimum extent");
      return failure();
    }
  }

  std::optional<unsigned> inputIndex;
  Value source = operation.getInput();
  int64_t sourceAxis = axis;
  while (true) {
    if (auto binary = source.getDefiningOp<BinaryOp>()) {
      if (binary.getWithScalar()) {
        source = binary.getInputs().front();
        continue;
      }
      auto firstType = cast<RankedTensorType>(binary.getInputs()[0].getType());
      auto secondType = cast<RankedTensorType>(binary.getInputs()[1].getType());
      if (firstType.getDimSize(sourceAxis) == 1) {
        source = binary.getInputs()[1];
      } else if (secondType.getDimSize(sourceAxis) == 1 ||
                 firstType.getDimSize(sourceAxis) ==
                   secondType.getDimSize(sourceAxis)) {
        source = binary.getInputs()[0];
      } else {
        break;
      }
      continue;
    }
    if (auto squeeze = source.getDefiningOp<SqueezeOp>()) {
      auto sourceType = cast<RankedTensorType>(squeeze.getInput().getType());
      std::set<int64_t> removedAxes;
      for (int64_t removedAxis : squeeze.getAxes()) {
        if (removedAxis < 0) {
          removedAxis += sourceType.getRank();
        }
        removedAxes.insert(removedAxis);
      }
      int64_t outputAxis = 0;
      bool mapped = false;
      for (int64_t inputAxis = 0; inputAxis < sourceType.getRank();
           ++inputAxis) {
        if (removedAxes.contains(inputAxis)) {
          continue;
        }
        if (outputAxis++ == sourceAxis) {
          sourceAxis = inputAxis;
          mapped = true;
          break;
        }
      }
      if (!mapped) {
        break;
      }
      source = squeeze.getInput();
      continue;
    }
    break;
  }
  if (auto input = source.getDefiningOp<InputOp>()) {
    if (auto model = operation->getParentOfType<ModelOp>()) {
      unsigned index = 0;
      for (InputOp candidate : model.getOps<InputOp>()) {
        if (candidate == input) {
          inputIndex = index;
          break;
        }
        ++index;
      }
    }
  } else if (auto argument = dyn_cast<BlockArgument>(source)) {
    if (operation->getParentOfType<func::FuncOp>()) {
      inputIndex = argument.getArgNumber();
    }
  }
  if (!inputIndex || sourceAxis < 0 || !std::in_range<uint32_t>(sourceAxis)) {
    operation.emitOpError("cannot trace dynamic slice axis to a model input");
    return failure();
  }
  return std::optional<InputDimensionRequirement>(
    InputDimensionRequirement{.input = *inputIndex,
                              .dimension = static_cast<uint32_t>(sourceAxis),
                              .minimum = requiredMinimum});
}

LogicalResult SliceOp::verify() {
  auto requirement = inferInputDimensionRequirement(*this);
  if (failed(requirement)) {
    return failure();
  }
  if (!*requirement) {
    return success();
  }
  Operation* constraintOwner = getOperation()->getParentOfType<ModelOp>();
  if (constraintOwner == nullptr) {
    constraintOwner = getOperation()->getParentOfType<func::FuncOp>();
  }
  auto constraints =
    constraintOwner
      ? constraintOwner->getAttrOfType<ArrayAttr>("ncnn.shape_constraints")
      : nullptr;
  if (constraints) {
    for (Attribute attribute : constraints) {
      auto constraint = dyn_cast<DimConstraintAttr>(attribute);
      if (constraint && constraint.getInput() == (*requirement)->input &&
          constraint.getDim() == (*requirement)->dimension &&
          constraint.getMin() >= (*requirement)->minimum) {
        return success();
      }
    }
  }
  return emitOpError()
         << "dynamic axis requires an input minimum extent of at least "
         << (*requirement)->minimum;
}

LogicalResult ReductionOp::inferReturnTypeComponents(
  MLIRContext*,
  std::optional<Location> location,
  ReductionOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto result = computeReductionResult(location,
                                       input,
                                       adaptor.getKind(),
                                       adaptor.getReduceAll(),
                                       adaptor.getAxes(),
                                       adaptor.getKeepdims());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.emplace_back(result->getShape(),
                                    result->getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// PoolingOp
//===----------------------------------------------------------------------===//

LogicalResult PoolingOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  PoolingOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  (void)context;
  auto input = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (input == nullptr) {
    return emitOptionalError(location, "pooling input must be a ranked tensor");
  }
  FailureOr<RankedTensorType> result =
    computePoolResult(location,
                      input,
                      static_cast<PoolKind>(adaptor.getKind()),
                      static_cast<PoolMode>(adaptor.getMode()),
                      adaptor.getKernelH(),
                      adaptor.getKernelW(),
                      adaptor.getStrideH(),
                      adaptor.getStrideW(),
                      adaptor.getPadTop(),
                      adaptor.getPadBottom(),
                      adaptor.getPadLeft(),
                      adaptor.getPadRight(),
                      adaptor.getPadMode(),
                      adaptor.getIncludePad());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypeComponents(
  MLIRContext* context,
  std::optional<Location> location,
  ConcatOp::Adaptor adaptor,
  SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  (void)context;
  FailureOr<RankedTensorType> result =
    computeConcatResult(location, adaptor.getInputs(), adaptor.getAxis());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::fold(FoldAdaptor,
                            SmallVectorImpl<OpFoldResult>& results) {
  for (OpResult result : getResults()) {
    (void)result;
    results.push_back(getInput());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  auto input = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (input == nullptr) {
    return emitOpError("input must be a ranked tensor");
  }
  const int64_t rank = input.getRank();
  int64_t axis = getAxis();
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return emitOpError("axis is outside operand rank");
  }
  return success();
}

}  // namespace mlir::ncnn

#define GET_OP_CLASSES
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.cpp.inc"
