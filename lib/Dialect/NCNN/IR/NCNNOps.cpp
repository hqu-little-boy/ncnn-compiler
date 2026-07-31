#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

#include <cstdint>
#include <limits>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
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
                                              int64_t padRight) {
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
  if (!wElem.isF32() && !wElem.isF16() && !wElem.isInteger(8)) {
    return fail("convolution weight has unsupported element type");
  }
  ArrayRef<int64_t> inShape = input.getShape();
  ArrayRef<int64_t> wShape = weight.getShape();
  for (int64_t dim : inShape) {
    if (dim <= 0) {
      return fail("convolution input dimensions must be positive");
    }
  }
  for (int64_t dim : wShape) {
    if (dim <= 0) {
      return fail("convolution weight dimensions must be positive");
    }
  }
  if (wShape[1] != inShape[0]) {
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
    const int64_t expectedSize = scaleIndex == 0 ? wShape[0] : 1;
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
  if (sameUpper || sameLower) {
    outputHeight = 1 + ((inShape[1] - 1) / strideH);
    outputWidth = 1 + ((inShape[2] - 1) / strideW);
  } else {
    FailureOr<int64_t> height = checkedAdd(inShape[1], padTop);
    if (succeeded(height)) {
      height = checkedAdd(*height, padBottom);
    }
    FailureOr<int64_t> width = checkedAdd(inShape[2], padLeft);
    if (succeeded(width)) {
      width = checkedAdd(*width, padRight);
    }
    if (failed(height) || failed(width)) {
      return fail("convolution padded dimension overflows");
    }
    if (*height < *extentHeight || *width < *extentWidth) {
      return fail("convolution kernel exceeds padded input");
    }
    outputHeight = 1 + ((*height - *extentHeight) / strideH);
    outputWidth = 1 + ((*width - *extentWidth) / strideW);
  }
  Type resultElem = requantized
                      ? static_cast<Type>(IntegerType::get(context, 8))
                      : static_cast<Type>(Float32Type::get(context));
  SmallVector<int64_t> shape = {wShape[0], outputHeight, outputWidth};
  return RankedTensorType::get(shape, resultElem);
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
  if (input <= 0) {
    return emitOptionalError(location, name, " input must be positive");
  }
  if (kernel <= 0) {
    return emitOptionalError(location, name, " kernel must be positive");
  }
  if (stride <= 0) {
    return emitOptionalError(location, name, " stride must be positive");
  }
  if (padBefore < 0 || padAfter < 0) {
    return emitOptionalError(location, name, " padding must be non-negative");
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
  if (!input.getElementType().isF32()) {
    return fail("pooling input must be f32");
  }
  if (kind != PoolKind::Maximum && kind != PoolKind::Average) {
    return fail("pooling kind is invalid");
  }
  ArrayRef<int64_t> inShape = input.getShape();
  for (int64_t dim : inShape) {
    if (dim <= 0) {
      return fail("pooling input dimensions must be positive");
    }
  }
  Type elem = input.getElementType();
  if (mode == PoolMode::Global) {
    return RankedTensorType::get(SmallVector<int64_t>{inShape[0]}, elem);
  }
  if (mode == PoolMode::Adaptive) {
    const int64_t outputHeight = kernelH == -233 ? inShape[1] : kernelH;
    const int64_t outputWidth = kernelW == -233 ? inShape[2] : kernelW;
    if (outputHeight <= 0 || outputWidth <= 0) {
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
      if (operand.getShape()[dimension] != first.getShape()[dimension]) {
        return fail("Concat non-axis dimensions must match");
      }
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

}  // namespace

//===----------------------------------------------------------------------===//
// 公开的形状推断入口（供 importer 在构建算子前计算结果类型）
//===----------------------------------------------------------------------===//

FailureOr<RankedTensorType> inferConvResultType(
  MLIRContext* context,
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType weight,
  ValueRange biasAndScales,
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
  int64_t padRight) {
  return computeConvResult(context,
                           location,
                           input,
                           weight,
                           biasAndScales,
                           hasBias,
                           term,
                           kernelH,
                           kernelW,
                           strideH,
                           strideW,
                           dilationH,
                           dilationW,
                           padTop,
                           padBottom,
                           padLeft,
                           padRight);
}

FailureOr<RankedTensorType> inferPoolResultType(
  std::optional<Location> location,
  RankedTensorType input,
  int64_t kind,
  int64_t mode,
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
  return computePoolResult(location,
                           input,
                           static_cast<PoolKind>(kind),
                           static_cast<PoolMode>(mode),
                           kernelH,
                           kernelW,
                           strideH,
                           strideW,
                           padTop,
                           padBottom,
                           padLeft,
                           padRight,
                           padMode,
                           includePad);
}

FailureOr<RankedTensorType> inferConcatResultType(
  std::optional<Location> location, ValueRange inputs, int64_t axis) {
  return computeConcatResult(location, inputs, axis);
}

//===----------------------------------------------------------------------===//
// Model boundary ops
//===----------------------------------------------------------------------===//

LogicalResult ModelOp::verify() {
  if (getBody().empty()) {
    return emitOpError("requires one body block");
  }
  Block& body = getBody().front();
  if (body.getNumArguments() != 0) {
    return emitOpError("body cannot have block arguments");
  }

  llvm::StringSet<> inputBlobs;
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
    }
    hasOutput = hasOutput || llvm::isa<OutputOp>(operation);
  }
  if (!hasOutput) {
    return emitOpError("requires at least one ncnn.output");
  }
  return success();
}

LogicalResult ConstOp::verify() {
  if (getValue().getType() != getOutput().getType()) {
    return emitOpError("value type ")
           << getValue().getType() << " does not match result type "
           << getOutput().getType();
  }
  return success();
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
                      adaptor.getPadRight());
  if (failed(result)) {
    return failure();
  }
  inferredReturnShapes.push_back(
    ShapedTypeComponents(result->getShape(), result->getElementType()));
  return success();
}

LogicalResult ConvolutionOp::verify() {
  auto input = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto weight = llvm::dyn_cast<RankedTensorType>(getWeight().getType());
  if (input == nullptr || weight == nullptr) {
    return emitOpError("input and weight must be ranked tensors");
  }
  FailureOr<RankedTensorType> result = computeConvResult(getContext(),
                                                         getLoc(),
                                                         input,
                                                         weight,
                                                         getBiasAndScales(),
                                                         getHasBias(),
                                                         getInt8ScaleTerm(),
                                                         getKernelH(),
                                                         getKernelW(),
                                                         getStrideH(),
                                                         getStrideW(),
                                                         getDilationH(),
                                                         getDilationW(),
                                                         getPadTop(),
                                                         getPadBottom(),
                                                         getPadLeft(),
                                                         getPadRight());
  if (failed(result)) {
    return failure();
  }
  if (getResult().getType() != *result) {
    return emitOpError("result type ")
           << getResult().getType() << " does not match inferred " << *result;
  }
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

LogicalResult PoolingOp::verify() {
  auto input = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (input == nullptr) {
    return emitOpError("input must be a ranked tensor");
  }
  FailureOr<RankedTensorType> result =
    computePoolResult(getLoc(),
                      input,
                      static_cast<PoolKind>(getKind()),
                      static_cast<PoolMode>(getMode()),
                      getKernelH(),
                      getKernelW(),
                      getStrideH(),
                      getStrideW(),
                      getPadTop(),
                      getPadBottom(),
                      getPadLeft(),
                      getPadRight(),
                      getPadMode(),
                      getIncludePad());
  if (failed(result)) {
    return failure();
  }
  if (getResult().getType() != *result) {
    return emitOpError("result type ")
           << getResult().getType() << " does not match inferred " << *result;
  }
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

LogicalResult ConcatOp::verify() {
  FailureOr<RankedTensorType> result =
    computeConcatResult(getLoc(), getInputs(), getAxis());
  if (failed(result)) {
    return failure();
  }
  if (getResult().getType() != *result) {
    return emitOpError("result type ")
           << getResult().getType() << " does not match inferred " << *result;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

LogicalResult SplitOp::verify() {
  if (getResults().size() < 2) {
    return emitOpError("requires at least two results");
  }
  const Type inputType = getInput().getType();
  for (OpResult result : getResults()) {
    if (result.getType() != inputType) {
      return emitOpError("all results must have the same type as the input");
    }
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
