#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_CONVERTNCNNTOTOSAPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

RankedTensorType getNHWCType(RankedTensorType chwType) {
  ArrayRef<int64_t> shape = chwType.getShape();
  return RankedTensorType::get({1, shape[1], shape[2], shape[0]},
                               chwType.getElementType());
}

RankedTensorType getOHWIType(RankedTensorType oihwType) {
  ArrayRef<int64_t> shape = oihwType.getShape();
  return RankedTensorType::get({shape[0], shape[2], shape[3], shape[1]},
                               oihwType.getElementType());
}

Value createShape(OpBuilder& builder,
                  Location location,
                  ArrayRef<int64_t> dimensions) {
  auto storageType = RankedTensorType::get(
    {static_cast<int64_t>(dimensions.size())}, builder.getIndexType());
  auto values = DenseIntElementsAttr::get(storageType, dimensions);
  auto shapeType = tosa::shapeType::get(builder.getContext(),
                                        static_cast<int>(dimensions.size()));
  Value result =
    builder.create<tosa::ConstShapeOp>(location, shapeType, values);
  return result;
}

Value createSplat(OpBuilder& builder,
                  Location location,
                  RankedTensorType type,
                  double value) {
  auto element = cast<FloatType>(type.getElementType());
  auto values =
    DenseElementsAttr::get(type, builder.getFloatAttr(element, value));
  Value result = builder.create<tosa::ConstOp>(location, type, values);
  return result;
}

Value createI8Zero(OpBuilder& builder, Location location) {
  auto type = RankedTensorType::get({1}, builder.getI8Type());
  auto value = DenseElementsAttr::get(type, builder.getI8IntegerAttr(0));
  Value result = builder.create<tosa::ConstOp>(location, type, value);
  return result;
}

RankedTensorType getBroadcastScalarType(RankedTensorType type) {
  SmallVector<int64_t> shape(type.getRank(), 1);
  return RankedTensorType::get(shape, type.getElementType());
}

Value convertCHWToNHWC(OpBuilder& builder, Location location, Value input);
Value convertNHWCToCHW(OpBuilder& builder,
                       Location location,
                       Value input,
                       RankedTensorType chwType);

bool isStaticF32Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getElementType().isF32();
}

FailureOr<int64_t> getRequiredIntegerAttr(Operation* operation,
                                          StringRef name) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  if (!attribute) {
    operation->emitOpError() << "requires '" << name << "' integer attribute";
    return failure();
  }
  return attribute.getInt();
}

int64_t getIntegerAttrOr(Operation* operation,
                         StringRef name,
                         int64_t fallback) {
  auto attribute = operation->getAttrOfType<IntegerAttr>(name);
  return attribute ? attribute.getInt() : fallback;
}

double getFloatAttrOr(Operation* operation, StringRef name, double fallback) {
  auto attribute = operation->getAttrOfType<FloatAttr>(name);
  return attribute ? attribute.getValueAsDouble() : fallback;
}

Value reshapeValue(OpBuilder& builder,
                   Location location,
                   Value input,
                   RankedTensorType outputType) {
  Value shape = createShape(builder, location, outputType.getShape());
  return {builder.create<tosa::ReshapeOp>(location, outputType, input, shape)};
}

Value restoreNCNNLayout(OpBuilder& builder,
                        Location location,
                        Value input,
                        RankedTensorType sourceType) {
  if (sourceType.getRank() == 3) {
    return convertNHWCToCHW(builder, location, input, sourceType);
  }
  return input;
}

Value convertNCNNLayout(OpBuilder& builder,
                        Location location,
                        Value input,
                        RankedTensorType sourceType) {
  if (sourceType.getRank() == 3) {
    return convertCHWToNHWC(builder, location, input);
  }
  return input;
}

FailureOr<Value> matchBroadcastRank(OpBuilder& builder,
                                    Location location,
                                    Value input,
                                    RankedTensorType outputType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() > outputType.getRank()) {
    return failure();
  }
  if (inputType.getRank() == outputType.getRank()) {
    return input;
  }
  SmallVector<int64_t> shape(outputType.getRank() - inputType.getRank(), 1);
  llvm::append_range(shape, inputType.getShape());
  return reshapeValue(builder,
                      location,
                      input,
                      RankedTensorType::get(shape, inputType.getElementType()));
}

Value convertCHWToNHWC(OpBuilder& builder, Location location, Value input) {
  auto chwType = cast<RankedTensorType>(input.getType());
  auto hwcType = RankedTensorType::get(
    {chwType.getShape()[1], chwType.getShape()[2], chwType.getShape()[0]},
    chwType.getElementType());
  Value transposed = builder.create<tosa::TransposeOp>(
    location, hwcType, input, ArrayRef<int32_t>{1, 2, 0});
  RankedTensorType nhwcType = getNHWCType(chwType);
  Value shape = createShape(builder, location, nhwcType.getShape());
  Value result =
    builder.create<tosa::ReshapeOp>(location, nhwcType, transposed, shape);
  return result;
}

Value convertNHWCToCHW(OpBuilder& builder,
                       Location location,
                       Value input,
                       RankedTensorType chwType) {
  auto nhwcType = cast<RankedTensorType>(input.getType());
  auto hwcType = RankedTensorType::get(
    {nhwcType.getShape()[1], nhwcType.getShape()[2], nhwcType.getShape()[3]},
    nhwcType.getElementType());
  Value shape = createShape(builder, location, hwcType.getShape());
  Value reshaped =
    builder.create<tosa::ReshapeOp>(location, hwcType, input, shape);
  return builder.create<tosa::TransposeOp>(
    location, chwType, reshaped, ArrayRef<int32_t>{2, 0, 1});
}

uint32_t convertAxis(int64_t axis, int64_t sourceRank) {
  if (sourceRank == 3) {
    static constexpr uint32_t kCHWToNHWC[] = {3, 1, 2};
    return kCHWToNHWC[axis];
  }
  return static_cast<uint32_t>(axis);
}

FailureOr<SmallVector<int64_t>> getPoolPadding(PoolingOp operation) {
  int64_t top = operation.getPadTopAttr().getInt();
  int64_t bottom = operation.getPadBottomAttr().getInt();
  int64_t left = operation.getPadLeftAttr().getInt();
  int64_t right = operation.getPadRightAttr().getInt();
  if (operation.getMode() != static_cast<int64_t>(PoolMode::Regular)) {
    return SmallVector<int64_t>{top, bottom, left, right};
  }

  auto input = cast<RankedTensorType>(operation.getInput().getType());
  auto output = cast<RankedTensorType>(operation.getOutput().getType());
  auto calculateTrailingPadding =
    [&](StringRef dimension,
        int64_t outputSize,
        int64_t stride,
        int64_t kernel,
        int64_t inputSize,
        int64_t leadingPadding) -> FailureOr<int64_t> {
    auto overflow = [&](StringRef arithmetic) {
      operation.emitOpError() << "pool padding " << dimension
                              << " arithmetic overflow during " << arithmetic;
      return failure();
    };
    int64_t outputOffset;
    if (llvm::SubOverflow(outputSize, int64_t{1}, outputOffset)) {
      return overflow("output size - 1");
    }
    int64_t stridedOffset;
    if (llvm::MulOverflow(outputOffset, stride, stridedOffset)) {
      return overflow("(output size - 1) * stride");
    }
    int64_t requiredSize;
    if (llvm::AddOverflow(stridedOffset, kernel, requiredSize)) {
      return overflow("strided offset + kernel");
    }
    int64_t padding;
    if (llvm::SubOverflow(requiredSize, inputSize, padding)) {
      return overflow("required size - input size");
    }
    if (llvm::SubOverflow(padding, leadingPadding, padding)) {
      return overflow("required padding - leading padding");
    }
    return padding;
  };
  FailureOr<int64_t> requiredBottom =
    calculateTrailingPadding("height",
                             output.getShape()[1],
                             operation.getStrideH(),
                             operation.getKernelH(),
                             input.getShape()[1],
                             top);
  if (failed(requiredBottom)) {
    return failure();
  }
  FailureOr<int64_t> requiredRight =
    calculateTrailingPadding("width",
                             output.getShape()[2],
                             operation.getStrideW(),
                             operation.getKernelW(),
                             input.getShape()[2],
                             left);
  if (failed(requiredRight)) {
    return failure();
  }
  bottom = std::max(bottom, *requiredBottom);
  right = std::max(right, *requiredRight);
  if (top < 0 || bottom < 0 || left < 0 || right < 0) {
    operation.emitOpError("pool padding must be non-negative");
    return failure();
  }
  auto makeDivisible = [](int64_t inputSize,
                          int64_t kernel,
                          int64_t stride,
                          int64_t leading,
                          int64_t& trailing) {
    int64_t remainder = (inputSize + leading + trailing - kernel) % stride;
    if (remainder < 0) {
      remainder += stride;
    }
    if (remainder != 0) {
      trailing += stride - remainder;
    }
  };
  makeDivisible(input.getShape()[1],
                operation.getKernelH(),
                operation.getStrideH(),
                top,
                bottom);
  makeDivisible(input.getShape()[2],
                operation.getKernelW(),
                operation.getStrideW(),
                left,
                right);
  return SmallVector<int64_t>{top, bottom, left, right};
}

class ConvertConvolution final : public OpConversionPattern<ConvolutionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ConvolutionOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    const int64_t padTop = operation.getPadTopAttr().getInt();
    const int64_t padBottom = operation.getPadBottomAttr().getInt();
    const int64_t padLeft = operation.getPadLeftAttr().getInt();
    const int64_t padRight = operation.getPadRightAttr().getInt();
    if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
      return operation.emitOpError(
        "requires explicit non-negative padding; run normalize-ncnn first");
    }
    Value input = adaptor.getInput();
    auto weightType = cast<RankedTensorType>(operation.getWeight().getType());
    auto ohwiType = getOHWIType(weightType);
    Value weight =
      rewriter.create<tosa::TransposeOp>(operation.getLoc(),
                                         ohwiType,
                                         adaptor.getWeight(),
                                         ArrayRef<int32_t>{0, 2, 3, 1});
    Value bias;
    if (operation.getHasBias()) {
      bias = adaptor.getBiasAndScales().front();
    } else {
      auto output = cast<RankedTensorType>(operation.getOutput().getType());
      auto biasType =
        RankedTensorType::get({output.getShape()[0]}, output.getElementType());
      bias = createSplat(rewriter, operation.getLoc(), biasType, 0.0);
    }
    auto sourceOutput = cast<RankedTensorType>(operation.getOutput().getType());
    auto outputType = getNHWCType(sourceOutput);
    Value inputZero =
      createSplat(rewriter,
                  operation.getLoc(),
                  RankedTensorType::get(
                    {1}, cast<ShapedType>(input.getType()).getElementType()),
                  0.0);
    Value weightZero =
      createSplat(rewriter,
                  operation.getLoc(),
                  RankedTensorType::get(
                    {1}, cast<ShapedType>(weight.getType()).getElementType()),
                  0.0);
    SmallVector<int64_t> padding{padTop, padBottom, padLeft, padRight};
    auto adjustTrailingPadding = [&](int64_t inputSize,
                                     int64_t kernel,
                                     int64_t stride,
                                     int64_t dilation,
                                     int64_t leading,
                                     int64_t& trailing) {
      int64_t effective = ((kernel - 1) * dilation) + 1;
      int64_t numerator = inputSize - 1 + leading + trailing - effective + 1;
      int64_t remainder = numerator % stride;
      if (remainder < 0) {
        remainder += stride;
      }
      if (remainder != 0) {
        trailing += stride - remainder;
      }
    };
    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    adjustTrailingPadding(inputShape[1],
                          operation.getKernelH(),
                          operation.getStrideH(),
                          operation.getDilationH(),
                          padTop,
                          padding[1]);
    adjustTrailingPadding(inputShape[2],
                          operation.getKernelW(),
                          operation.getStrideW(),
                          operation.getDilationW(),
                          padLeft,
                          padding[3]);
    int64_t effectiveH =
      ((operation.getKernelH() - 1) * operation.getDilationH()) + 1;
    int64_t effectiveW =
      ((operation.getKernelW() - 1) * operation.getDilationW()) + 1;
    int64_t paddedHeight =
      ((inputShape[1] + padding[0] + padding[1] - effectiveH) /
       operation.getStrideH()) +
      1;
    int64_t paddedWidth =
      ((inputShape[2] + padding[2] + padding[3] - effectiveW) /
       operation.getStrideW()) +
      1;
    auto paddedOutputType = RankedTensorType::get(
      {1, paddedHeight, paddedWidth, sourceOutput.getShape()[0]},
      sourceOutput.getElementType());
    Value result = rewriter.create<tosa::Conv2DOp>(
      operation.getLoc(),
      paddedOutputType,
      input,
      weight,
      bias,
      inputZero,
      weightZero,
      padding,
      ArrayRef<int64_t>{static_cast<int64_t>(operation.getStrideH()),
                        static_cast<int64_t>(operation.getStrideW())},
      ArrayRef<int64_t>{static_cast<int64_t>(operation.getDilationH()),
                        static_cast<int64_t>(operation.getDilationW())},
      rewriter.getF32Type());
    if (paddedOutputType != outputType) {
      Value start = createShape(rewriter, operation.getLoc(), {0, 0, 0, 0});
      Value size =
        createShape(rewriter, operation.getLoc(), outputType.getShape());
      result = rewriter.create<tosa::SliceOp>(
        operation.getLoc(), outputType, result, start, size);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertRelu final : public OpConversionPattern<ReluOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ReluOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Value input = adaptor.getInput();
    auto type = cast<RankedTensorType>(input.getType());
    const double slope = operation.getNegativeSlope().convertToDouble();
    if (slope == 0.0) {
      auto element = cast<FloatType>(type.getElementType());
      Value result = rewriter.create<tosa::ClampOp>(
        operation.getLoc(),
        type,
        input,
        rewriter.getFloatAttr(element, 0.0),
        rewriter.getFloatAttr(element,
                              std::numeric_limits<double>::infinity()));
      rewriter.replaceOp(operation, result);
      return success();
    }
    auto scalarType = getBroadcastScalarType(type);
    Value zero = createSplat(rewriter, operation.getLoc(), scalarType, 0.0);
    Value slopeValue =
      createSplat(rewriter, operation.getLoc(), scalarType, slope);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value negative = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, input, slopeValue, shift);
    auto conditionType =
      RankedTensorType::get(type.getShape(), rewriter.getI1Type());
    Value condition = rewriter.create<tosa::GreaterEqualOp>(
      operation.getLoc(), conditionType, input, zero);
    Value result = rewriter.create<tosa::SelectOp>(
      operation.getLoc(), type, condition, input, negative);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertPooling final : public OpConversionPattern<PoolingOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    PoolingOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Value input = adaptor.getInput();
    auto sourceOutput = cast<RankedTensorType>(operation.getOutput().getType());
    const bool global =
      operation.getMode() == static_cast<int64_t>(PoolMode::Global);
    auto inputType = cast<RankedTensorType>(input.getType());
    RankedTensorType outputType =
      global ? RankedTensorType::get({1, 1, 1, inputType.getShape()[3]},
                                     inputType.getElementType())
             : getNHWCType(sourceOutput);
    SmallVector<int64_t> kernel =
      global
        ? SmallVector<int64_t>{inputType.getShape()[1], inputType.getShape()[2]}
        : SmallVector<int64_t>{static_cast<int64_t>(operation.getKernelH()),
                               static_cast<int64_t>(operation.getKernelW())};
    SmallVector<int64_t> stride =
      global
        ? SmallVector<int64_t>{1, 1}
        : SmallVector<int64_t>{static_cast<int64_t>(operation.getStrideH()),
                               static_cast<int64_t>(operation.getStrideW())};
    FailureOr<SmallVector<int64_t>> padding = getPoolPadding(operation);
    if (failed(padding)) {
      return failure();
    }
    RankedTensorType pooledType = outputType;
    if (!global) {
      const int64_t pooledHeight =
        ((inputType.getShape()[1] + (*padding)[0] + (*padding)[1] - kernel[0]) /
         stride[0]) +
        1;
      const int64_t pooledWidth =
        ((inputType.getShape()[2] + (*padding)[2] + (*padding)[3] - kernel[1]) /
         stride[1]) +
        1;
      pooledType = RankedTensorType::get(
        {1, pooledHeight, pooledWidth, inputType.getShape()[3]},
        inputType.getElementType());
    }
    Value poolInput = input;
    SmallVector<int64_t> poolPadding = *padding;
    const bool materializeMaxPadding =
      !global &&
      operation.getKind() == static_cast<int64_t>(PoolKind::Maximum) &&
      (poolPadding[0] >= kernel[0] || poolPadding[1] >= kernel[0] ||
       poolPadding[2] >= kernel[1] || poolPadding[3] >= kernel[1]);
    if (materializeMaxPadding) {
      auto paddedInputType = RankedTensorType::get(
        {inputType.getShape()[0],
         inputType.getShape()[1] + poolPadding[0] + poolPadding[1],
         inputType.getShape()[2] + poolPadding[2] + poolPadding[3],
         inputType.getShape()[3]},
        inputType.getElementType());
      Value paddingShape = createShape(rewriter,
                                       operation.getLoc(),
                                       {0,
                                        0,
                                        poolPadding[0],
                                        poolPadding[1],
                                        poolPadding[2],
                                        poolPadding[3],
                                        0,
                                        0});
      Value padValue =
        createSplat(rewriter,
                    operation.getLoc(),
                    RankedTensorType::get({1}, inputType.getElementType()),
                    -std::numeric_limits<double>::infinity());
      poolInput = rewriter.create<tosa::PadOp>(
        operation.getLoc(), paddedInputType, input, paddingShape, padValue);
      poolPadding.assign(4, 0);
    }
    Value result;
    if (operation.getKind() == static_cast<int64_t>(PoolKind::Maximum)) {
      result = rewriter.create<tosa::MaxPool2dOp>(
        operation.getLoc(), pooledType, poolInput, kernel, stride, poolPadding);
    } else {
      Value zero =
        createSplat(rewriter,
                    operation.getLoc(),
                    RankedTensorType::get({1}, inputType.getElementType()),
                    0.0);
      result = rewriter.create<tosa::AvgPool2dOp>(operation.getLoc(),
                                                  pooledType,
                                                  input,
                                                  zero,
                                                  zero,
                                                  kernel,
                                                  stride,
                                                  *padding,
                                                  rewriter.getF32Type());
    }
    if (pooledType != outputType) {
      Value start = createShape(rewriter, operation.getLoc(), {0, 0, 0, 0});
      Value size =
        createShape(rewriter, operation.getLoc(), outputType.getShape());
      result = rewriter.create<tosa::SliceOp>(
        operation.getLoc(), outputType, result, start, size);
    }
    if (global) {
      Value shape =
        createShape(rewriter, operation.getLoc(), sourceOutput.getShape());
      result = rewriter.create<tosa::ReshapeOp>(
        operation.getLoc(), sourceOutput, result, shape);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertSplit final : public OpConversionPattern<SplitOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    SplitOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    SmallVector<Value> replacements(operation->getNumResults(),
                                    adaptor.getInput());
    rewriter.replaceOp(operation, replacements);
    return success();
  }
};

class ConvertConcat final : public OpConversionPattern<ConcatOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ConcatOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto sourceType = cast<RankedTensorType>(operation.getOutput().getType());
    auto outputType =
      sourceType.getRank() == 3 ? getNHWCType(sourceType) : sourceType;
    uint32_t axis = convertAxis(operation.getAxis(), sourceType.getRank());
    Value result = rewriter.create<tosa::ConcatOp>(
      operation.getLoc(), outputType, adaptor.getInputs(), axis);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertDropout final : public OpConversionPattern<DropoutOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    DropoutOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Value input = adaptor.getInput();
    const double scale = operation.getScale().convertToDouble();
    if (scale == 1.0) {
      rewriter.replaceOp(operation, input);
      return success();
    }
    auto type = cast<RankedTensorType>(input.getType());
    Value factor = createSplat(
      rewriter, operation.getLoc(), getBroadcastScalarType(type), scale);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, input, factor, shift);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertSoftmax final : public OpConversionPattern<SoftmaxOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    SoftmaxOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Value input = adaptor.getInput();
    auto type = cast<RankedTensorType>(input.getType());
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    const uint32_t axis =
      convertAxis(operation.getAxis(), sourceType.getRank());
    SmallVector<int64_t> reducedShape(type.getShape());
    reducedShape[axis] = 1;
    auto reducedType =
      RankedTensorType::get(reducedShape, type.getElementType());
    Value maximum = rewriter.create<tosa::ReduceMaxOp>(
      operation.getLoc(), reducedType, input, axis);
    Value shifted =
      rewriter.create<tosa::SubOp>(operation.getLoc(), type, input, maximum);
    Value exponent =
      rewriter.create<tosa::ExpOp>(operation.getLoc(), type, shifted);
    Value sum = rewriter.create<tosa::ReduceSumOp>(
      operation.getLoc(), reducedType, exponent, axis);
    Value reciprocal =
      rewriter.create<tosa::ReciprocalOp>(operation.getLoc(), reducedType, sum);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, exponent, reciprocal, shift);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertHardActivation final : public ConversionPattern {
 public:
  ConvertHardActivation(const TypeConverter& typeConverter,
                        MLIRContext* context,
                        StringRef operationName,
                        bool swish)
    : ConversionPattern(typeConverter, operationName, 1, context),
      swish_(swish) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() != 1 || operation->getNumResults() != 1 ||
        !isStaticF32Tensor(operands.front().getType())) {
      return operation->emitOpError("supports one static f32 tensor only");
    }
    Value input = operands.front();
    auto type = cast<RankedTensorType>(input.getType());
    const double alpha = getFloatAttrOr(operation, "alpha", 0.2);
    const double beta = getFloatAttrOr(operation, "beta", 0.5);
    Value alphaValue = createSplat(
      rewriter, operation->getLoc(), getBroadcastScalarType(type), alpha);
    Value betaValue = createSplat(
      rewriter, operation->getLoc(), getBroadcastScalarType(type), beta);
    Value shift = createI8Zero(rewriter, operation->getLoc());
    Value scaled = rewriter.create<tosa::MulOp>(
      operation->getLoc(), type, input, alphaValue, shift);
    Value affine = rewriter.create<tosa::AddOp>(
      operation->getLoc(), type, scaled, betaValue);
    Value gate = rewriter.create<tosa::ClampOp>(operation->getLoc(),
                                                type,
                                                affine,
                                                rewriter.getF32FloatAttr(0.0),
                                                rewriter.getF32FloatAttr(1.0));
    if (swish_) {
      gate = rewriter.create<tosa::MulOp>(
        operation->getLoc(), type, input, gate, shift);
    }
    rewriter.replaceOp(operation, gate);
    return success();
  }

 private:
  bool swish_;
};

class ConvertDepthwiseConvolution final : public ConversionPattern {
 public:
  ConvertDepthwiseConvolution(const TypeConverter& typeConverter,
                              MLIRContext* context,
                              StringRef operationName)
    : ConversionPattern(typeConverter, operationName, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() < 2 || operation->getNumResults() != 1 ||
        !isStaticF32Tensor(operands[0].getType()) ||
        !isStaticF32Tensor(operands[1].getType())) {
      return operation->emitOpError(
        "supports static f32 input and weight tensors only");
    }
    auto inputType = cast<RankedTensorType>(operands[0].getType());
    auto weightType = cast<RankedTensorType>(operands[1].getType());
    auto sourceOutput =
      cast<RankedTensorType>(operation->getResult(0).getType());
    if (inputType.getRank() != 4 || weightType.getRank() != 4 ||
        sourceOutput.getRank() != 3 || weightType.getShape()[1] != 1) {
      return operation->emitOpError(
        "requires CHW input/output and [O,1,H,W] weights");
    }
    const int64_t channels = inputType.getShape()[3];
    const int64_t outputs = weightType.getShape()[0];
    if (channels <= 0 || outputs % channels != 0 ||
        getIntegerAttrOr(operation, "group", channels) != channels) {
      return operation->emitOpError(
        "requires group equal to input channels and divisible output channels");
    }
    const int64_t multiplier = outputs / channels;
    auto groupedWeightType = RankedTensorType::get({channels,
                                                    multiplier,
                                                    weightType.getShape()[2],
                                                    weightType.getShape()[3]},
                                                   weightType.getElementType());
    Value groupedWeight = reshapeValue(
      rewriter, operation->getLoc(), operands[1], groupedWeightType);
    auto hwcmType = RankedTensorType::get({weightType.getShape()[2],
                                           weightType.getShape()[3],
                                           channels,
                                           multiplier},
                                          weightType.getElementType());
    Value weight =
      rewriter.create<tosa::TransposeOp>(operation->getLoc(),
                                         hwcmType,
                                         groupedWeight,
                                         ArrayRef<int32_t>{2, 3, 0, 1});
    const bool hasBias = getIntegerAttrOr(operation, "has_bias", 0) != 0;
    Value bias;
    if (hasBias) {
      if (operands.size() < 3 || !isStaticF32Tensor(operands[2].getType())) {
        return operation->emitOpError("has_bias requires an f32 bias operand");
      }
      bias = operands[2];
    } else {
      bias = createSplat(
        rewriter,
        operation->getLoc(),
        RankedTensorType::get({outputs}, inputType.getElementType()),
        0.0);
    }
    SmallVector<int64_t> pad;
    for (StringRef name : {"pad_top", "pad_bottom", "pad_left", "pad_right"}) {
      FailureOr<int64_t> value = getRequiredIntegerAttr(operation, name);
      if (failed(value) || *value < 0) {
        return operation->emitOpError(
          "requires explicit non-negative padding; run normalize-ncnn first");
      }
      pad.push_back(*value);
    }
    auto getPair = [&](StringRef first,
                       StringRef second) -> FailureOr<SmallVector<int64_t>> {
      FailureOr<int64_t> a = getRequiredIntegerAttr(operation, first);
      FailureOr<int64_t> b = getRequiredIntegerAttr(operation, second);
      if (failed(a) || failed(b)) {
        return failure();
      }
      return SmallVector<int64_t>{*a, *b};
    };
    FailureOr<SmallVector<int64_t>> stride = getPair("stride_h", "stride_w");
    FailureOr<SmallVector<int64_t>> dilation =
      getPair("dilation_h", "dilation_w");
    if (failed(stride) || failed(dilation)) {
      return failure();
    }
    auto adjustTrailingPadding = [&](int64_t inputSize,
                                     int64_t kernel,
                                     int64_t strideValue,
                                     int64_t dilationValue,
                                     int64_t leading,
                                     int64_t& trailing) {
      int64_t effective = ((kernel - 1) * dilationValue) + 1;
      int64_t numerator = inputSize - 1 + leading + trailing - effective + 1;
      int64_t remainder = numerator % strideValue;
      if (remainder < 0) {
        remainder += strideValue;
      }
      if (remainder != 0) {
        trailing += strideValue - remainder;
      }
    };
    adjustTrailingPadding(inputType.getShape()[1],
                          weightType.getShape()[2],
                          (*stride)[0],
                          (*dilation)[0],
                          pad[0],
                          pad[1]);
    adjustTrailingPadding(inputType.getShape()[2],
                          weightType.getShape()[3],
                          (*stride)[1],
                          (*dilation)[1],
                          pad[2],
                          pad[3]);
    Value inputZero =
      createSplat(rewriter,
                  operation->getLoc(),
                  RankedTensorType::get({1}, inputType.getElementType()),
                  0.0);
    Value weightZero =
      createSplat(rewriter,
                  operation->getLoc(),
                  RankedTensorType::get({1}, weightType.getElementType()),
                  0.0);
    auto outputType = getNHWCType(sourceOutput);
    int64_t effectiveH = ((weightType.getShape()[2] - 1) * (*dilation)[0]) + 1;
    int64_t effectiveW = ((weightType.getShape()[3] - 1) * (*dilation)[1]) + 1;
    int64_t paddedHeight =
      ((inputType.getShape()[1] + pad[0] + pad[1] - effectiveH) /
       (*stride)[0]) +
      1;
    int64_t paddedWidth =
      ((inputType.getShape()[2] + pad[2] + pad[3] - effectiveW) /
       (*stride)[1]) +
      1;
    auto paddedOutputType = RankedTensorType::get(
      {1, paddedHeight, paddedWidth, outputs}, sourceOutput.getElementType());
    Value result =
      rewriter.create<tosa::DepthwiseConv2DOp>(operation->getLoc(),
                                               paddedOutputType,
                                               operands[0],
                                               weight,
                                               bias,
                                               inputZero,
                                               weightZero,
                                               pad,
                                               *stride,
                                               *dilation,
                                               rewriter.getF32Type());
    if (paddedOutputType != outputType) {
      Value start = createShape(rewriter, operation->getLoc(), {0, 0, 0, 0});
      Value size =
        createShape(rewriter, operation->getLoc(), outputType.getShape());
      result = rewriter.create<tosa::SliceOp>(
        operation->getLoc(), outputType, result, start, size);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertReshape final : public ConversionPattern {
 public:
  ConvertReshape(const TypeConverter& typeConverter,
                 MLIRContext* context,
                 StringRef operationName)
    : ConversionPattern(typeConverter, operationName, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() != 1 || operation->getNumResults() != 1 ||
        !isStaticF32Tensor(operation->getOperand(0).getType()) ||
        !isStaticF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one static f32 tensor only");
    }
    auto inputType = cast<RankedTensorType>(operation->getOperand(0).getType());
    auto outputType = cast<RankedTensorType>(operation->getResult(0).getType());
    Value input = restoreNCNNLayout(
      rewriter, operation->getLoc(), operands.front(), inputType);
    Value reshaped =
      reshapeValue(rewriter, operation->getLoc(), input, outputType);
    rewriter.replaceOp(
      operation,
      convertNCNNLayout(rewriter, operation->getLoc(), reshaped, outputType));
    return success();
  }
};

class ConvertBinary final : public ConversionPattern {
 public:
  ConvertBinary(const TypeConverter& typeConverter,
                MLIRContext* context,
                StringRef operationName)
    : ConversionPattern(typeConverter, operationName, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.empty() || operands.size() > 2 ||
        operation->getNumResults() != 1 ||
        !isStaticF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError(
        "supports one result and one or two operands");
    }
    auto sourceOutput =
      cast<RankedTensorType>(operation->getResult(0).getType());
    auto outputType =
      sourceOutput.getRank() == 3 ? getNHWCType(sourceOutput) : sourceOutput;
    FailureOr<int64_t> opType = getRequiredIntegerAttr(operation, "op_type");
    if (failed(opType)) {
      return failure();
    }
    SmallVector<Value> values(operands.begin(), operands.end());
    if (values.size() == 1) {
      values.push_back(createSplat(rewriter,
                                   operation->getLoc(),
                                   getBroadcastScalarType(outputType),
                                   getFloatAttrOr(operation, "scalar", 0.0)));
    }
    FailureOr<Value> lhs =
      matchBroadcastRank(rewriter, operation->getLoc(), values[0], outputType);
    FailureOr<Value> rhs =
      matchBroadcastRank(rewriter, operation->getLoc(), values[1], outputType);
    if (failed(lhs) || failed(rhs)) {
      return operation->emitOpError("operand ranks cannot broadcast to result");
    }
    Value shift = createI8Zero(rewriter, operation->getLoc());
    auto multiply = [&](Value a, Value b) -> Value {
      return {rewriter.create<tosa::MulOp>(
        operation->getLoc(), outputType, a, b, shift)};
    };
    auto divide = [&](Value a, Value b) -> Value {
      Value reciprocal = rewriter.create<tosa::ReciprocalOp>(
        operation->getLoc(), cast<RankedTensorType>(b.getType()), b);
      return multiply(a, reciprocal);
    };
    Value result;
    switch (*opType) {
      case 0:
        result = rewriter.create<tosa::AddOp>(
          operation->getLoc(), outputType, *lhs, *rhs);
        break;
      case 1:
        result = rewriter.create<tosa::SubOp>(
          operation->getLoc(), outputType, *lhs, *rhs);
        break;
      case 2:
        result = multiply(*lhs, *rhs);
        break;
      case 3:
        result = divide(*lhs, *rhs);
        break;
      case 4:
        result = rewriter.create<tosa::MaximumOp>(
          operation->getLoc(), outputType, *lhs, *rhs);
        break;
      case 5:
        result = rewriter.create<tosa::MinimumOp>(
          operation->getLoc(), outputType, *lhs, *rhs);
        break;
      case 6:
        result = rewriter.create<tosa::PowOp>(
          operation->getLoc(), outputType, *lhs, *rhs);
        break;
      case 7:
        result = rewriter.create<tosa::SubOp>(
          operation->getLoc(), outputType, *rhs, *lhs);
        break;
      case 8:
        result = divide(*rhs, *lhs);
        break;
      case 9:
        result = rewriter.create<tosa::PowOp>(
          operation->getLoc(), outputType, *rhs, *lhs);
        break;
      default:
        return operation->emitOpError(
          "supports ADD/SUB/MUL/DIV/MAX/MIN/POW and reverse variants only");
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertInnerProduct final : public ConversionPattern {
 public:
  ConvertInnerProduct(const TypeConverter& typeConverter,
                      MLIRContext* context,
                      StringRef operationName)
    : ConversionPattern(typeConverter, operationName, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() < 2 || operation->getNumResults() != 1 ||
        !isStaticF32Tensor(operation->getOperand(0).getType()) ||
        !isStaticF32Tensor(operation->getOperand(1).getType()) ||
        !isStaticF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError(
        "supports static f32 input, weight, and result tensors only");
    }
    auto sourceInput =
      cast<RankedTensorType>(operation->getOperand(0).getType());
    auto weightType =
      cast<RankedTensorType>(operation->getOperand(1).getType());
    auto outputType = cast<RankedTensorType>(operation->getResult(0).getType());
    if (weightType.getRank() != 2 || outputType.getRank() != 1) {
      return operation->emitOpError(
        "requires [O,K] weights and a rank-1 output");
    }
    const int64_t outputs = weightType.getShape()[0];
    const int64_t inputs = weightType.getShape()[1];
    if (sourceInput.getNumElements() != inputs ||
        outputType.getShape()[0] != outputs) {
      return operation->emitOpError("input/weight/output sizes do not match");
    }
    Value input = restoreNCNNLayout(
      rewriter, operation->getLoc(), operands[0], sourceInput);
    auto matrixInputType =
      RankedTensorType::get({1, 1, inputs}, sourceInput.getElementType());
    input = reshapeValue(rewriter, operation->getLoc(), input, matrixInputType);
    auto transposedWeightType =
      RankedTensorType::get({inputs, outputs}, weightType.getElementType());
    Value transposedWeight =
      rewriter.create<tosa::TransposeOp>(operation->getLoc(),
                                         transposedWeightType,
                                         operands[1],
                                         ArrayRef<int32_t>{1, 0});
    auto matrixWeightType =
      RankedTensorType::get({1, inputs, outputs}, weightType.getElementType());
    Value matrixWeight = reshapeValue(
      rewriter, operation->getLoc(), transposedWeight, matrixWeightType);
    auto matrixOutputType =
      RankedTensorType::get({1, 1, outputs}, outputType.getElementType());
    Value result = rewriter.create<tosa::MatMulOp>(
      operation->getLoc(), matrixOutputType, input, matrixWeight);
    const bool hasBias = getIntegerAttrOr(operation, "has_bias", 0) != 0;
    if (hasBias) {
      if (operands.size() < 3 || !isStaticF32Tensor(operands[2].getType())) {
        return operation->emitOpError("has_bias requires an f32 bias operand");
      }
      auto biasType =
        RankedTensorType::get({1, 1, outputs}, outputType.getElementType());
      Value bias =
        reshapeValue(rewriter, operation->getLoc(), operands[2], biasType);
      result = rewriter.create<tosa::AddOp>(
        operation->getLoc(), matrixOutputType, result, bias);
    }
    rewriter.replaceOp(
      operation,
      reshapeValue(rewriter, operation->getLoc(), result, outputType));
    return success();
  }
};

class ConvertNCNNToTosaPass final
  : public impl::ConvertNCNNToTosaPassBase<ConvertNCNNToTosaPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<ModelOp>().empty()) {
      module.emitError(
        "convert-ncnn-to-tosa requires ncnn.model to be "
        "converted to func.func first");
      signalPassFailure();
      return;
    }

    MLIRContext* context = module.getContext();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return type.getRank() == 3 ? getNHWCType(type) : type;
    });
    typeConverter.addTargetMaterialization([](OpBuilder& builder,
                                              RankedTensorType resultType,
                                              ValueRange inputs,
                                              Location location,
                                              Type) -> Value {
      if (inputs.size() != 1 || resultType.getRank() != 4) {
        return {};
      }
      auto inputType = dyn_cast<RankedTensorType>(inputs.front().getType());
      if (!inputType || inputType.getRank() != 3 ||
          getNHWCType(inputType) != resultType) {
        return {};
      }
      return convertCHWToNHWC(builder, location, inputs.front());
    });
    typeConverter.addSourceMaterialization([](OpBuilder& builder,
                                              RankedTensorType resultType,
                                              ValueRange inputs,
                                              Location location) -> Value {
      if (inputs.size() != 1 || resultType.getRank() != 3) {
        return {};
      }
      auto inputType = dyn_cast<RankedTensorType>(inputs.front().getType());
      if (!inputType || inputType.getRank() != 4 ||
          getNHWCType(resultType) != inputType) {
        return {};
      }
      return convertNHWCToCHW(builder, location, inputs.front(), resultType);
    });

    RewritePatternSet patterns(context);
    patterns.add<ConvertConvolution,
                 ConvertRelu,
                 ConvertPooling,
                 ConvertSplit,
                 ConvertConcat,
                 ConvertDropout,
                 ConvertSoftmax>(typeConverter, context);
    patterns.add<ConvertHardActivation>(
      typeConverter, context, "ncnn.hard_sigmoid", false);
    patterns.add<ConvertHardActivation>(
      typeConverter, context, "ncnn.hard_swish", true);
    patterns.add<ConvertDepthwiseConvolution>(
      typeConverter, context, "ncnn.convolution_depthwise");
    patterns.add<ConvertReshape>(typeConverter, context, "ncnn.reshape");
    patterns.add<ConvertBinary>(typeConverter, context, "ncnn.binary");
    patterns.add<ConvertInnerProduct>(
      typeConverter, context, "ncnn.inner_product");

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect,
                           func::FuncDialect,
                           tosa::TosaDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<ConvolutionOp>([](ConvolutionOp operation) {
      return operation.getInt8ScaleTerm() != 0;
    });
    target.addDynamicallyLegalOp<PoolingOp>([](PoolingOp operation) {
      return operation.getMode() == static_cast<int64_t>(PoolMode::Adaptive) ||
             (operation.getKind() == static_cast<int64_t>(PoolKind::Average) &&
              operation.getIncludePad());
    });
    target.addIllegalOp<ReluOp, SplitOp, ConcatOp, DropoutOp, SoftmaxOp>();
    for (StringRef name : {"ncnn.hard_sigmoid",
                           "ncnn.hard_swish",
                           "ncnn.convolution_depthwise",
                           "ncnn.reshape",
                           "ncnn.binary",
                           "ncnn.inner_product"}) {
      target.addIllegalOp(OperationName(name, context));
    }

    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPartialConversion(module, target, frozen))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
