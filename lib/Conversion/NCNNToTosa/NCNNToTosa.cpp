#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

SmallVector<OpFoldResult> getDynamicSizes(OpBuilder& builder,
                                          Location location,
                                          Value source,
                                          RankedTensorType type) {
  SmallVector<OpFoldResult> sizes;
  for (auto [index, extent] : llvm::enumerate(type.getShape())) {
    sizes.push_back(
      ShapedType::isDynamic(extent)
        ? OpFoldResult(builder.create<tensor::DimOp>(location, source, index))
        : OpFoldResult(builder.getIndexAttr(extent)));
  }
  return sizes;
}

SmallVector<Value> getDynamicSizeValues(OpBuilder& builder,
                                        Location location,
                                        Value source,
                                        RankedTensorType type) {
  SmallVector<Value> sizes;
  for (auto [index, extent] : llvm::enumerate(type.getShape())) {
    if (ShapedType::isDynamic(extent)) {
      sizes.push_back(builder.create<tensor::DimOp>(location, source, index));
    }
  }
  return sizes;
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

bool isRankedF32Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getElementType().isF32();
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
  if (!chwType.hasStaticShape()) {
    SmallVector<OpFoldResult> outputShape;
    outputShape.push_back(builder.getIndexAttr(1));
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
      int64_t extent = hwcType.getShape()[dimension];
      outputShape.push_back(ShapedType::isDynamic(extent)
                              ? OpFoldResult(builder.create<tensor::DimOp>(
                                  location, transposed, dimension))
                              : OpFoldResult(builder.getIndexAttr(extent)));
    }
    SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}, {3}};
    return {builder.create<tensor::ExpandShapeOp>(
      location, nhwcType, transposed, reassociation, outputShape)};
  }
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
  if (!nhwcType.hasStaticShape()) {
    SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}, {3}};
    Value reshaped = builder.create<tensor::CollapseShapeOp>(
      location, hwcType, input, reassociation);
    return builder.create<tosa::TransposeOp>(
      location, chwType, reshaped, ArrayRef<int32_t>{2, 0, 1});
  }
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
  const bool dynamicSpatial = input.isDynamicDim(1) || input.isDynamicDim(2);
  if (dynamicSpatial) {
    if (operation.getPadMode() == 0 &&
        (operation.getStrideH() != 1 || operation.getStrideW() != 1)) {
      operation.emitOpError(
        "dynamic tail pooling requires unit spatial strides");
      return failure();
    }
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
      operation.emitOpError("pool padding must be non-negative");
      return failure();
    }
    return SmallVector<int64_t>{top, bottom, left, right};
  }
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
    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    const bool dynamicSpatial = ShapedType::isDynamic(inputShape[1]) ||
                                ShapedType::isDynamic(inputShape[2]);
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
    if (!dynamicSpatial) {
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
    }
    int64_t effectiveH =
      ((operation.getKernelH() - 1) * operation.getDilationH()) + 1;
    int64_t effectiveW =
      ((operation.getKernelW() - 1) * operation.getDilationW()) + 1;
    auto paddedOutputType = outputType;
    if (!dynamicSpatial) {
      int64_t paddedHeight =
        ((inputShape[1] + padding[0] + padding[1] - effectiveH) /
         operation.getStrideH()) +
        1;
      int64_t paddedWidth =
        ((inputShape[2] + padding[2] + padding[3] - effectiveW) /
         operation.getStrideW()) +
        1;
      paddedOutputType = RankedTensorType::get(
        {1, paddedHeight, paddedWidth, sourceOutput.getShape()[0]},
        sourceOutput.getElementType());
    }
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
    if (!global && inputType.hasStaticShape()) {
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
      auto paddedDimension =
        [](int64_t dimension, int64_t before, int64_t after) {
          return ShapedType::isDynamic(dimension) ? ShapedType::kDynamic
                                                  : dimension + before + after;
        };
      auto paddedInputType = RankedTensorType::get(
        {inputType.getShape()[0],
         paddedDimension(
           inputType.getShape()[1], poolPadding[0], poolPadding[1]),
         paddedDimension(
           inputType.getShape()[2], poolPadding[2], poolPadding[3]),
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
    int64_t sourceAxis = operation.getAxis();
    if (sourceAxis < 0) {
      sourceAxis += sourceType.getRank();
    }
    const uint32_t axis = convertAxis(sourceAxis, sourceType.getRank());
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

class ConvertPadding final : public ConversionPattern {
 public:
  ConvertPadding(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, "ncnn.padding", 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() != 1 || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked CHW f32 tensor only");
    }
    auto sourceInput =
      cast<RankedTensorType>(operation->getOperand(0).getType());
    auto sourceOutput =
      cast<RankedTensorType>(operation->getResult(0).getType());
    if (sourceInput.getRank() != 3 || sourceOutput.getRank() != 3) {
      return operation->emitOpError("requires CHW input and output");
    }
    SmallVector<int64_t> pad;
    for (StringRef name : {"top", "bottom", "left", "right"}) {
      FailureOr<int64_t> value = getRequiredIntegerAttr(operation, name);
      if (failed(value)) {
        return failure();
      }
      if (*value < 0) {
        return operation->emitOpError("padding must be non-negative");
      }
      pad.push_back(*value);
    }
    if (sourceOutput.getShape()[0] != sourceInput.getShape()[0] ||
        (!sourceInput.isDynamicDim(1) &&
         sourceOutput.getShape()[1] !=
           sourceInput.getShape()[1] + pad[0] + pad[1]) ||
        (!sourceInput.isDynamicDim(2) &&
         sourceOutput.getShape()[2] !=
           sourceInput.getShape()[2] + pad[2] + pad[3])) {
      return operation->emitOpError("padding does not match result shape");
    }
    auto value = operation->getAttrOfType<FloatAttr>("value");
    if (!value) {
      return operation->emitOpError("requires 'value' float attribute");
    }
    auto inputType = cast<RankedTensorType>(operands.front().getType());
    Value padding = createShape(rewriter,
                                operation->getLoc(),
                                {0, 0, pad[0], pad[1], pad[2], pad[3], 0, 0});
    Value padValue =
      createSplat(rewriter,
                  operation->getLoc(),
                  RankedTensorType::get({1}, inputType.getElementType()),
                  value.getValueAsDouble());
    rewriter.replaceOp(operation,
                       rewriter.create<tosa::PadOp>(operation->getLoc(),
                                                    getNHWCType(sourceOutput),
                                                    operands.front(),
                                                    padding,
                                                    padValue));
    return success();
  }
};

class ConvertInterp final : public ConversionPattern {
 public:
  ConvertInterp(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, "ncnn.interp", 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() != 1 || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked CHW f32 tensor only");
    }
    auto sourceInput =
      cast<RankedTensorType>(operation->getOperand(0).getType());
    auto sourceOutput =
      cast<RankedTensorType>(operation->getResult(0).getType());
    if (sourceInput.getRank() != 3 || sourceOutput.getRank() != 3 ||
        sourceInput.getShape()[0] != sourceOutput.getShape()[0]) {
      return operation->emitOpError(
        "requires CHW input/output with unchanged channels");
    }
    FailureOr<int64_t> scaleH =
      getRequiredIntegerAttr(operation, "height_scale");
    FailureOr<int64_t> scaleW =
      getRequiredIntegerAttr(operation, "width_scale");
    if (failed(scaleH) || failed(scaleW)) {
      return failure();
    }
    if (*scaleH <= 0 || *scaleW <= 0) {
      return operation->emitOpError("resize scales must be positive");
    }
    int64_t expectedH = sourceInput.isDynamicDim(1)
                          ? ShapedType::kDynamic
                          : sourceInput.getShape()[1] * *scaleH;
    int64_t expectedW = sourceInput.isDynamicDim(2)
                          ? ShapedType::kDynamic
                          : sourceInput.getShape()[2] * *scaleW;
    if (sourceOutput.getShape()[1] != expectedH ||
        sourceOutput.getShape()[2] != expectedW) {
      return operation->emitOpError(
        "height_scale/width_scale do not match result shape");
    }

    auto outputType = getNHWCType(sourceOutput);
    if (!sourceInput.hasStaticShape()) {
      Value input = operands.front();
      Value inputH =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 1);
      Value inputW =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 2);
      Value heightFactor =
        rewriter.create<arith::ConstantIndexOp>(operation->getLoc(), *scaleH);
      Value widthFactor =
        rewriter.create<arith::ConstantIndexOp>(operation->getLoc(), *scaleW);
      Value outputH = rewriter.create<arith::MulIOp>(
        operation->getLoc(), inputH, heightFactor);
      Value outputW = rewriter.create<arith::MulIOp>(
        operation->getLoc(), inputW, widthFactor);
      Value empty = rewriter.create<tensor::EmptyOp>(
        operation->getLoc(), outputType, ValueRange{outputH, outputW});
      AffineMap identity = rewriter.getMultiDimIdentityMap(4);
      SmallVector<utils::IteratorType> iterators(4,
                                                 utils::IteratorType::parallel);
      auto result = rewriter.create<linalg::GenericOp>(
        operation->getLoc(),
        outputType,
        ValueRange{},
        ValueRange{empty},
        ArrayRef<AffineMap>{identity},
        iterators,
        [&](OpBuilder& nested, Location location, ValueRange) {
          Value batch = nested.create<linalg::IndexOp>(location, 0);
          Value outputHeight = nested.create<linalg::IndexOp>(location, 1);
          Value outputWidth = nested.create<linalg::IndexOp>(location, 2);
          Value channel = nested.create<linalg::IndexOp>(location, 3);
          Value inputHeight =
            nested.create<arith::DivUIOp>(location, outputHeight, heightFactor);
          Value inputWidth =
            nested.create<arith::DivUIOp>(location, outputWidth, widthFactor);
          Value value = nested.create<tensor::ExtractOp>(
            location,
            input,
            ValueRange{batch, inputHeight, inputWidth, channel});
          nested.create<linalg::YieldOp>(location, value);
        });
      rewriter.replaceOp(operation, result.getResults());
      return success();
    }

    // ncnn nearest uses floor(out / scale). TOSA nearest rounds to the closest
    // sample, so shift by half a scale and extend the border to preserve floor.
    Value scale =
      createShape(rewriter, operation->getLoc(), {*scaleH, 1, *scaleW, 1});
    Value offset = createShape(
      rewriter, operation->getLoc(), {-(*scaleH / 2), -(*scaleW / 2)});
    Value border =
      createShape(rewriter,
                  operation->getLoc(),
                  {*scaleH - 1 - (*scaleH / 2), *scaleW - 1 - (*scaleW / 2)});
    rewriter.replaceOp(operation,
                       rewriter.create<tosa::ResizeOp>(operation->getLoc(),
                                                       outputType,
                                                       operands.front(),
                                                       scale,
                                                       offset,
                                                       border,
                                                       "NEAREST_NEIGHBOR"));
    return success();
  }
};

class ConvertDeconvolution final : public ConversionPattern {
 public:
  ConvertDeconvolution(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, "ncnn.deconvolution", 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() < 2 || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isStaticF32Tensor(operation->getOperand(1).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError(
        "supports ranked f32 input/result and static f32 weight tensors only");
    }
    auto sourceInput =
      cast<RankedTensorType>(operation->getOperand(0).getType());
    auto weightType =
      cast<RankedTensorType>(operation->getOperand(1).getType());
    auto sourceOutput =
      cast<RankedTensorType>(operation->getResult(0).getType());
    if (sourceInput.getRank() != 3 || weightType.getRank() != 4 ||
        sourceOutput.getRank() != 3 ||
        weightType.getShape()[0] != sourceOutput.getShape()[0] ||
        weightType.getShape()[1] != sourceInput.getShape()[0]) {
      return operation->emitOpError(
        "requires CHW input/output and ncnn [O,I,KH,KW] weights");
    }
    const int64_t kernelH = weightType.getShape()[2];
    const int64_t kernelW = weightType.getShape()[3];
    if (getIntegerAttrOr(operation, "kernel_h", kernelH) != kernelH ||
        getIntegerAttrOr(operation, "kernel_w", kernelW) != kernelW) {
      return operation->emitOpError(
        "kernel_h/kernel_w do not match the weight shape");
    }
    if (getIntegerAttrOr(operation, "dilation_h", 1) != 1 ||
        getIntegerAttrOr(operation, "dilation_w", 1) != 1) {
      return operation->emitOpError(
        "TOSA transpose_conv2d does not support dilation");
    }
    FailureOr<int64_t> strideH = getRequiredIntegerAttr(operation, "stride_h");
    FailureOr<int64_t> strideW = getRequiredIntegerAttr(operation, "stride_w");
    if (failed(strideH) || failed(strideW) || *strideH <= 0 || *strideW <= 0) {
      return operation->emitOpError("stride must be positive");
    }
    SmallVector<int64_t> crop;
    for (StringRef name : {"pad_top", "pad_bottom", "pad_left", "pad_right"}) {
      FailureOr<int64_t> value = getRequiredIntegerAttr(operation, name);
      if (failed(value)) {
        return failure();
      }
      if (*value < 0) {
        return operation->emitOpError("padding must be non-negative");
      }
      crop.push_back(*value);
    }
    const int64_t outputPadH =
      getIntegerAttrOr(operation, "output_pad_bottom", 0);
    const int64_t outputPadW =
      getIntegerAttrOr(operation, "output_pad_right", 0);
    if (outputPadH < 0 || outputPadW < 0) {
      return operation->emitOpError("output padding must be non-negative");
    }
    SmallVector<int64_t> outPad{
      -crop[0], outputPadH - crop[1], -crop[2], outputPadW - crop[3]};
    if (outPad[0] <= -kernelH || outPad[1] <= -kernelH ||
        outPad[2] <= -kernelW || outPad[3] <= -kernelW) {
      return operation->emitOpError(
        "crop exceeds the TOSA transpose_conv2d out_pad range");
    }
    const bool dynamicSpatial =
      sourceInput.isDynamicDim(1) || sourceInput.isDynamicDim(2);
    const int64_t expectedH = dynamicSpatial
                                ? ShapedType::kDynamic
                                : ((sourceInput.getShape()[1] - 1) * *strideH) +
                                    kernelH + outPad[0] + outPad[1];
    const int64_t expectedW = dynamicSpatial
                                ? ShapedType::kDynamic
                                : ((sourceInput.getShape()[2] - 1) * *strideW) +
                                    kernelW + outPad[2] + outPad[3];
    if ((!sourceInput.isDynamicDim(1) &&
         expectedH != sourceOutput.getShape()[1]) ||
        (!sourceInput.isDynamicDim(2) &&
         expectedW != sourceOutput.getShape()[2])) {
      return operation->emitOpError(
        "stride, padding, and output padding do not match result shape");
    }

    const bool hasBias = getIntegerAttrOr(operation, "has_bias", 0) != 0;
    if ((hasBias && operands.size() != 3) ||
        (!hasBias && operands.size() != 2)) {
      return operation->emitOpError("operand count does not match has_bias");
    }
    Value bias;
    if (hasBias) {
      auto biasType = dyn_cast<RankedTensorType>(operands[2].getType());
      if (!biasType || !biasType.getElementType().isF32() ||
          biasType.getRank() != 1 ||
          biasType.getShape()[0] != sourceOutput.getShape()[0]) {
        return operation->emitOpError("requires [O] f32 bias");
      }
      bias = operands[2];
    } else {
      bias = createSplat(rewriter,
                         operation->getLoc(),
                         RankedTensorType::get({sourceOutput.getShape()[0]},
                                               sourceOutput.getElementType()),
                         0.0);
    }
    auto outputType = getNHWCType(sourceOutput);
    if (dynamicSpatial) {
      if (kernelH != 2 || kernelW != 2 || *strideH != 2 || *strideW != 2 ||
          llvm::any_of(crop, [](int64_t value) { return value != 0; }) ||
          outputPadH != 0 || outputPadW != 0) {
        return operation->emitOpError(
          "dynamic Deconvolution only supports 2x2 kernel, stride 2, and zero "
          "padding/output padding");
      }

      Value input = operands[0];
      Location location = operation->getLoc();
      Value inputH = rewriter.create<tensor::DimOp>(location, input, 1);
      Value inputW = rewriter.create<tensor::DimOp>(location, input, 2);
      Value two = rewriter.create<arith::ConstantIndexOp>(location, 2);
      Value outputH = rewriter.create<arith::MulIOp>(location, inputH, two);
      Value outputW = rewriter.create<arith::MulIOp>(location, inputW, two);
      Value empty = rewriter.create<tensor::EmptyOp>(
        location, outputType, ValueRange{outputH, outputW});

      AffineExpr n = rewriter.getAffineDimExpr(0);
      AffineExpr oh = rewriter.getAffineDimExpr(1);
      AffineExpr ow = rewriter.getAffineDimExpr(2);
      AffineExpr outputChannel = rewriter.getAffineDimExpr(3);
      AffineExpr inputChannel = rewriter.getAffineDimExpr(4);
      AffineMap biasMap = AffineMap::get(
        4, 0, {rewriter.getAffineDimExpr(3)}, rewriter.getContext());
      AffineMap outputMap = rewriter.getMultiDimIdentityMap(4);
      SmallVector<utils::IteratorType> parallelIterators(
        4, utils::IteratorType::parallel);
      auto initialized = rewriter.create<linalg::GenericOp>(
        location,
        outputType,
        ValueRange{bias},
        ValueRange{empty},
        ArrayRef<AffineMap>{biasMap, outputMap},
        parallelIterators,
        [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
          nested.create<linalg::YieldOp>(nestedLocation, arguments[0]);
        });

      AffineMap inputMap =
        AffineMap::get(5,
                       0,
                       {n, oh.floorDiv(2), ow.floorDiv(2), inputChannel},
                       rewriter.getContext());
      AffineMap weightMap =
        AffineMap::get(5,
                       0,
                       {outputChannel, inputChannel, oh % 2, ow % 2},
                       rewriter.getContext());
      AffineMap resultMap =
        AffineMap::get(5, 0, {n, oh, ow, outputChannel}, rewriter.getContext());
      SmallVector<utils::IteratorType> iterators(4,
                                                 utils::IteratorType::parallel);
      iterators.push_back(utils::IteratorType::reduction);
      auto result = rewriter.create<linalg::GenericOp>(
        location,
        outputType,
        ValueRange{input, operands[1]},
        ValueRange{initialized.getResult(0)},
        ArrayRef<AffineMap>{inputMap, weightMap, resultMap},
        iterators,
        [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
          Value product = nested.create<arith::MulFOp>(
            nestedLocation, arguments[0], arguments[1]);
          Value sum =
            nested.create<arith::AddFOp>(nestedLocation, product, arguments[2]);
          nested.create<linalg::YieldOp>(nestedLocation, sum);
        });
      Value dynamicResult = result.getResult(0);
      const int64_t activationType =
        getIntegerAttrOr(operation, "activation_type", 0);
      if (activationType == 1) {
        dynamicResult = rewriter.create<tosa::ClampOp>(
          location,
          outputType,
          dynamicResult,
          rewriter.getF32FloatAttr(0.0),
          rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity()));
      } else if (activationType != 0) {
        return operation->emitOpError(
          "only no activation and ReLU activation_type=1 are supported");
      }
      rewriter.replaceOp(operation, dynamicResult);
      return success();
    }
    auto ohwiType = getOHWIType(weightType);
    Value weight =
      rewriter.create<tosa::TransposeOp>(operation->getLoc(),
                                         ohwiType,
                                         operands[1],
                                         ArrayRef<int32_t>{0, 2, 3, 1});
    Value inputZero =
      createSplat(rewriter,
                  operation->getLoc(),
                  RankedTensorType::get({1}, sourceInput.getElementType()),
                  0.0);
    Value weightZero =
      createSplat(rewriter,
                  operation->getLoc(),
                  RankedTensorType::get({1}, weightType.getElementType()),
                  0.0);
    Value result = rewriter.create<tosa::TransposeConv2DOp>(
      operation->getLoc(),
      outputType,
      operands[0],
      weight,
      bias,
      inputZero,
      weightZero,
      outPad,
      ArrayRef<int64_t>{*strideH, *strideW},
      rewriter.getF32Type());
    const int64_t activationType =
      getIntegerAttrOr(operation, "activation_type", 0);
    if (activationType == 1) {
      result = rewriter.create<tosa::ClampOp>(
        operation->getLoc(),
        outputType,
        result,
        rewriter.getF32FloatAttr(0.0),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity()));
    } else if (activationType != 0) {
      return operation->emitOpError(
        "only no activation and ReLU activation_type=1 are supported");
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertSigmoid final : public ConversionPattern {
 public:
  ConvertSigmoid(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, "ncnn.sigmoid", 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() != 1 || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked f32 tensor only");
    }
    auto type = cast<RankedTensorType>(operands.front().getType());
    Value clamped = rewriter.create<tosa::ClampOp>(
      operation->getLoc(),
      type,
      operands.front(),
      rewriter.getF32FloatAttr(-88.3762626647949F),
      rewriter.getF32FloatAttr(88.3762626647949F));
    rewriter.replaceOp(
      operation,
      rewriter.create<tosa::SigmoidOp>(operation->getLoc(), type, clamped));
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
        !isRankedF32Tensor(operands.front().getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked f32 tensor only");
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
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isStaticF32Tensor(operands[1].getType())) {
      return operation->emitOpError(
        "supports ranked f32 input and static f32 weight tensors only");
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
    const bool dynamicSpatial =
      inputType.isDynamicDim(1) || inputType.isDynamicDim(2);
    if (!dynamicSpatial) {
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
    }
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
    auto paddedOutputType = outputType;
    if (!dynamicSpatial) {
      int64_t effectiveH =
        ((weightType.getShape()[2] - 1) * (*dilation)[0]) + 1;
      int64_t effectiveW =
        ((weightType.getShape()[3] - 1) * (*dilation)[1]) + 1;
      int64_t paddedHeight =
        ((inputType.getShape()[1] + pad[0] + pad[1] - effectiveH) /
         (*stride)[0]) +
        1;
      int64_t paddedWidth =
        ((inputType.getShape()[2] + pad[2] + pad[3] - effectiveW) /
         (*stride)[1]) +
        1;
      paddedOutputType = RankedTensorType::get(
        {1, paddedHeight, paddedWidth, outputs}, sourceOutput.getElementType());
    }
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
    if (operands.empty() || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("requires ranked f32 input and output");
    }
    auto inputType = cast<RankedTensorType>(operation->getOperand(0).getType());
    auto outputType = cast<RankedTensorType>(operation->getResult(0).getType());
    Value input = restoreNCNNLayout(
      rewriter, operation->getLoc(), operands.front(), inputType);
    Value reshaped;
    auto shapeSources =
      operation->getAttrOfType<DenseI64ArrayAttr>("shape_sources");
    if (shapeSources) {
      if (shapeSources.size() != (outputType.getRank() * 2)) {
        return operation->emitOpError("has invalid shape source metadata");
      }
      SmallVector<Value> dimensions;
      ArrayRef<int64_t> sources = shapeSources.asArrayRef();
      for (int64_t outputDimension = 0; outputDimension < outputType.getRank();
           ++outputDimension) {
        int64_t inputIndex = sources[outputDimension * 2];
        int64_t sourceDimension = sources[(outputDimension * 2) + 1];
        if (inputIndex < 0 ||
            static_cast<uint64_t>(inputIndex) >= operands.size()) {
          return operation->emitOpError("shape source input is out of range");
        }
        Value source = operands[inputIndex];
        auto sourceType =
          cast<RankedTensorType>(operation->getOperand(inputIndex).getType());
        if (sourceType.getRank() == 3) {
          static constexpr int64_t kCHWToNHWCDimension[] = {3, 1, 2};
          sourceDimension = kCHWToNHWCDimension[sourceDimension];
        }
        dimensions.push_back(rewriter.create<tensor::DimOp>(
          operation->getLoc(), source, sourceDimension));
      }
      auto shapeType =
        RankedTensorType::get({outputType.getRank()}, rewriter.getIndexType());
      Value shape = rewriter.create<tensor::FromElementsOp>(
        operation->getLoc(), shapeType, dimensions);
      reshaped = rewriter.create<tensor::ReshapeOp>(
        operation->getLoc(), outputType, input, shape);
    } else {
      if (operands.size() != 1 || !inputType.hasStaticShape() ||
          !outputType.hasStaticShape()) {
        return operation->emitOpError(
          "dynamic Reshape requires shape expression metadata");
      }
      reshaped = reshapeValue(rewriter, operation->getLoc(), input, outputType);
    }
    rewriter.replaceOp(
      operation,
      convertNCNNLayout(rewriter, operation->getLoc(), reshaped, outputType));
    return success();
  }
};

class ConvertShapeChange final : public ConversionPattern {
 public:
  ConvertShapeChange(const TypeConverter& typeConverter,
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

class ConvertPermute final : public OpConversionPattern<PermuteOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    PermuteOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto inputType = cast<RankedTensorType>(operation.getInput().getType());
    auto outputType = cast<RankedTensorType>(operation.getOutput().getType());
    Value input = restoreNCNNLayout(
      rewriter, operation.getLoc(), adaptor.getInput(), inputType);
    SmallVector<int32_t> permutation;
    for (int64_t axis : operation.getPermutation()) {
      permutation.push_back(static_cast<int32_t>(axis));
    }
    Value result = rewriter.create<tosa::TransposeOp>(
      operation.getLoc(), outputType, input, permutation);
    rewriter.replaceOp(
      operation,
      convertNCNNLayout(rewriter, operation.getLoc(), result, outputType));
    return success();
  }
};

class ConvertGELU final : public OpConversionPattern<GELUOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    GELUOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto type = cast<RankedTensorType>(adaptor.getInput().getType());
    Value half = createSplat(
      rewriter, operation.getLoc(), getBroadcastScalarType(type), 0.5);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value activation;
    if (!operation.getFast()) {
      Value inverseSqrtTwo = createSplat(rewriter,
                                         operation.getLoc(),
                                         getBroadcastScalarType(type),
                                         -0.7071067811865476);
      Value scaled = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, adaptor.getInput(), inverseSqrtTwo, shift);
      Value init = rewriter.create<tensor::EmptyOp>(
        operation.getLoc(),
        type.getShape(),
        type.getElementType(),
        getDynamicSizeValues(rewriter, operation.getLoc(), scaled, type));
      auto erfc = rewriter.create<linalg::MapOp>(
        operation.getLoc(),
        ValueRange{scaled},
        init,
        [](OpBuilder& builder, Location location, ValueRange values) {
          Value value = builder.create<math::ErfcOp>(location, values.front());
          builder.create<linalg::YieldOp>(location, value);
        });
      activation = erfc->getResult(0);
    } else {
      Value one = createSplat(
        rewriter, operation.getLoc(), getBroadcastScalarType(type), 1.0);
      Value cubic = rewriter.create<tosa::MulOp>(operation.getLoc(),
                                                 type,
                                                 adaptor.getInput(),
                                                 adaptor.getInput(),
                                                 shift);
      cubic = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, cubic, adaptor.getInput(), shift);
      Value cubicScale = createSplat(
        rewriter, operation.getLoc(), getBroadcastScalarType(type), 0.044715);
      cubic = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, cubic, cubicScale, shift);
      Value sum = rewriter.create<tosa::AddOp>(
        operation.getLoc(), type, adaptor.getInput(), cubic);
      Value tanhScale = createSplat(rewriter,
                                    operation.getLoc(),
                                    getBroadcastScalarType(type),
                                    0.7978845608028654);
      sum = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, sum, tanhScale, shift);
      Value tanh = rewriter.create<tosa::TanhOp>(operation.getLoc(), type, sum);
      activation =
        rewriter.create<tosa::AddOp>(operation.getLoc(), type, one, tanh);
    }
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, adaptor.getInput(), activation, shift);
    result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, result, half, shift);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertBatchNorm final : public OpConversionPattern<BatchNormOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    BatchNormOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    auto outputType = cast<RankedTensorType>(adaptor.getInput().getType());
    SmallVector<int64_t> parameterShape(outputType.getRank(), 1);
    const int64_t channelAxis =
      sourceType.getRank() == 3 ? outputType.getRank() - 1 : 0;
    parameterShape[channelAxis] = sourceType.getShape()[0];
    auto parameterType =
      RankedTensorType::get(parameterShape, outputType.getElementType());
    auto reshapeParameter = [&](Value value) {
      return reshapeValue(rewriter, operation.getLoc(), value, parameterType);
    };
    Value slope = reshapeParameter(adaptor.getSlope());
    Value mean = reshapeParameter(adaptor.getMean());
    Value variance = reshapeParameter(adaptor.getVariance());
    Value bias = reshapeParameter(adaptor.getBias());
    Value epsilon = createSplat(rewriter,
                                operation.getLoc(),
                                getBroadcastScalarType(outputType),
                                operation.getEpsilon().convertToDouble());
    Value varianceWithEpsilon = rewriter.create<tosa::AddOp>(
      operation.getLoc(), parameterType, variance, epsilon);
    Value exponent =
      createSplat(rewriter, operation.getLoc(), parameterType, -0.5);
    Value inverseStd = rewriter.create<tosa::PowOp>(
      operation.getLoc(), parameterType, varianceWithEpsilon, exponent);
    Value zero = createSplat(rewriter, operation.getLoc(), parameterType, 0.0);
    Value fallback =
      createSplat(rewriter, operation.getLoc(), parameterType, 10000.0);
    auto conditionType =
      RankedTensorType::get(parameterShape, rewriter.getI1Type());
    Value isZero = rewriter.create<tosa::EqualOp>(
      operation.getLoc(), conditionType, varianceWithEpsilon, zero);
    inverseStd = rewriter.create<tosa::SelectOp>(
      operation.getLoc(), parameterType, isZero, fallback, inverseStd);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value scale = rewriter.create<tosa::MulOp>(
      operation.getLoc(), parameterType, slope, inverseStd, shift);
    Value scaledMean = rewriter.create<tosa::MulOp>(
      operation.getLoc(), parameterType, scale, mean, shift);
    Value offset = rewriter.create<tosa::SubOp>(
      operation.getLoc(), parameterType, bias, scaledMean);
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), outputType, adaptor.getInput(), scale, shift);
    result = rewriter.create<tosa::AddOp>(
      operation.getLoc(), outputType, result, offset);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertGemm final : public OpConversionPattern<GemmOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    GemmOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto weightType = cast<RankedTensorType>(adaptor.getWeight().getType());
    auto outputType = cast<RankedTensorType>(operation.getOutput().getType());
    const int64_t m = inputType.getShape()[0];
    const int64_t k = inputType.getShape()[1];
    const int64_t n = weightType.getShape()[0];
    auto matrixInputType =
      RankedTensorType::get({1, m, k}, inputType.getElementType());
    Value input = reshapeValue(
      rewriter, operation.getLoc(), adaptor.getInput(), matrixInputType);
    auto matrixWeightType =
      RankedTensorType::get({1, n, k}, weightType.getElementType());
    Value weight = reshapeValue(
      rewriter, operation.getLoc(), adaptor.getWeight(), matrixWeightType);
    auto transposedWeightType =
      RankedTensorType::get({1, k, n}, weightType.getElementType());
    weight = rewriter.create<tosa::TransposeOp>(operation.getLoc(),
                                                transposedWeightType,
                                                weight,
                                                ArrayRef<int32_t>{0, 2, 1});
    auto matrixOutputType =
      RankedTensorType::get({1, m, n}, outputType.getElementType());
    Value result = rewriter.create<tosa::MatMulOp>(
      operation.getLoc(), matrixOutputType, input, weight);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    auto biasType =
      RankedTensorType::get({1, 1, n}, outputType.getElementType());
    Value bias =
      reshapeValue(rewriter, operation.getLoc(), adaptor.getBias(), biasType);
    if (operation.getBeta().convertToDouble() != 1.0) {
      Value beta = createSplat(rewriter,
                               operation.getLoc(),
                               getBroadcastScalarType(matrixOutputType),
                               operation.getBeta().convertToDouble());
      bias = rewriter.create<tosa::MulOp>(
        operation.getLoc(), biasType, bias, beta, shift);
    }
    result = rewriter.create<tosa::AddOp>(
      operation.getLoc(), matrixOutputType, result, bias);
    if (operation.getAlpha().convertToDouble() != 1.0) {
      Value alpha = createSplat(rewriter,
                                operation.getLoc(),
                                getBroadcastScalarType(matrixOutputType),
                                operation.getAlpha().convertToDouble());
      result = rewriter.create<tosa::MulOp>(
        operation.getLoc(), matrixOutputType, result, alpha, shift);
    }
    rewriter.replaceOp(
      operation,
      reshapeValue(rewriter, operation.getLoc(), result, outputType));
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
        !isRankedF32Tensor(operation->getResult(0).getType())) {
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

class ConvertShuffleChannel final
  : public OpConversionPattern<ShuffleChannelOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ShuffleChannelOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto type = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!type || type.getRank() != 4 || !type.getElementType().isF32()) {
      return operation.emitOpError("requires a rank-4 NHWC f32 input");
    }
    const int64_t channels = type.getShape()[3];
    const int64_t group = operation.getReverse()
                            ? channels / operation.getGroup()
                            : operation.getGroup();
    if (group <= 0 || channels % group != 0) {
      return operation.emitOpError("group must divide channel count");
    }
    auto groupedType = RankedTensorType::get({type.getShape()[0],
                                              type.getShape()[1],
                                              type.getShape()[2],
                                              group,
                                              channels / group},
                                             type.getElementType());
    Value grouped;
    if (type.hasStaticShape()) {
      grouped = reshapeValue(
        rewriter, operation.getLoc(), adaptor.getInput(), groupedType);
    } else {
      SmallVector<ReassociationIndices> reassociation = {{0}, {1}, {2}, {3, 4}};
      grouped = rewriter.create<tensor::ExpandShapeOp>(
        operation.getLoc(),
        groupedType,
        adaptor.getInput(),
        reassociation,
        getDynamicSizes(
          rewriter, operation.getLoc(), adaptor.getInput(), groupedType));
    }
    auto shuffledType = RankedTensorType::get({type.getShape()[0],
                                               type.getShape()[1],
                                               type.getShape()[2],
                                               channels / group,
                                               group},
                                              type.getElementType());
    Value shuffled =
      rewriter.create<tosa::TransposeOp>(operation.getLoc(),
                                         shuffledType,
                                         grouped,
                                         ArrayRef<int32_t>{0, 1, 2, 4, 3});
    Value restored;
    if (type.hasStaticShape()) {
      restored = reshapeValue(rewriter, operation.getLoc(), shuffled, type);
    } else {
      SmallVector<ReassociationIndices> reassociation = {{0}, {1}, {2}, {3, 4}};
      restored = rewriter.create<tensor::CollapseShapeOp>(
        operation.getLoc(), type, shuffled, reassociation);
    }
    rewriter.replaceOp(operation, restored);
    return success();
  }
};

class ConvertSlice final : public OpConversionPattern<SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    SliceOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    int64_t sourceAxis = operation.getAxis();
    if (sourceAxis < 0) {
      sourceAxis += sourceType.getRank();
    }
    const uint32_t axis = convertAxis(sourceAxis, sourceType.getRank());
    int64_t offset = 0;
    SmallVector<Value> results;
    for (Value result : operation.getResults()) {
      auto sourceResultType = cast<RankedTensorType>(result.getType());
      auto resultType = sourceResultType.getRank() == 3
                          ? getNHWCType(sourceResultType)
                          : sourceResultType;
      SmallVector<int64_t> start(inputType.getRank(), 0);
      start[axis] = offset;
      results.push_back(rewriter.create<tosa::SliceOp>(
        operation.getLoc(),
        resultType,
        adaptor.getInput(),
        createShape(rewriter, operation.getLoc(), start),
        createShape(rewriter, operation.getLoc(), resultType.getShape())));
      offset += resultType.getShape()[axis];
    }
    rewriter.replaceOp(operation, results);
    return success();
  }
};

class ConvertReduction final : public OpConversionPattern<ReductionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ReductionOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto sourceInput = cast<RankedTensorType>(operation.getInput().getType());
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    if (operation.getKind() != 3 || !inputType.getElementType().isF32()) {
      return operation.emitOpError("only static f32 mean is supported");
    }
    SmallVector<int64_t> sourceAxes;
    if (operation.getReduceAll()) {
      for (int64_t axis = 0; axis < sourceInput.getRank(); ++axis) {
        sourceAxes.push_back(axis);
      }
    } else {
      llvm::append_range(sourceAxes, operation.getAxes());
    }
    SmallVector<uint32_t> axes;
    int64_t reducedElements = 1;
    for (int64_t sourceAxis : sourceAxes) {
      if (sourceAxis < 0) {
        sourceAxis += sourceInput.getRank();
      }
      axes.push_back(convertAxis(sourceAxis, sourceInput.getRank()));
      if (llvm::MulOverflow(reducedElements,
                            sourceInput.getShape()[sourceAxis],
                            reducedElements)) {
        return operation.emitOpError("reduced element count overflows");
      }
    }
    llvm::sort(axes);
    Value result = adaptor.getInput();
    SmallVector<int64_t> reducedShape(inputType.getShape());
    for (uint32_t axis : axes) {
      reducedShape[axis] = 1;
      auto reducedType =
        RankedTensorType::get(reducedShape, inputType.getElementType());
      result = rewriter.create<tosa::ReduceSumOp>(
        operation.getLoc(), reducedType, result, axis);
    }
    auto reducedType = cast<RankedTensorType>(result.getType());
    const double scale = operation.getCoeff().convertToDouble() /
                         static_cast<double>(reducedElements);
    Value factor = createSplat(
      rewriter, operation.getLoc(), getBroadcastScalarType(reducedType), scale);
    result =
      rewriter.create<tosa::MulOp>(operation.getLoc(),
                                   reducedType,
                                   result,
                                   factor,
                                   createI8Zero(rewriter, operation.getLoc()));
    auto outputType = cast<RankedTensorType>(operation.getOutput().getType());
    auto convertedOutputType =
      outputType.getRank() == 3 ? getNHWCType(outputType) : outputType;
    if (reducedType != convertedOutputType) {
      result =
        reshapeValue(rewriter, operation.getLoc(), result, convertedOutputType);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

#include "DetectionOutputLowering.inc"

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
                 ConvertSoftmax,
                 ConvertShuffleChannel,
                 ConvertSlice,
                 ConvertReduction,
                 ConvertGELU,
                 ConvertBatchNorm,
                 ConvertPermute,
                 ConvertGemm>(typeConverter, context);
    patterns.add<ConvertHardActivation>(
      typeConverter, context, "ncnn.hard_sigmoid", false);
    patterns.add<ConvertHardActivation>(
      typeConverter, context, "ncnn.hard_swish", true);
    patterns.add<ConvertDepthwiseConvolution>(
      typeConverter, context, "ncnn.convolution_depthwise");
    patterns.add<ConvertReshape>(typeConverter, context, "ncnn.reshape");
    patterns.add<ConvertShapeChange>(typeConverter, context, "ncnn.squeeze");
    patterns.add<ConvertShapeChange>(
      typeConverter, context, "ncnn.expand_dims");
    patterns.add<ConvertBinary>(typeConverter, context, "ncnn.binary");
    patterns.add<ConvertInnerProduct>(
      typeConverter, context, "ncnn.inner_product");
    patterns.add<ConvertPadding,
                 ConvertInterp,
                 ConvertDeconvolution,
                 ConvertSigmoid,
                 ConvertDetectionOutput>(typeConverter, context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect,
                           func::FuncDialect,
                           linalg::LinalgDialect,
                           math::MathDialect,
                           scf::SCFDialect,
                           tensor::TensorDialect,
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
    target.addIllegalOp<ReluOp,
                        DetectionOutputOp,
                        SplitOp,
                        ConcatOp,
                        DropoutOp,
                        SoftmaxOp,
                        ShuffleChannelOp,
                        SliceOp,
                        ReductionOp,
                        GELUOp,
                        BatchNormOp,
                        PermuteOp,
                        GemmOp>();
    for (StringRef name : {"ncnn.hard_sigmoid",
                           "ncnn.hard_swish",
                           "ncnn.convolution_depthwise",
                           "ncnn.reshape",
                           "ncnn.squeeze",
                           "ncnn.expand_dims",
                           "ncnn.binary",
                           "ncnn.inner_product",
                           "ncnn.padding",
                           "ncnn.interp",
                           "ncnn.deconvolution",
                           "ncnn.sigmoid"}) {
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
