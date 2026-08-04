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
    auto outputType =
      getNHWCType(cast<RankedTensorType>(operation.getOutput().getType()));
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
    Value result = rewriter.create<tosa::Conv2DOp>(
      operation.getLoc(),
      outputType,
      input,
      weight,
      bias,
      inputZero,
      weightZero,
      ArrayRef<int64_t>{padTop, padBottom, padLeft, padRight},
      ArrayRef<int64_t>{static_cast<int64_t>(operation.getStrideH()),
                        static_cast<int64_t>(operation.getStrideW())},
      ArrayRef<int64_t>{static_cast<int64_t>(operation.getDilationH()),
                        static_cast<int64_t>(operation.getDilationW())},
      rewriter.getF32Type());
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
    Value result;
    if (operation.getKind() == static_cast<int64_t>(PoolKind::Maximum)) {
      result = rewriter.create<tosa::MaxPool2dOp>(
        operation.getLoc(), outputType, input, kernel, stride, *padding);
    } else {
      Value zero =
        createSplat(rewriter,
                    operation.getLoc(),
                    RankedTensorType::get({1}, inputType.getElementType()),
                    0.0);
      result = rewriter.create<tosa::AvgPool2dOp>(operation.getLoc(),
                                                  outputType,
                                                  input,
                                                  zero,
                                                  zero,
                                                  kernel,
                                                  stride,
                                                  *padding,
                                                  rewriter.getF32Type());
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

    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPartialConversion(module, target, frozen))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
