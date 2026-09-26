// 量化族降低：batchnorm、quantize、dequantize、requantize、cast、
// zero-point cast。
//
// 职责
//   在保持 ncnn 标度/零点语义的前提下，把量化边界显式化为 tosa 的
//   逐元素算子。
//
// 不变量
//   * 标度与零点只能来自操作数，不得从类型推断；
//   * roundStoragePrecision 统一收口存储精度，单个 pattern 不得自插 cast；
//   * requantize 的舍入模式固定（与 ncnn 一致），不随 fastmath 漂移。
//
// 顺序依赖
//   * 必须在 FuseQuantChainNCNN 之后（量化链已折叠）；
//   * 之后由 MatmulKernelNCNN 的 int8 内核消费。
//
// 明确不做
//   * 不做量化策略决策（标度是输入）；
//   * 不做伪量化训练相关变换。

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosaPatterns.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/TosaLoweringUtils.hpp"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"
#include "ncnn-mlir/Support/ConstantFold.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/ModelLedger.hpp"
#include "ncnn-mlir/Support/Precision.hpp"

namespace mlir::ncnn {
namespace {

// Shared lowering helpers used by this family (clang-tidy forbids
// `using namespace`, so imports are per-name).
using tosa_lowering::applyLowPrecisionBoundary;
using tosa_lowering::convertFloatingTensor;
using tosa_lowering::createI8Zero;
using tosa_lowering::createSplat;
using tosa_lowering::dequantizeNcnn;
using tosa_lowering::getBroadcastScalarType;
using tosa_lowering::getDynamicSizeValues;
using tosa_lowering::quantizeSignedI8;
using tosa_lowering::reshapeValue;

Value convertSignedI8ToF32(OpBuilder& builder, Location location, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto outputType = inputType.clone(builder.getF32Type());
  Value empty = builder.create<tensor::EmptyOp>(
    location,
    outputType.getShape(),
    outputType.getElementType(),
    getDynamicSizeValues(builder, location, input, outputType));
  auto converted = builder.create<linalg::MapOp>(
    location,
    ValueRange{input},
    empty,
    [](OpBuilder& nested, Location nestedLocation, ValueRange values) {
      Value result = nested.create<arith::SIToFPOp>(
        nestedLocation, nested.getF32Type(), values.front());
      nested.create<linalg::YieldOp>(nestedLocation, result);
    });
  return converted->getResult(0);
}

class ConvertBatchNorm final : public OpConversionPattern<BatchNormOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    BatchNormOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    Value input = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, adaptor.getInput());
    auto outputType = cast<RankedTensorType>(input.getType());
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
      operation.getLoc(), outputType, input, scale, shift);
    result = rewriter.create<tosa::AddOp>(
      operation.getLoc(), outputType, result, offset);
    result = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, result);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertQuantize final : public OpConversionPattern<QuantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    QuantizeOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto scaleType = cast<RankedTensorType>(adaptor.getScale().getType());
    std::optional<unsigned> scaleDimension;
    if (scaleType.getShape()[0] != 1) {
      scaleDimension = inputType.getRank() == 4 ? 3U : 0U;
    }
    Value result = quantizeSignedI8(rewriter,
                                    operation.getLoc(),
                                    adaptor.getInput(),
                                    adaptor.getScale(),
                                    scaleDimension);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertDequantize final : public OpConversionPattern<DequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    DequantizeOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Value bias =
      adaptor.getBias().empty() ? Value{} : adaptor.getBias().front();
    Value result = dequantizeNcnn(rewriter,
                                  operation.getLoc(),
                                  adaptor.getInput(),
                                  adaptor.getScale(),
                                  bias);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertRequantize final : public OpConversionPattern<RequantizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    RequantizeOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    if (operation.getActivationType() != 0 &&
        operation.getActivationType() != 1) {
      return operation.emitOpError(
        "lowering supports no activation or ReLU only");
    }
    Value bias =
      adaptor.getBias().empty() ? Value{} : adaptor.getBias().front();
    Value floating = dequantizeNcnn(rewriter,
                                    operation.getLoc(),
                                    adaptor.getInput(),
                                    adaptor.getInputScale(),
                                    bias);
    auto floatingType = cast<RankedTensorType>(floating.getType());
    if (operation.getActivationType() == 1) {
      floating = rewriter.create<tosa::ClampOp>(
        operation.getLoc(),
        floatingType,
        floating,
        rewriter.getF32FloatAttr(0.0),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity()));
    }
    auto scaleType = cast<RankedTensorType>(adaptor.getOutputScale().getType());
    std::optional<unsigned> scaleDimension;
    if (scaleType.getShape()[0] != 1) {
      scaleDimension = floatingType.getRank() == 4 ? 3U : 0U;
    }
    Value result = quantizeSignedI8(rewriter,
                                    operation.getLoc(),
                                    floating,
                                    adaptor.getOutputScale(),
                                    scaleDimension);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertCast final : public OpConversionPattern<CastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    CastOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType = cast<RankedTensorType>(
      getTypeConverter()->convertType(operation.getOutput().getType()));
    if (inputType.getElementType() == outputType.getElementType()) {
      rewriter.replaceOp(operation, adaptor.getInput());
      return success();
    }
    Value result =
      inputType.getElementType().isSignlessInteger(8)
        ? convertSignedI8ToF32(rewriter, operation.getLoc(), adaptor.getInput())
        : convertFloatingTensor(rewriter,
                                operation.getLoc(),
                                adaptor.getInput(),
                                outputType.getElementType());
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertZeroPointCast final : public OpConversionPattern<ZeroPointCastOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ZeroPointCastOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto outputType = cast<RankedTensorType>(
      getTypeConverter()->convertType(operation.getOutput().getType()));
    Value empty = rewriter.create<tensor::EmptyOp>(
      operation.getLoc(),
      outputType.getShape(),
      outputType.getElementType(),
      getDynamicSizeValues(
        rewriter, operation.getLoc(), adaptor.getInput(), outputType));
    const bool toUnsigned = operation.getToUnsigned();
    const int64_t zeroPoint = operation.getZeroPoint();
    auto converted = rewriter.create<linalg::MapOp>(
      operation.getLoc(),
      ValueRange{adaptor.getInput()},
      empty,
      [toUnsigned, zeroPoint](
        OpBuilder& nested, Location nestedLocation, ValueRange values) {
        Type wide = nested.getI16Type();
        Value source = values.front();
        if (!toUnsigned) {
          source = nested
                     .create<UnrealizedConversionCastOp>(
                       nestedLocation, nested.getI8Type(), source)
                     .getResult(0);
        }
        Value widened =
          toUnsigned
            ? Value(nested.create<arith::ExtSIOp>(nestedLocation, wide, source))
            : Value(
                nested.create<arith::ExtUIOp>(nestedLocation, wide, source));
        Value offset = nested.create<arith::ConstantOp>(
          nestedLocation, nested.getI16IntegerAttr(zeroPoint));
        Value rebased = toUnsigned ? Value(nested.create<arith::AddIOp>(
                                       nestedLocation, widened, offset))
                                   : Value(nested.create<arith::SubIOp>(
                                       nestedLocation, widened, offset));
        const int64_t minimum = toUnsigned ? 0 : -128;
        const int64_t maximum = toUnsigned ? 255 : 127;
        Value lower = nested.create<arith::ConstantOp>(
          nestedLocation, nested.getI16IntegerAttr(minimum));
        Value upper = nested.create<arith::ConstantOp>(
          nestedLocation, nested.getI16IntegerAttr(maximum));
        Value clamped = nested.create<arith::MaxSIOp>(
          nestedLocation,
          nested.create<arith::MinSIOp>(nestedLocation, rebased, upper),
          lower);
        Value result = nested.create<arith::TruncIOp>(
          nestedLocation, nested.getI8Type(), clamped);
        if (toUnsigned) {
          result =
            nested
              .create<UnrealizedConversionCastOp>(
                nestedLocation,
                IntegerType::get(nested.getContext(),
                                 8,
                                 IntegerType::SignednessSemantics::Unsigned),
                result)
              .getResult(0);
        }
        nested.create<linalg::YieldOp>(nestedLocation, result);
      });
    rewriter.replaceOp(operation, converted.getResults());
    return success();
  }
};

}  // namespace

void populateQuantToTosaPatterns(RewritePatternSet& patterns,
                                 const TypeConverter& typeConverter,
                                 MLIRContext* context) {
  patterns.add<ConvertBatchNorm,
               ConvertQuantize,
               ConvertDequantize,
               ConvertRequantize,
               ConvertCast,
               ConvertZeroPointCast>(typeConverter, context);
}

}  // namespace mlir::ncnn
