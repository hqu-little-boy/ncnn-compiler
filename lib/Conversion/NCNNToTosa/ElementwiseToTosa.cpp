// 逐元素族降低：relu、sigmoid、hard-sigmoid/hard-swish、swish、tanh、
// GELU、dropout、binary、unary。
//
// 职责
//   把逐元素算子一对一映射为 tosa 的对应算子，必要时插入低精度边界。
//
// 不变量
//   * 广播规则按 ncnn 语义（matchBroadcastRank），不做 numpy 式扩展；
//   * 激活的数值近似（如 GELU 的 tanh 近似）必须与 ncnn 一致。
//
// 顺序依赖
//   * 无特殊顺序约束；op 名互不重叠。
//
// 明确不做
//   * 不融合激活进 conv/matmul（FuseLinalgEpilogue 的职责）；
//   * 不做激活的数学等价替换。

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
using tosa_lowering::createI8Zero;
using tosa_lowering::createSplat;
using tosa_lowering::getBroadcastScalarType;
using tosa_lowering::getDynamicSizeValues;
using tosa_lowering::getFloatAttrOr;
using tosa_lowering::getNHWCType;
using tosa_lowering::getRequiredIntegerAttr;
using tosa_lowering::isRankedF32Tensor;
using tosa_lowering::reshapeValue;

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
      Attribute minimum;
      Attribute maximum;
      if (auto element = dyn_cast<FloatType>(type.getElementType())) {
        minimum = rewriter.getFloatAttr(element, 0.0);
        maximum = rewriter.getFloatAttr(
          element, std::numeric_limits<double>::infinity());
      } else if (auto element = dyn_cast<IntegerType>(type.getElementType())) {
        minimum = rewriter.getIntegerAttr(element, 0);
        maximum = rewriter.getIntegerAttr(
          element, (1LL << (element.getWidth() - 1)) - 1);
      } else {
        return operation.emitOpError("requires floating or integer input");
      }
      Value result = rewriter.create<tosa::ClampOp>(
        operation.getLoc(), type, input, minimum, maximum);
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

class ConvertSigmoid final : public ConversionPattern {
 public:
  ConvertSigmoid(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, contract::kLayerSigmoid, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() != 1 || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked f32 tensor only");
    }
    Value input = applyLowPrecisionBoundary(
      rewriter, operation->getLoc(), operation, operands.front());
    auto type = cast<RankedTensorType>(input.getType());
    Value clamped = rewriter.create<tosa::ClampOp>(
      operation->getLoc(),
      type,
      input,
      rewriter.getF32FloatAttr(-88.3762626647949F),
      rewriter.getF32FloatAttr(88.3762626647949F));
    Value result =
      rewriter.create<tosa::SigmoidOp>(operation->getLoc(), type, clamped);
    result = applyLowPrecisionBoundary(
      rewriter, operation->getLoc(), operation, result);
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
        !isRankedF32Tensor(operands.front().getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked f32 tensor only");
    }
    Value input = applyLowPrecisionBoundary(
      rewriter, operation->getLoc(), operation, operands.front());
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
    gate =
      applyLowPrecisionBoundary(rewriter, operation->getLoc(), operation, gate);
    rewriter.replaceOp(operation, gate);
    return success();
  }

 private:
  bool swish_;
};

class ConvertSwish final : public OpConversionPattern<SwishOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    SwishOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto type = cast<RankedTensorType>(adaptor.getInput().getType());
    Value sigmoid = rewriter.create<tosa::SigmoidOp>(
      operation.getLoc(), type, adaptor.getInput());
    Value shift = createI8Zero(rewriter, operation.getLoc());
    rewriter.replaceOp(
      operation,
      rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, adaptor.getInput(), sigmoid, shift));
    return success();
  }
};

class ConvertTanH final : public OpConversionPattern<TanHOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    TanHOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto type = cast<RankedTensorType>(adaptor.getInput().getType());
    rewriter.replaceOp(operation,
                       rewriter.create<tosa::TanhOp>(
                         operation.getLoc(), type, adaptor.getInput()));
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
    Value input = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, adaptor.getInput());
    auto type = cast<RankedTensorType>(input.getType());
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
        operation.getLoc(), type, input, inverseSqrtTwo, shift);
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
      Value cubic = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, input, input, shift);
      cubic = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, cubic, input, shift);
      Value cubicScale = createSplat(
        rewriter, operation.getLoc(), getBroadcastScalarType(type), 0.044715);
      cubic = rewriter.create<tosa::MulOp>(
        operation.getLoc(), type, cubic, cubicScale, shift);
      Value sum =
        rewriter.create<tosa::AddOp>(operation.getLoc(), type, input, cubic);
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
      operation.getLoc(), type, input, activation, shift);
    result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, result, half, shift);
    result = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, result);
    rewriter.replaceOp(operation, result);
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
        operation->getNumResults() != 1) {
      return operation->emitOpError(
        "supports one result and one or two operands");
    }
    auto sourceOutput =
      cast<RankedTensorType>(operation->getResult(0).getType());
    const bool signedI8 = sourceOutput.getElementType().isSignlessInteger(8);
    if (!sourceOutput.getElementType().isF32() && !signedI8) {
      return operation->emitOpError("supports f32 or signed i8 tensors only");
    }
    auto outputType =
      sourceOutput.getRank() == 3 ? getNHWCType(sourceOutput) : sourceOutput;
    FailureOr<int64_t> opType = getRequiredIntegerAttr(operation, "op_type");
    if (failed(opType)) {
      return failure();
    }
    if (signedI8 && (operands.size() != 2 || *opType != 4)) {
      return operation->emitOpError(
        "signed i8 BinaryOp only supports two-input maximum");
    }
    SmallVector<Value> values(operands.begin(), operands.end());
    if (!signedI8) {
      for (Value& value : values) {
        value = applyLowPrecisionBoundary(
          rewriter, operation->getLoc(), operation, value);
      }
    }
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
    if (!signedI8) {
      result = applyLowPrecisionBoundary(
        rewriter, operation->getLoc(), operation, result);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertUnary final : public OpConversionPattern<UnaryOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    UnaryOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    if (!isRankedF32Tensor(operation.getInput().getType()) ||
        !isRankedF32Tensor(operation.getOutput().getType())) {
      return operation->emitOpError("requires a ranked f32 tensor");
    }
    if (operation.getOpType() != 4) {
      return operation->emitOpError(
        "only the square operation (op_type=4) is supported");
    }
    Value input = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, adaptor.getInput());
    auto outputType = cast<RankedTensorType>(input.getType());
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), outputType, input, input, shift);
    result = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, result);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

}  // namespace

void populateElementwiseToTosaPatterns(RewritePatternSet& patterns,
                                       const TypeConverter& typeConverter,
                                       MLIRContext* context) {
  patterns.add<ConvertRelu,
               ConvertDropout,
               ConvertSigmoid,
               ConvertSwish,
               ConvertTanH,
               ConvertGELU>(typeConverter, context);
  patterns.add<ConvertHardActivation>(
    typeConverter, context, contract::kLayerHardSigmoid, false);
  patterns.add<ConvertHardActivation>(
    typeConverter, context, contract::kLayerHardSwish, true);
  patterns.add<ConvertBinary>(typeConverter, context, contract::kLayerBinary);
  patterns.add<ConvertUnary>(typeConverter, context);
}

}  // namespace mlir::ncnn
