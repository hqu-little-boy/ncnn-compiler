#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace mlir::ncnn {
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
  const int64_t requiredHeight =
    ((output.getShape()[1] - 1) * operation.getStrideH()) +
    operation.getKernelH();
  const int64_t requiredWidth =
    ((output.getShape()[2] - 1) * operation.getStrideW()) +
    operation.getKernelW();
  bottom = std::max(bottom, requiredHeight - input.getShape()[1] - top);
  right = std::max(right, requiredWidth - input.getShape()[2] - left);
  if (top < 0 || bottom < 0 || left < 0 || right < 0) {
    return failure();
  }
  return SmallVector<int64_t>{top, bottom, left, right};
}

class ConvertNCNNToTosaPass final
  : public PassWrapper<ConvertNCNNToTosaPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNCNNToTosaPass)

  StringRef getArgument() const final { return "convert-ncnn-to-tosa"; }
  StringRef getDescription() const final {
    return "Convert the SqueezeNet ncnn op set to TOSA";
  }

  void getDependentDialects(DialectRegistry& registry) const final {
    registry
      .insert<arith::ArithDialect, func::FuncDialect, tosa::TosaDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<ModelOp>().empty()) {
      module.emitError(
        "convert-ncnn-to-tosa requires ncnn.model to be "
        "converted to func.func first");
      signalPassFailure();
      return;
    }
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (failed(convertFunction(function))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  static LogicalResult convertFunction(func::FuncOp function) {
    if (function.isExternal()) {
      return success();
    }
    bool hasNCNNOps = false;
    function.walk([&](Operation* operation) {
      hasNCNNOps =
        hasNCNNOps || (operation->getDialect() != nullptr &&
                       operation->getDialect()->getNamespace() == "ncnn");
    });
    if (!hasNCNNOps) {
      return success();
    }
    if (!llvm::hasSingleElement(function.getBody())) {
      return function.emitOpError(
        "convert-ncnn-to-tosa requires a single-block function");
    }
    DenseMap<Value, Value> values;
    Block& block = function.getBody().front();
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToStart(&block);
    for (BlockArgument argument : block.getArguments()) {
      auto type = dyn_cast<RankedTensorType>(argument.getType());
      if (type != nullptr && type.getRank() == 3) {
        values[argument] =
          convertCHWToNHWC(builder, function.getLoc(), argument);
      } else {
        values[argument] = argument;
      }
    }

    SmallVector<Operation*> erase;
    for (Operation& operation : llvm::make_early_inc_range(block)) {
      if (isa<arith::ConstantOp, func::ReturnOp>(operation) ||
          (operation.getDialect() != nullptr &&
           operation.getDialect()->getNamespace() == "tosa")) {
        continue;
      }
      builder.setInsertionPoint(&operation);
      if (auto convolution = dyn_cast<ConvolutionOp>(operation)) {
        if (failed(convertConvolution(convolution, builder, values))) {
          return failure();
        }
      } else if (auto relu = dyn_cast<ReluOp>(operation)) {
        if (failed(convertRelu(relu, builder, values))) {
          return failure();
        }
      } else if (auto pooling = dyn_cast<PoolingOp>(operation)) {
        if (failed(convertPooling(pooling, builder, values))) {
          return failure();
        }
      } else if (auto split = dyn_cast<SplitOp>(operation)) {
        Value input = lookup(split, split.getInput(), values);
        if (!input) {
          return failure();
        }
        for (OpResult result : split.getResults()) {
          values[result] = input;
        }
      } else if (auto concat = dyn_cast<ConcatOp>(operation)) {
        if (failed(convertConcat(concat, builder, values))) {
          return failure();
        }
      } else if (auto dropout = dyn_cast<DropoutOp>(operation)) {
        if (failed(convertDropout(dropout, builder, values))) {
          return failure();
        }
      } else if (auto softmax = dyn_cast<SoftmaxOp>(operation)) {
        if (failed(convertSoftmax(softmax, builder, values))) {
          return failure();
        }
      } else if (operation.getDialect() != nullptr &&
                 operation.getDialect()->getNamespace() == "ncnn") {
        return operation.emitOpError(
          "is not supported by convert-ncnn-to-tosa");
      } else {
        continue;
      }
      erase.push_back(&operation);
    }

    auto returnOp = cast<func::ReturnOp>(block.getTerminator());
    builder.setInsertionPoint(returnOp);
    SmallVector<Value> returns;
    for (Value value : returnOp.getOperands()) {
      Value converted = values.lookup(value);
      if (!converted) {
        returns.push_back(value);
        continue;
      }
      auto originalType = dyn_cast<RankedTensorType>(value.getType());
      if (originalType != nullptr && originalType.getRank() == 3) {
        converted =
          convertNHWCToCHW(builder, returnOp.getLoc(), converted, originalType);
      }
      returns.push_back(converted);
    }
    returnOp.getOperandsMutable().assign(returns);
    for (Operation* operation : llvm::reverse(erase)) {
      operation->erase();
    }
    return success();
  }

  static Value lookup(Operation* operation,
                      Value source,
                      const DenseMap<Value, Value>& values) {
    Value converted = values.lookup(source);
    if (!converted) {
      operation->emitOpError("operand has no converted TOSA value");
    }
    return converted;
  }

  static LogicalResult convertConvolution(ConvolutionOp operation,
                                          OpBuilder& builder,
                                          DenseMap<Value, Value>& values) {
    if (operation.getInt8ScaleTerm() != 0) {
      return operation.emitOpError("quantized convolution is not supported");
    }
    const int64_t padTop = operation.getPadTopAttr().getInt();
    const int64_t padBottom = operation.getPadBottomAttr().getInt();
    const int64_t padLeft = operation.getPadLeftAttr().getInt();
    const int64_t padRight = operation.getPadRightAttr().getInt();
    if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
      return operation.emitOpError(
        "requires explicit non-negative padding; run normalize-ncnn first");
    }
    Value input = lookup(operation, operation.getInput(), values);
    if (!input) {
      return failure();
    }
    auto weightType = cast<RankedTensorType>(operation.getWeight().getType());
    auto ohwiType = getOHWIType(weightType);
    Value weight =
      builder.create<tosa::TransposeOp>(operation.getLoc(),
                                        ohwiType,
                                        operation.getWeight(),
                                        ArrayRef<int32_t>{0, 2, 3, 1});
    Value bias;
    if (operation.getHasBias()) {
      bias = operation.getBiasAndScales().front();
    } else {
      auto output = cast<RankedTensorType>(operation.getOutput().getType());
      auto biasType =
        RankedTensorType::get({output.getShape()[0]}, output.getElementType());
      bias = createSplat(builder, operation.getLoc(), biasType, 0.0);
    }
    auto outputType =
      getNHWCType(cast<RankedTensorType>(operation.getOutput().getType()));
    Value inputZero =
      createSplat(builder,
                  operation.getLoc(),
                  RankedTensorType::get(
                    {1}, cast<ShapedType>(input.getType()).getElementType()),
                  0.0);
    Value weightZero =
      createSplat(builder,
                  operation.getLoc(),
                  RankedTensorType::get(
                    {1}, cast<ShapedType>(weight.getType()).getElementType()),
                  0.0);
    Value result = builder.create<tosa::Conv2DOp>(
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
      builder.getF32Type());
    values[operation.getOutput()] = result;
    return success();
  }

  static LogicalResult convertRelu(ReluOp operation,
                                   OpBuilder& builder,
                                   DenseMap<Value, Value>& values) {
    Value input = lookup(operation, operation.getInput(), values);
    if (!input) {
      return failure();
    }
    auto type = cast<RankedTensorType>(input.getType());
    const double slope = operation.getNegativeSlope().convertToDouble();
    if (slope == 0.0) {
      auto element = cast<FloatType>(type.getElementType());
      values[operation.getOutput()] = builder.create<tosa::ClampOp>(
        operation.getLoc(),
        type,
        input,
        builder.getFloatAttr(element, 0.0),
        builder.getFloatAttr(element, std::numeric_limits<double>::infinity()));
      return success();
    }
    auto scalarType = getBroadcastScalarType(type);
    Value zero = createSplat(builder, operation.getLoc(), scalarType, 0.0);
    Value slopeValue =
      createSplat(builder, operation.getLoc(), scalarType, slope);
    Value shift = createI8Zero(builder, operation.getLoc());
    Value negative = builder.create<tosa::MulOp>(
      operation.getLoc(), type, input, slopeValue, shift);
    auto conditionType =
      RankedTensorType::get(type.getShape(), builder.getI1Type());
    Value condition = builder.create<tosa::GreaterEqualOp>(
      operation.getLoc(), conditionType, input, zero);
    values[operation.getOutput()] = builder.create<tosa::SelectOp>(
      operation.getLoc(), type, condition, input, negative);
    return success();
  }

  static LogicalResult convertPooling(PoolingOp operation,
                                      OpBuilder& builder,
                                      DenseMap<Value, Value>& values) {
    if (operation.getMode() == static_cast<int64_t>(PoolMode::Adaptive)) {
      return operation.emitOpError("adaptive pooling is not supported");
    }
    if (operation.getKind() == static_cast<int64_t>(PoolKind::Average) &&
        operation.getIncludePad()) {
      return operation.emitOpError(
        "average pooling with include_pad=true is not supported");
    }
    Value input = lookup(operation, operation.getInput(), values);
    if (!input) {
      return failure();
    }
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
      return operation.emitOpError("cannot map pooling padding to TOSA");
    }
    Value result;
    if (operation.getKind() == static_cast<int64_t>(PoolKind::Maximum)) {
      result = builder.create<tosa::MaxPool2dOp>(
        operation.getLoc(), outputType, input, kernel, stride, *padding);
    } else {
      Value zero =
        createSplat(builder,
                    operation.getLoc(),
                    RankedTensorType::get({1}, inputType.getElementType()),
                    0.0);
      result = builder.create<tosa::AvgPool2dOp>(operation.getLoc(),
                                                 outputType,
                                                 input,
                                                 zero,
                                                 zero,
                                                 kernel,
                                                 stride,
                                                 *padding,
                                                 builder.getF32Type());
    }
    if (global) {
      Value shape =
        createShape(builder, operation.getLoc(), sourceOutput.getShape());
      result = builder.create<tosa::ReshapeOp>(
        operation.getLoc(), sourceOutput, result, shape);
    }
    values[operation.getOutput()] = result;
    return success();
  }

  static LogicalResult convertConcat(ConcatOp operation,
                                     OpBuilder& builder,
                                     DenseMap<Value, Value>& values) {
    SmallVector<Value> inputs;
    for (Value input : operation.getInputs()) {
      Value converted = lookup(operation, input, values);
      if (!converted) {
        return failure();
      }
      inputs.push_back(converted);
    }
    auto sourceType = cast<RankedTensorType>(operation.getOutput().getType());
    auto outputType =
      sourceType.getRank() == 3 ? getNHWCType(sourceType) : sourceType;
    uint32_t axis = convertAxis(operation.getAxis(), sourceType.getRank());
    values[operation.getOutput()] = builder.create<tosa::ConcatOp>(
      operation.getLoc(), outputType, inputs, axis);
    return success();
  }

  static LogicalResult convertDropout(DropoutOp operation,
                                      OpBuilder& builder,
                                      DenseMap<Value, Value>& values) {
    Value input = lookup(operation, operation.getInput(), values);
    if (!input) {
      return failure();
    }
    const double scale = operation.getScale().convertToDouble();
    if (scale == 1.0) {
      values[operation.getOutput()] = input;
      return success();
    }
    auto type = cast<RankedTensorType>(input.getType());
    Value factor = createSplat(
      builder, operation.getLoc(), getBroadcastScalarType(type), scale);
    Value shift = createI8Zero(builder, operation.getLoc());
    values[operation.getOutput()] = builder.create<tosa::MulOp>(
      operation.getLoc(), type, input, factor, shift);
    return success();
  }

  static LogicalResult convertSoftmax(SoftmaxOp operation,
                                      OpBuilder& builder,
                                      DenseMap<Value, Value>& values) {
    Value input = lookup(operation, operation.getInput(), values);
    if (!input) {
      return failure();
    }
    auto type = cast<RankedTensorType>(input.getType());
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    const uint32_t axis =
      convertAxis(operation.getAxis(), sourceType.getRank());
    SmallVector<int64_t> reducedShape(type.getShape());
    reducedShape[axis] = 1;
    auto reducedType =
      RankedTensorType::get(reducedShape, type.getElementType());
    Value maximum = builder.create<tosa::ReduceMaxOp>(
      operation.getLoc(), reducedType, input, axis);
    Value shifted =
      builder.create<tosa::SubOp>(operation.getLoc(), type, input, maximum);
    Value exponent =
      builder.create<tosa::ExpOp>(operation.getLoc(), type, shifted);
    Value sum = builder.create<tosa::ReduceSumOp>(
      operation.getLoc(), reducedType, exponent, axis);
    Value reciprocal =
      builder.create<tosa::ReciprocalOp>(operation.getLoc(), reducedType, sum);
    Value shift = createI8Zero(builder, operation.getLoc());
    values[operation.getOutput()] = builder.create<tosa::MulOp>(
      operation.getLoc(), type, exponent, reciprocal, shift);
    return success();
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertNCNNToTosaPass() {
  return std::make_unique<ConvertNCNNToTosaPass>();
}

void registerNCNNToTosaPasses() {
  static PassRegistration<ConvertNCNNToTosaPass> registration;
}

}  // namespace mlir::ncnn
