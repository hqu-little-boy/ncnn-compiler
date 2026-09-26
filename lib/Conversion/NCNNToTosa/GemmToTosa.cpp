// GEMM 降低。
//
// 职责
//   把 ncnn.gemm 降低为 tosa 的矩阵乘 + 广播偏置，并按需插入行标度
//   反量化（computeGemmRowScales / dequantizeGemmAccumulator）。
//
// 不变量
//   * 累加器宽度按输入标度推导，不得窄化；
//   * 偏置广播语义与 ncnn 一致（按行）。
//
// 顺序依赖
//   * 之后由 StrategyNCNN / MatmulKernelNCNN 选择内核。
//
// 明确不做
//   * 不做转置/折叠决策（StrategyNCNN）；
//   * 不做 packing（PackStaticMatmulNCNN）。

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
using tosa_lowering::convertI32ToF32;
using tosa_lowering::createI8Zero;
using tosa_lowering::createIntegerZero;
using tosa_lowering::createSplat;
using tosa_lowering::getBroadcastScalarType;
using tosa_lowering::getDynamicSizeValues;
using tosa_lowering::quantizeSignedI8;
using tosa_lowering::reshapeValue;
using tosa_lowering::transposeOrFoldConstant;

Value computeGemmRowScales(OpBuilder& builder, Location location, Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto scaleType =
    RankedTensorType::get({inputType.getShape()[0]}, builder.getF32Type());
  Value empty = builder.create<tensor::EmptyOp>(
    location,
    scaleType.getShape(),
    scaleType.getElementType(),
    getDynamicSizeValues(builder, location, input, scaleType));
  Value zero =
    builder.create<arith::ConstantOp>(location, builder.getF32FloatAttr(0.0));
  Value initialized =
    builder.create<linalg::FillOp>(location, zero, empty).getResult(0);
  AffineExpr row = builder.getAffineDimExpr(0);
  AffineExpr column = builder.getAffineDimExpr(1);
  AffineMap inputMap =
    AffineMap::get(2, 0, {row, column}, builder.getContext());
  AffineMap outputMap = AffineMap::get(2, 0, row, builder.getContext());
  auto maximum = builder.create<linalg::GenericOp>(
    location,
    scaleType,
    ValueRange{input},
    ValueRange{initialized},
    ArrayRef<AffineMap>{inputMap, outputMap},
    ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                  utils::IteratorType::reduction},
    [](OpBuilder& nested, Location nestedLocation, ValueRange values) {
      Value absolute =
        nested.create<math::AbsFOp>(nestedLocation, values.front());
      Value result = nested.create<arith::MaximumFOp>(
        nestedLocation, absolute, values.back());
      nested.create<linalg::YieldOp>(nestedLocation, result);
    });
  Value resultEmpty = builder.create<tensor::EmptyOp>(
    location,
    scaleType.getShape(),
    scaleType.getElementType(),
    getDynamicSizeValues(builder, location, input, scaleType));
  auto scales = builder.create<linalg::MapOp>(
    location,
    maximum.getResults(),
    resultEmpty,
    [](OpBuilder& nested, Location nestedLocation, ValueRange values) {
      Value zero = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getF32FloatAttr(0.0));
      Value one = nested.create<arith::ConstantOp>(nestedLocation,
                                                   nested.getF32FloatAttr(1.0));
      Value maximum = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getF32FloatAttr(127.0));
      Value empty = nested.create<arith::CmpFOp>(
        nestedLocation, arith::CmpFPredicate::OEQ, values.front(), zero);
      Value computed =
        nested.create<arith::DivFOp>(nestedLocation, maximum, values.front());
      Value result =
        nested.create<arith::SelectOp>(nestedLocation, empty, one, computed);
      nested.create<linalg::YieldOp>(nestedLocation, result);
    });
  return scales->getResult(0);
}

Value dequantizeGemmAccumulator(OpBuilder& builder,
                                Location location,
                                Value accumulator,
                                Value rowScale,
                                Value weightScale) {
  Value converted = convertI32ToF32(builder, location, accumulator);
  auto outputType = cast<RankedTensorType>(converted.getType());
  Value empty = builder.create<tensor::EmptyOp>(
    location,
    outputType.getShape(),
    outputType.getElementType(),
    getDynamicSizeValues(builder, location, converted, outputType));
  AffineMap identity = builder.getMultiDimIdentityMap(3);
  AffineMap rowMap =
    AffineMap::get(3, 0, builder.getAffineDimExpr(1), builder.getContext());
  AffineMap scalarMap = AffineMap::get(
    3, 0, builder.getAffineConstantExpr(0), builder.getContext());
  SmallVector<utils::IteratorType> iterators(3, utils::IteratorType::parallel);
  auto result = builder.create<linalg::GenericOp>(
    location,
    outputType,
    ValueRange{converted, rowScale, weightScale},
    ValueRange{empty},
    ArrayRef<AffineMap>{identity, rowMap, scalarMap, identity},
    iterators,
    [](OpBuilder& nested, Location nestedLocation, ValueRange values) {
      Value combined =
        nested.create<arith::MulFOp>(nestedLocation, values[1], values[2]);
      Value value =
        nested.create<arith::DivFOp>(nestedLocation, values[0], combined);
      nested.create<linalg::YieldOp>(nestedLocation, value);
    });
  return result->getResult(0);
}

class ConvertGemm final : public OpConversionPattern<GemmOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    GemmOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    const bool quantized = operation.getInt8ScaleTerm() != 0;
    Value sourceInput = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, adaptor.getInput());
    Value sourceWeight = adaptor.getWeight();
    auto weightType = cast<RankedTensorType>(sourceWeight.getType());
    if (!quantized && !weightType.getElementType().isF32()) {
      sourceWeight = convertFloatingTensor(
        rewriter, operation.getLoc(), sourceWeight, rewriter.getF32Type());
      weightType = cast<RankedTensorType>(sourceWeight.getType());
    }
    auto inputType = cast<RankedTensorType>(sourceInput.getType());
    auto outputType = cast<RankedTensorType>(operation.getOutput().getType());
    const int64_t m = inputType.getShape()[0];
    const int64_t k = inputType.getShape()[1];
    const int64_t n = weightType.getShape()[0];
    Value rowScale;
    if (quantized) {
      rowScale =
        computeGemmRowScales(rewriter, operation.getLoc(), sourceInput);
      sourceInput = quantizeSignedI8(
        rewriter, operation.getLoc(), sourceInput, rowScale, 0);
      inputType = cast<RankedTensorType>(sourceInput.getType());
    }
    auto matrixInputType =
      RankedTensorType::get({1, m, k}, inputType.getElementType());
    Value input;
    if (inputType.hasStaticShape()) {
      input = reshapeValue(
        rewriter, operation.getLoc(), sourceInput, matrixInputType);
    } else {
      SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
      SmallVector<OpFoldResult> outputShape;
      outputShape.push_back(rewriter.getIndexAttr(1));
      Value dynamicM =
        rewriter.create<tensor::DimOp>(operation.getLoc(), sourceInput, 0);
      outputShape.push_back(dynamicM);
      outputShape.push_back(rewriter.getIndexAttr(k));
      input = rewriter.create<tensor::ExpandShapeOp>(operation.getLoc(),
                                                     matrixInputType,
                                                     sourceInput,
                                                     reassociation,
                                                     outputShape);
    }
    auto matrixWeightType =
      RankedTensorType::get({1, n, k}, weightType.getElementType());
    Value weight = reshapeValue(
      rewriter, operation.getLoc(), sourceWeight, matrixWeightType);
    auto transposedWeightType =
      RankedTensorType::get({1, k, n}, weightType.getElementType());
    weight = transposeOrFoldConstant(rewriter,
                                     operation.getLoc(),
                                     weight,
                                     transposedWeightType,
                                     ArrayRef<int32_t>{0, 2, 1});
    auto matrixOutputType =
      RankedTensorType::get({1, m, n},
                            quantized ? static_cast<Type>(rewriter.getI32Type())
                                      : outputType.getElementType());
    Value result;
    if (quantized) {
      Value zero =
        createIntegerZero(rewriter,
                          operation.getLoc(),
                          RankedTensorType::get({1}, rewriter.getI8Type()));
      result = rewriter.create<tosa::MatMulOp>(
        operation.getLoc(), matrixOutputType, input, weight, zero, zero);
      result = dequantizeGemmAccumulator(rewriter,
                                         operation.getLoc(),
                                         result,
                                         rowScale,
                                         adaptor.getScales().front());
      matrixOutputType = cast<RankedTensorType>(result.getType());
    } else {
      result = rewriter.create<tosa::MatMulOp>(
        operation.getLoc(), matrixOutputType, input, weight);
    }
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
    if (outputType.hasStaticShape()) {
      result = reshapeValue(rewriter, operation.getLoc(), result, outputType);
    } else {
      SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
      result = rewriter.create<tensor::CollapseShapeOp>(
        operation.getLoc(), outputType, result, reassociation);
    }
    result = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, result);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

}  // namespace

void populateGemmToTosaPatterns(RewritePatternSet& patterns,
                                const TypeConverter& typeConverter,
                                MLIRContext* context) {
  patterns.add<ConvertGemm>(typeConverter, context);
}

}  // namespace mlir::ncnn
