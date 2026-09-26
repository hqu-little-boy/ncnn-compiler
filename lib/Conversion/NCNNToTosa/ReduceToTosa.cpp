// 归约与 inner-product 降低。
//
// 职责
//   把 ncnn.reduction / inner_product 降低为 tosa 的归约与矩阵乘组合。
//
// 不变量
//   * 归约轴按 ncnn 的 axis 语义转换（convertAxis），不做 numpy 广播；
//   * inner_product 的权重布局转换不改变数值。
//
// 顺序依赖
//   * 之后的矩阵乘内核选择归 StrategyNCNN / MatmulKernelNCNN。
//
// 明确不做
//   * 不做归约的数值等价改写（如求和顺序重排）。

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
using tosa_lowering::convertAxis;
using tosa_lowering::createI8Zero;
using tosa_lowering::createIndexConstant;
using tosa_lowering::createIntegerZero;
using tosa_lowering::createSplat;
using tosa_lowering::dequantizeAccumulator;
using tosa_lowering::getBroadcastScalarType;
using tosa_lowering::getIntegerAttrOr;
using tosa_lowering::getNHWCType;
using tosa_lowering::isRankedF32Tensor;
using tosa_lowering::isStaticF32Tensor;
using tosa_lowering::quantizeSignedI8;
using tosa_lowering::reshapeValue;
using tosa_lowering::restoreNCNNLayout;
using tosa_lowering::transposeOrFoldConstant;

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
      return operation.emitOpError("only f32 mean is supported");
    }
    SmallVector<int64_t> sourceAxes;
    if (operation.getReduceAll()) {
      for (int64_t axis = 0; axis < sourceInput.getRank(); ++axis) {
        sourceAxes.push_back(axis);
      }
    } else {
      for (int64_t axis : operation.getAxes()) {
        sourceAxes.push_back(axis < 0 ? axis + sourceInput.getRank() : axis);
      }
    }
    SmallVector<uint32_t> axes;
    int64_t reducedElements = 1;
    bool hasDynamicReducedExtent = false;
    for (int64_t sourceAxis : sourceAxes) {
      if (sourceAxis < 0) {
        sourceAxis += sourceInput.getRank();
      }
      axes.push_back(convertAxis(sourceAxis, sourceInput.getRank()));
      const int64_t extent = sourceInput.getShape()[sourceAxis];
      if (ShapedType::isDynamic(extent)) {
        hasDynamicReducedExtent = true;
        continue;
      }
      if (llvm::MulOverflow(reducedElements, extent, reducedElements)) {
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
    auto scalarType = getBroadcastScalarType(reducedType);
    Value factor;
    if (hasDynamicReducedExtent) {
      Value count =
        createIndexConstant(rewriter, operation.getLoc(), reducedElements);
      for (int64_t sourceAxis : sourceAxes) {
        if (!sourceInput.isDynamicDim(sourceAxis)) {
          continue;
        }
        Value extent = rewriter.create<tensor::DimOp>(
          operation.getLoc(),
          adaptor.getInput(),
          convertAxis(sourceAxis, sourceInput.getRank()));
        count =
          rewriter.create<arith::MulIOp>(operation.getLoc(), count, extent);
      }
      Value countI64 = rewriter.create<arith::IndexCastUIOp>(
        operation.getLoc(), rewriter.getI64Type(), count);
      Value countFloat = rewriter.create<arith::UIToFPOp>(
        operation.getLoc(), rewriter.getF32Type(), countI64);
      Value coefficient = rewriter.create<arith::ConstantFloatOp>(
        operation.getLoc(), rewriter.getF32Type(), operation.getCoeff());
      Value scale = rewriter.create<arith::DivFOp>(
        operation.getLoc(), coefficient, countFloat);
      factor = rewriter.create<tensor::FromElementsOp>(
        operation.getLoc(), scalarType, scale);
    } else {
      const double scale = operation.getCoeff().convertToDouble() /
                           static_cast<double>(reducedElements);
      factor = createSplat(rewriter, operation.getLoc(), scalarType, scale);
    }
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
      if (convertedOutputType.hasStaticShape()) {
        result = reshapeValue(
          rewriter, operation.getLoc(), result, convertedOutputType);
      } else {
        llvm::SmallDenseSet<int64_t> reducedSourceAxes(sourceAxes.begin(),
                                                       sourceAxes.end());
        SmallVector<int64_t> retainedSourceAxes;
        SmallVector<unsigned> retainedPhysicalAxes;
        for (int64_t sourceAxis = 0; sourceAxis < sourceInput.getRank();
             ++sourceAxis) {
          if (!reducedSourceAxes.contains(sourceAxis)) {
            retainedSourceAxes.push_back(sourceAxis);
            retainedPhysicalAxes.push_back(
              convertAxis(sourceAxis, sourceInput.getRank()));
          }
        }
        SmallVector<unsigned> physicalOrder(retainedPhysicalAxes);
        llvm::sort(physicalOrder);
        SmallVector<int64_t> physicalShape;
        SmallVector<std::optional<unsigned>> sourceDimensions;
        for (unsigned physicalAxis : physicalOrder) {
          physicalShape.push_back(reducedType.getShape()[physicalAxis]);
          sourceDimensions.push_back(physicalAxis);
        }
        auto physicalType = RankedTensorType::get(
          physicalShape, convertedOutputType.getElementType());
        result = reshapeValue(
          rewriter, operation.getLoc(), result, physicalType, sourceDimensions);
        SmallVector<int32_t> permutation;
        for (int64_t sourceAxis : retainedSourceAxes) {
          unsigned physicalAxis =
            convertAxis(sourceAxis, sourceInput.getRank());
          permutation.push_back(static_cast<int32_t>(
            llvm::find(physicalOrder, physicalAxis) - physicalOrder.begin()));
        }
        if (!llvm::is_sorted(permutation)) {
          result = rewriter.create<tosa::TransposeOp>(
            operation.getLoc(), convertedOutputType, result, permutation);
        }
      }
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
    const int64_t scaleTerm = getIntegerAttrOr(operation, "int8_scale_term", 0);
    const bool quantized = scaleTerm != 0;
    auto inputElement =
      operands.empty()
        ? Type{}
        : cast<ShapedType>(operands[0].getType()).getElementType();
    auto weightElement =
      operands.size() < 2
        ? Type{}
        : cast<ShapedType>(operands[1].getType()).getElementType();
    if (operands.size() < 2 || operation->getNumResults() != 1 ||
        (quantized ? (!inputElement.isF32() && !inputElement.isInteger(8))
                   : !isRankedF32Tensor(operation->getOperand(0).getType())) ||
        (quantized ? (!weightElement.isF32() && !weightElement.isInteger(8))
                   : !isStaticF32Tensor(operation->getOperand(1).getType())) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError(
        "supports ranked f32/i8 input and static f32/i8 weight tensors for "
        "quantized InnerProduct");
    }
    auto sourceInput =
      cast<RankedTensorType>(operation->getOperand(0).getType());
    auto weightType =
      cast<RankedTensorType>(operation->getOperand(1).getType());
    auto outputType = cast<RankedTensorType>(operation->getResult(0).getType());
    const bool dynamicMatrix =
      sourceInput.getRank() == 2 && sourceInput.isDynamicDim(0) &&
      !sourceInput.isDynamicDim(1) && outputType.getRank() == 2;
    if (weightType.getRank() != 2 ||
        (!dynamicMatrix && outputType.getRank() != 1)) {
      return operation->emitOpError(
        "requires [O,K] weights and a rank-1 output");
    }
    const int64_t outputs = weightType.getShape()[0];
    const int64_t inputs = weightType.getShape()[1];
    if ((!dynamicMatrix && sourceInput.getNumElements() != inputs) ||
        (dynamicMatrix && sourceInput.getShape()[1] != inputs) ||
        outputType.getShape().back() != outputs) {
      return operation->emitOpError("input/weight/output sizes do not match");
    }
    Value input = restoreNCNNLayout(
      rewriter, operation->getLoc(), operands[0], sourceInput);
    const bool hasBias = getIntegerAttrOr(operation, "has_bias", 0) != 0;
    const unsigned scaleOffset = hasBias ? 3 : 2;
    Value weightScale;
    Value inputScale;
    Value sourceWeight = operands[1];
    if (quantized) {
      weightScale = operands[scaleOffset];
      inputScale = operands[scaleOffset + 1];
      input =
        quantizeSignedI8(rewriter, operation->getLoc(), input, inputScale);
      sourceWeight = quantizeSignedI8(
        rewriter, operation->getLoc(), sourceWeight, weightScale, 0);
    }
    const int64_t rows = dynamicMatrix ? ShapedType::kDynamic : 1;
    auto matrixInputType = RankedTensorType::get(
      {1, rows, inputs}, cast<ShapedType>(input.getType()).getElementType());
    if (dynamicMatrix) {
      SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
      Value dynamicRows =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 0);
      input = rewriter.create<tensor::ExpandShapeOp>(
        operation->getLoc(),
        matrixInputType,
        input,
        reassociation,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                  dynamicRows,
                                  rewriter.getIndexAttr(inputs)});
    } else {
      input =
        reshapeValue(rewriter, operation->getLoc(), input, matrixInputType);
    }
    auto transposedWeightType = RankedTensorType::get(
      {inputs, outputs},
      cast<ShapedType>(sourceWeight.getType()).getElementType());
    Value transposedWeight = transposeOrFoldConstant(rewriter,
                                                     operation->getLoc(),
                                                     sourceWeight,
                                                     transposedWeightType,
                                                     ArrayRef<int32_t>{1, 0});
    auto matrixWeightType = RankedTensorType::get(
      {1, inputs, outputs}, transposedWeightType.getElementType());
    Value matrixWeight = reshapeValue(
      rewriter, operation->getLoc(), transposedWeight, matrixWeightType);
    auto matrixOutputType = RankedTensorType::get(
      {1, rows, outputs},
      quantized ? rewriter.getI32Type() : outputType.getElementType());
    Value result;
    if (quantized) {
      Value zero =
        createIntegerZero(rewriter,
                          operation->getLoc(),
                          RankedTensorType::get({1}, rewriter.getI8Type()));
      result = rewriter.create<tosa::MatMulOp>(
        operation->getLoc(), matrixOutputType, input, matrixWeight, zero, zero);
    } else {
      result = rewriter.create<tosa::MatMulOp>(
        operation->getLoc(), matrixOutputType, input, matrixWeight);
    }
    if (quantized) {
      result = dequantizeAccumulator(
        rewriter, operation->getLoc(), result, weightScale, inputScale, 2);
      matrixOutputType = cast<RankedTensorType>(result.getType());
    }
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
    if (dynamicMatrix) {
      SmallVector<ReassociationIndices> reassociation = {{0, 1}, {2}};
      result = rewriter.create<tensor::CollapseShapeOp>(
        operation->getLoc(), outputType, result, reassociation);
    } else {
      result = reshapeValue(rewriter, operation->getLoc(), result, outputType);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

}  // namespace

void populateReduceToTosaPatterns(RewritePatternSet& patterns,
                                  const TypeConverter& typeConverter,
                                  MLIRContext* context) {
  patterns.add<ConvertReduction>(typeConverter, context);
  patterns.add<ConvertInnerProduct>(
    typeConverter, context, contract::kLayerInnerProduct);
}

}  // namespace mlir::ncnn
