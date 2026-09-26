// NCNN -> TOSA 共享 lowering helper 的实现。
//
// 职责
//   各算子族都需要的构建块：形状重塑、CHW<->NHWC 布局转换、穿过视图
//   的常量折叠、低精度边界、动态尺寸搬运。单族专用的 helper 留在各族
//   TU 里，本文件只放真共享的部分。
//
// 不变量
//   * 常量查找只穿透纯视图（tensor.cast / collapse_shape / expand_shape），
//     不穿透 extract_slice（子集）与 from_elements（合成）；
//   * 低精度边界统一由 applyLowPrecisionBoundary 收口，helper 之外不插 cast；
//   * 动态尺寸只从 tensor.dim 取，不得从内容推断。
//
// 顺序依赖
//   * 无 pass 顺序约束（纯函数库）；
//   * 被所有 NCNNToTosa 族 TU 通过 using 声明引用（禁用 using namespace）。
//
// 明确不做
//   * 不做量化策略决策；
//   * 不做布局决策；
//   * 不持有状态（全部自由函数）。

#include "ncnn-mlir/Conversion/NCNNToTosa/TosaLoweringUtils.hpp"

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
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"
#include "ncnn-mlir/Support/ConstantFold.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/ModelLedger.hpp"
#include "ncnn-mlir/Support/Precision.hpp"

namespace mlir::ncnn::tosa_lowering {

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

bool usesFP16Arithmetic(Operation* operation) {
  auto function = operation->getParentOfType<func::FuncOp>();
  auto precision = function->getAttrOfType<StringAttr>(contract::kPrecision);
  auto accumulator =
    function->getAttrOfType<StringAttr>(contract::kFp16Accumulator);
  return precision && precision.getValue() == "fp16" && accumulator &&
         accumulator.getValue() == "f16";
}

Type getLowPrecisionStorageType(OpBuilder& builder, Operation* operation) {
  auto function = operation->getParentOfType<func::FuncOp>();
  auto precision = function->getAttrOfType<StringAttr>(contract::kPrecision);
  if (!precision) {
    return {};
  }
  return llvm::StringSwitch<Type>(precision.getValue())
    .Case("fp16", builder.getF16Type())
    .Case("bf16", builder.getBF16Type())
    .Default(Type());
}

bool usesLowPrecisionBoundary(Operation* operation) {
  StringRef name = operation->getName().getStringRef();
  auto function = operation->getParentOfType<func::FuncOp>();
  auto precision = function->getAttrOfType<StringAttr>(contract::kPrecision);
  const bool lowPrecision = precision && (precision.getValue() == "fp16" ||
                                          precision.getValue() == "bf16");
  return lowPrecision &&
         ncnn_mlir::operator_precision_capability(
           std::string_view(name.data(), name.size())) ==
           ncnn_mlir::OperatorPrecisionCapability::LowPrecisionBoundary;
}

Value applyLowPrecisionBoundary(OpBuilder& builder,
                                Location location,
                                Operation* operation,
                                Value value) {
  Type storageType = getLowPrecisionStorageType(builder, operation);
  return usesLowPrecisionBoundary(operation)
           ? roundStoragePrecision(builder, location, value, storageType)
           : value;
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

Value createIntegerZero(OpBuilder& builder,
                        Location location,
                        RankedTensorType type) {
  auto element = cast<IntegerType>(type.getElementType());
  auto value = DenseElementsAttr::get(type, builder.getIntegerAttr(element, 0));
  return {builder.create<tosa::ConstOp>(location, type, value)};
}

ElementsAttr getConstantTensorElements(Value value) {
  Operation* defining = value.getDefiningOp();
  if (!defining) {
    return {};
  }
  if (auto constant = dyn_cast<arith::ConstantOp>(defining)) {
    return dyn_cast<ElementsAttr>(constant.getValue());
  }
  if (auto constant = dyn_cast<tosa::ConstOp>(defining)) {
    return constant.getValues();
  }
  return {};
}

Value foldConstantReshape(OpBuilder& builder,
                          Location location,
                          Value input,
                          RankedTensorType outputType) {
  auto sourceType = dyn_cast<RankedTensorType>(input.getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      !outputType.hasStaticShape() ||
      !ncnn_mlir::is_foldable_element_type(outputType.getElementType()) ||
      sourceType.getNumElements() != outputType.getNumElements()) {
    return {};
  }
  ElementsAttr elements = getConstantTensorElements(input);
  if (!elements || (!elements.isSplat() && !isa<DenseElementsAttr>(elements))) {
    return {};
  }
  DenseElementsAttr folded =
    ncnn_mlir::reshape_dense_elements(elements, outputType);
  if (!folded) {
    return {};
  }
  return builder.create<arith::ConstantOp>(location, outputType, folded);
}

Value foldConstantTranspose(OpBuilder& builder,
                            Location location,
                            Value input,
                            ArrayRef<int32_t> permutation) {
  auto sourceType = dyn_cast<RankedTensorType>(input.getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      permutation.size() != static_cast<size_t>(sourceType.getRank())) {
    return {};
  }
  for (int32_t dimension : permutation) {
    if (dimension < 0 || dimension >= sourceType.getRank()) {
      return {};
    }
  }
  ElementsAttr elements = getConstantTensorElements(input);
  if (!elements || !cast<ShapedType>(elements.getType()).hasStaticShape()) {
    return {};
  }
  SmallVector<int64_t> resultShape;
  resultShape.reserve(permutation.size());
  for (int32_t dimension : permutation) {
    resultShape.push_back(sourceType.getShape()[dimension]);
  }
  auto resultType =
    RankedTensorType::get(resultShape, sourceType.getElementType());
  DenseElementsAttr folded = ncnn_mlir::transpose_dense_elements(
    elements, sourceType, permutation, resultType);
  if (!folded) {
    return {};
  }
  return builder.create<arith::ConstantOp>(location, resultType, folded);
}

Value transposeOrFoldConstant(OpBuilder& builder,
                              Location location,
                              Value input,
                              RankedTensorType resultType,
                              ArrayRef<int32_t> permutation) {
  if (Value folded =
        foldConstantTranspose(builder, location, input, permutation)) {
    return folded;
  }
  return {builder.create<tosa::TransposeOp>(
    location, resultType, input, permutation)};
}

DenseElementsAttr quantizeConstantToI8(ElementsAttr elements,
                                       RankedTensorType sourceType,
                                       ElementsAttr scaleElements,
                                       std::optional<unsigned> scaleDimension) {
  const int64_t rank = sourceType.getRank();
  if (!elements || !scaleElements || !sourceType.hasStaticShape() ||
      rank == 0 || !sourceType.getElementType().isF32() ||
      (scaleDimension && *scaleDimension != 0)) {
    return {};
  }
  auto scaleType = dyn_cast<RankedTensorType>(scaleElements.getType());
  if (!scaleType || !scaleType.hasStaticShape() ||
      !scaleType.getElementType().isF32() || scaleType.getRank() != 1) {
    return {};
  }
  const int64_t channels = sourceType.getShape()[0];
  if (scaleType.getShape()[0] != 1 &&
      (!scaleDimension || scaleType.getShape()[0] != channels)) {
    return {};
  }
  auto dense = dyn_cast<DenseElementsAttr>(elements);
  auto denseScale = dyn_cast<DenseElementsAttr>(scaleElements);
  if (!dense || !denseScale) {
    return {};
  }
  const bool scalarScale = scaleType.getShape()[0] == 1;
  const int64_t count = sourceType.getNumElements();
  const int64_t rowSize = channels > 0 ? count / channels : 0;
  APFloat half(0.5f);
  APFloat negativeHalf(-0.5f);
  APFloat zero(0.0f);
  APFloat maximum(127.0f);
  APFloat minimum(-127.0f);
  SmallVector<APInt> quantized;
  quantized.reserve(count);
  auto inputValues = dense.getValues<APFloat>();
  auto scaleValues = denseScale.getValues<APFloat>();
  auto inputValue = inputValues.begin();
  for (int64_t offset = 0; offset < count; ++offset, ++inputValue) {
    const int64_t scaleIndex = scalarScale ? 0 : offset / rowSize;
    if (scaleIndex >= scaleType.getShape()[0]) {
      return {};
    }
    APFloat scaled(*inputValue);
    scaled.multiply(*(scaleValues.begin() + scaleIndex),
                    APFloat::rmNearestTiesToEven);
    if (!scaled.isFinite()) {
      return {};
    }
    APFloat positive(scaled);
    positive.add(half, APFloat::rmNearestTiesToEven);
    positive.roundToIntegral(llvm::RoundingMode::TowardNegative);
    APFloat negative(scaled);
    negative.add(negativeHalf, APFloat::rmNearestTiesToEven);
    negative.roundToIntegral(llvm::RoundingMode::TowardPositive);
    APFloat rounded =
      scaled.compare(zero) == APFloat::cmpLessThan ? negative : positive;
    if (rounded.compare(maximum) == APFloat::cmpGreaterThan) {
      rounded = maximum;
    } else if (rounded.compare(minimum) == APFloat::cmpLessThan) {
      rounded = minimum;
    }
    const double integral = rounded.convertToDouble();
    quantized.push_back(APInt(8, static_cast<int64_t>(integral), true));
  }
  return DenseElementsAttr::get(
    sourceType.clone(IntegerType::get(sourceType.getContext(), 8)), quantized);
}

Value foldConstantQuantizeI8(OpBuilder& builder,
                             Location location,
                             Value input,
                             Value scale,
                             std::optional<unsigned> scaleDimension) {
  auto sourceType = dyn_cast<RankedTensorType>(input.getType());
  if (!sourceType || !sourceType.hasStaticShape()) {
    return {};
  }
  ElementsAttr elements = getConstantTensorElements(input);
  ElementsAttr scaleElements = getConstantTensorElements(scale);
  DenseElementsAttr folded =
    quantizeConstantToI8(elements, sourceType, scaleElements, scaleDimension);
  if (!folded) {
    return {};
  }
  return builder.create<arith::ConstantOp>(location, folded.getType(), folded);
}

Value quantizeSignedI8(OpBuilder& builder,
                       Location location,
                       Value input,
                       Value scale,
                       std::optional<unsigned> scaleDimension) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.getElementType().isInteger(8)) {
    return input;
  }
  if (Value folded = foldConstantQuantizeI8(
        builder, location, input, scale, scaleDimension)) {
    return folded;
  }
  auto outputType = inputType.clone(builder.getI8Type());
  Value empty = builder.create<tensor::EmptyOp>(
    location,
    outputType.getShape(),
    outputType.getElementType(),
    getDynamicSizeValues(builder, location, input, outputType));
  AffineMap inputMap = builder.getMultiDimIdentityMap(inputType.getRank());
  AffineExpr scaleIndex = scaleDimension
                            ? builder.getAffineDimExpr(*scaleDimension)
                            : builder.getAffineConstantExpr(0);
  AffineMap scaleMap =
    AffineMap::get(inputType.getRank(), 0, scaleIndex, builder.getContext());
  SmallVector<utils::IteratorType> iterators(inputType.getRank(),
                                             utils::IteratorType::parallel);
  auto quantized = builder.create<linalg::GenericOp>(
    location,
    outputType,
    ValueRange{input, scale},
    ValueRange{empty},
    ArrayRef<AffineMap>{inputMap, scaleMap, inputMap},
    iterators,
    [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
      Value scaled = nested.create<arith::MulFOp>(
        nestedLocation, arguments[0], arguments[1]);
      Type type = scaled.getType();
      Value zero = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getFloatAttr(type, 0.0));
      // round-half-away-from-zero 的纯 arith 形态：符号分支后 ±0.5 再截断
      // （fptosi 向零截断，正数即 floor、负数即 ceil），与 floor/ceil
      // select 形态对全部有限输入逐位等价；f32 侧先收拢到 ±128 保证
      // fptosi 无未定义行为，i32 侧再收紧到 ncnn 的 ±127 饱和域。全程
      // arith op 使 body 可被行级向量化提升，无需向量数学库覆盖 floor。
      Value bound = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getFloatAttr(type, 128.0));
      Value negativeBound = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getFloatAttr(type, -128.0));
      Value half = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getFloatAttr(type, 0.5));
      Value negativeHalf = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getFloatAttr(type, -0.5));
      Value bounded = nested.create<arith::MaximumFOp>(
        nestedLocation,
        nested.create<arith::MinimumFOp>(nestedLocation, scaled, bound),
        negativeBound);
      Value nonnegative = nested.create<arith::CmpFOp>(
        nestedLocation, arith::CmpFPredicate::OGE, scaled, zero);
      Value shifted = nested.create<arith::SelectOp>(
        nestedLocation,
        nonnegative,
        nested.create<arith::AddFOp>(nestedLocation, bounded, half),
        nested.create<arith::AddFOp>(nestedLocation, bounded, negativeHalf));
      Value truncated = nested.create<arith::FPToSIOp>(
        nestedLocation, nested.getI32Type(), shifted);
      Value minimum = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getI32IntegerAttr(127));
      Value maximum = nested.create<arith::ConstantOp>(
        nestedLocation, nested.getI32IntegerAttr(-127));
      Value clamped = nested.create<arith::MaxSIOp>(
        nestedLocation,
        nested.create<arith::MinSIOp>(nestedLocation, truncated, minimum),
        maximum);
      Value result = nested.create<arith::TruncIOp>(
        nestedLocation, nested.getI8Type(), clamped);
      nested.create<linalg::YieldOp>(nestedLocation, result);
    });
  return quantized->getResult(0);
}

Value convertI32ToF32(OpBuilder& builder, Location location, Value input) {
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

Value dequantizeNcnn(
  OpBuilder& builder, Location location, Value input, Value scale, Value bias) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto outputType = inputType.clone(builder.getF32Type());
  Value empty = builder.create<tensor::EmptyOp>(
    location,
    outputType.getShape(),
    outputType.getElementType(),
    getDynamicSizeValues(builder, location, input, outputType));
  const unsigned scaleDimension = inputType.getRank() == 4 ? 3 : 0;
  auto scaleType = cast<RankedTensorType>(scale.getType());
  AffineExpr scaleIndex = scaleType.getShape()[0] == 1
                            ? builder.getAffineConstantExpr(0)
                            : builder.getAffineDimExpr(scaleDimension);
  AffineMap identity = builder.getMultiDimIdentityMap(inputType.getRank());
  AffineMap parameterMap =
    AffineMap::get(inputType.getRank(), 0, scaleIndex, builder.getContext());
  SmallVector<Value> inputs{input, scale};
  SmallVector<AffineMap> maps{identity, parameterMap};
  if (bias) {
    inputs.push_back(bias);
    maps.push_back(parameterMap);
  }
  maps.push_back(identity);
  SmallVector<utils::IteratorType> iterators(inputType.getRank(),
                                             utils::IteratorType::parallel);
  auto converted = builder.create<linalg::GenericOp>(
    location,
    outputType,
    inputs,
    ValueRange{empty},
    maps,
    iterators,
    [hasBias = static_cast<bool>(bias)](
      OpBuilder& nested, Location nestedLocation, ValueRange values) {
      Value floating = nested.create<arith::SIToFPOp>(
        nestedLocation, nested.getF32Type(), values[0]);
      Value result =
        nested.create<arith::MulFOp>(nestedLocation, floating, values[1]);
      if (hasBias) {
        result =
          nested.create<arith::AddFOp>(nestedLocation, result, values[2]);
      }
      nested.create<linalg::YieldOp>(nestedLocation, result);
    });
  return converted->getResult(0);
}

Value dequantizeAccumulator(OpBuilder& builder,
                            Location location,
                            Value accumulator,
                            Value weightScale,
                            Value inputScale,
                            unsigned channelDimension) {
  auto accumulatorType = cast<RankedTensorType>(accumulator.getType());
  auto weightScaleType = cast<RankedTensorType>(weightScale.getType());
  Value shift = createI8Zero(builder, location);
  Value combined = builder.create<tosa::MulOp>(
    location, weightScaleType, weightScale, inputScale, shift);
  Value reciprocal =
    builder.create<tosa::ReciprocalOp>(location, weightScaleType, combined);
  Value zero = createSplat(builder, location, weightScaleType, 0.0);
  auto conditionType =
    RankedTensorType::get(weightScaleType.getShape(), builder.getI1Type());
  Value zeroWeightScale =
    builder.create<tosa::EqualOp>(location, conditionType, weightScale, zero);
  reciprocal = builder.create<tosa::SelectOp>(
    location, weightScaleType, zeroWeightScale, zero, reciprocal);
  auto outputType = accumulatorType.clone(builder.getF32Type());
  Value converted = convertI32ToF32(builder, location, accumulator);
  if (!outputType.hasStaticShape()) {
    Value empty = builder.create<tensor::EmptyOp>(
      location,
      outputType.getShape(),
      outputType.getElementType(),
      getDynamicSizeValues(builder, location, converted, outputType));
    AffineMap identity = builder.getMultiDimIdentityMap(outputType.getRank());
    AffineExpr scaleIndex = weightScaleType.getShape()[0] == 1
                              ? builder.getAffineConstantExpr(0)
                              : builder.getAffineDimExpr(channelDimension);
    AffineMap scaleMap =
      AffineMap::get(outputType.getRank(), 0, scaleIndex, builder.getContext());
    SmallVector<utils::IteratorType> iterators(outputType.getRank(),
                                               utils::IteratorType::parallel);
    return builder
      .create<linalg::GenericOp>(
        location,
        outputType,
        ValueRange{converted, reciprocal},
        ValueRange{empty},
        ArrayRef<AffineMap>{identity, scaleMap, identity},
        iterators,
        [](OpBuilder& nested, Location nestedLocation, ValueRange values) {
          Value result = nested.create<arith::MulFOp>(
            nestedLocation, values.front(), values[1]);
          nested.create<linalg::YieldOp>(nestedLocation, result);
        })
      .getResult(0);
  }
  SmallVector<int64_t> scaleShape(accumulatorType.getRank(), 1);
  scaleShape[channelDimension] = weightScaleType.getShape()[0];
  Value broadcastScale =
    reshapeValue(builder,
                 location,
                 reciprocal,
                 RankedTensorType::get(scaleShape, builder.getF32Type()));
  return {builder.create<tosa::MulOp>(
    location, outputType, converted, broadcastScale, shift)};
}

Value convertFloatingTensor(OpBuilder& builder,
                            Location location,
                            Value input,
                            Type targetElement) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (inputType.getElementType() == targetElement) {
    return input;
  }
  auto outputType = inputType.clone(targetElement);
  Value init = builder.create<tensor::EmptyOp>(
    location,
    outputType.getShape(),
    targetElement,
    getDynamicSizeValues(builder, location, input, outputType));
  auto converted = builder.create<linalg::MapOp>(
    location,
    ValueRange{input},
    init,
    [targetElement](
      OpBuilder& nested, Location nestedLocation, ValueRange values) {
      auto source = cast<FloatType>(values.front().getType());
      auto target = cast<FloatType>(targetElement);
      Value value = source.getWidth() < target.getWidth()
                      ? static_cast<Value>(nested.create<arith::ExtFOp>(
                          nestedLocation, target, values.front()))
                      : static_cast<Value>(nested.create<arith::TruncFOp>(
                          nestedLocation, target, values.front()));
      nested.create<linalg::YieldOp>(nestedLocation, value);
    });
  return converted->getResult(0);
}

Value roundStoragePrecision(OpBuilder& builder,
                            Location location,
                            Value input,
                            Type storageElement) {
  Value stored =
    convertFloatingTensor(builder, location, input, storageElement);
  return convertFloatingTensor(
    builder,
    location,
    stored,
    cast<ShapedType>(input.getType()).getElementType());
}

Value initializeConvolutionOutput(OpBuilder& builder,
                                  Location location,
                                  RankedTensorType outputType,
                                  Value bias) {
  Value empty = builder.create<tensor::EmptyOp>(
    location, outputType.getShape(), outputType.getElementType());
  AffineExpr channel = builder.getAffineDimExpr(3);
  AffineMap biasMap = AffineMap::get(4, 0, {channel}, builder.getContext());
  AffineMap outputMap = builder.getMultiDimIdentityMap(4);
  SmallVector<utils::IteratorType> iterators(4, utils::IteratorType::parallel);
  auto initialized = builder.create<linalg::GenericOp>(
    location,
    outputType,
    ValueRange{bias},
    ValueRange{empty},
    ArrayRef<AffineMap>{biasMap, outputMap},
    iterators,
    [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
      nested.create<linalg::YieldOp>(nestedLocation, arguments.front());
    });
  return initialized.getResult(0);
}

DenseIntElementsAttr createI64PairAttr(OpBuilder& builder,
                                       ArrayRef<int64_t> values) {
  return DenseIntElementsAttr::get(
    RankedTensorType::get({2}, builder.getI64Type()), values);
}

Value createIndexConstant(OpBuilder& builder,
                          Location location,
                          int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
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

Value getConvolutionOutputExtent(OpBuilder& builder,
                                 Location location,
                                 Value input,
                                 int64_t dimension,
                                 int64_t leadingPadding,
                                 int64_t trailingPadding,
                                 int64_t kernel,
                                 int64_t stride,
                                 int64_t dilation) {
  Value inputExtent = builder.create<tensor::DimOp>(location, input, dimension);
  const int64_t effectiveKernel = ((kernel - 1) * dilation) + 1;
  Value padded = builder.create<arith::AddIOp>(
    location,
    inputExtent,
    createIndexConstant(builder, location, leadingPadding + trailingPadding));
  Value numerator = builder.create<arith::SubIOp>(
    location, padded, createIndexConstant(builder, location, effectiveKernel));
  Value quotient = builder.create<arith::DivUIOp>(
    location, numerator, createIndexConstant(builder, location, stride));
  return builder.create<arith::AddIOp>(
    location, quotient, createIndexConstant(builder, location, 1));
}

Value initializeDynamicConvolutionOutput(OpBuilder& builder,
                                         Location location,
                                         RankedTensorType outputType,
                                         Value input,
                                         ArrayRef<int64_t> padding,
                                         ArrayRef<int64_t> kernel,
                                         ArrayRef<int64_t> stride,
                                         ArrayRef<int64_t> dilation) {
  SmallVector<Value> dynamicSizes;
  if (outputType.isDynamicDim(1)) {
    dynamicSizes.push_back(getConvolutionOutputExtent(builder,
                                                      location,
                                                      input,
                                                      1,
                                                      padding[0],
                                                      padding[1],
                                                      kernel[0],
                                                      stride[0],
                                                      dilation[0]));
  }
  if (outputType.isDynamicDim(2)) {
    dynamicSizes.push_back(getConvolutionOutputExtent(builder,
                                                      location,
                                                      input,
                                                      2,
                                                      padding[2],
                                                      padding[3],
                                                      kernel[1],
                                                      stride[1],
                                                      dilation[1]));
  }
  Value empty = builder.create<tensor::EmptyOp>(
    location, outputType.getShape(), outputType.getElementType(), dynamicSizes);
  Value zero = builder.create<arith::ConstantOp>(
    location, builder.getZeroAttr(outputType.getElementType()));
  return builder.create<linalg::FillOp>(location, zero, empty).getResult(0);
}

RankedTensorType getBroadcastScalarType(RankedTensorType type) {
  SmallVector<int64_t> shape(type.getRank(), 1);
  return RankedTensorType::get(shape, type.getElementType());
}

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
  if (Value folded =
        foldConstantReshape(builder, location, input, outputType)) {
    return folded;
  }
  Value shape = createShape(builder, location, outputType.getShape());
  return {builder.create<tosa::ReshapeOp>(location, outputType, input, shape)};
}

Value reshapeValue(OpBuilder& builder,
                   Location location,
                   Value input,
                   RankedTensorType outputType,
                   ArrayRef<std::optional<unsigned>> sourceDimensions) {
  SmallVector<Value> dimensions;
  dimensions.reserve(outputType.getRank());
  for (auto [index, extent] : llvm::enumerate(outputType.getShape())) {
    if (ShapedType::isDynamic(extent)) {
      dimensions.push_back(builder.create<tensor::DimOp>(
        location, input, *sourceDimensions[index]));
    } else {
      dimensions.push_back(
        builder.create<arith::ConstantIndexOp>(location, extent));
    }
  }
  auto shapeType =
    RankedTensorType::get({outputType.getRank()}, builder.getIndexType());
  Value shape =
    builder.create<tensor::FromElementsOp>(location, shapeType, dimensions);
  return {
    builder.create<tensor::ReshapeOp>(location, outputType, input, shape)};
}

Value restoreNCNNLayout(OpBuilder& builder,
                        Location location,
                        Value input,
                        RankedTensorType sourceType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (sourceType.getRank() == 3 && inputType.getRank() == 4) {
    return convertNHWCToCHW(builder, location, input, sourceType);
  }
  return input;
}

Value convertNCNNLayout(OpBuilder& builder,
                        Location location,
                        Value input,
                        RankedTensorType sourceType) {
  auto inputType = cast<RankedTensorType>(input.getType());
  if (sourceType.getRank() == 3 && inputType.getRank() == 3) {
    return convertCHWToNHWC(builder, location, input);
  }
  return input;
}

Value convertCHWToNHWC(OpBuilder& builder, Location location, Value input) {
  auto chwType = cast<RankedTensorType>(input.getType());
  auto hwcType = RankedTensorType::get(
    {chwType.getShape()[1], chwType.getShape()[2], chwType.getShape()[0]},
    chwType.getElementType());
  Value transposed = transposeOrFoldConstant(
    builder, location, input, hwcType, ArrayRef<int32_t>{1, 2, 0});
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

}  // namespace mlir::ncnn::tosa_lowering
