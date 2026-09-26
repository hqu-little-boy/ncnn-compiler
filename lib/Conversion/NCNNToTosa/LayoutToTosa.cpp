// 形状 / 布局族降低：split、concat、padding、interp、grid_sample、
// reshape、squeeze/expand_dims、permute、slice、shuffle_channel。
//
// 职责
//   这一族只改形状与布局，不改数值；统一经 restoreNCNNLayout /
//   convertNCNNLayout 与 NCNN 的 CHW 布局约定对接。
//
// 不变量
//   * 形状变化必须可静态推导或有显式动态尺寸来源；
//   * 常量输入的重排尽量折叠成常量（foldConstantTranspose 等），
//     折叠不了的保持运行期算子而不是猜。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后（形状约束已解）；
//   * 之后 TosaToLinalg 转 tensor 算子，供 BufferizeNCNN 使用。
//
// 明确不做
//   * 不做布局优化（layout island 由 StrategyNCNN 管）；
//   * 不做常量传播（那是 fold 系列 pass）。

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
using tosa_lowering::convertAxis;
using tosa_lowering::convertNCNNLayout;
using tosa_lowering::createIndexConstant;
using tosa_lowering::createShape;
using tosa_lowering::createSplat;
using tosa_lowering::getNHWCType;
using tosa_lowering::getRequiredIntegerAttr;
using tosa_lowering::isRankedF32Tensor;
using tosa_lowering::reshapeValue;
using tosa_lowering::restoreNCNNLayout;

Value getTensorElementCount(OpBuilder& builder,
                            Location location,
                            Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  Value count = createIndexConstant(builder, location, 1);
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
    Value extent = builder.create<tensor::DimOp>(location, value, dimension);
    count = builder.create<arith::MulIOp>(location, count, extent);
  }
  return count;
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
    auto sourceAxis = static_cast<int64_t>(operation.getAxis());
    if (sourceAxis < 0) {
      sourceAxis += sourceType.getRank();
    }
    uint32_t axis = convertAxis(sourceAxis, sourceType.getRank());
    SmallVector<Value> inputs(adaptor.getInputs().begin(),
                              adaptor.getInputs().end());
    if (sourceType.getElementType().isF32()) {
      for (Value& input : inputs) {
        input = applyLowPrecisionBoundary(
          rewriter, operation.getLoc(), operation, input);
      }
    }
    Value result = rewriter.create<tosa::ConcatOp>(
      operation.getLoc(), outputType, inputs, axis);
    if (sourceType.getElementType().isF32()) {
      result = applyLowPrecisionBoundary(
        rewriter, operation.getLoc(), operation, result);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertPadding final : public ConversionPattern {
 public:
  ConvertPadding(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, contract::kLayerPadding, 1, context) {}

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
    FailureOr<int64_t> paddingType =
      getRequiredIntegerAttr(operation, "padding_type");
    if (failed(paddingType)) {
      return failure();
    }
    if (*paddingType == 2) {
      if (!sourceInput.hasStaticShape() ||
          pad[0] >= sourceInput.getShape()[1] ||
          pad[1] >= sourceInput.getShape()[1] ||
          pad[2] >= sourceInput.getShape()[2] ||
          pad[3] >= sourceInput.getShape()[2]) {
        return operation->emitOpError(
          "reflection padding requires static extents larger than padding");
      }
      auto outputType = getNHWCType(sourceOutput);
      Value empty = rewriter.create<tensor::EmptyOp>(
        operation->getLoc(), outputType, ValueRange{});
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
          Value outputY = nested.create<linalg::IndexOp>(location, 1);
          Value outputX = nested.create<linalg::IndexOp>(location, 2);
          Value channel = nested.create<linalg::IndexOp>(location, 3);
          auto reflect = [&](
                           Value output, int64_t before, int64_t inputExtent) {
            Value beforeValue = createIndexConstant(nested, location, before);
            Value extentValue =
              createIndexConstant(nested, location, inputExtent);
            Value coordinate =
              nested.create<arith::SubIOp>(location, output, beforeValue);
            Value zero = createIndexConstant(nested, location, 0);
            Value negative = nested.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::slt, coordinate, zero);
            Value reflectedBefore =
              nested.create<arith::SubIOp>(location, zero, coordinate);
            Value upper = nested.create<arith::SubIOp>(
              location,
              nested.create<arith::MulIOp>(
                location,
                extentValue,
                createIndexConstant(nested, location, 2)),
              createIndexConstant(nested, location, 2));
            Value beyond = nested.create<arith::CmpIOp>(
              location, arith::CmpIPredicate::sge, coordinate, extentValue);
            Value reflectedAfter =
              nested.create<arith::SubIOp>(location, upper, coordinate);
            coordinate = nested.create<arith::SelectOp>(
              location, negative, reflectedBefore, coordinate);
            return Value(nested.create<arith::SelectOp>(
              location, beyond, reflectedAfter, coordinate));
          };
          Value inputY = reflect(outputY, pad[0], sourceInput.getShape()[1]);
          Value inputX = reflect(outputX, pad[2], sourceInput.getShape()[2]);
          Value sample = nested.create<tensor::ExtractOp>(
            location,
            operands.front(),
            ValueRange{batch, inputY, inputX, channel});
          nested.create<linalg::YieldOp>(location, sample);
        });
      rewriter.replaceOp(operation, result.getResults());
      return success();
    }
    if (*paddingType != 0) {
      return operation->emitOpError("unsupported padding type");
    }
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
    : ConversionPattern(typeConverter, contract::kLayerInterp, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if ((operands.size() != 1 && operands.size() != 2) ||
        operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError(
        "supports ranked CHW f32 input and optional size reference only");
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
    FailureOr<int64_t> explicitH =
      getRequiredIntegerAttr(operation, "output_h");
    FailureOr<int64_t> explicitW =
      getRequiredIntegerAttr(operation, "output_w");
    FailureOr<int64_t> resizeType =
      getRequiredIntegerAttr(operation, "resize_type");
    auto alignCorner = operation->getAttrOfType<BoolAttr>("align_corner");
    if (failed(scaleH) || failed(scaleW) || failed(explicitH) ||
        failed(explicitW) || failed(resizeType) || !alignCorner) {
      return failure();
    }
    if (*scaleH <= 0 || *scaleW <= 0 || *explicitH < 0 || *explicitW < 0) {
      return operation->emitOpError(
        "resize scales must be positive and output dimensions non-negative");
    }
    auto referenceType =
      operands.size() == 2
        ? dyn_cast<RankedTensorType>(operation->getOperand(1).getType())
        : RankedTensorType{};
    if (operands.size() == 2 &&
        (!referenceType || referenceType.getRank() != 3)) {
      return operation->emitOpError("size reference must be a CHW tensor");
    }
    int64_t expectedH =
      referenceType ? referenceType.getShape()[1]
      : *explicitH != 0
        ? *explicitH
        : (sourceInput.isDynamicDim(1) ? ShapedType::kDynamic
                                       : sourceInput.getShape()[1] * *scaleH);
    int64_t expectedW =
      referenceType ? referenceType.getShape()[2]
      : *explicitW != 0
        ? *explicitW
        : (sourceInput.isDynamicDim(2) ? ShapedType::kDynamic
                                       : sourceInput.getShape()[2] * *scaleW);
    if (sourceOutput.getShape()[1] != expectedH ||
        sourceOutput.getShape()[2] != expectedW) {
      return operation->emitOpError(
        "height_scale/width_scale do not match result shape");
    }

    auto outputType = getNHWCType(sourceOutput);
    if (*resizeType == 2) {
      Value input = operands.front();
      Value inputH =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 1);
      Value inputW =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 2);
      Value outputH =
        referenceType
          ? rewriter.create<tensor::DimOp>(operation->getLoc(), operands[1], 1)
        : *explicitH != 0
          ? createIndexConstant(rewriter, operation->getLoc(), *explicitH)
          : rewriter.create<arith::MulIOp>(
              operation->getLoc(),
              inputH,
              createIndexConstant(rewriter, operation->getLoc(), *scaleH));
      Value outputW =
        referenceType
          ? rewriter.create<tensor::DimOp>(operation->getLoc(), operands[1], 2)
        : *explicitW != 0
          ? createIndexConstant(rewriter, operation->getLoc(), *explicitW)
          : rewriter.create<arith::MulIOp>(
              operation->getLoc(),
              inputW,
              createIndexConstant(rewriter, operation->getLoc(), *scaleW));
      SmallVector<Value> dynamicSizes;
      if (outputType.isDynamicDim(1)) {
        dynamicSizes.push_back(outputH);
      }
      if (outputType.isDynamicDim(2)) {
        dynamicSizes.push_back(outputW);
      }
      Value empty = rewriter.create<tensor::EmptyOp>(
        operation->getLoc(), outputType, dynamicSizes);
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
          Value outputY = nested.create<linalg::IndexOp>(location, 1);
          Value outputX = nested.create<linalg::IndexOp>(location, 2);
          Value channel = nested.create<linalg::IndexOp>(location, 3);
          auto coordinate = [&](Value outputIndex,
                                Value inputSize,
                                Value outputSize) {
            Type f32 = nested.getF32Type();
            Type i64 = nested.getI64Type();
            Value zeroF = nested.create<arith::ConstantOp>(
              location, nested.getF32FloatAttr(0.0F));
            Value oneF = nested.create<arith::ConstantOp>(
              location, nested.getF32FloatAttr(1.0F));
            Value halfF = nested.create<arith::ConstantOp>(
              location, nested.getF32FloatAttr(0.5F));
            Value inputI64 =
              nested.create<arith::IndexCastOp>(location, i64, inputSize);
            Value outputI64 =
              nested.create<arith::IndexCastOp>(location, i64, outputSize);
            Value indexI64 =
              nested.create<arith::IndexCastOp>(location, i64, outputIndex);
            Value inputF =
              nested.create<arith::SIToFPOp>(location, f32, inputI64);
            Value outputF =
              nested.create<arith::SIToFPOp>(location, f32, outputI64);
            Value indexF =
              nested.create<arith::SIToFPOp>(location, f32, indexI64);
            Value inputMaximum =
              nested.create<arith::SubFOp>(location, inputF, oneF);
            Value source;
            if (alignCorner.getValue()) {
              Value outputMaximum =
                nested.create<arith::SubFOp>(location, outputF, oneF);
              source = nested.create<arith::DivFOp>(
                location,
                nested.create<arith::MulFOp>(location, indexF, inputMaximum),
                outputMaximum);
            } else {
              source = nested.create<arith::SubFOp>(
                location,
                nested.create<arith::DivFOp>(
                  location,
                  nested.create<arith::MulFOp>(
                    location,
                    nested.create<arith::AddFOp>(location, indexF, halfF),
                    inputF),
                  outputF),
                halfF);
            }
            source = nested.create<arith::MaximumFOp>(location, source, zeroF);
            source =
              nested.create<arith::MinimumFOp>(location, source, inputMaximum);
            Value floor = nested.create<math::FloorOp>(location, source);
            Value baseI64 =
              nested.create<arith::FPToSIOp>(location, i64, floor);
            Value upper = nested.create<arith::SubIOp>(
              location,
              inputI64,
              nested.create<arith::ConstantIntOp>(location, 2, 64));
            baseI64 = nested.create<arith::MinSIOp>(location, baseI64, upper);
            Value base = nested.create<arith::IndexCastOp>(
              location, nested.getIndexType(), baseI64);
            Value baseF =
              nested.create<arith::SIToFPOp>(location, f32, baseI64);
            Value fraction =
              nested.create<arith::SubFOp>(location, source, baseF);
            return std::pair<Value, Value>{base, fraction};
          };
          auto [sourceY, fractionY] = coordinate(outputY, inputH, outputH);
          auto [sourceX, fractionX] = coordinate(outputX, inputW, outputW);
          Value one = nested.create<arith::ConstantOp>(
            location, nested.getF32FloatAttr(1.0F));
          Value nextY = nested.create<arith::AddIOp>(
            location, sourceY, createIndexConstant(nested, location, 1));
          Value nextX = nested.create<arith::AddIOp>(
            location, sourceX, createIndexConstant(nested, location, 1));
          auto extract = [&](Value y, Value x) {
            return nested.create<tensor::ExtractOp>(
              location, input, ValueRange{batch, y, x, channel});
          };
          Value top = nested.create<arith::AddFOp>(
            location,
            nested.create<arith::MulFOp>(
              location,
              extract(sourceY, sourceX),
              nested.create<arith::SubFOp>(location, one, fractionX)),
            nested.create<arith::MulFOp>(
              location, extract(sourceY, nextX), fractionX));
          Value bottom = nested.create<arith::AddFOp>(
            location,
            nested.create<arith::MulFOp>(
              location,
              extract(nextY, sourceX),
              nested.create<arith::SubFOp>(location, one, fractionX)),
            nested.create<arith::MulFOp>(
              location, extract(nextY, nextX), fractionX));
          Value value = nested.create<arith::AddFOp>(
            location,
            nested.create<arith::MulFOp>(
              location,
              top,
              nested.create<arith::SubFOp>(location, one, fractionY)),
            nested.create<arith::MulFOp>(location, bottom, fractionY));
          nested.create<linalg::YieldOp>(location, value);
        });
      rewriter.replaceOp(operation, result.getResults());
      return success();
    }
    if (*resizeType != 1 || alignCorner.getValue()) {
      return operation->emitOpError("unsupported resize mode or alignment");
    }
    if (!sourceInput.hasStaticShape() || *explicitH != 0 || *explicitW != 0) {
      Value input = operands.front();
      Value inputH =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 1);
      Value inputW =
        rewriter.create<tensor::DimOp>(operation->getLoc(), input, 2);
      Value heightFactor =
        createIndexConstant(rewriter, operation->getLoc(), *scaleH);
      Value widthFactor =
        createIndexConstant(rewriter, operation->getLoc(), *scaleW);
      Value outputH =
        *explicitH != 0
          ? createIndexConstant(rewriter, operation->getLoc(), *explicitH)
          : rewriter.create<arith::MulIOp>(
              operation->getLoc(), inputH, heightFactor);
      Value outputW =
        *explicitW != 0
          ? createIndexConstant(rewriter, operation->getLoc(), *explicitW)
          : rewriter.create<arith::MulIOp>(
              operation->getLoc(), inputW, widthFactor);
      SmallVector<Value> dynamicSizes;
      if (outputType.isDynamicDim(1)) {
        dynamicSizes.push_back(outputH);
      }
      if (outputType.isDynamicDim(2)) {
        dynamicSizes.push_back(outputW);
      }
      Value empty = rewriter.create<tensor::EmptyOp>(
        operation->getLoc(), outputType, dynamicSizes);
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
          Value inputHeight;
          Value inputWidth;
          if (*explicitH != 0) {
            Value numerator =
              nested.create<arith::MulIOp>(location, outputHeight, inputH);
            inputHeight =
              nested.create<arith::DivUIOp>(location, numerator, outputH);
          } else {
            inputHeight = nested.create<arith::DivUIOp>(
              location, outputHeight, heightFactor);
          }
          if (*explicitW != 0) {
            Value numerator =
              nested.create<arith::MulIOp>(location, outputWidth, inputW);
            inputWidth =
              nested.create<arith::DivUIOp>(location, numerator, outputW);
          } else {
            inputWidth =
              nested.create<arith::DivUIOp>(location, outputWidth, widthFactor);
          }
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

class ConvertGridSample final : public OpConversionPattern<GridSampleOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    GridSampleOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    if (operation.getSampleType() != 1 || operation.getPaddingMode() != 1 ||
        !operation.getPermuteFusion()) {
      return operation.emitOpError(
        "supports bilinear zero padding with permute fusion only");
    }
    Value input = adaptor.getInput();
    Value grid = adaptor.getGrid();
    auto sourceOutput = cast<RankedTensorType>(operation.getOutput().getType());
    auto outputType = getNHWCType(sourceOutput);
    Value inputH = rewriter.create<tensor::DimOp>(operation.getLoc(), input, 1);
    Value inputW = rewriter.create<tensor::DimOp>(operation.getLoc(), input, 2);
    Value outputH = rewriter.create<tensor::DimOp>(operation.getLoc(), grid, 1);
    Value outputW = rewriter.create<tensor::DimOp>(operation.getLoc(), grid, 2);
    SmallVector<Value> dynamicSizes;
    if (outputType.isDynamicDim(1)) {
      dynamicSizes.push_back(outputH);
    }
    if (outputType.isDynamicDim(2)) {
      dynamicSizes.push_back(outputW);
    }
    Value empty = rewriter.create<tensor::EmptyOp>(
      operation.getLoc(), outputType, dynamicSizes);
    AffineMap identity = rewriter.getMultiDimIdentityMap(4);
    SmallVector<utils::IteratorType> iterators(4,
                                               utils::IteratorType::parallel);
    auto result = rewriter.create<linalg::GenericOp>(
      operation.getLoc(),
      outputType,
      ValueRange{},
      ValueRange{empty},
      ArrayRef<AffineMap>{identity},
      iterators,
      [&](OpBuilder& nested, Location location, ValueRange) {
        Type f32 = nested.getF32Type();
        Type i64 = nested.getI64Type();
        Value batch = nested.create<linalg::IndexOp>(location, 0);
        Value outputY = nested.create<linalg::IndexOp>(location, 1);
        Value outputX = nested.create<linalg::IndexOp>(location, 2);
        Value channel = nested.create<linalg::IndexOp>(location, 3);
        Value zeroIndex = createIndexConstant(nested, location, 0);
        Value oneIndex = createIndexConstant(nested, location, 1);
        Value gridX = nested.create<tensor::ExtractOp>(
          location, grid, ValueRange{batch, outputY, outputX, zeroIndex});
        Value gridY = nested.create<tensor::ExtractOp>(
          location, grid, ValueRange{batch, outputY, outputX, oneIndex});
        auto coordinate = [&](Value normalized, Value inputSize) {
          Value oneF = nested.create<arith::ConstantOp>(
            location, nested.getF32FloatAttr(1.0F));
          Value halfF = nested.create<arith::ConstantOp>(
            location, nested.getF32FloatAttr(0.5F));
          Value inputI64 =
            nested.create<arith::IndexCastOp>(location, i64, inputSize);
          Value inputF =
            nested.create<arith::SIToFPOp>(location, f32, inputI64);
          Value rebased =
            nested.create<arith::AddFOp>(location, normalized, oneF);
          Value source;
          if (operation.getAlignCorner()) {
            source = nested.create<arith::MulFOp>(
              location,
              nested.create<arith::MulFOp>(location, rebased, halfF),
              nested.create<arith::SubFOp>(location, inputF, oneF));
          } else {
            source = nested.create<arith::MulFOp>(
              location,
              nested.create<arith::SubFOp>(
                location,
                nested.create<arith::MulFOp>(location, rebased, inputF),
                oneF),
              halfF);
          }
          Value floor = nested.create<math::FloorOp>(location, source);
          Value lower = nested.create<arith::FPToSIOp>(location, i64, floor);
          Value fraction =
            nested.create<arith::SubFOp>(location, source, floor);
          return std::pair<Value, Value>{lower, fraction};
        };
        auto [sourceY, fractionY] = coordinate(gridY, inputH);
        auto [sourceX, fractionX] = coordinate(gridX, inputW);
        Value oneI64 = nested.create<arith::ConstantIntOp>(location, 1, 64);
        Value nextY = nested.create<arith::AddIOp>(location, sourceY, oneI64);
        Value nextX = nested.create<arith::AddIOp>(location, sourceX, oneI64);
        Value inputHI64 =
          nested.create<arith::IndexCastOp>(location, i64, inputH);
        Value inputWI64 =
          nested.create<arith::IndexCastOp>(location, i64, inputW);
        Value zeroI64 = nested.create<arith::ConstantIntOp>(location, 0, 64);
        Value oneF = nested.create<arith::ConstantOp>(
          location, nested.getF32FloatAttr(1.0F));
        Value zeroF = nested.create<arith::ConstantOp>(
          location, nested.getF32FloatAttr(0.0F));
        auto extract = [&](Value y, Value x) {
          Value validYLow = nested.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::sge, y, zeroI64);
          Value validYHigh = nested.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::slt, y, inputHI64);
          Value validXLow = nested.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::sge, x, zeroI64);
          Value validXHigh = nested.create<arith::CmpIOp>(
            location, arith::CmpIPredicate::slt, x, inputWI64);
          Value valid = nested.create<arith::AndIOp>(
            location,
            nested.create<arith::AndIOp>(location, validYLow, validYHigh),
            nested.create<arith::AndIOp>(location, validXLow, validXHigh));
          Value maximumY =
            nested.create<arith::SubIOp>(location, inputHI64, oneI64);
          Value maximumX =
            nested.create<arith::SubIOp>(location, inputWI64, oneI64);
          Value boundedY = nested.create<arith::MaxSIOp>(
            location,
            nested.create<arith::MinSIOp>(location, y, maximumY),
            zeroI64);
          Value boundedX = nested.create<arith::MaxSIOp>(
            location,
            nested.create<arith::MinSIOp>(location, x, maximumX),
            zeroI64);
          Value yIndex = nested.create<arith::IndexCastOp>(
            location, nested.getIndexType(), boundedY);
          Value xIndex = nested.create<arith::IndexCastOp>(
            location, nested.getIndexType(), boundedX);
          Value sample = nested.create<tensor::ExtractOp>(
            location, input, ValueRange{batch, yIndex, xIndex, channel});
          return Value(
            nested.create<arith::SelectOp>(location, valid, sample, zeroF));
        };
        Value top = nested.create<arith::AddFOp>(
          location,
          nested.create<arith::MulFOp>(
            location,
            extract(sourceY, sourceX),
            nested.create<arith::SubFOp>(location, oneF, fractionX)),
          nested.create<arith::MulFOp>(
            location, extract(sourceY, nextX), fractionX));
        Value bottom = nested.create<arith::AddFOp>(
          location,
          nested.create<arith::MulFOp>(
            location,
            extract(nextY, sourceX),
            nested.create<arith::SubFOp>(location, oneF, fractionX)),
          nested.create<arith::MulFOp>(
            location, extract(nextY, nextX), fractionX));
        Value value = nested.create<arith::AddFOp>(
          location,
          nested.create<arith::MulFOp>(
            location,
            top,
            nested.create<arith::SubFOp>(location, oneF, fractionY)),
          nested.create<arith::MulFOp>(location, bottom, fractionY));
        nested.create<linalg::YieldOp>(location, value);
      });
    rewriter.replaceOp(operation, result.getResults());
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
    } else if (auto shapeSpec =
                 operation->getAttrOfType<DenseI64ArrayAttr>("shape_spec")) {
      auto zeroSources =
        operation->getAttrOfType<DenseI64ArrayAttr>("shape_zero_sources");
      if (!zeroSources || shapeSpec.size() != outputType.getRank() ||
          zeroSources.size() != shapeSpec.size()) {
        return operation->emitOpError("has invalid shape_spec metadata");
      }
      SmallVector<Value> dimensions;
      Value knownCount = createIndexConstant(rewriter, operation->getLoc(), 1);
      int64_t inferredDimension = -1;
      for (auto [index, specification] :
           llvm::enumerate(shapeSpec.asArrayRef())) {
        Value dimension;
        if (specification == 0) {
          dimension = rewriter.create<tensor::DimOp>(
            operation->getLoc(), input, zeroSources.asArrayRef()[index]);
        } else if (specification == -1) {
          inferredDimension = index;
          dimensions.push_back(Value{});
          continue;
        } else {
          dimension =
            createIndexConstant(rewriter, operation->getLoc(), specification);
        }
        dimensions.push_back(dimension);
        knownCount = rewriter.create<arith::MulIOp>(
          operation->getLoc(), knownCount, dimension);
      }
      Value inputCount =
        getTensorElementCount(rewriter, operation->getLoc(), input);
      if (inferredDimension >= 0) {
        dimensions[inferredDimension] = rewriter.create<arith::DivUIOp>(
          operation->getLoc(), inputCount, knownCount);
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
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError("supports one ranked f32 tensor only");
    }
    auto inputType = cast<RankedTensorType>(operation->getOperand(0).getType());
    auto outputType = cast<RankedTensorType>(operation->getResult(0).getType());
    Value input = restoreNCNNLayout(
      rewriter, operation->getLoc(), operands.front(), inputType);
    Value reshaped;
    if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
      reshaped = reshapeValue(rewriter, operation->getLoc(), input, outputType);
    } else {
      SmallVector<std::optional<unsigned>> sourceDimensions(
        outputType.getRank());
      if (auto expand = dyn_cast<ExpandDimsOp>(operation)) {
        llvm::SmallDenseSet<int64_t> axes;
        for (int64_t axis : expand.getAxes()) {
          axes.insert(axis < 0 ? axis + outputType.getRank() : axis);
        }
        unsigned sourceDimension = 0;
        for (int64_t axis = 0; axis < outputType.getRank(); ++axis) {
          if (!axes.contains(axis)) {
            sourceDimensions[axis] = sourceDimension++;
          }
        }
      } else {
        auto squeeze = cast<SqueezeOp>(operation);
        llvm::SmallDenseSet<int64_t> axes;
        for (int64_t axis : squeeze.getAxes()) {
          axis = axis < 0 ? axis + inputType.getRank() : axis;
          if (inputType.getShape()[axis] == 1) {
            axes.insert(axis);
          }
        }
        unsigned outputDimension = 0;
        for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
          if (!axes.contains(axis)) {
            sourceDimensions[outputDimension++] = axis;
          }
        }
      }
      reshaped = reshapeValue(
        rewriter, operation->getLoc(), input, outputType, sourceDimensions);
    }
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
    auto sourceAxis = static_cast<int64_t>(operation.getAxis());
    if (sourceAxis < 0) {
      sourceAxis += sourceType.getRank();
    }
    const uint32_t axis = convertAxis(sourceAxis, sourceType.getRank());
    int64_t staticOffset = 0;
    Value dynamicOffset = createIndexConstant(rewriter, operation.getLoc(), 0);
    Value axisExtent = rewriter.create<tensor::DimOp>(
      operation.getLoc(), adaptor.getInput(), axis);
    ArrayRef<int64_t> requestedSlices = operation.getSlices();
    SmallVector<Value> results;
    for (auto [resultIndex, result] : llvm::enumerate(operation.getResults())) {
      auto sourceResultType = cast<RankedTensorType>(result.getType());
      auto resultType = sourceResultType.getRank() == 3
                          ? getNHWCType(sourceResultType)
                          : sourceResultType;
      SmallVector<int64_t> start(inputType.getRank(), 0);
      start[axis] = staticOffset;
      if (inputType.hasStaticShape()) {
        results.push_back(rewriter.create<tosa::SliceOp>(
          operation.getLoc(),
          resultType,
          adaptor.getInput(),
          createShape(rewriter, operation.getLoc(), start),
          createShape(rewriter, operation.getLoc(), resultType.getShape())));
        staticOffset += resultType.getDimSize(axis);
      } else {
        SmallVector<OpFoldResult> offsets(inputType.getRank(),
                                          rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(inputType.getRank(),
                                          rewriter.getIndexAttr(1));
        offsets[axis] = dynamicOffset;
        Value axisSize;
        OpFoldResult axisSizeFolded;
        if (requestedSlices[resultIndex] == -233) {
          Value remaining = rewriter.create<arith::SubIOp>(
            operation.getLoc(), axisExtent, dynamicOffset);
          axisSize = rewriter.create<arith::DivUIOp>(
            operation.getLoc(),
            remaining,
            createIndexConstant(rewriter,
                                operation.getLoc(),
                                requestedSlices.size() - resultIndex));
          axisSizeFolded = resultType.isDynamicDim(axis)
                             ? OpFoldResult(axisSize)
                             : OpFoldResult(rewriter.getIndexAttr(
                                 resultType.getDimSize(axis)));
        } else {
          axisSize = createIndexConstant(
            rewriter, operation.getLoc(), requestedSlices[resultIndex]);
          axisSizeFolded = rewriter.getIndexAttr(requestedSlices[resultIndex]);
        }
        for (auto [dimension, extent] :
             llvm::enumerate(resultType.getShape())) {
          sizes.push_back(
            dimension == axis ? axisSizeFolded
            : ShapedType::isDynamic(extent)
              ? OpFoldResult(rewriter.create<tensor::DimOp>(
                  operation.getLoc(), adaptor.getInput(), dimension))
              : OpFoldResult(rewriter.getIndexAttr(extent)));
        }
        results.push_back(
          rewriter.create<tensor::ExtractSliceOp>(operation.getLoc(),
                                                  resultType,
                                                  adaptor.getInput(),
                                                  offsets,
                                                  sizes,
                                                  strides));
        dynamicOffset = rewriter.create<arith::AddIOp>(
          operation.getLoc(), dynamicOffset, axisSize);
      }
    }
    rewriter.replaceOp(operation, results);
    return success();
  }
};

}  // namespace

void populateLayoutToTosaPatterns(RewritePatternSet& patterns,
                                  const TypeConverter& typeConverter,
                                  MLIRContext* context) {
  patterns.add<ConvertSplit,
               ConvertConcat,
               ConvertPadding,
               ConvertInterp,
               ConvertGridSample,
               ConvertPermute,
               ConvertShuffleChannel,
               ConvertSlice>(typeConverter, context);
  patterns.add<ConvertReshape>(typeConverter, context, contract::kLayerReshape);
  patterns.add<ConvertShapeChange>(
    typeConverter, context, contract::kLayerSqueeze);
  patterns.add<ConvertShapeChange>(
    typeConverter, context, contract::kLayerExpandDims);
}

}  // namespace mlir::ncnn
