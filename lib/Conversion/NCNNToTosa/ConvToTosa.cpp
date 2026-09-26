// Convolution 族降低：conv2d、depthwise。
//
// 职责
//   把 ncnn.convolution / convolution_depthwise 降低为
//   tosa.conv2d / depthwise_conv2d，含权重布局转换（OIHW -> OHWI / HWCF）
//   与可选的动态形状输出物化。转置卷积在 DeconvToTosa.cpp。
//
// 不变量
//   * 权重必须能追溯到常量或运行期值，布局转换不改变数值；
//   * 动态空间维走 initializeDynamicConvolutionOutput，绝不写死尺寸；
//   * int8 卷积先 quantizeSignedI8 归一到同一标度，再进 tosa。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后（padding/stride/dilation 已解析）；
//   * 之后由 TosaToLinalg 转成 linalg.conv_2d_nhwc_hwcf 供
//     StrategyNCNN 消费。
//
// 明确不做
//   * 不做 Winograd / im2col 决策（StrategyNCNN 的职责）；
//   * 不做 requant 融合（FuseQuantChainNCNN 的职责）。

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
using tosa_lowering::createI64PairAttr;
using tosa_lowering::createIntegerZero;
using tosa_lowering::createShape;
using tosa_lowering::createSplat;
using tosa_lowering::dequantizeAccumulator;
using tosa_lowering::getIntegerAttrOr;
using tosa_lowering::getNHWCType;
using tosa_lowering::getOHWIType;
using tosa_lowering::getRequiredIntegerAttr;
using tosa_lowering::initializeConvolutionOutput;
using tosa_lowering::initializeDynamicConvolutionOutput;
using tosa_lowering::isRankedF32Tensor;
using tosa_lowering::isStaticF32Tensor;
using tosa_lowering::quantizeSignedI8;
using tosa_lowering::reshapeValue;
using tosa_lowering::roundStoragePrecision;
using tosa_lowering::transposeOrFoldConstant;
using tosa_lowering::usesFP16Arithmetic;

RankedTensorType getHWCFType(RankedTensorType oihwType) {
  ArrayRef<int64_t> shape = oihwType.getShape();
  return RankedTensorType::get({shape[2], shape[3], shape[1], shape[0]},
                               oihwType.getElementType());
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
    input =
      applyLowPrecisionBoundary(rewriter, operation.getLoc(), operation, input);
    auto weightType = cast<RankedTensorType>(operation.getWeight().getType());
    Type storageElement = weightType.getElementType();
    const bool lowPrecisionWeight =
      storageElement.isF16() || storageElement.isBF16();
    const bool fp16Arithmetic =
      storageElement.isF16() && usesFP16Arithmetic(operation);
    Value sourceWeight = adaptor.getWeight();
    auto sourceOutput = cast<RankedTensorType>(operation.getOutput().getType());
    auto outputType = getNHWCType(sourceOutput);
    const int64_t scaleTerm = operation.getInt8ScaleTerm();
    if (scaleTerm != 0) {
      Location location = operation.getLoc();
      ValueRange tail = adaptor.getBiasAndScales();
      const unsigned scaleOffset = operation.getHasBias() ? 1 : 0;
      Value weightScale = tail[scaleOffset];
      Value inputScale = tail[scaleOffset + 1];
      Value quantizedInput =
        quantizeSignedI8(rewriter, location, input, inputScale);
      auto weightScaleType = cast<RankedTensorType>(weightScale.getType());
      Value quantizedWeight = quantizeSignedI8(
        rewriter,
        location,
        sourceWeight,
        weightScale,
        weightScaleType.getShape()[0] == 1 ? std::nullopt
                                           : std::optional<unsigned>(0));
      auto quantizedWeightType =
        cast<RankedTensorType>(quantizedWeight.getType());
      auto ohwiType = getOHWIType(quantizedWeightType);
      Value weight = transposeOrFoldConstant(rewriter,
                                             location,
                                             quantizedWeight,
                                             ohwiType,
                                             ArrayRef<int32_t>{0, 2, 3, 1});
      Value inputZero = createIntegerZero(
        rewriter, location, RankedTensorType::get({1}, rewriter.getI8Type()));
      Value weightZero = createIntegerZero(
        rewriter, location, RankedTensorType::get({1}, rewriter.getI8Type()));
      auto i32BiasType = RankedTensorType::get({sourceOutput.getShape()[0]},
                                               rewriter.getI32Type());
      Value contractionBias =
        createIntegerZero(rewriter, location, i32BiasType);
      SmallVector<int64_t> padding{padTop, padBottom, padLeft, padRight};
      auto inputShape =
        cast<RankedTensorType>(quantizedInput.getType()).getShape();
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
      auto accumulatorType = outputType.clone(rewriter.getI32Type());
      Value accumulator;
      if (dynamicSpatial) {
        auto paddedInputType =
          RankedTensorType::get({inputShape[0],
                                 ShapedType::isDynamic(inputShape[1])
                                   ? ShapedType::kDynamic
                                   : inputShape[1] + padding[0] + padding[1],
                                 ShapedType::isDynamic(inputShape[2])
                                   ? ShapedType::kDynamic
                                   : inputShape[2] + padding[2] + padding[3],
                                 inputShape[3]},
                                rewriter.getI8Type());
        Value paddingShape = createShape(
          rewriter,
          location,
          {0, 0, padding[0], padding[1], padding[2], padding[3], 0, 0});
        Value zero = createIntegerZero(
          rewriter, location, RankedTensorType::get({1}, rewriter.getI8Type()));
        Value paddedInput = rewriter.create<tosa::PadOp>(
          location, paddedInputType, quantizedInput, paddingShape, zero);
        auto hwcfType = getHWCFType(quantizedWeightType);
        Value linalgWeight =
          transposeOrFoldConstant(rewriter,
                                  location,
                                  quantizedWeight,
                                  hwcfType,
                                  ArrayRef<int32_t>{2, 3, 1, 0});
        Value initialized = initializeDynamicConvolutionOutput(
          rewriter,
          location,
          accumulatorType,
          quantizedInput,
          padding,
          {static_cast<int64_t>(operation.getKernelH()),
           static_cast<int64_t>(operation.getKernelW())},
          {static_cast<int64_t>(operation.getStrideH()),
           static_cast<int64_t>(operation.getStrideW())},
          {static_cast<int64_t>(operation.getDilationH()),
           static_cast<int64_t>(operation.getDilationW())});
        accumulator =
          rewriter
            .create<linalg::Conv2DNhwcHwcfOp>(
              location,
              TypeRange{accumulatorType},
              ValueRange{paddedInput, linalgWeight},
              ValueRange{initialized},
              createI64PairAttr(rewriter,
                                {static_cast<int64_t>(operation.getStrideH()),
                                 static_cast<int64_t>(operation.getStrideW())}),
              createI64PairAttr(
                rewriter,
                {static_cast<int64_t>(operation.getDilationH()),
                 static_cast<int64_t>(operation.getDilationW())}))
            .getResult(0);
      }
      if (!dynamicSpatial) {
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
        accumulatorType = RankedTensorType::get(
          {1, paddedHeight, paddedWidth, sourceOutput.getShape()[0]},
          rewriter.getI32Type());
      }
      if (!dynamicSpatial) {
        accumulator = rewriter.create<tosa::Conv2DOp>(
          location,
          accumulatorType,
          quantizedInput,
          weight,
          contractionBias,
          inputZero,
          weightZero,
          padding,
          ArrayRef<int64_t>{static_cast<int64_t>(operation.getStrideH()),
                            static_cast<int64_t>(operation.getStrideW())},
          ArrayRef<int64_t>{static_cast<int64_t>(operation.getDilationH()),
                            static_cast<int64_t>(operation.getDilationW())},
          rewriter.getI32Type());
      }
      Value result = dequantizeAccumulator(
        rewriter, location, accumulator, weightScale, inputScale, 3);
      if (operation.getHasBias()) {
        auto biasType = RankedTensorType::get(
          {1, 1, 1, sourceOutput.getShape()[0]}, rewriter.getF32Type());
        Value bias = reshapeValue(rewriter, location, tail.front(), biasType);
        result = rewriter.create<tosa::AddOp>(
          location, cast<RankedTensorType>(result.getType()), result, bias);
      }
      auto floatOutputType = outputType.clone(rewriter.getF32Type());
      if (cast<RankedTensorType>(result.getType()) != floatOutputType) {
        Value start = createShape(rewriter, location, {0, 0, 0, 0});
        Value size =
          createShape(rewriter, location, floatOutputType.getShape());
        result = rewriter.create<tosa::SliceOp>(
          location, floatOutputType, result, start, size);
      }
      if (scaleTerm > 100) {
        Value outputScale = tail[scaleOffset + 2];
        result = quantizeSignedI8(rewriter, location, result, outputScale);
      }
      rewriter.replaceOp(operation, result);
      return success();
    }
    Value bias;
    if (operation.getHasBias()) {
      bias = adaptor.getBiasAndScales().front();
    } else {
      auto biasType = RankedTensorType::get({sourceOutput.getShape()[0]},
                                            sourceOutput.getElementType());
      bias = createSplat(rewriter, operation.getLoc(), biasType, 0.0);
    }
    if (fp16Arithmetic) {
      Location location = operation.getLoc();
      input = convertFloatingTensor(rewriter, location, input, storageElement);
      bias = convertFloatingTensor(rewriter, location, bias, storageElement);
      auto hwcfType = getHWCFType(weightType);
      Value weight = transposeOrFoldConstant(rewriter,
                                             location,
                                             sourceWeight,
                                             hwcfType,
                                             ArrayRef<int32_t>{2, 3, 1, 0});
      SmallVector<int64_t> padding{padTop, padBottom, padLeft, padRight};
      auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
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
      const bool dynamicSpatial = ShapedType::isDynamic(inputShape[1]) ||
                                  ShapedType::isDynamic(inputShape[2]);
      if (dynamicSpatial) {
        return operation.emitOpError(
          "FP16 arithmetic convolution requires static spatial dimensions");
      }
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
      auto paddedInputType =
        RankedTensorType::get({inputShape[0],
                               inputShape[1] + padding[0] + padding[1],
                               inputShape[2] + padding[2] + padding[3],
                               inputShape[3]},
                              storageElement);
      Value paddingShape = createShape(
        rewriter,
        location,
        {0, 0, padding[0], padding[1], padding[2], padding[3], 0, 0});
      Value zero = createSplat(
        rewriter, location, RankedTensorType::get({1}, storageElement), 0.0);
      Value paddedInput = rewriter.create<tosa::PadOp>(
        location, paddedInputType, input, paddingShape, zero);
      auto fp16OutputType = outputType.clone(storageElement);
      Value initialized =
        initializeConvolutionOutput(rewriter, location, fp16OutputType, bias);
      auto convolution = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
        location,
        TypeRange{fp16OutputType},
        ValueRange{paddedInput, weight},
        ValueRange{initialized},
        createI64PairAttr(rewriter,
                          {static_cast<int64_t>(operation.getStrideH()),
                           static_cast<int64_t>(operation.getStrideW())}),
        createI64PairAttr(rewriter,
                          {static_cast<int64_t>(operation.getDilationH()),
                           static_cast<int64_t>(operation.getDilationW())}));
      Value result = convertFloatingTensor(
        rewriter, location, convolution.getResult(0), rewriter.getF32Type());
      rewriter.replaceOp(operation, result);
      return success();
    }
    if (lowPrecisionWeight) {
      input = roundStoragePrecision(
        rewriter, operation.getLoc(), input, storageElement);
    }
    if (lowPrecisionWeight) {
      sourceWeight = convertFloatingTensor(
        rewriter, operation.getLoc(), sourceWeight, rewriter.getF32Type());
      weightType = cast<RankedTensorType>(sourceWeight.getType());
    }
    auto ohwiType = getOHWIType(weightType);
    Value weight = transposeOrFoldConstant(rewriter,
                                           operation.getLoc(),
                                           sourceWeight,
                                           ohwiType,
                                           ArrayRef<int32_t>{0, 2, 3, 1});
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
    if (!ShapedType::isDynamic(inputShape[1])) {
      adjustTrailingPadding(inputShape[1],
                            operation.getKernelH(),
                            operation.getStrideH(),
                            operation.getDilationH(),
                            padTop,
                            padding[1]);
    }
    if (!ShapedType::isDynamic(inputShape[2])) {
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
    int64_t paddedHeight =
      ShapedType::isDynamic(inputShape[1])
        ? ShapedType::kDynamic
        : ((inputShape[1] + padding[0] + padding[1] - effectiveH) /
           operation.getStrideH()) +
            1;
    int64_t paddedWidth =
      ShapedType::isDynamic(inputShape[2])
        ? ShapedType::kDynamic
        : ((inputShape[2] + padding[2] + padding[3] - effectiveW) /
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
      if (dynamicSpatial) {
        SmallVector<OpFoldResult> offsets(outputType.getRank(),
                                          rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(outputType.getRank(),
                                          rewriter.getIndexAttr(1));
        for (auto [dimension, extent] :
             llvm::enumerate(outputType.getShape())) {
          sizes.push_back(ShapedType::isDynamic(extent)
                            ? OpFoldResult(rewriter.create<tensor::DimOp>(
                                operation.getLoc(), result, dimension))
                            : OpFoldResult(rewriter.getIndexAttr(extent)));
        }
        result = rewriter.create<tensor::ExtractSliceOp>(
          operation.getLoc(), outputType, result, offsets, sizes, strides);
      } else {
        Value start = createShape(rewriter, operation.getLoc(), {0, 0, 0, 0});
        Value size =
          createShape(rewriter, operation.getLoc(), outputType.getShape());
        result = rewriter.create<tosa::SliceOp>(
          operation.getLoc(), outputType, result, start, size);
      }
    }
    if (lowPrecisionWeight) {
      result = roundStoragePrecision(
        rewriter, operation.getLoc(), result, storageElement);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
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
    auto sourceWeightType =
      operands.size() >= 2 ? dyn_cast<RankedTensorType>(operands[1].getType())
                           : RankedTensorType{};
    const int64_t scaleTerm = getIntegerAttrOr(operation, "int8_scale_term", 0);
    const bool quantized = scaleTerm != 0;
    const bool lowPrecisionWeight =
      sourceWeightType && sourceWeightType.hasStaticShape() &&
      (sourceWeightType.getElementType().isF16() ||
       sourceWeightType.getElementType().isBF16());
    const bool fp16Arithmetic = sourceWeightType &&
                                sourceWeightType.getElementType().isF16() &&
                                usesFP16Arithmetic(operation);
    auto sourceInputType =
      operands.empty() ? RankedTensorType{}
                       : dyn_cast<RankedTensorType>(operands[0].getType());
    const bool supportedQuantizedInput =
      sourceInputType && (sourceInputType.getElementType().isF32() ||
                          sourceInputType.getElementType().isInteger(8));
    const bool supportedQuantizedWeight =
      sourceWeightType && sourceWeightType.hasStaticShape() &&
      (sourceWeightType.getElementType().isF32() ||
       sourceWeightType.getElementType().isInteger(8));
    if (operands.size() < 2 || operation->getNumResults() != 1 ||
        (quantized ? !supportedQuantizedInput
                   : !isRankedF32Tensor(operation->getOperand(0).getType())) ||
        (quantized ? !supportedQuantizedWeight
                   : (!isStaticF32Tensor(operands[1].getType()) &&
                      !lowPrecisionWeight))) {
      return operation->emitOpError(
        "supports ranked f32/i8 input and static f32/i8 weight tensors for "
        "quantized convolution");
    }
    Type storageElement = sourceWeightType.getElementType();
    Value input = operands[0];
    Value sourceWeight = operands[1];
    const bool hasBias = getIntegerAttrOr(operation, "has_bias", 0) != 0;
    const unsigned scaleOffset = hasBias ? 3 : 2;
    Value weightScale;
    Value inputScale;
    if (quantized) {
      weightScale = operands[scaleOffset];
      inputScale = operands[scaleOffset + 1];
      input =
        quantizeSignedI8(rewriter, operation->getLoc(), input, inputScale);
      auto weightScaleType = cast<RankedTensorType>(weightScale.getType());
      sourceWeight = quantizeSignedI8(rewriter,
                                      operation->getLoc(),
                                      sourceWeight,
                                      weightScale,
                                      weightScaleType.getShape()[0] == 1
                                        ? std::nullopt
                                        : std::optional<unsigned>(0));
      sourceWeightType = cast<RankedTensorType>(sourceWeight.getType());
    } else if (fp16Arithmetic) {
      input = convertFloatingTensor(
        rewriter, operation->getLoc(), input, storageElement);
    } else if (lowPrecisionWeight) {
      input = roundStoragePrecision(
        rewriter, operation->getLoc(), input, storageElement);
      sourceWeight = convertFloatingTensor(
        rewriter, operation->getLoc(), sourceWeight, rewriter.getF32Type());
    }
    auto inputType = cast<RankedTensorType>(input.getType());
    auto weightType = cast<RankedTensorType>(sourceWeight.getType());
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
      rewriter, operation->getLoc(), sourceWeight, groupedWeightType);
    auto hwcmType = RankedTensorType::get({weightType.getShape()[2],
                                           weightType.getShape()[3],
                                           channels,
                                           multiplier},
                                          weightType.getElementType());
    Value weight = transposeOrFoldConstant(rewriter,
                                           operation->getLoc(),
                                           groupedWeight,
                                           hwcmType,
                                           ArrayRef<int32_t>{2, 3, 0, 1});
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
        RankedTensorType::get(
          {outputs},
          quantized ? rewriter.getF32Type() : inputType.getElementType()),
        0.0);
    }
    if (fp16Arithmetic) {
      bias = convertFloatingTensor(
        rewriter, operation->getLoc(), bias, storageElement);
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
    auto outputType = getNHWCType(sourceOutput);
    if (quantized) {
      auto accumulatorType = outputType.clone(rewriter.getI32Type());
      Value accumulator;
      if (dynamicSpatial) {
        auto paddedInputType =
          RankedTensorType::get({inputType.getShape()[0],
                                 inputType.isDynamicDim(1)
                                   ? ShapedType::kDynamic
                                   : inputType.getShape()[1] + pad[0] + pad[1],
                                 inputType.isDynamicDim(2)
                                   ? ShapedType::kDynamic
                                   : inputType.getShape()[2] + pad[2] + pad[3],
                                 inputType.getShape()[3]},
                                rewriter.getI8Type());
        Value paddingShape =
          createShape(rewriter,
                      operation->getLoc(),
                      {0, 0, pad[0], pad[1], pad[2], pad[3], 0, 0});
        Value zero =
          createIntegerZero(rewriter,
                            operation->getLoc(),
                            RankedTensorType::get({1}, rewriter.getI8Type()));
        Value paddedInput = rewriter.create<tosa::PadOp>(
          operation->getLoc(), paddedInputType, input, paddingShape, zero);
        auto expandedAccumulatorType =
          RankedTensorType::get({accumulatorType.getShape()[0],
                                 accumulatorType.getShape()[1],
                                 accumulatorType.getShape()[2],
                                 channels,
                                 multiplier},
                                rewriter.getI32Type());
        Value initialized = initializeDynamicConvolutionOutput(
          rewriter,
          operation->getLoc(),
          expandedAccumulatorType,
          input,
          pad,
          {weightType.getShape()[2], weightType.getShape()[3]},
          *stride,
          *dilation);
        Value expanded = rewriter
                           .create<linalg::DepthwiseConv2DNhwcHwcmOp>(
                             operation->getLoc(),
                             TypeRange{expandedAccumulatorType},
                             ValueRange{paddedInput, weight},
                             ValueRange{initialized},
                             createI64PairAttr(rewriter, *stride),
                             createI64PairAttr(rewriter, *dilation))
                           .getResult(0);
        SmallVector<ReassociationIndices> reassociation = {
          {0}, {1}, {2}, {3, 4}};
        accumulator = rewriter.create<tensor::CollapseShapeOp>(
          operation->getLoc(), accumulatorType, expanded, reassociation);
      }
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
        accumulatorType = RankedTensorType::get(
          {1, paddedHeight, paddedWidth, outputs}, rewriter.getI32Type());
      }
      Value zero =
        createIntegerZero(rewriter,
                          operation->getLoc(),
                          RankedTensorType::get({1}, rewriter.getI8Type()));
      Value contractionBias = createIntegerZero(
        rewriter,
        operation->getLoc(),
        RankedTensorType::get({outputs}, rewriter.getI32Type()));
      if (!dynamicSpatial) {
        accumulator =
          rewriter.create<tosa::DepthwiseConv2DOp>(operation->getLoc(),
                                                   accumulatorType,
                                                   input,
                                                   weight,
                                                   contractionBias,
                                                   zero,
                                                   zero,
                                                   pad,
                                                   *stride,
                                                   *dilation,
                                                   rewriter.getI32Type());
      }
      Value result = dequantizeAccumulator(
        rewriter, operation->getLoc(), accumulator, weightScale, inputScale, 3);
      if (hasBias) {
        auto biasType =
          RankedTensorType::get({1, 1, 1, outputs}, rewriter.getF32Type());
        Value broadcastBias =
          reshapeValue(rewriter, operation->getLoc(), bias, biasType);
        result =
          rewriter.create<tosa::AddOp>(operation->getLoc(),
                                       cast<RankedTensorType>(result.getType()),
                                       result,
                                       broadcastBias);
      }
      auto floatOutputType = outputType.clone(rewriter.getF32Type());
      if (cast<RankedTensorType>(result.getType()) != floatOutputType) {
        Value start = createShape(rewriter, operation->getLoc(), {0, 0, 0, 0});
        Value size = createShape(
          rewriter, operation->getLoc(), floatOutputType.getShape());
        result = rewriter.create<tosa::SliceOp>(
          operation->getLoc(), floatOutputType, result, start, size);
      }
      if (scaleTerm > 100) {
        result = quantizeSignedI8(
          rewriter, operation->getLoc(), result, operands[scaleOffset + 2]);
      }
      rewriter.replaceOp(operation, result);
      return success();
    }
    if (fp16Arithmetic) {
      if (dynamicSpatial) {
        return operation->emitOpError(
          "FP16 arithmetic depthwise convolution requires static spatial "
          "dimensions");
      }
      auto paddedInputType =
        RankedTensorType::get({inputType.getShape()[0],
                               inputType.getShape()[1] + pad[0] + pad[1],
                               inputType.getShape()[2] + pad[2] + pad[3],
                               inputType.getShape()[3]},
                              storageElement);
      Value paddingShape =
        createShape(rewriter,
                    operation->getLoc(),
                    {0, 0, pad[0], pad[1], pad[2], pad[3], 0, 0});
      Value zero = createSplat(rewriter,
                               operation->getLoc(),
                               RankedTensorType::get({1}, storageElement),
                               0.0);
      Value paddedInput = rewriter.create<tosa::PadOp>(
        operation->getLoc(), paddedInputType, input, paddingShape, zero);
      auto fp16OutputType = outputType.clone(storageElement);
      Value initialized = initializeConvolutionOutput(
        rewriter, operation->getLoc(), fp16OutputType, bias);
      auto convolution = rewriter.create<linalg::DepthwiseConv2DNhwcHwcmOp>(
        operation->getLoc(),
        TypeRange{fp16OutputType},
        ValueRange{paddedInput, weight},
        ValueRange{initialized},
        createI64PairAttr(rewriter, *stride),
        createI64PairAttr(rewriter, *dilation));
      Value result = convertFloatingTensor(rewriter,
                                           operation->getLoc(),
                                           convolution.getResult(0),
                                           rewriter.getF32Type());
      rewriter.replaceOp(operation, result);
      return success();
    }
    if (dynamicSpatial) {
      if (!inputType.isDynamicDim(1)) {
        adjustTrailingPadding(inputType.getShape()[1],
                              weightType.getShape()[2],
                              (*stride)[0],
                              (*dilation)[0],
                              pad[0],
                              pad[1]);
      }
      if (!inputType.isDynamicDim(2)) {
        adjustTrailingPadding(inputType.getShape()[2],
                              weightType.getShape()[3],
                              (*stride)[1],
                              (*dilation)[1],
                              pad[2],
                              pad[3]);
      }
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
    int64_t effectiveH = ((weightType.getShape()[2] - 1) * (*dilation)[0]) + 1;
    int64_t effectiveW = ((weightType.getShape()[3] - 1) * (*dilation)[1]) + 1;
    int64_t paddedHeight =
      inputType.isDynamicDim(1)
        ? ShapedType::kDynamic
        : ((inputType.getShape()[1] + pad[0] + pad[1] - effectiveH) /
           (*stride)[0]) +
            1;
    int64_t paddedWidth =
      inputType.isDynamicDim(2)
        ? ShapedType::kDynamic
        : ((inputType.getShape()[2] + pad[2] + pad[3] - effectiveW) /
           (*stride)[1]) +
            1;
    auto paddedOutputType = RankedTensorType::get(
      {1, paddedHeight, paddedWidth, outputs}, sourceOutput.getElementType());
    Value result =
      rewriter.create<tosa::DepthwiseConv2DOp>(operation->getLoc(),
                                               paddedOutputType,
                                               input,
                                               weight,
                                               bias,
                                               inputZero,
                                               weightZero,
                                               pad,
                                               *stride,
                                               *dilation,
                                               rewriter.getF32Type());
    if (paddedOutputType != outputType) {
      if (dynamicSpatial) {
        SmallVector<OpFoldResult> offsets(outputType.getRank(),
                                          rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides(outputType.getRank(),
                                          rewriter.getIndexAttr(1));
        for (auto [dimension, extent] :
             llvm::enumerate(outputType.getShape())) {
          sizes.push_back(ShapedType::isDynamic(extent)
                            ? OpFoldResult(rewriter.create<tensor::DimOp>(
                                operation->getLoc(), result, dimension))
                            : OpFoldResult(rewriter.getIndexAttr(extent)));
        }
        result = rewriter.create<tensor::ExtractSliceOp>(
          operation->getLoc(), outputType, result, offsets, sizes, strides);
      } else {
        Value start = createShape(rewriter, operation->getLoc(), {0, 0, 0, 0});
        Value size =
          createShape(rewriter, operation->getLoc(), outputType.getShape());
        result = rewriter.create<tosa::SliceOp>(
          operation->getLoc(), outputType, result, start, size);
      }
    }
    if (lowPrecisionWeight) {
      result = roundStoragePrecision(
        rewriter, operation->getLoc(), result, storageElement);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

}  // namespace

void populateConvToTosaPatterns(RewritePatternSet& patterns,
                                const TypeConverter& typeConverter,
                                MLIRContext* context) {
  patterns.add<ConvertConvolution>(typeConverter, context);
  patterns.add<ConvertDepthwiseConvolution>(
    typeConverter, context, contract::kLayerConvolutionDepthwise);
}

}  // namespace mlir::ncnn
