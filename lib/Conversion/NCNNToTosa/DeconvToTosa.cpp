// Deconvolution 族降低（转置卷积）。
//
// 职责
//   把 ncnn.deconvolution 降低为 tosa.transpose_conv2d，含权重布局转换
//   （OIHW -> OHWI）与 padding/stride 展开。
//
// 不变量
//   * 权重布局转换不改变数值；
//   * padding 只在静态可解时展开，否则报错而不是猜；
//   * 低精度边界统一走 applyLowPrecisionBoundary。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后（padding/stride/dilation 已解析）；
//   * 之后由 TosaToLinalg 转 tensor 算子。
//
// 明确不做
//   * 不做输出形状推导之外的改写；
//   * 不融合 bias/激活（FuseLinalgEpilogue 的职责）。

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
using tosa_lowering::createSplat;
using tosa_lowering::getIntegerAttrOr;
using tosa_lowering::getNHWCType;
using tosa_lowering::getOHWIType;
using tosa_lowering::getRequiredIntegerAttr;
using tosa_lowering::isRankedF32Tensor;
using tosa_lowering::transposeOrFoldConstant;

class ConvertDeconvolution final : public ConversionPattern {
 public:
  ConvertDeconvolution(const TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(
        typeConverter, contract::kLayerDeconvolution, 1, context) {}

  LogicalResult matchAndRewrite(
    Operation* operation,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const final {
    if (operands.size() < 2 || operation->getNumResults() != 1 ||
        !isRankedF32Tensor(operation->getOperand(0).getType()) ||
        !isa<FloatType>(cast<ShapedType>(operation->getOperand(1).getType())
                          .getElementType()) ||
        !isRankedF32Tensor(operation->getResult(0).getType())) {
      return operation->emitOpError(
        "supports ranked f32 input/result and static f32 weight tensors only");
    }
    auto sourceInput =
      cast<RankedTensorType>(operation->getOperand(0).getType());
    auto weightType = cast<RankedTensorType>(operands[1].getType());
    Value sourceWeight = operands[1];
    if (!weightType.getElementType().isF32()) {
      sourceWeight = convertFloatingTensor(
        rewriter, operation->getLoc(), sourceWeight, rewriter.getF32Type());
      weightType = cast<RankedTensorType>(sourceWeight.getType());
    }
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

      Value input = applyLowPrecisionBoundary(
        rewriter, operation->getLoc(), operation, operands[0]);
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
        ValueRange{input, sourceWeight},
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
      dynamicResult = applyLowPrecisionBoundary(
        rewriter, operation->getLoc(), operation, dynamicResult);
      rewriter.replaceOp(operation, dynamicResult);
      return success();
    }
    auto ohwiType = getOHWIType(weightType);
    Value weight = transposeOrFoldConstant(rewriter,
                                           operation->getLoc(),
                                           sourceWeight,
                                           ohwiType,
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
      applyLowPrecisionBoundary(
        rewriter, operation->getLoc(), operation, operands[0]),
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
    result = applyLowPrecisionBoundary(
      rewriter, operation->getLoc(), operation, result);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

}  // namespace

void populateDeconvToTosaPatterns(RewritePatternSet& patterns,
                                  const TypeConverter& typeConverter,
                                  MLIRContext* context) {
  patterns.add<ConvertDeconvolution>(typeConverter, context);
}

}  // namespace mlir::ncnn
