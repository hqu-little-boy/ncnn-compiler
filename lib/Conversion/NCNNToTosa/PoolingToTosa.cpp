// Pooling 降低。
//
// 职责
//   把 ncnn.pooling 降低为 tosa.max_pool2d / avg_pool2d，含 padding
//   展开（getPoolPadding）与 count_include_pad 语义对齐。
//
// 不变量
//   * padding 只在静态可解时展开，否则报错而不是猜；
//   * Average + include_pad + 静态形状的实例被判为「可不转换」，
//     由 ConversionTarget 动态合法化保留。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后（kernel/stride/pad 已解析）。
//
// 明确不做
//   * 不做全局池化之外的降采样策略；
//   * 不改写为 conv（那会改变数值路径）。

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
using tosa_lowering::createIndexConstant;
using tosa_lowering::createShape;
using tosa_lowering::createSplat;
using tosa_lowering::getNHWCType;

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
  const bool dynamicSpatial = input.isDynamicDim(1) || input.isDynamicDim(2);
  if (dynamicSpatial) {
    if (operation.getPadMode() == 0 &&
        (operation.getStrideH() != 1 || operation.getStrideW() != 1)) {
      operation.emitOpError(
        "dynamic tail pooling requires unit spatial strides");
      return failure();
    }
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
      operation.emitOpError("pool padding must be non-negative");
      return failure();
    }
    return SmallVector<int64_t>{top, bottom, left, right};
  }
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
  auto makeDivisible = [](int64_t inputSize,
                          int64_t kernel,
                          int64_t stride,
                          int64_t leading,
                          int64_t& trailing) {
    int64_t remainder = (inputSize + leading + trailing - kernel) % stride;
    if (remainder < 0) {
      remainder += stride;
    }
    if (remainder != 0) {
      trailing += stride - remainder;
    }
  };
  makeDivisible(input.getShape()[1],
                operation.getKernelH(),
                operation.getStrideH(),
                top,
                bottom);
  makeDivisible(input.getShape()[2],
                operation.getKernelW(),
                operation.getStrideW(),
                left,
                right);
  return SmallVector<int64_t>{top, bottom, left, right};
}

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
    const bool adaptive =
      operation.getMode() == static_cast<int64_t>(PoolMode::Adaptive);
    const auto kernelH = static_cast<int64_t>(operation.getKernelH());
    const auto kernelW = static_cast<int64_t>(operation.getKernelW());
    auto inputType = cast<RankedTensorType>(input.getType());
    const bool dynamicRegular =
      !global && !adaptive && !inputType.hasStaticShape();
    const bool signedI8 = inputType.getElementType().isSignlessInteger(8);
    if (!signedI8) {
      input = applyLowPrecisionBoundary(
        rewriter, operation.getLoc(), operation, input);
    }
    if ((global && !inputType.hasStaticShape()) || adaptive || dynamicRegular) {
      if (signedI8) {
        return operation.emitOpError(
          "dynamic or adaptive signed i8 pooling is not supported");
      }
      Location location = operation.getLoc();
      const bool maximum =
        operation.getKind() == static_cast<int64_t>(PoolKind::Maximum);
      Value inputH = rewriter.create<tensor::DimOp>(location, input, 1);
      Value inputW = rewriter.create<tensor::DimOp>(location, input, 2);
      Value inputC = rewriter.create<tensor::DimOp>(location, input, 3);
      Value outputH;
      Value outputW;
      RankedTensorType runtimeOutputType;
      if (global) {
        runtimeOutputType = sourceOutput;
      } else {
        if (dynamicRegular) {
          auto outputExtent = [&](Value inputExtent,
                                  int64_t kernel,
                                  int64_t stride,
                                  int64_t padBefore,
                                  int64_t padAfter) {
            Value padded = rewriter.create<arith::AddIOp>(
              location,
              inputExtent,
              createIndexConstant(rewriter, location, padBefore + padAfter));
            Value reduced = rewriter.create<arith::SubIOp>(
              location,
              padded,
              createIndexConstant(rewriter, location, kernel));
            if (operation.getPadMode() == 0) {
              reduced = rewriter.create<arith::AddIOp>(
                location,
                reduced,
                createIndexConstant(rewriter, location, stride - 1));
            }
            Value quotient = rewriter.create<arith::DivUIOp>(
              location,
              reduced,
              createIndexConstant(rewriter, location, stride));
            return rewriter.create<arith::AddIOp>(
              location, quotient, createIndexConstant(rewriter, location, 1));
          };
          outputH =
            outputExtent(inputH,
                         kernelH,
                         static_cast<int64_t>(operation.getStrideH()),
                         static_cast<int64_t>(operation.getPadTop()),
                         static_cast<int64_t>(operation.getPadBottom()));
          outputW = outputExtent(inputW,
                                 kernelW,
                                 static_cast<int64_t>(operation.getStrideW()),
                                 static_cast<int64_t>(operation.getPadLeft()),
                                 static_cast<int64_t>(operation.getPadRight()));
        } else {
          outputH = kernelH == -233
                      ? inputH
                      : createIndexConstant(rewriter, location, kernelH);
          outputW = kernelW == -233
                      ? inputW
                      : createIndexConstant(rewriter, location, kernelW);
        }
        runtimeOutputType = getNHWCType(sourceOutput);
      }

      SmallVector<Value> dynamicSizes;
      for (int64_t dimension = 0; dimension < runtimeOutputType.getRank();
           ++dimension) {
        if (!runtimeOutputType.isDynamicDim(dimension)) {
          continue;
        }
        if (global) {
          dynamicSizes.push_back(inputC);
        } else if (dimension == 1) {
          dynamicSizes.push_back(outputH);
        } else if (dimension == 2) {
          dynamicSizes.push_back(outputW);
        } else {
          dynamicSizes.push_back(inputC);
        }
      }
      Value empty = rewriter.create<tensor::EmptyOp>(
        location, runtimeOutputType, dynamicSizes);
      AffineMap outputMap =
        rewriter.getMultiDimIdentityMap(runtimeOutputType.getRank());
      SmallVector<utils::IteratorType> iterators(runtimeOutputType.getRank(),
                                                 utils::IteratorType::parallel);
      auto result = rewriter.create<linalg::GenericOp>(
        location,
        runtimeOutputType,
        ValueRange{},
        ValueRange{empty},
        ArrayRef<AffineMap>{outputMap},
        iterators,
        [&](OpBuilder& nested, Location bodyLocation, ValueRange) {
          Value channel = nested.create<linalg::IndexOp>(
            bodyLocation, runtimeOutputType.getRank() - 1);
          Value heightBegin = createIndexConstant(nested, bodyLocation, 0);
          Value heightEnd = inputH;
          Value widthBegin = createIndexConstant(nested, bodyLocation, 0);
          Value widthEnd = inputW;
          if (adaptive) {
            Value oh = nested.create<linalg::IndexOp>(bodyLocation, 1);
            Value ow = nested.create<linalg::IndexOp>(bodyLocation, 2);
            Value one = createIndexConstant(nested, bodyLocation, 1);
            Type wideType = nested.getIntegerType(128);
            auto widen = [&](Value value) {
              return nested.create<arith::IndexCastUIOp>(
                bodyLocation, wideType, value);
            };
            auto boundary = [&](Value position,
                                Value inputExtent,
                                Value outputExtent,
                                bool ceiling) {
              Value wideOutput = widen(outputExtent);
              Value numerator = nested.create<arith::MulIOp>(
                bodyLocation, widen(position), widen(inputExtent));
              if (ceiling) {
                Value wideOne = nested.create<arith::ConstantOp>(
                  bodyLocation, nested.getIntegerAttr(wideType, 1));
                numerator = nested.create<arith::AddIOp>(
                  bodyLocation,
                  numerator,
                  nested.create<arith::SubIOp>(
                    bodyLocation, wideOutput, wideOne));
              }
              Value quotient = nested.create<arith::DivUIOp>(
                bodyLocation, numerator, wideOutput);
              return nested.create<arith::IndexCastUIOp>(
                bodyLocation, nested.getIndexType(), quotient);
            };
            Value nextH = nested.create<arith::AddIOp>(bodyLocation, oh, one);
            Value nextW = nested.create<arith::AddIOp>(bodyLocation, ow, one);
            if (kernelH == -233) {
              heightBegin = oh;
              heightEnd = nextH;
            } else {
              heightBegin = boundary(oh, inputH, outputH, false);
              heightEnd = boundary(nextH, inputH, outputH, true);
            }
            if (kernelW == -233) {
              widthBegin = ow;
              widthEnd = nextW;
            } else {
              widthBegin = boundary(ow, inputW, outputW, false);
              widthEnd = boundary(nextW, inputW, outputW, true);
            }
          } else if (dynamicRegular) {
            Value oh = nested.create<linalg::IndexOp>(bodyLocation, 1);
            Value ow = nested.create<linalg::IndexOp>(bodyLocation, 2);
            Value zero = createIndexConstant(nested, bodyLocation, 0);
            Value strideH =
              createIndexConstant(nested,
                                  bodyLocation,
                                  static_cast<int64_t>(operation.getStrideH()));
            Value strideW =
              createIndexConstant(nested,
                                  bodyLocation,
                                  static_cast<int64_t>(operation.getStrideW()));
            Value logicalHeightBegin = nested.create<arith::SubIOp>(
              bodyLocation,
              nested.create<arith::MulIOp>(bodyLocation, oh, strideH),
              createIndexConstant(nested,
                                  bodyLocation,
                                  static_cast<int64_t>(operation.getPadTop())));
            Value logicalWidthBegin = nested.create<arith::SubIOp>(
              bodyLocation,
              nested.create<arith::MulIOp>(bodyLocation, ow, strideW),
              createIndexConstant(
                nested,
                bodyLocation,
                static_cast<int64_t>(operation.getPadLeft())));
            Value logicalHeightEnd = nested.create<arith::AddIOp>(
              bodyLocation,
              logicalHeightBegin,
              createIndexConstant(nested, bodyLocation, kernelH));
            Value logicalWidthEnd = nested.create<arith::AddIOp>(
              bodyLocation,
              logicalWidthBegin,
              createIndexConstant(nested, bodyLocation, kernelW));
            heightBegin = nested.create<arith::MinSIOp>(
              bodyLocation,
              nested.create<arith::MaxSIOp>(
                bodyLocation, logicalHeightBegin, zero),
              inputH);
            widthBegin = nested.create<arith::MinSIOp>(
              bodyLocation,
              nested.create<arith::MaxSIOp>(
                bodyLocation, logicalWidthBegin, zero),
              inputW);
            heightEnd = nested.create<arith::MinSIOp>(
              bodyLocation,
              nested.create<arith::MaxSIOp>(
                bodyLocation, logicalHeightEnd, zero),
              inputH);
            widthEnd = nested.create<arith::MinSIOp>(
              bodyLocation,
              nested.create<arith::MaxSIOp>(
                bodyLocation, logicalWidthEnd, zero),
              inputW);
          }

          Value initial = nested.create<arith::ConstantFloatOp>(
            bodyLocation,
            nested.getF32Type(),
            APFloat(maximum ? -std::numeric_limits<float>::infinity() : 0.0F));
          auto rows = nested.create<scf::ForOp>(
            bodyLocation,
            heightBegin,
            heightEnd,
            createIndexConstant(nested, bodyLocation, 1),
            ValueRange{initial},
            [&](OpBuilder& rowBuilder,
                Location rowLocation,
                Value row,
                ValueRange rowState) {
              auto columns = rowBuilder.create<scf::ForOp>(
                rowLocation,
                widthBegin,
                widthEnd,
                createIndexConstant(rowBuilder, rowLocation, 1),
                rowState,
                [&](OpBuilder& columnBuilder,
                    Location columnLocation,
                    Value column,
                    ValueRange columnState) {
                  Value batch =
                    createIndexConstant(columnBuilder, columnLocation, 0);
                  Value element = columnBuilder.create<tensor::ExtractOp>(
                    columnLocation,
                    input,
                    ValueRange{batch, row, column, channel});
                  Value accumulated =
                    maximum ? Value(columnBuilder.create<arith::MaximumFOp>(
                                columnLocation, columnState.front(), element))
                            : Value(columnBuilder.create<arith::AddFOp>(
                                columnLocation, columnState.front(), element));
                  columnBuilder.create<scf::YieldOp>(columnLocation,
                                                     accumulated);
                });
              rowBuilder.create<scf::YieldOp>(rowLocation,
                                              columns.getResult(0));
            });
          Value pooled = rows.getResult(0);
          if (!maximum) {
            Value rowsExtent =
              dynamicRegular && operation.getIncludePad()
                ? createIndexConstant(nested, bodyLocation, kernelH)
                : nested.create<arith::SubIOp>(
                    bodyLocation, heightEnd, heightBegin);
            Value columnsExtent =
              dynamicRegular && operation.getIncludePad()
                ? createIndexConstant(nested, bodyLocation, kernelW)
                : nested.create<arith::SubIOp>(
                    bodyLocation, widthEnd, widthBegin);
            Value rowsCount = nested.create<arith::IndexCastOp>(
              bodyLocation, nested.getI64Type(), rowsExtent);
            Value columnsCount = nested.create<arith::IndexCastOp>(
              bodyLocation, nested.getI64Type(), columnsExtent);
            Value count = nested.create<arith::MulIOp>(
              bodyLocation, rowsCount, columnsCount);
            Value countFloat = nested.create<arith::UIToFPOp>(
              bodyLocation, nested.getF32Type(), count);
            pooled =
              nested.create<arith::DivFOp>(bodyLocation, pooled, countFloat);
          }
          nested.create<linalg::YieldOp>(bodyLocation, pooled);
        });
      Value dynamicResult = result.getResult(0);
      dynamicResult = applyLowPrecisionBoundary(
        rewriter, operation.getLoc(), operation, dynamicResult);
      rewriter.replaceOp(operation, dynamicResult);
      return success();
    }
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
    RankedTensorType pooledType = outputType;
    if (!global && inputType.hasStaticShape()) {
      const int64_t pooledHeight =
        ((inputType.getShape()[1] + (*padding)[0] + (*padding)[1] - kernel[0]) /
         stride[0]) +
        1;
      const int64_t pooledWidth =
        ((inputType.getShape()[2] + (*padding)[2] + (*padding)[3] - kernel[1]) /
         stride[1]) +
        1;
      pooledType = RankedTensorType::get(
        {1, pooledHeight, pooledWidth, inputType.getShape()[3]},
        inputType.getElementType());
    }
    Value poolInput = input;
    SmallVector<int64_t> poolPadding = *padding;
    const bool materializeMaxPadding =
      !global &&
      operation.getKind() == static_cast<int64_t>(PoolKind::Maximum) &&
      (poolPadding[0] >= kernel[0] || poolPadding[1] >= kernel[0] ||
       poolPadding[2] >= kernel[1] || poolPadding[3] >= kernel[1]);
    if (signedI8 && materializeMaxPadding) {
      return operation.emitOpError(
        "signed i8 max pooling padding exceeds the kernel");
    }
    if (materializeMaxPadding) {
      auto paddedDimension =
        [](int64_t dimension, int64_t before, int64_t after) {
          return ShapedType::isDynamic(dimension) ? ShapedType::kDynamic
                                                  : dimension + before + after;
        };
      auto paddedInputType = RankedTensorType::get(
        {inputType.getShape()[0],
         paddedDimension(
           inputType.getShape()[1], poolPadding[0], poolPadding[1]),
         paddedDimension(
           inputType.getShape()[2], poolPadding[2], poolPadding[3]),
         inputType.getShape()[3]},
        inputType.getElementType());
      Value paddingShape = createShape(rewriter,
                                       operation.getLoc(),
                                       {0,
                                        0,
                                        poolPadding[0],
                                        poolPadding[1],
                                        poolPadding[2],
                                        poolPadding[3],
                                        0,
                                        0});
      Value padValue =
        createSplat(rewriter,
                    operation.getLoc(),
                    RankedTensorType::get({1}, inputType.getElementType()),
                    -std::numeric_limits<double>::infinity());
      poolInput = rewriter.create<tosa::PadOp>(
        operation.getLoc(), paddedInputType, input, paddingShape, padValue);
      poolPadding.assign(4, 0);
    }
    Value result;
    if (operation.getKind() == static_cast<int64_t>(PoolKind::Maximum)) {
      result = rewriter.create<tosa::MaxPool2dOp>(
        operation.getLoc(), pooledType, poolInput, kernel, stride, poolPadding);
    } else {
      Value zero =
        createSplat(rewriter,
                    operation.getLoc(),
                    RankedTensorType::get({1}, inputType.getElementType()),
                    0.0);
      result = rewriter.create<tosa::AvgPool2dOp>(operation.getLoc(),
                                                  pooledType,
                                                  input,
                                                  zero,
                                                  zero,
                                                  kernel,
                                                  stride,
                                                  *padding,
                                                  rewriter.getF32Type());
    }
    if (pooledType != outputType) {
      Value start = createShape(rewriter, operation.getLoc(), {0, 0, 0, 0});
      Value size =
        createShape(rewriter, operation.getLoc(), outputType.getShape());
      result = rewriter.create<tosa::SliceOp>(
        operation.getLoc(), outputType, result, start, size);
    }
    if (global) {
      Value shape =
        createShape(rewriter, operation.getLoc(), sourceOutput.getShape());
      result = rewriter.create<tosa::ReshapeOp>(
        operation.getLoc(), sourceOutput, result, shape);
    }
    if (!signedI8) {
      result = applyLowPrecisionBoundary(
        rewriter, operation.getLoc(), operation, result);
    }
    rewriter.replaceOp(operation, result);
    return success();
  }
};

}  // namespace

void populatePoolingToTosaPatterns(RewritePatternSet& patterns,
                                   const TypeConverter& typeConverter,
                                   MLIRContext* context) {
  patterns.add<ConvertPooling>(typeConverter, context);
}

}  // namespace mlir::ncnn
