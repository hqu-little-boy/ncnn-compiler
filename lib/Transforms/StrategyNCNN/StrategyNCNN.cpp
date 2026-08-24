#include "ncnn-mlir/Transforms/StrategyNCNN/StrategyNCNN.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_STRATEGYNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// 算子形态策略（A1）：卷积的窗口索引映射含加法算术，不满足
// vector.contract 的投影置换前置条件，也无法进入 GEMM 的循环序访存。
// 本 pass 在向量化之前把可改写的 linalg.conv_2d_nhwc_hwcf 变换为
// matmul 形态：
//   路径一  1×1 s1：collapse(image)+collapse(weight) 直接成为
//           [N·H·W,IC]×[IC,OC] matmul（无数据复制，纯视图；动态 H/W 同样
//           成立）；
//   路径二  k×k 静态空间维且命中 ncnn prefer_gemm 启发式：linalg.generic
//           gather 构造 im2col [N·H·W, KH·KW·IC]，与常量折叠的
//           collapse(weight) 做 matmul；
//   方案 a  epilogue 衔接：conv 结果的唯一用户若是恒等映射的逐元素
//           generic（ReLU/Sigmoid 等），则该 consumer 一并改写进折叠的
//           二维域（matmul → 2D generic → expand_shape），使激活随
//           matmul 主循环落位、并保持向量化器可提升的形态。
// 深度卷积 lower 为独立的 linalg::DepthwiseConv2DNhwcHwcmOp，天然不进
// 本策略；Winograd 仅保留开关位（数值预算未验证，默认关闭）。
enum class ConvStrategy { Auto, Gemm, Conv, Winograd };

std::optional<ConvStrategy> parseStrategy(StringRef strategy) {
  if (strategy == "auto") {
    return ConvStrategy::Auto;
  }
  if (strategy == "gemm") {
    return ConvStrategy::Gemm;
  }
  if (strategy == "conv") {
    return ConvStrategy::Conv;
  }
  if (strategy == "winograd") {
    return ConvStrategy::Winograd;
  }
  return std::nullopt;
}

bool isLiftableElementwiseBody(linalg::GenericOp consumer) {
  Region& region = consumer->getRegion(0);
  if (!region.hasOneBlock()) {
    return false;
  }
  Block& block = region.front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    return false;
  }
  if (block.getNumArguments() != 2) {
    return false;
  }
  Value input = block.getArgument(0);
  Value output = block.getArgument(1);
  if (input.use_empty() || !output.use_empty()) {
    return false;
  }
  bool hasComputation = false;
  for (Operation& operation : block.without_terminator()) {
    const StringRef dialect = operation.getName().getDialect()->getNamespace();
    if (dialect != "arith" && dialect != "math") {
      return false;
    }
    hasComputation |= !isa<arith::ConstantOp>(operation);
  }
  return hasComputation;
}

// conv 结果的唯一消费者须是静态、恒等映射、单输入单初始化的逐元素
// generic——与 fuse-linalg-epilogue 的匹配条件一致，保证改写进折叠域后
// 仍处于既有融合/向量化设施的可处理形态。
bool isCollapsibleElementwiseConsumer(linalg::GenericOp consumer,
                                      linalg::Conv2DNhwcHwcfOp convolution) {
  Value producerResult = convolution.getResult(0);
  auto producerType = dyn_cast<RankedTensorType>(producerResult.getType());
  if (!producerResult.hasOneUse() ||
      *producerResult.user_begin() != consumer.getOperation()) {
    return false;
  }
  if (consumer.getNumDpsInputs() != 1 || consumer.getNumDpsInits() != 1) {
    return false;
  }
  const unsigned rank = producerType.getRank();
  for (utils::IteratorType iteratorType : consumer.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  auto maps = consumer.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity() ||
      maps[0].getNumDims() != rank || maps[0].getNumResults() != rank) {
    return false;
  }
  auto resultType = dyn_cast<RankedTensorType>(consumer.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      resultType != producerType) {
    return false;
  }
  return isLiftableElementwiseBody(consumer);
}

// 行域折叠：[N,H,W,C] → [N·H·W, C]。全静态时是零拷贝 collapse_shape 视图；
// 任一维为动态时改用 tensor.reshape——collapse_shape 的重联组含多个动态维
// 会触发下游 ExpandStridedMetadata 的上游缺陷（ReinterpretCast 构造崩溃），
// 而 reshape 走运行时 shape 张量，与既有动态 Reshape layer 同路。
Value collapseToRows(RewriterBase& rewriter,
                     Location location,
                     Value tensorValue,
                     ArrayRef<int64_t> shape) {
  auto elementType =
    cast<RankedTensorType>(tensorValue.getType()).getElementType();
  int64_t rows = 1;
  bool dynamicRows = false;
  for (unsigned dimension : {0u, 1u, 2u}) {
    if (shape[dimension] == ShapedType::kDynamic) {
      dynamicRows = true;
      break;
    }
    rows *= shape[dimension];
  }
  if (!dynamicRows) {
    return rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get({rows, shape[3]}, elementType),
      tensorValue,
      SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
  }
  Value rowCount;
  for (unsigned dimension : {0u, 1u, 2u}) {
    Value extent = shape[dimension] == ShapedType::kDynamic
                     ? Value(rewriter.create<tensor::DimOp>(
                         location, tensorValue, dimension))
                     : Value(rewriter.create<arith::ConstantIndexOp>(
                         location, shape[dimension]));
    rowCount =
      rowCount
        ? Value(rewriter.create<arith::MulIOp>(location, rowCount, extent))
        : extent;
  }
  auto shapeType = RankedTensorType::get({2}, rewriter.getIndexType());
  auto collapsedType =
    RankedTensorType::get({ShapedType::kDynamic, shape[3]}, elementType);
  return rewriter.create<tensor::ReshapeOp>(
    location,
    collapsedType,
    tensorValue,
    rewriter.create<tensor::FromElementsOp>(
      location,
      shapeType,
      SmallVector<Value>{
        rowCount,
        rewriter.create<arith::ConstantIndexOp>(location, shape[3])}));
}

// [N·H·W, C] → [N,H,W,C]。全静态时是零拷贝 expand_shape 视图；动态空间维
// 以参照张量（与结果同型的 conv 初始化）的 tensor.dim 组装运行时 shape
// 张量，同样规避多动态维重联的上游限制。
Value expandFromRows(RewriterBase& rewriter,
                     Location location,
                     Value tensorValue,
                     ArrayRef<int64_t> shape,
                     Value referenceForDynamicDims) {
  auto elementType =
    cast<RankedTensorType>(tensorValue.getType()).getElementType();
  auto expandedType = RankedTensorType::get(shape, elementType);
  bool dynamicSpatial =
    llvm::any_of(ArrayRef<unsigned>{0u, 1u, 2u}, [&](unsigned dimension) {
      return shape[dimension] == ShapedType::kDynamic;
    });
  if (!dynamicSpatial) {
    SmallVector<OpFoldResult> outputShape;
    for (unsigned dimension : {0u, 1u, 2u, 3u}) {
      outputShape.push_back(
        OpFoldResult(rewriter.getIndexAttr(shape[dimension])));
    }
    return rewriter.create<tensor::ExpandShapeOp>(
      location,
      expandedType,
      tensorValue,
      SmallVector<ReassociationIndices>{{0, 1, 2}, {3}},
      outputShape);
  }
  SmallVector<Value> extents;
  for (unsigned dimension : {0u, 1u, 2u, 3u}) {
    extents.push_back(shape[dimension] == ShapedType::kDynamic
                        ? Value(rewriter.create<tensor::DimOp>(
                            location, referenceForDynamicDims, dimension))
                        : Value(rewriter.create<arith::ConstantIndexOp>(
                            location, shape[dimension])));
  }
  auto shapeType = RankedTensorType::get({4}, rewriter.getIndexType());
  return {rewriter.create<tensor::ReshapeOp>(
    location,
    expandedType,
    tensorValue,
    rewriter.create<tensor::FromElementsOp>(location, shapeType, extents))};
}

// im2col gather 的窗口映射：(oh,ow,kh,kw,ic) → (n=0, oh·sh+kh·dh,
// ow·sw+kw·dw, ic)，与 conv_2d_nhwc_hwcf 的 memoized 映射逐项一致
// （输入是 TosaToLinalg 已显式 pad 后的张量）。
AffineMap makeWindowMap(MLIRContext* context,
                        int64_t strideHeight,
                        int64_t strideWidth,
                        int64_t dilationHeight,
                        int64_t dilationWidth) {
  AffineExpr oh = getAffineDimExpr(0, context);
  AffineExpr ow = getAffineDimExpr(1, context);
  AffineExpr kh = getAffineDimExpr(2, context);
  AffineExpr kw = getAffineDimExpr(3, context);
  AffineExpr ic = getAffineDimExpr(4, context);
  return AffineMap::get(5,
                        0,
                        {getAffineConstantExpr(0, context),
                         oh * strideHeight + kh * dilationHeight,
                         ow * strideWidth + kw * dilationWidth,
                         ic},
                        context);
}

class StrategyNCNNPass final
  : public impl::StrategyNCNNPassBase<StrategyNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    const std::optional<ConvStrategy> strategy =
      parseStrategy(this->strategy.getValue());
    if (!strategy) {
      module.emitOpError() << "invalid strategy-ncnn strategy '"
                           << this->strategy.getValue()
                           << "': expected one of auto, gemm, conv, winograd";
      signalPassFailure();
      return;
    }
    SmallVector<linalg::Conv2DNhwcHwcfOp> candidates;
    module.walk([&](func::FuncOp function) {
      function.walk([&](linalg::Conv2DNhwcHwcfOp convolution) {
        candidates.push_back(convolution);
      });
    });
    if (candidates.empty()) {
      return;
    }
    IRRewriter rewriter(&getContext());
    for (linalg::Conv2DNhwcHwcfOp convolution : candidates) {
      rewriteConvolution(rewriter, convolution, *strategy);
    }
  }

 private:
  // ncnn convolution_x86 prefer_sgemm 启发式的编译期化：权重工作集超过
  // L2 预算，或任一通道数大于 16 时选择 im2col+GEMM。
  bool preferGemm(int64_t inputChannels,
                  int64_t outputChannels,
                  int64_t kernelHeight,
                  int64_t kernelWidth,
                  int64_t dilationHeight,
                  int64_t dilationWidth,
                  int64_t strideHeight,
                  int64_t strideWidth) const {
    const int64_t weightBytes = inputChannels * outputChannels * kernelHeight *
                                kernelWidth * dilationHeight * dilationWidth *
                                strideHeight * strideWidth * sizeof(float) * 2;
    return weightBytes > this->gemmL2Bytes.getValue() || inputChannels > 16 ||
           outputChannels > 16;
  }

  void rewriteConvolution(RewriterBase& rewriter,
                          linalg::Conv2DNhwcHwcfOp convolution,
                          ConvStrategy strategy) const {
    auto imageType =
      dyn_cast<RankedTensorType>(convolution.getInputs()[0].getType());
    auto weightType =
      dyn_cast<RankedTensorType>(convolution.getInputs()[1].getType());
    auto resultType =
      dyn_cast<RankedTensorType>(convolution.getResult(0).getType());
    if (!imageType || !weightType || !resultType || imageType.getRank() != 4 ||
        weightType.getRank() != 4 || resultType.getRank() != 4 ||
        llvm::any_of(weightType.getShape(), [](int64_t extent) {
          return ShapedType::isDynamic(extent);
        })) {
      // 权重恒为静态常量（TosaToLinalg 已物化）；动态权重实例保持原路径。
      return;
    }
    const ArrayRef<int64_t> imageShape = imageType.getShape();
    const ArrayRef<int64_t> weightShape = weightType.getShape();
    const ArrayRef<int64_t> resultShape = resultType.getShape();
    const int64_t kernelHeight = weightShape[0];
    const int64_t kernelWidth = weightShape[1];
    const int64_t inputChannels = weightShape[2];
    const int64_t outputChannels = weightShape[3];
    SmallVector<int64_t> strides;
    SmallVector<int64_t> dilations;
    llvm::append_range(strides, convolution.getStrides().getValues<int64_t>());
    llvm::append_range(dilations,
                       convolution.getDilations().getValues<int64_t>());

    // 路径判定：1×1 s1 无条件走视图折叠（对齐 ncnn 的恒 GEMM 分支）；
    // 其余 k×k 需要静态空间维（OH·OW 可知）且命中阈值或显式 gemm 策略。
    const bool unitKernelStrideOne = kernelHeight == 1 && kernelWidth == 1 &&
                                     strides[0] == 1 && strides[1] == 1;
    const bool staticSpatial = !ShapedType::isDynamic(imageShape[1]) &&
                               !ShapedType::isDynamic(imageShape[2]) &&
                               !ShapedType::isDynamic(resultShape[1]) &&
                               !ShapedType::isDynamic(resultShape[2]);
    bool useUnitView = false;
    bool useIm2col = false;
    switch (strategy) {
      case ConvStrategy::Auto:
        useUnitView = unitKernelStrideOne;
        useIm2col = !unitKernelStrideOne && staticSpatial &&
                    preferGemm(inputChannels,
                               outputChannels,
                               kernelHeight,
                               kernelWidth,
                               dilations[0],
                               dilations[1],
                               strides[0],
                               strides[1]);
        break;
      case ConvStrategy::Gemm:
        useUnitView = unitKernelStrideOne;
        useIm2col = !unitKernelStrideOne && staticSpatial;
        break;
      case ConvStrategy::Conv:
      case ConvStrategy::Winograd:
        // Winograd 仅预留开关位：F(6,3) 改变累加结构，数值预算单独验证
        // 前保持直接卷积路径。
        return;
    }
    if (!useUnitView && !useIm2col) {
      return;
    }

    Location location = convolution.getLoc();
    rewriter.setInsertionPoint(convolution);

    Value collapsedImage = collapseToRows(
      rewriter, location, convolution.getInputs()[0], imageShape);
    Value collapsedWeight = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get(
        {useUnitView ? inputChannels
                     : kernelHeight * kernelWidth * inputChannels,
         outputChannels},
        weightType.getElementType()),
      convolution.getInputs()[1],
      SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
    Value collapsedInit = collapseToRows(
      rewriter, location, convolution.getDpsInitOperand(0)->get(), resultShape);

    Value contraction = collapsedImage;
    if (useIm2col) {
      // gather 产出 [OH,OW,KH,KW,IC]，折叠为 [OH·OW, KH·KW·IC] 与权重
      // 折叠序（kh,kw,ic）对齐。
      SmallVector<int64_t> windowShape{resultShape[1],
                                       resultShape[2],
                                       kernelHeight,
                                       kernelWidth,
                                       inputChannels};
      auto windowType =
        RankedTensorType::get(windowShape, imageType.getElementType());
      AffineMap inputMap = makeWindowMap(rewriter.getContext(),
                                         strides[0],
                                         strides[1],
                                         dilations[0],
                                         dilations[1]);
      auto identityMap = AffineMap::getMultiDimIdentityMap(
        windowShape.size(), rewriter.getContext());
      Value windowBuffer = rewriter.create<tensor::EmptyOp>(
        location, windowShape, imageType.getElementType());
      auto gather = rewriter.create<linalg::GenericOp>(
        location,
        TypeRange{windowType},
        ValueRange{convolution.getInputs()[0]},
        ValueRange{windowBuffer},
        SmallVector<AffineMap>{inputMap, identityMap},
        SmallVector<utils::IteratorType>(windowShape.size(),
                                         utils::IteratorType::parallel),
        [&](OpBuilder& bodyBuilder, Location bodyLocation, ValueRange values) {
          bodyBuilder.create<linalg::YieldOp>(bodyLocation, values[0]);
        });
      contraction = rewriter.create<tensor::CollapseShapeOp>(
        location,
        RankedTensorType::get({resultShape[1] * resultShape[2],
                               kernelHeight * kernelWidth * inputChannels},
                              imageType.getElementType()),
        gather.getResult(0),
        SmallVector<ReassociationIndices>{{0, 1}, {2, 3, 4}});
    }

    auto contractionOutputType =
      cast<RankedTensorType>(collapsedInit.getType());
    auto matmul = rewriter.create<linalg::MatmulOp>(
      location,
      TypeRange{contractionOutputType},
      ValueRange{contraction, collapsedWeight},
      ValueRange{collapsedInit});
    Value contracted = matmul.getResult(0);

    // epilogue 衔接（方案 a）：唯一用户是恒等逐元素 generic 时，把
    // consumer 整体搬进折叠二维域，expand_shape 推迟到其后，下游继续看
    // 到原四维形状。fuse-linalg-epilogue 的视图守卫会跳过本 pass 产物，
    // 不会二次融合。
    Value convResult = convolution.getResult(0);
    linalg::GenericOp consumer;
    if (convResult.hasOneUse()) {
      consumer = dyn_cast<linalg::GenericOp>(*convResult.getUsers().begin());
    }
    if (consumer && isCollapsibleElementwiseConsumer(consumer, convolution)) {
      // 折叠后的激活须位于 consumer 之前：其 outs 初始化可能定义在 conv
      // 之后，插入点取 consumer 自身位置保证支配关系。
      rewriter.setInsertionPoint(consumer);
      // consumer 的 outs 若复用 conv 结果本身则直接以 matmul 结果为初始化，
      // 否则对原初始化取折叠视图（纯视图，无复制）。
      Value consumerInit = consumer.getDpsInitOperand(0)->get();
      Value collapsedConsumerInit =
        consumerInit == convResult
          ? contracted
          : collapseToRows(rewriter, location, consumerInit, resultShape);
      auto identityTwoDim =
        AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
      Value consumerBuffer = rewriter.create<tensor::EmptyOp>(
        location,
        contractionOutputType.getShape(),
        contractionOutputType.getElementType());
      auto lifted = rewriter.create<linalg::GenericOp>(
        consumer.getLoc(),
        TypeRange{consumerBuffer.getType()},
        ValueRange{contracted},
        ValueRange{collapsedConsumerInit},
        SmallVector<AffineMap>{identityTwoDim, identityTwoDim},
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel));
      IRMapping mapping;
      consumer->getRegion(0).cloneInto(&lifted->getRegion(0), mapping);
      Value expanded = expandFromRows(rewriter,
                                      location,
                                      lifted.getResult(0),
                                      resultShape,
                                      convolution.getDpsInitOperand(0)->get());
      rewriter.replaceAllUsesWith(consumer.getResult(0), expanded);
      rewriter.eraseOp(consumer);
    } else {
      Value expanded = expandFromRows(rewriter,
                                      location,
                                      contracted,
                                      resultShape,
                                      convolution.getDpsInitOperand(0)->get());
      rewriter.replaceAllUsesWith(convResult, expanded);
    }
    rewriter.eraseOp(convolution);
  }
};

}  // namespace

}  // namespace mlir::ncnn
