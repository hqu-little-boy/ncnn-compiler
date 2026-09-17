#include "ncnn-mlir/Transforms/StrategyNCNN/StrategyNCNN.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

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

void copyConvContract(Operation* source, Operation* target) {
  for (StringRef attribute : {contract::kOperationFamily,
                              contract::kImplementation,
                              contract::kKernelStatic,
                              contract::kKernelHeight,
                              contract::kKernelWidth,
                              contract::kStrideHeight,
                              contract::kStrideWidth,
                              contract::kDilationHeight,
                              contract::kDilationWidth,
                              contract::kInputChannels,
                              contract::kOutputChannels,
                              contract::kMultiplier,
                              contract::kFallback}) {
    if (Attribute value = source->getAttr(attribute)) {
      target->setAttr(attribute, value);
    }
  }
  for (StringRef attribute :
       {StringRef("ncnn.name"), StringRef("ncnn.source_layer")}) {
    if (Attribute value = source->getAttr(attribute)) {
      target->setAttr(attribute, value);
    }
  }
}

void annotateConvContract(Operation* operation,
                          StringRef implementation,
                          int64_t kernelHeight,
                          int64_t kernelWidth,
                          int64_t strideHeight,
                          int64_t strideWidth,
                          int64_t dilationHeight,
                          int64_t dilationWidth,
                          int64_t inputChannels,
                          int64_t outputChannels) {
  contract::annotateOperationFamily(operation, "conv", implementation);
  contract::annotateGeometry(operation,
                             kernelHeight,
                             kernelWidth,
                             strideHeight,
                             strideWidth,
                             dilationHeight,
                             dilationWidth,
                             inputChannels,
                             outputChannels);
}

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

// Winograd F(6,3)（P7）：ncnn conv3x3s1_winograd63 的编译期化。3×3 s1
// 卷积改写为 输入变换 → 批量 GEMM → 输出变换 三段，算术强度从每权重
// 1 次 MAC/输出提升到 36/输出。矩阵抄录自 vendored ncnn
// convolution_3x3_winograd.h（G=ktm、Bᵀ=itm、Aᵀ=otm，interpolation 点
// ±1, ±2, ±1/2, ∞ 的有理系数）。数值上变换改变累加结构（不再是逐 k 串
// 行 FMA 链），f32 相对误差 ~1e-3 量级，由逐模型数值预算对账把关
// （docs/ncnn-performance-parity-plan.md §3-P7：预算过不了的模型不启
// 用，不做全局默认）。
namespace winograd63 {

// G（kernel transform，ktm[8][3]）。
constexpr float kG[8][3] = {{1.0F, 0.0F, 0.0F},
                            {-2.0F / 9, -2.0F / 9, -2.0F / 9},
                            {-2.0F / 9, 2.0F / 9, -2.0F / 9},
                            {1.0F / 90, 1.0F / 45, 2.0F / 45},
                            {1.0F / 90, -1.0F / 45, 2.0F / 45},
                            {1.0F / 45, 1.0F / 90, 1.0F / 180},
                            {1.0F / 45, -1.0F / 90, 1.0F / 180},
                            {0.0F, 0.0F, 1.0F}};

// Bᵀ（input transform，itm[8][8]）。
constexpr float kBt[8][8] = {
  {1.0F, 0.0F, -5.25F, 0.0F, 5.25F, 0.0F, -1.0F, 0.0F},
  {0.0F, 1.0F, 1.0F, -4.25F, -4.25F, 1.0F, 1.0F, 0.0F},
  {0.0F, -1.0F, 1.0F, 4.25F, -4.25F, -1.0F, 1.0F, 0.0F},
  {0.0F, 0.5F, 0.25F, -2.5F, -1.25F, 2.0F, 1.0F, 0.0F},
  {0.0F, -0.5F, 0.25F, 2.5F, -1.25F, -2.0F, 1.0F, 0.0F},
  {0.0F, 2.0F, 4.0F, -2.5F, -5.0F, 0.5F, 1.0F, 0.0F},
  {0.0F, -2.0F, 4.0F, 2.5F, -5.0F, -0.5F, 1.0F, 0.0F},
  {0.0F, -1.0F, 0.0F, 5.25F, 0.0F, -5.25F, 0.0F, 1.0F}};

// Aᵀ（output transform，otm[6][8]）。
constexpr float kAt[6][8] = {
  {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 32.0F, 32.0F, 0.0F},
  {0.0F, 1.0F, -1.0F, 2.0F, -2.0F, 16.0F, -16.0F, 0.0F},
  {0.0F, 1.0F, 1.0F, 4.0F, 4.0F, 8.0F, 8.0F, 0.0F},
  {0.0F, 1.0F, -1.0F, 8.0F, -8.0F, 4.0F, -4.0F, 0.0F},
  {0.0F, 1.0F, 1.0F, 16.0F, 16.0F, 2.0F, 2.0F, 0.0F},
  {0.0F, 1.0F, -1.0F, 32.0F, -32.0F, 1.0F, -1.0F, 1.0F}};

constexpr int64_t kAlpha = 8;   // 变换域边长（tile 6 + 界 2）
constexpr int64_t kTile = 6;    // 输出 tile 边长
constexpr int64_t kBatch = 64;  // 变换域批数 α²

// ncnn dispatch 判据的编译期化（convolution_x86.cpp:640）：3×3 s1 d1 且
// IC>8 或 OC>8 才值得 Winograd——小通道层的变换开销吃不掉 GEMM 收益。
bool eligible(int64_t kernelHeight,
              int64_t kernelWidth,
              int64_t dilationHeight,
              int64_t dilationWidth,
              int64_t strideHeight,
              int64_t strideWidth,
              int64_t inputChannels,
              int64_t outputChannels) {
  return kernelHeight == 3 && kernelWidth == 3 && dilationHeight == 1 &&
         dilationWidth == 1 && strideHeight == 1 && strideWidth == 1 &&
         (inputChannels > 8 || outputChannels > 8);
}

// 编译期权重变换：G·w·Gᵀ 折叠 [64, OC, IC] 常量（ncnn create_pipeline
// 预变换的编译期等价物）。输入 [3,3,IC,OC]（MLIR 布局，kh-major），
// 输出批维最外与 batch_matmul 的 A 面板一致（行内 oc*IC+ic）。
DenseFPElementsAttr transformWeight(RankedTensorType weightType,
                                    DenseFPElementsAttr weights) {
  const int64_t inputChannels = weightType.getShape()[2];
  const int64_t outputChannels = weightType.getShape()[3];
  const int64_t depth = inputChannels * outputChannels;
  const llvm::fltSemantics& semantics =
    cast<FloatType>(weightType.getElementType()).getFloatSemantics();
  SmallVector<APFloat> transformed(kBatch * depth, APFloat(semantics));
  auto values = weights.getValues<APFloat>();
  // 先行段（kw 方向）：W1[kh][jp][ic][oc] = Σ_kw G[jp][kw]·w[kh][kw][ic][oc]
  SmallVector<APFloat> stage1(3 * kAlpha * depth, APFloat(semantics));
  auto at1 = [&](int64_t kh, int64_t jp, int64_t index) -> APFloat& {
    return stage1[(((kh * kAlpha) + jp) * depth) + index];
  };
  int64_t sourceIndex = 0;
  for (int64_t kh = 0; kh < 3; ++kh) {
    for (int64_t kw = 0; kw < 3; ++kw) {
      for (int64_t ic = 0; ic < inputChannels; ++ic) {
        for (int64_t oc = 0; oc < outputChannels; ++oc) {
          const APFloat& value = values[sourceIndex++];
          for (int64_t jp = 0; jp < kAlpha; ++jp) {
            const float coefficient = kG[jp][kw];
            if (coefficient != 0.0F) {
              APFloat product(value);
              product.multiply(APFloat(coefficient),
                               llvm::APFloat::rmNearestTiesToEven);
              APFloat& slot = at1(kh, jp, (ic * outputChannels) + oc);
              slot.add(product, llvm::APFloat::rmNearestTiesToEven);
            }
          }
        }
      }
    }
  }
  // 再列段（kh 方向）：V[i·8+jp][oc][ic] = Σ_kh G[i][kh]·W1[kh][jp][ic][oc]
  for (int64_t kh = 0; kh < 3; ++kh) {
    for (int64_t jp = 0; jp < kAlpha; ++jp) {
      for (int64_t ic = 0; ic < inputChannels; ++ic) {
        for (int64_t oc = 0; oc < outputChannels; ++oc) {
          const APFloat& value = at1(kh, jp, (ic * outputChannels) + oc);
          for (int64_t i = 0; i < kAlpha; ++i) {
            const float coefficient = kG[i][kh];
            if (coefficient != 0.0F) {
              APFloat product(value);
              product.multiply(APFloat(coefficient),
                               llvm::APFloat::rmNearestTiesToEven);
              APFloat& slot = transformed[(((i * kAlpha) + jp) * depth) +
                                          ((oc * inputChannels) + ic)];
              slot.add(product, llvm::APFloat::rmNearestTiesToEven);
            }
          }
        }
      }
    }
  }
  auto transformedType = RankedTensorType::get(
    {kBatch, outputChannels, inputChannels}, weightType.getElementType());
  return DenseFPElementsAttr::get(transformedType, transformed);
}

}  // namespace winograd63

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
    SmallVector<linalg::MatmulOp> int8Matmuls;
    SmallVector<linalg::BatchMatmulOp> unitBatches;
    module.walk([&](func::FuncOp function) {
      function.walk([&](linalg::Conv2DNhwcHwcfOp convolution) {
        candidates.push_back(convolution);
      });
      function.walk([&](linalg::MatmulOp matmul) {
        // int8 InnerProduct/Gemm 路径（P4）：B 常量 [K,N] 转置物化为
        // [N,K] 并改写 matmul_transpose_b，与卷积路径共用 row-dot 内核。
        if (isInt8ConstantBMatmul(matmul)) {
          int8Matmuls.push_back(matmul);
        }
      });
      // P6-C：MHA 静态路径的 batch=1 batch_matmul 语义上就是普通
      // matmul——通用下降把它展开成标量 mul+add 循环且逐 k 读写回 C，
      // clang 无法向量化（medium_rec 实测 29% 热点）。B 折叠后进入
      // TileMatmulForall + A1b 内核的既有路径。
      function.walk([&](linalg::BatchMatmulOp batch) {
        auto aType = dyn_cast<RankedTensorType>(batch.getInputs()[0].getType());
        if (aType && aType.hasStaticShape() && aType.getShape()[0] == 1 &&
            batch.hasPureTensorSemantics()) {
          unitBatches.push_back(batch);
        }
      });
    });
    IRRewriter rewriter(&getContext());
    for (linalg::BatchMatmulOp batch : unitBatches) {
      rewriteUnitBatchMatmul(rewriter, batch);
    }
    for (linalg::MatmulOp matmul : int8Matmuls) {
      rewriteInt8Matmul(rewriter, matmul);
    }
    for (linalg::Conv2DNhwcHwcfOp convolution : candidates) {
      rewriteConvolution(rewriter, convolution, *strategy);
    }
  }

  // [1,M,K]×[1,K,N] → [M,K]×[K,N] 的 linalg.matmul（前后配 collapse/
  // expand 视图，纯视图零拷贝）。B 维静态为 1 时 batch_matmul 的结果与
  // 折叠后 matmul 逐位一致（batch 维唯一）。
  void rewriteUnitBatchMatmul(RewriterBase& rewriter,
                              linalg::BatchMatmulOp batch) const {
    Location location = batch.getLoc();
    const auto aType = cast<RankedTensorType>(batch.getInputs()[0].getType());
    const auto bType = cast<RankedTensorType>(batch.getInputs()[1].getType());
    const auto cType =
      cast<RankedTensorType>(batch.getOutputs().front().getType());
    const int64_t rows = aType.getShape()[1];
    const int64_t depth = aType.getShape()[2];
    const int64_t columns = bType.getShape()[2];
    if (bType.getShape()[0] != 1 || cType.getShape()[0] != 1 ||
        cType.getShape()[1] != rows || cType.getShape()[2] != columns) {
      return;
    }
    rewriter.setInsertionPoint(batch);
    // rank-3 输入折掉 B 维：[[0,1],[2]]（维度全部成组）。
    SmallVector<ReassociationIndices> dropBatch{{0, 1}, {2}};
    Value collapsedA = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get({rows, depth}, aType.getElementType()),
      batch.getInputs()[0],
      dropBatch);
    Value collapsedB = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get({depth, columns}, bType.getElementType()),
      batch.getInputs()[1],
      dropBatch);
    Value collapsedC = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get({rows, columns}, cType.getElementType()),
      batch.getOutputs().front(),
      dropBatch);
    Value contracted =
      rewriter
        .create<linalg::MatmulOp>(location,
                                  TypeRange{collapsedC.getType()},
                                  ValueRange{collapsedA, collapsedB},
                                  ValueRange{collapsedC})
        .getResult(0);
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
      location, cType, contracted, dropBatch);
    rewriter.replaceAllUsesWith(batch.getResult(0), expanded);
    rewriter.eraseOp(batch);
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

  // 权重常量 [KH,KW,IC,OC] 直接重排为 [OC, KH·KW·IC] 的稠密常量（k 序
  // kh,kw,ic 与 im2col gather 的折叠序一致），供整数路径的
  // matmul_transpose_b 使用——B 面按 [N,K] 存放使内核沿 k 连续读取
  // （ncnn transpose_pack_B 的编译期等价物）。权重非常量（不应出现，
  // TosaToLinalg 已物化）返回空值，调用方回退。
  std::optional<Value> buildTransposedWeightConstant(
    RewriterBase& rewriter,
    Location location,
    Value weight,
    int64_t kernelHeight,
    int64_t kernelWidth,
    int64_t inputChannels,
    int64_t outputChannels) const {
    auto constant = dyn_cast<arith::ConstantOp>(weight.getDefiningOp());
    if (!constant) {
      return std::nullopt;
    }
    auto elements = dyn_cast<DenseElementsAttr>(constant.getValueAttr());
    if (!elements || !isa<::mlir::IntegerType>(elements.getElementType())) {
      return std::nullopt;
    }
    const unsigned bitWidth = elements.getElementType().getIntOrFloatBitWidth();
    const int64_t depthExtent = kernelHeight * kernelWidth * inputChannels;
    SmallVector<APInt> transposed(outputChannels * depthExtent,
                                  APInt(bitWidth, 0));
    auto values = elements.getValues<APInt>();
    int64_t sourceIndex = 0;
    for (int64_t kh = 0; kh < kernelHeight; ++kh) {
      for (int64_t kw = 0; kw < kernelWidth; ++kw) {
        for (int64_t ic = 0; ic < inputChannels; ++ic) {
          for (int64_t oc = 0; oc < outputChannels; ++oc) {
            transposed[(oc * depthExtent) +
                       ((kh * kernelWidth) * inputChannels) +
                       (kw * inputChannels) + ic] = values[sourceIndex];
            ++sourceIndex;
          }
        }
      }
    }
    auto transposedType = RankedTensorType::get({outputChannels, depthExtent},
                                                elements.getElementType());
    return rewriter
      .create<arith::ConstantOp>(
        location,
        transposedType,
        DenseIntElementsAttr::get(transposedType, transposed))
      .getResult();
  }

  // i8 linalg.matmul 且 B 为静态常量（InnerProduct/Gemm 权重）。
  static bool isInt8ConstantBMatmul(linalg::MatmulOp matmul) {
    if (!matmul.hasPureTensorSemantics() || matmul.getInputs().size() != 2 ||
        matmul.getOutputs().size() != 1) {
      return false;
    }
    auto rhsType = dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
    auto lhsType = dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
    auto accType = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!rhsType || !lhsType || !accType || !rhsType.hasStaticShape() ||
        rhsType.getRank() != 2 || !rhsType.getElementType().isInteger(8) ||
        !lhsType.getElementType().isInteger(8) ||
        !accType.getElementType().isInteger(32)) {
      return false;
    }
    auto constant =
      dyn_cast<arith::ConstantOp>(matmul.getInputs()[1].getDefiningOp());
    auto elements =
      constant ? dyn_cast<DenseElementsAttr>(constant.getValueAttr()) : nullptr;
    return elements && matmul->getParentOfType<scf::ForallOp>() == nullptr &&
           matmul->getParentOfType<scf::ForOp>() == nullptr;
  }

  // B 常量 [K,N] 直接重排为 [N,K]，matmul → matmul_transpose_b。
  void rewriteInt8Matmul(RewriterBase& rewriter,
                         linalg::MatmulOp matmul) const {
    auto rhsType = cast<RankedTensorType>(matmul.getInputs()[1].getType());
    const int64_t depth = rhsType.getShape()[0];
    const int64_t columns = rhsType.getShape()[1];
    auto weightNK = buildTransposedWeightConstant(
      rewriter, matmul.getLoc(), matmul.getInputs()[1], 1, 1, depth, columns);
    if (!weightNK) {
      return;
    }
    rewriter.setInsertionPoint(matmul);
    auto transposed = rewriter.create<linalg::MatmulTransposeBOp>(
      matmul.getLoc(),
      TypeRange{matmul.getResult(0).getType()},
      ValueRange{matmul.getInputs()[0], *weightNK},
      ValueRange{matmul.getOutputs().front()});
    rewriter.replaceAllUsesWith(matmul.getResult(0), transposed.getResult(0));
    rewriter.eraseOp(matmul);
  }

  // P7 Winograd F(6,3) 改写。IR 结构（全部 tensor 层，向量化/并行化由
  // 既有后续 pass 接管）：
  //   1. pad 图像到 tile 网格 [1, TH·6+2, TW·6+2, IC]（tile 对齐余量 +
  //      右/下各 2 的变换窗口越界；输入已含 TosaToLinalg 显式 pad）；
  //   2. 输入变换两段 generic（8-tap 全展开直线 body）：U = Bᵀ·(d·B)，
  //      iterator (th, tw, i, jp, c) 全 parallel、c 最内——归约形态每内
  //      层迭代只有 1 MAC 无法向量化（实测 vmovss 16k 条、resnet18
  //      winograd 变体 129ms vs 基线 32ms），展开后 8 次 mul+add 由
  //      clang auto-vec 命中；
  //   3. 中心 linalg.batch_matmul [64,OC,IC]×[64,IC,T]→[64,OC,T]——
  //      命中 P6-C kernelizeBatchMatmul 的 forall(b)+A1b 形态内核
  //      （复用 P3 M×N 寄存器分块微内核）；
  //   4. 输出变换两段 generic（同展开形态），[TH,6,TW,6,OC] 交错空间
  //      布局（collapse {{0,1},{2,3},{4}} 直接折出 [TH·6,TW·6,OC]）；
  //   5. 折回裁剪到 [1,OH,OW,OC]，与 conv init（逐通道 bias）逐元素加。
  // 数值：f32 全链，相对 ncnn 直接卷积 ~1e-3 量级（F(6,3) 有理插值点），
  // 数值契约由启用侧的 golden 预算承担（默认不启用，见 dispatch）。
  bool rewriteWinograd(RewriterBase& rewriter,
                       linalg::Conv2DNhwcHwcfOp convolution) const {
    using winograd63::kAlpha;
    using winograd63::kAt;
    using winograd63::kBatch;
    using winograd63::kBt;
    using winograd63::kTile;
    auto imageType =
      dyn_cast<RankedTensorType>(convolution.getInputs()[0].getType());
    auto weightType =
      dyn_cast<RankedTensorType>(convolution.getInputs()[1].getType());
    auto resultType =
      dyn_cast<RankedTensorType>(convolution.getResult(0).getType());
    if (!imageType || !weightType || !resultType || imageType.getRank() != 4 ||
        weightType.getRank() != 4 || resultType.getRank() != 4 ||
        !imageType.hasStaticShape() || !resultType.hasStaticShape() ||
        !weightType.hasStaticShape()) {
      return false;
    }
    const ArrayRef<int64_t> imageShape = imageType.getShape();
    const ArrayRef<int64_t> resultShape = resultType.getShape();
    const int64_t inputChannels = weightType.getShape()[2];
    const int64_t outputChannels = weightType.getShape()[3];
    const int64_t batches = imageShape[0];
    const int64_t outputHeight = resultShape[1];
    const int64_t outputWidth = resultShape[2];
    const auto strides = convolution.getStrides().getValues<int64_t>();
    const auto dilations = convolution.getDilations().getValues<int64_t>();
    if (batches != 1 || inputChannels < 1 || outputChannels < 1) {
      // tile 域以 N=1 折叠展开；批维实例回退既有路径。
      return false;
    }
    auto weightConstant =
      dyn_cast<arith::ConstantOp>(convolution.getInputs()[1].getDefiningOp());
    auto weightElements =
      weightConstant
        ? dyn_cast<DenseFPElementsAttr>(weightConstant.getValueAttr())
        : nullptr;
    if (!weightElements) {
      return false;
    }
    annotateConvContract(convolution,
                         "winograd",
                         3,
                         3,
                         strides[0],
                         strides[1],
                         dilations[0],
                         dilations[1],
                         inputChannels,
                         outputChannels);

    const int64_t tileRows = (outputHeight + kTile - 1) / kTile;
    const int64_t tileColumns = (outputWidth + kTile - 1) / kTile;
    const int64_t paddedHeight = (tileRows * kTile) + 2;
    const int64_t paddedWidth = (tileColumns * kTile) + 2;
    const int64_t tiles = tileRows * tileColumns;
    const int64_t inputHeight = imageShape[1];
    const int64_t inputWidth = imageShape[2];

    Location location = convolution.getLoc();
    rewriter.setInsertionPoint(convolution);
    MLIRContext* context = rewriter.getContext();
    const auto elementType = imageType.getElementType();
    auto zeroScalar = [&]() {
      return rewriter
        .create<arith::ConstantOp>(location,
                                   rewriter.getFloatAttr(elementType, 0.0))
        .getResult();
    };
    auto dim = [&](int64_t position) {
      return getAffineDimExpr(position, context);
    };
    auto zeroInit = [&](ArrayRef<int64_t> shape) {
      return rewriter
        .create<linalg::FillOp>(location,
                                ValueRange{zeroScalar()},
                                ValueRange{rewriter.create<tensor::EmptyOp>(
                                  location, shape, elementType)})
        .getResult(0);
    };

    // 1. pad 到 tile 网格。
    auto pad = tensor::PadOp::create(
      rewriter,
      location,
      RankedTensorType::get({1, paddedHeight, paddedWidth, inputChannels},
                            elementType),
      convolution.getInputs()[0],
      SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(0)),
      SmallVector<OpFoldResult>{
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(paddedHeight - inputHeight),
        rewriter.getIndexAttr(paddedWidth - inputWidth),
        rewriter.getIndexAttr(0)},
      zeroScalar());
    // 批维折掉变 [PH, PW, IC]（纯视图），供 affine 投影消费。
    Value inputRows = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get({paddedHeight, paddedWidth, inputChannels},
                            elementType),
      pad,
      SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}});

    // 2. 输入变换两段，8-tap 全展开直线 body（Σ_k input_k·coeff[k]，
    //    系数为 body 标量字面常量），iterator 全 parallel、c 最内维。
    //    发射器 transformRows：对输出行 r 逐行发射一个 generic——行维
    //    （source 的 rowDim 维）以常量 row 出现在所有 tap 投影中，从迭
    //    代空间完全移除；K 个 tap 各占一个 operand 槽（linalg generic
    //    one-map-per-operand，同一 source 重复 K 次），输出域 = 去掉
    //    rowDim 维的 outputDomain，assemble 时 insert_slice 拼回。
    //    rowTapMaps(row) 返回 K 个 4 维 map（迭代器 = outputDomain 去
    //    rowDim 维，c 最内）；输出的恒等投影由本发射器补齐。
    auto transformRows =
      [&](Value source,
          ArrayRef<int64_t> outputDomain,
          int64_t rows,
          unsigned rowDim,
          const std::function<SmallVector<AffineMap>(int64_t)>& rowTapMaps,
          const std::function<ArrayRef<float>(int64_t)>& coefficientsForRow)
      -> Value {
      Value assembled =
        rewriter.create<tensor::EmptyOp>(location, outputDomain, elementType)
          .getResult();
      SmallVector<int64_t> rowDomain(outputDomain);
      rowDomain.erase(rowDomain.begin() + static_cast<std::ptrdiff_t>(rowDim));
      const auto iteratorCount = static_cast<unsigned>(rowDomain.size());
      for (int64_t row = 0; row < rows; ++row) {
        SmallVector<AffineMap> maps = rowTapMaps(row);
        maps.push_back(
          AffineMap::getMultiDimIdentityMap(iteratorCount, context));
        auto outputType = RankedTensorType::get(rowDomain, elementType);
        auto init = zeroInit(rowDomain);
        SmallVector<Value> tapOperands(maps.size() - 1, source);
        auto generic = rewriter.create<linalg::GenericOp>(
          location,
          TypeRange{outputType},
          tapOperands,
          ValueRange{init},
          maps,
          SmallVector<utils::IteratorType>(iteratorCount,
                                           utils::IteratorType::parallel),
          [&, row](OpBuilder& bodyBuilder,
                   Location bodyLocation,
                   ValueRange bodyValues) {
            ArrayRef<float> coefficients = coefficientsForRow(row);
            Value accumulator;
            for (auto [tap, coefficient] : llvm::enumerate(coefficients)) {
              auto product = bodyBuilder.create<arith::MulFOp>(
                bodyLocation,
                bodyValues[tap],
                bodyBuilder.create<arith::ConstantOp>(
                  bodyLocation,
                  bodyBuilder.getFloatAttr(elementType, coefficient)));
              accumulator = tap == 0 ? product.getResult()
                                     : bodyBuilder
                                         .create<arith::AddFOp>(
                                           bodyLocation, accumulator, product)
                                         .getResult();
            }
            bodyBuilder.create<linalg::YieldOp>(bodyLocation, accumulator);
          });
        SmallVector<OpFoldResult> offsets(outputDomain.size(),
                                          rewriter.getIndexAttr(0));
        offsets[rowDim] = rewriter.getIndexAttr(row);
        SmallVector<OpFoldResult> sizes;
        for (std::size_t dimension = 0; dimension < outputDomain.size();
             ++dimension) {
          sizes.push_back(rewriter.getIndexAttr(
            dimension == rowDim ? 1 : outputDomain[dimension]));
        }
        SmallVector<OpFoldResult> strides(outputDomain.size(),
                                          rewriter.getIndexAttr(1));
        assembled = rewriter
                      .create<tensor::InsertSliceOp>(location,
                                                     generic->getResult(0),
                                                     assembled,
                                                     offsets,
                                                     sizes,
                                                     strides)
                      ->getResult(0);
      }
      return assembled;
    };

    // 2a. 第一段（行方向）：M1[th,tw,i,jp,c] = Σ_j B[j,jp]·d[th·6+i,
    //     tw·6+k, c]。tap k 投影 d[(th·6+i), (tw·6+k), c]；输出行 jp 的
    //     系数 = B[:,jp] = kBt[jp,:]（B = kBtᵀ，bRowMajor 转置序）。
    const auto inputDomain = SmallVector<int64_t>{
      tileRows, tileColumns, kAlpha, kAlpha, inputChannels};
    // B（bRowMajor）行主序 B[j][jp] = kBt[jp][j]；输出行 jp 的 tap 系
    // 数 = B 的第 jp 列（k 序）。
    SmallVector<float> bColumnsFlat(kAlpha * kAlpha);
    for (int64_t jp = 0; jp < kAlpha; ++jp) {
      for (int64_t k = 0; k < kAlpha; ++k) {
        bColumnsFlat[(jp * kAlpha) + k] = kBt[jp][k];
      }
    }
    Value stage1 = transformRows(
      inputRows,
      inputDomain,
      kAlpha,
      3,
      [&](int64_t) {
        SmallVector<AffineMap> maps;
        for (int64_t tap = 0; tap < kAlpha; ++tap) {
          maps.push_back(
            AffineMap::get(4,
                           0,
                           {(dim(0) * 6) + dim(2), (dim(1) * 6) + tap, dim(3)},
                           context));
        }
        return maps;
      },
      [&](int64_t row) {
        return ArrayRef<float>(bColumnsFlat.data() + (row * kAlpha), kAlpha);
      });
    // 2b. 第二段（列方向）：U[i',jp',c] = Σ_i Bt[i',i]·M1[i, jp', c]。
    //     tap k 投影 M1[th, tw, k, jp', c]；输出行 i' 的系数 = kBt[i',:]。
    SmallVector<float> btRowsFlat(kAlpha * kAlpha);
    for (int64_t row = 0; row < kAlpha; ++row) {
      for (int64_t column = 0; column < kAlpha; ++column) {
        btRowsFlat[(row * kAlpha) + column] = kBt[row][column];
      }
    }
    Value inputTransformed = transformRows(
      stage1,
      inputDomain,
      kAlpha,
      2,
      [&](int64_t) {
        SmallVector<AffineMap> maps;
        for (int64_t tap = 0; tap < kAlpha; ++tap) {
          maps.push_back(AffineMap::get(4,
                                        0,
                                        {dim(0),
                                         dim(1),
                                         getAffineConstantExpr(tap, context),
                                         dim(2),
                                         dim(3)},
                                        context));
        }
        return maps;
      },
      [&](int64_t row) {
        return ArrayRef<float>(btRowsFlat.data() + (row * kAlpha), kAlpha);
      });

    // 3. 中心 batch_matmul：U 折叠 [T,64,IC] → transpose [64,IC,T] →
    //    与常量 V [64,OC,IC] 批量收缩 → Y [64,OC,T]。
    Value uTiles = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get({tiles, kBatch, inputChannels}, elementType),
      inputTransformed,
      SmallVector<ReassociationIndices>{{0, 1}, {2, 3}, {4}});
    Value uInit = rewriter.create<tensor::EmptyOp>(
      location,
      RankedTensorType::get({kBatch, inputChannels, tiles}, elementType)
        .getShape(),
      elementType);
    Value uBatched = rewriter
                       .create<linalg::TransposeOp>(
                         location, uTiles, uInit, ArrayRef<int64_t>{1, 2, 0})
                       ->getResult(0);
    Value weightTransformed =
      rewriter
        .create<arith::ConstantOp>(
          location,
          winograd63::transformWeight(
            weightType, cast<DenseFPElementsAttr>(weightElements)))
        .getResult();
    // linalg 收缩语义：outs 是初始累加器，必须零填充（tensor.empty 未
    // 初始化会把堆内存并进结果）。
    Value yInit = zeroInit(SmallVector<int64_t>{kBatch, outputChannels, tiles});
    Value yBatched = rewriter
                       .create<linalg::BatchMatmulOp>(
                         location,
                         TypeRange{RankedTensorType::get(
                           {kBatch, outputChannels, tiles}, elementType)},
                         ValueRange{weightTransformed, uBatched},
                         ValueRange{yInit})
                       .getResult(0);
    copyConvContract(convolution, yBatched.getDefiningOp());

    // 4. 输出变换两段（8-tap 展开形态，输出域交错布局）。Y [64,OC,T]
    //    → [T,OC,64] 转置 → expand [TH,TW,OC,8,8] → 转置换轴到
    //    [TH,TW,i,j,OC]。
    Value yTiled = rewriter.create<tensor::EmptyOp>(
      location,
      RankedTensorType::get({tiles, outputChannels, kBatch}, elementType)
        .getShape(),
      elementType);
    Value ySwapped = rewriter
                       .create<linalg::TransposeOp>(
                         location, yBatched, yTiled, ArrayRef<int64_t>{2, 1, 0})
                       ->getResult(0);
    Value ySpatial = rewriter.create<tensor::ExpandShapeOp>(
      location,
      RankedTensorType::get(
        {tileRows, tileColumns, outputChannels, kAlpha, kAlpha}, elementType),
      ySwapped,
      SmallVector<ReassociationIndices>{{0, 1}, {2}, {3, 4}});
    Value yChannelInnerInit = rewriter.create<tensor::EmptyOp>(
      location,
      RankedTensorType::get(
        {tileRows, tileColumns, kAlpha, kAlpha, outputChannels}, elementType)
        .getShape(),
      elementType);
    Value yTiles =
      rewriter
        .create<linalg::TransposeOp>(location,
                                     ySpatial,
                                     yChannelInnerInit,
                                     ArrayRef<int64_t>{0, 1, 3, 4, 2})
        ->getResult(0);

    // 4a. 列段：P[th,tw,i,c,oc] = Σ_jp Y[i,jp,oc]·At[c,jp]（kAt 行主
    //     序 [6,8]）。输出域 [TH,6,TW,6,OC] 交错（含 tw 维在 c 前）。
    //     注意本段的输出域 = 最终交错布局（列段直接产 [TH,c,TW? …]：
    //     行段再换轴代价更高，改为列段产 [TH,TW,i,c,oc]、行段产交错）。
    const auto outputMidDomain = SmallVector<int64_t>{
      tileRows, tileColumns, kAlpha, kTile, outputChannels};
    // 空间交错布局 [TH, r, TW, c, OC]：collapse {{0,1},{2,3},{4}} 直接
    // 折出 [TH·6, TW·6, OC]（无中间 transpose）。
    const auto outputDomain =
      SmallVector<int64_t>{tileRows, kTile, tileColumns, kTile, outputChannels};
    SmallVector<float> atRowMajor(kTile * kAlpha);
    for (int64_t r = 0; r < kTile; ++r) {
      for (int64_t c = 0; c < kAlpha; ++c) {
        atRowMajor[(r * kAlpha) + c] = kAt[r][c];
      }
    }
    Value outputStage1 = transformRows(
      yTiles,
      outputMidDomain,
      kTile,
      3,
      [&](int64_t) {
        SmallVector<AffineMap> maps;
        for (int64_t tap = 0; tap < kAlpha; ++tap) {
          maps.push_back(AffineMap::get(4,
                                        0,
                                        {dim(0),
                                         dim(1),
                                         dim(2),
                                         getAffineConstantExpr(tap, context),
                                         dim(3)},
                                        context));
        }
        return maps;
      },
      [&](int64_t row) {
        return ArrayRef<float>(atRowMajor.data() + (row * kAlpha), kAlpha);
      });
    // 4b. 行段：O[th,r,tw,c,oc] = Σ_i P[th,tw,i,c,oc]·At[r,i]（kAt
    //     行主序）。输出域 [TH,6,TW,6,OC]，rowDim=1（r 维）从迭代空
    // 间移除，迭代器 (th, tw, c, oc)；P 投影 [th, tw, tap, c, oc]（5
    // 结果，P 域 [TH,TW,8,6,OC]）。
    Value outputTransformed = transformRows(
      outputStage1,
      outputDomain,
      kTile,
      1,
      [&](int64_t) {
        SmallVector<AffineMap> maps;
        for (int64_t tap = 0; tap < kAlpha; ++tap) {
          maps.push_back(AffineMap::get(4,
                                        0,
                                        {dim(0),
                                         dim(1),
                                         getAffineConstantExpr(tap, context),
                                         dim(2),
                                         dim(3)},
                                        context));
        }
        return maps;
      },
      [&](int64_t row) {
        return ArrayRef<float>(atRowMajor.data() + (row * kAlpha), kAlpha);
      });

    // 5. 折回 [1, TH·6, TW·6, OC] 后裁剪到 [1, OH, OW, OC]。
    Value outputRows = rewriter.create<tensor::CollapseShapeOp>(
      location,
      RankedTensorType::get(
        {(tileRows * kTile), (tileColumns * kTile), outputChannels},
        elementType),
      outputTransformed,
      SmallVector<ReassociationIndices>{{0, 1}, {2, 3}, {4}});
    Value outputExpanded = rewriter.create<tensor::ExpandShapeOp>(
      location,
      RankedTensorType::get(
        {1, (tileRows * kTile), (tileColumns * kTile), outputChannels},
        elementType),
      outputRows,
      SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}});
    auto trimmed = rewriter.create<tensor::ExtractSliceOp>(
      location,
      RankedTensorType::get({1, outputHeight, outputWidth, outputChannels},
                            elementType),
      outputExpanded,
      SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(0)),
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(outputHeight),
                                rewriter.getIndexAttr(outputWidth),
                                rewriter.getIndexAttr(outputChannels)},
      SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1)),
      std::nullopt);
    // bias 补加：conv 的 init operand 含逐通道 bias（NCNNToTosa 以广播
    // 填充进入 conv 的 outs）。中心 GEMM 域是变换域不含空间维，在空间
    // 折叠恢复后与 init 逐元素相加（等价于直接卷积的 acc+bias 语义）。
    Value convInit = convolution.getDpsInitOperand(0)->get();
    Value biased =
      rewriter
        .create<linalg::GenericOp>(
          location,
          TypeRange{trimmed->getResult(0).getType()},
          ValueRange{trimmed->getResult(0), convInit},
          ValueRange{rewriter.create<tensor::EmptyOp>(
            location,
            SmallVector<int64_t>{1, outputHeight, outputWidth, outputChannels},
            elementType)},
          SmallVector<AffineMap>{AffineMap::getMultiDimIdentityMap(4, context),
                                 AffineMap::getMultiDimIdentityMap(4, context),
                                 AffineMap::getMultiDimIdentityMap(4, context)},
          SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
          [&](OpBuilder& bodyBuilder,
              Location bodyLocation,
              ValueRange bodyValues) {
            auto sum = bodyBuilder.create<arith::AddFOp>(
              bodyLocation, bodyValues[0], bodyValues[1]);
            bodyBuilder.create<linalg::YieldOp>(bodyLocation, sum.getResult());
          })
        ->getResult(0);
    // 类型一致（conv 输出即裁剪目标形状），CastOp 在 canonicalize 折叠。
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
      convolution, resultType, biased);
    return true;
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
      contract::annotateOperationFamily(convolution, "conv", "fallback");
      contract::annotateFallback(convolution, "dynamic_or_invalid_geometry");
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
    annotateConvContract(convolution,
                         "unknown",
                         kernelHeight,
                         kernelWidth,
                         strides[0],
                         strides[1],
                         dilations[0],
                         dilations[1],
                         inputChannels,
                         outputChannels);

    // P7 Winograd：显式 winograd 策略对判据内实例优先改写（opt-in，
    // 数值预算逐模型对账后由启用侧决定）；判据 = ncnn convolution_x86
    // 的 3×3 s1 d1 且 IC>8 || OC>8。判据外实例回退常规 dispatch。
    if (strategy == ConvStrategy::Winograd &&
        !imageType.getElementType().isInteger(8) &&
        winograd63::eligible(kernelHeight,
                             kernelWidth,
                             dilations[0],
                             dilations[1],
                             strides[0],
                             strides[1],
                             inputChannels,
                             outputChannels) &&
        !ShapedType::isDynamic(imageShape[0]) && imageShape[0] == 1 &&
        !ShapedType::isDynamic(imageShape[1]) &&
        !ShapedType::isDynamic(imageShape[2])) {
      if (rewriteWinograd(rewriter, convolution)) {
        return;
      }
    }

    // 路径判定：1×1 s1 无条件走视图折叠（对齐 ncnn 的恒 GEMM 分支）；
    // 其余 k×k 需要静态空间维且命中阈值或显式 gemm 策略。整数（int8
    // 量化）卷积无直接卷积内核可用，任何 k×k 静态形状一律 im2col+GEMM
    // （对齐 ncnn int8 主路径的算子形态）。
    const bool integerElements = imageType.getElementType().isInteger(8);
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
                    (integerElements || preferGemm(inputChannels,
                                                   outputChannels,
                                                   kernelHeight,
                                                   kernelWidth,
                                                   dilations[0],
                                                   dilations[1],
                                                   strides[0],
                                                   strides[1]));
        break;
      case ConvStrategy::Gemm:
        useUnitView = unitKernelStrideOne;
        useIm2col = !unitKernelStrideOne && staticSpatial;
        break;
      case ConvStrategy::Conv:
        // 直接卷积：保持 linalg.conv_2d_nhwc_hwcf 原路径。
        annotateConvContract(convolution,
                             "direct",
                             kernelHeight,
                             kernelWidth,
                             strides[0],
                             strides[1],
                             dilations[0],
                             dilations[1],
                             inputChannels,
                             outputChannels);
        return;
      case ConvStrategy::Winograd:
        // 判据外实例（非 3×3 s1 d1、批维非 1、动态空间维、int8）在
        // rewriteWinograd 守卫中未被改写时落回常规 dispatch：显式
        // 策略不比 auto 更少优化。
        useUnitView = unitKernelStrideOne;
        useIm2col = !unitKernelStrideOne && staticSpatial &&
                    (integerElements || preferGemm(inputChannels,
                                                   outputChannels,
                                                   kernelHeight,
                                                   kernelWidth,
                                                   dilations[0],
                                                   dilations[1],
                                                   strides[0],
                                                   strides[1]));
        break;
    }
    if (!useUnitView && !useIm2col) {
      if (staticSpatial) {
        annotateConvContract(convolution,
                             "direct",
                             kernelHeight,
                             kernelWidth,
                             strides[0],
                             strides[1],
                             dilations[0],
                             dilations[1],
                             inputChannels,
                             outputChannels);
      } else {
        contract::annotateOperationFamily(convolution, "conv", "fallback");
        contract::annotateFallback(convolution, "dynamic_spatial");
      }
      return;
    }
    annotateConvContract(convolution,
                         "gemm",
                         kernelHeight,
                         kernelWidth,
                         strides[0],
                         strides[1],
                         dilations[0],
                         dilations[1],
                         inputChannels,
                         outputChannels);

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
    // 整数路径：权重常量直接重排为 [N,K]（B 面转置物化），收缩走
    // matmul_transpose_b——A1b 的 i8 row-dot 内核依赖 B 沿 k 连续。
    // 浮点路径维持 [K,N] 折叠 + linalg.matmul（f32 内核沿 n 向量化）。
    Value contracted;
    if (integerElements) {
      auto weightNK = buildTransposedWeightConstant(rewriter,
                                                    location,
                                                    convolution.getInputs()[1],
                                                    kernelHeight,
                                                    kernelWidth,
                                                    inputChannels,
                                                    outputChannels);
      if (weightNK) {
        contracted = rewriter
                       .create<linalg::MatmulTransposeBOp>(
                         location,
                         TypeRange{contractionOutputType},
                         ValueRange{contraction, *weightNK},
                         ValueRange{collapsedInit})
                       .getResult(0);
      }
    }
    if (!contracted) {
      contracted =
        rewriter
          .create<linalg::MatmulOp>(location,
                                    TypeRange{contractionOutputType},
                                    ValueRange{contraction, collapsedWeight},
                                    ValueRange{collapsedInit})
          .getResult(0);
    }
    copyConvContract(convolution, contracted.getDefiningOp());

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
