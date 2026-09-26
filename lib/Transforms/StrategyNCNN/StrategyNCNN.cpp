#include "ncnn-mlir/Transforms/StrategyNCNN/StrategyNCNN.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
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
#include "ncnn-mlir/Support/CheckedMath.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/Winograd63.hpp"
#include "ncnn-mlir/Transforms/Winograd63NCNN/Winograd63NCNN.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_STRATEGYNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// Keep strategy rewrites bounded: all of them materialize tensor views or
// temporary buffers whose static element counts must be both representable and
// reasonably sized.  A failed proof leaves the original convolution in place.
constexpr int64_t kMaxStrategyElements = 1LL << 28;
// Keep im2col bounded separately: a window buffer may exceed the per-tensor
// strategy limit even when the GEMM path is the established faster lowering.
constexpr int64_t kMaxIm2colWindowElements = 1LL << 29;

// Bounds proofs all run through one implementation (see CheckedMath.hpp).
// The result type is FailureOr; inside `mlir::ncnn` the `failed`/`succeeded`
// helpers resolve unqualified from the enclosing `mlir` namespace.
using ncnn_mlir::checkedAdd;
using ncnn_mlir::checkedMul;
using ncnn_mlir::checkedProduct;

StringRef knownLayout(Operation* operation) {
  for (StringRef attribute :
       {contract::kOutputLayout, contract::kLayout, contract::kInputLayout}) {
    if (auto value = operation->getAttrOfType<StringAttr>(attribute)) {
      return value.getValue();
    }
  }
  return {};
}

std::optional<int64_t> knownPackFactor(Operation* operation) {
  for (StringRef attribute :
       {contract::kPackFactor, contract::kLayoutPackFactor}) {
    if (auto value = operation->getAttrOfType<IntegerAttr>(attribute)) {
      return value.getInt();
    }
  }
  return std::nullopt;
}

bool layoutMetadataCompatible(Operation* producer, Operation* consumer) {
  StringRef producerLayout = knownLayout(producer);
  StringRef consumerLayout = knownLayout(consumer);
  if (!producerLayout.empty() && !consumerLayout.empty() &&
      producerLayout != consumerLayout) {
    return false;
  }
  auto producerFactor = knownPackFactor(producer);
  auto consumerFactor = knownPackFactor(consumer);
  return !producerFactor || !consumerFactor ||
         *producerFactor == *consumerFactor;
}

// 算子形态策略（A1）：卷积的窗口索引映射含加法
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
  return llvm::StringSwitch<std::optional<ConvStrategy>>(strategy)
    .Case("auto", ConvStrategy::Auto)
    .Case("gemm", ConvStrategy::Gemm)
    .Case("conv", ConvStrategy::Conv)
    .Case("winograd", ConvStrategy::Winograd)
    .Default(std::nullopt);
}


bool isLiftableElementwiseBody(linalg::GenericOp consumer) {
  Region& region = consumer->getRegion(0);
  if (!region.hasOneBlock()) {
    return false;
  }
  Block& block = region.front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      block.getNumArguments() !=
        consumer.getNumDpsInputs() + consumer.getNumDpsInits()) {
    return false;
  }
  bool hasInputUse = false;
  for (unsigned index = 0; index < consumer.getNumDpsInputs(); ++index) {
    hasInputUse |= !block.getArgument(index).use_empty();
  }
  if (!hasInputUse) {
    return false;
  }
  for (unsigned index = consumer.getNumDpsInputs();
       index < block.getNumArguments();
       ++index) {
    if (!block.getArgument(index).use_empty()) {
      return false;
    }
  }
  bool hasComputation = false;
  unsigned bodyOperations = 0;
  for (Operation& operation : block.without_terminator()) {
    if (++bodyOperations > 8) {
      return false;
    }
    if (!isa<arith::MaximumFOp,
             arith::MinimumFOp,
             arith::SelectOp,
             arith::CmpFOp,
             arith::MulFOp,
             arith::AddFOp,
             arith::SubFOp,
             arith::DivFOp,
             arith::NegFOp,
             arith::ConstantOp,
             math::ExpOp,
             math::TanhOp,
             math::ErfOp>(operation)) {
      return false;
    }
    hasComputation |= !isa<arith::ConstantOp>(operation);
  }
  return hasComputation;
}

bool isIdentityMapForRank(AffineMap map, int64_t rank) {
  return map.isIdentity() && std::cmp_equal(map.getNumDims(), rank) &&
         std::cmp_equal(map.getNumResults(), rank);
}

// Do not lift a residual that is itself another computed convolution branch.
// Such branches can be rewritten later in this pass; retaining the direct
// producer/consumer form lets the epilogue pass handle them without creating a
// collapse view whose operand is subsequently replaced by a 2-D contraction.
bool hasUnsafeTensorView(Value value) {
  Operation* definition = value.getDefiningOp();
  return definition && isa<tensor::CollapseShapeOp,
                           tensor::ExpandShapeOp,
                           tensor::ExtractSliceOp,
                           tensor::InsertSliceOp,
                           tensor::CastOp,
                           tensor::ReshapeOp>(definition);
}

bool hasComputedTensorProducer(Value value) {
  Operation* definition = value.getDefiningOp();
  for (unsigned depth = 0; definition && depth < 8; ++depth) {
    StringRef name = definition->getName().getStringRef();
    if (name == "linalg.conv_2d_nhwc_hwcf" || name == "linalg.matmul" ||
        name == "linalg.matmul_transpose_b" || name == "linalg.batch_matmul") {
      return true;
    }
    if (isa<tensor::CollapseShapeOp,
            tensor::ExpandShapeOp,
            tensor::ExtractSliceOp,
            tensor::CastOp,
            tensor::ReshapeOp>(definition) &&
        definition->getNumOperands() == 1) {
      definition = definition->getOperand(0).getDefiningOp();
      continue;
    }
    return false;
  }
  return false;
}

bool hasOnlyDpsInputUses(Value result, linalg::GenericOp consumer) {
  for (OpOperand& use : result.getUses()) {
    if (use.getOwner() != consumer.getOperation()) {
      return false;
    }
    bool isDpsInput = false;
    for (OpOperand* input : consumer.getDpsInputOperands()) {
      if (input == &use) {
        isDpsInput = true;
        break;
      }
    }
    if (!isDpsInput) {
      return false;
    }
  }
  return true;
}

// conv 结果的全部使用须落在同一个静态、并行、可保序搬到二维域的逐元素
// generic DPS 输入中；允许该输入值在 consumer 内重复出现。除同形状 residual
// 外，只接受最后 channel 的 rank-1 广播；其它 map 由前置 FuseLinalgEpilogue
// 尝试融合，拒绝后保留独立 generic。
bool isCollapsibleElementwiseConsumer(linalg::GenericOp consumer,
                                      linalg::Conv2DNhwcHwcfOp convolution) {
  Value producerResult = convolution.getResult(0);
  auto producerType = dyn_cast<RankedTensorType>(producerResult.getType());
  if (!producerType || !hasOnlyDpsInputUses(producerResult, consumer) ||
      consumer.getNumDpsInputs() < 1 || consumer.getNumDpsInits() != 1 ||
      consumer.getDpsInputOperand(0)->get() != producerResult) {
    return false;
  }
  Value consumerInit = consumer.getDpsInitOperand(0)->get();
  if (!layoutMetadataCompatible(convolution, consumer) ||
      hasUnsafeTensorView(consumerInit)) {
    return false;
  }
  for (Value input : consumer.getDpsInputs()) {
    if (input == consumerInit) {
      return false;
    }
  }
  const unsigned rank = producerType.getRank();
  // This rewrite is defined for the original NHWC convolution result only.
  // A previous Strategy rewrite may leave a 2-D matmul result feeding a
  // generated generic; never try to collapse that already-collapsed domain a
  // second time.
  if (rank != 4 || consumer->hasAttr(contract::kStrategyLiftedEpilogue)) {
    return false;
  }
  for (utils::IteratorType iteratorType : consumer.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  auto maps = consumer.getIndexingMapsArray();
  if (!std::cmp_equal(maps.size(), consumer.getNumDpsInputs() + 1) ||
      !isIdentityMapForRank(maps.front(), rank) ||
      !isIdentityMapForRank(maps.back(), rank)) {
    return false;
  }
  auto resultType = dyn_cast<RankedTensorType>(consumer.getResult(0).getType());
  auto initType =
    dyn_cast<RankedTensorType>(consumer.getDpsInitOperand(0)->get().getType());
  if (!resultType || !initType || !resultType.hasStaticShape() ||
      resultType != producerType || initType != producerType ||
      !producerType.hasStaticShape() ||
      !producerType.getElementType().isF32() ||
      producerType.getShape()[3] <= 0) {
    return false;
  }
  for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
    auto operandType =
      dyn_cast<RankedTensorType>(consumer.getDpsInputs()[index].getType());
    if (!operandType || !operandType.hasStaticShape() ||
        !operandType.getElementType().isF32() ||
        hasUnsafeTensorView(consumer.getDpsInputs()[index])) {
      return false;
    }
    if (operandType.getRank() == static_cast<int64_t>(rank)) {
      if (operandType != producerType ||
          !isIdentityMapForRank(maps[index], rank) ||
          (consumer.getDpsInputs()[index] != producerResult &&
           hasComputedTensorProducer(consumer.getDpsInputs()[index]))) {
        return false;
      }
      continue;
    }
    if (operandType.getRank() != 1 || maps[index].getNumResults() != 1 ||
        !std::cmp_equal(maps[index].getNumDims(), rank)) {
      return false;
    }
    auto channel = dyn_cast<AffineDimExpr>(maps[index].getResult(0));
    if (!channel || channel.getPosition() != rank - 1 ||
        operandType.getShape()[0] != producerType.getShape()[rank - 1]) {
      return false;
    }
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
  auto tensorType = dyn_cast<RankedTensorType>(tensorValue.getType());
  if (!tensorType || tensorType.getRank() != 4 || shape.size() != 4 ||
      ShapedType::isDynamic(shape[3]) ||
      ShapedType::isDynamic(tensorType.getShape()[3]) ||
      tensorType.getShape()[3] != shape[3]) {
    return {};
  }
  auto elementType = tensorType.getElementType();
  // Avoid forming collapse(expand(x)) around a view already produced by an
  // earlier Strategy rewrite. Canonicalization may replace the expand operand
  // with x while retaining the collapse shell, yielding an invalid rank-2 to
  // rank-2 collapse with rank-4 reassociation.
  if (auto expand = tensorValue.getDefiningOp<tensor::ExpandShapeOp>()) {
    auto sourceType = dyn_cast<RankedTensorType>(expand.getSrc().getType());
    if (sourceType && sourceType.getRank() == 2 &&
        tensorType.hasStaticShape() && sourceType.hasStaticShape()) {
      auto foldedRows = checkedProduct(
        ArrayRef<int64_t>{shape[0], shape[1], shape[2]}, kMaxStrategyElements);
      if (succeeded(foldedRows) && sourceType.getShape()[0] == *foldedRows &&
          sourceType.getShape()[1] == shape[3]) {
        return expand.getSrc();
      }
    }
  }
  int64_t rows = 1;
  bool dynamicRows = false;
  for (unsigned dimension : {0u, 1u, 2u}) {
    if (shape[dimension] == ShapedType::kDynamic) {
      dynamicRows = true;
      break;
    }
    auto accumulated = checkedMul(rows, shape[dimension]);
    if (failed(accumulated)) {
      return {};
    }
    rows = *accumulated;
  }
  if (!dynamicRows && rows > kMaxStrategyElements) {
    return {};
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
  auto tensorType = dyn_cast<RankedTensorType>(tensorValue.getType());
  auto referenceType =
    dyn_cast<RankedTensorType>(referenceForDynamicDims.getType());
  if (!tensorType || !referenceType || tensorType.getRank() != 2 ||
      referenceType.getRank() != 4 || shape.size() != 4 ||
      ShapedType::isDynamic(shape[3]) ||
      ShapedType::isDynamic(tensorType.getShape()[1]) ||
      ShapedType::isDynamic(referenceType.getShape()[3]) ||
      tensorType.getShape()[1] != shape[3] ||
      referenceType.getShape()[3] != shape[3]) {
    return {};
  }
  auto elementType = tensorType.getElementType();
  auto expandedType = RankedTensorType::get(shape, elementType);
  bool dynamicSpatial =
    llvm::any_of(ArrayRef<unsigned>{0u, 1u, 2u}, [&](unsigned dimension) {
      return shape[dimension] == ShapedType::kDynamic;
    });
  if (!dynamicSpatial) {
    auto foldedRows = checkedProduct(
      ArrayRef<int64_t>{shape[0], shape[1], shape[2]}, kMaxStrategyElements);
    if (failed(foldedRows) || tensorType.getShape()[0] != *foldedRows) {
      return {};
    }
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
        auto bType = dyn_cast<RankedTensorType>(batch.getInputs()[1].getType());
        auto cType =
          dyn_cast<RankedTensorType>(batch.getOutputs().front().getType());
        if (aType && bType && cType && aType.hasStaticShape() &&
            bType.hasStaticShape() && cType.hasStaticShape() &&
            aType.getShape()[0] == 1 && bType.getShape()[0] == 1 &&
            cType.getShape()[0] == 1 &&
            cType.getShape()[1] == aType.getShape()[1] &&
            aType.getShape()[2] == bType.getShape()[1] &&
            cType.getShape()[2] == bType.getShape()[2] &&
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
    if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
        !cType.hasStaticShape() ||
        failed(checkedProduct(aType.getShape(), kMaxStrategyElements)) ||
        failed(checkedProduct(bType.getShape(), kMaxStrategyElements)) ||
        failed(checkedProduct(cType.getShape(), kMaxStrategyElements))) {
      return;
    }
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
    // Pure overflow proof here: this predicate only needs "representable", not
    // "small enough to rewrite", so the budget is the type's ceiling.
    auto weightElements = checkedProduct(ArrayRef<int64_t>{inputChannels,
                                                           outputChannels,
                                                           kernelHeight,
                                                           kernelWidth,
                                                           dilationHeight,
                                                           dilationWidth,
                                                           strideHeight,
                                                           strideWidth},
                                         std::numeric_limits<int64_t>::max());
    if (failed(weightElements)) {
      return true;
    }
    auto weightBytes = checkedMul(*weightElements, sizeof(float) * 2);
    if (failed(weightBytes)) {
      return true;
    }
    return *weightBytes > this->gemmL2Bytes.getValue() || inputChannels > 16 ||
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
    auto depthExtentProof = checkedProduct(
      ArrayRef<int64_t>{kernelHeight, kernelWidth, inputChannels},
      kMaxStrategyElements);
    if (failed(depthExtentProof)) {
      return std::nullopt;
    }
    const int64_t depthExtent = *depthExtentProof;
    auto elementCountProof = checkedMul(outputChannels, depthExtent);
    if (failed(elementCountProof) ||
        *elementCountProof > kMaxStrategyElements ||
        !std::cmp_equal(elements.getNumElements(), *elementCountProof)) {
      return std::nullopt;
    }
    const int64_t elementCount = *elementCountProof;
    SmallVector<APInt> transposed(static_cast<size_t>(elementCount),
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
    rewriter.setInsertionPoint(matmul);
    auto weightNK = buildTransposedWeightConstant(
      rewriter, matmul.getLoc(), matmul.getInputs()[1], 1, 1, depth, columns);
    if (!weightNK) {
      return;
    }
    auto transposed = rewriter.create<linalg::MatmulTransposeBOp>(
      matmul.getLoc(),
      TypeRange{matmul.getResult(0).getType()},
      ValueRange{matmul.getInputs()[0], *weightNK},
      ValueRange{matmul.getOutputs().front()});
    rewriter.replaceAllUsesWith(matmul.getResult(0), transposed.getResult(0));
    rewriter.eraseOp(matmul);
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
    auto initType = dyn_cast<RankedTensorType>(
      convolution.getDpsInitOperand(0)->get().getType());
    if (!imageType || !weightType || !resultType || !initType ||
        imageType.getRank() != 4 || weightType.getRank() != 4 ||
        resultType.getRank() != 4 || initType.getRank() != 4 ||
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
    if (strides.size() != 2 || dilations.size() != 2 ||
        llvm::any_of(strides, [](int64_t value) { return value <= 0; }) ||
        llvm::any_of(dilations, [](int64_t value) { return value <= 0; })) {
      contract::annotateOperationFamily(convolution, "conv", "fallback");
      contract::annotateFallback(convolution, "invalid_geometry");
      return;
    }
    contract::annotateConvContract(convolution,
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
      if (winograd63::rewriteWinograd(
            rewriter, convolution, kMaxStrategyElements)) {
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
    const ArrayRef<int64_t> initShape = initType.getShape();
    const bool staticChannels =
      !ShapedType::isDynamic(imageShape[3]) &&
      !ShapedType::isDynamic(resultShape[3]) &&
      !ShapedType::isDynamic(initShape[3]) && imageShape[3] == inputChannels &&
      resultShape[3] == outputChannels && initShape[3] == outputChannels;
    const bool staticBatchOne =
      !ShapedType::isDynamic(imageShape[0]) && imageShape[0] == 1 &&
      !ShapedType::isDynamic(resultShape[0]) && resultShape[0] == 1 &&
      !ShapedType::isDynamic(initShape[0]) && initShape[0] == 1;
    // Every rewrite below materializes a dense view or workspace sized by these
    // shapes, so each must multiply out within the per-tensor budget.
    const bool unitViewShapesSafe =
      staticChannels && imageType.hasStaticShape() &&
      resultType.hasStaticShape() && initType.hasStaticShape() &&
      imageShape[0] == resultShape[0] && imageShape[0] == initShape[0] &&
      imageShape[1] == resultShape[1] && imageShape[1] == initShape[1] &&
      imageShape[2] == resultShape[2] && imageShape[2] == initShape[2] &&
      succeeded(checkedProduct(imageShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(weightShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(resultShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(initShape, kMaxStrategyElements));
    const bool im2colShapesSafe =
      staticChannels && staticSpatial && staticBatchOne &&
      imageType.hasStaticShape() && resultType.hasStaticShape() &&
      initType.hasStaticShape() && initType == resultType &&
      succeeded(checkedProduct(imageShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(weightShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(resultShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(initShape, kMaxStrategyElements)) &&
      succeeded(checkedProduct(ArrayRef<int64_t>{resultShape[1],
                                                 resultShape[2],
                                                 kernelHeight,
                                                 kernelWidth,
                                                 inputChannels},
                               kMaxIm2colWindowElements));
    bool useUnitView = false;
    bool useIm2col = false;
    switch (strategy) {
      case ConvStrategy::Auto:
        useUnitView = unitKernelStrideOne && unitViewShapesSafe;
        useIm2col = !unitKernelStrideOne && im2colShapesSafe &&
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
        useUnitView = unitKernelStrideOne && unitViewShapesSafe;
        useIm2col = !unitKernelStrideOne && im2colShapesSafe;
        break;
      case ConvStrategy::Conv:
        // 直接卷积：保持 linalg.conv_2d_nhwc_hwcf 原路径。
        contract::annotateConvContract(convolution,
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
        useUnitView = unitKernelStrideOne && unitViewShapesSafe;
        useIm2col = !unitKernelStrideOne && im2colShapesSafe &&
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
        contract::annotateConvContract(convolution,
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
    contract::annotateConvContract(convolution,
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
    contract::copyConvContract(convolution, contracted.getDefiningOp());

    // epilogue 衔接（方案 a）：唯一用户是恒等逐元素 generic 时，把
    // consumer 整体搬进折叠二维域，expand_shape 推迟到其后，下游继续看
    // 到原四维形状。fuse-linalg-epilogue 的视图守卫会跳过本 pass 产物，
    // 不会二次融合。
    Value convResult = convolution.getResult(0);
    linalg::GenericOp consumer;
    Operation* onlyUser = nullptr;
    bool hasMultipleUsers = false;
    for (OpOperand& use : convResult.getUses()) {
      if (!onlyUser) {
        onlyUser = use.getOwner();
      } else if (onlyUser != use.getOwner()) {
        hasMultipleUsers = true;
        break;
      }
    }
    if (!hasMultipleUsers) {
      consumer = dyn_cast_or_null<linalg::GenericOp>(onlyUser);
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
      SmallVector<Value> liftedInputs{contracted};
      SmallVector<AffineMap> liftedMaps{identityTwoDim};
      for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
        Value input = consumer.getDpsInputs()[index];
        if (input == convResult) {
          liftedInputs.push_back(contracted);
          liftedMaps.push_back(identityTwoDim);
          continue;
        }
        auto inputType = cast<RankedTensorType>(input.getType());
        if (inputType.getRank() == resultType.getRank()) {
          liftedInputs.push_back(
            collapseToRows(rewriter, location, input, resultShape));
          liftedMaps.push_back(identityTwoDim);
          continue;
        }
        // A rank-1 channel/column operand is already in the collapsed
        // coordinate space. Its original map was proven to select the last
        // output dimension by isCollapsibleElementwiseConsumer.
        liftedInputs.push_back(input);
        liftedMaps.push_back(
          AffineMap::get(2,
                         0,
                         {getAffineDimExpr(1, rewriter.getContext())},
                         rewriter.getContext()));
      }
      liftedMaps.push_back(identityTwoDim);
      auto lifted = rewriter.create<linalg::GenericOp>(
        consumer.getLoc(),
        TypeRange{collapsedConsumerInit.getType()},
        liftedInputs,
        ValueRange{collapsedConsumerInit},
        liftedMaps,
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel));
      lifted->setAttr(contract::kStrategyLiftedEpilogue,
                      rewriter.getUnitAttr());
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
