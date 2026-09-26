// Attention / 归一化族降低：softmax、layernorm、embed、multi-head attention。
//
// 职责
//   把注意力与归一化算子降低为 tosa 的逐元素/归约组合，并按需插入
//   layout 转换（restoreNCNNLayout / convertNCNNLayout）。
//
// 不变量
//   * attention 分段记录（appendAttentionSegmentRecords）只写 ledger，
//     不影响数值；
//   * 动态序列长走 lowerDynamic，绝不假设静态；
//   * head_dim 必须整除 embed，否则拒绝并回退。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后；
//   * 之后由 TosaToLinalg 与 StrategyNCNN 的 batch_matmul 路径接管。
//
// 明确不做
//   * 不做 KV-cache / 增量推理；
//   * 不融合 attention 与后续算子（FuseLinalgEpilogue 的职责）。

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
using tosa_lowering::createI8Zero;
using tosa_lowering::createSplat;
using tosa_lowering::getBroadcastScalarType;
using tosa_lowering::reshapeValue;
using tosa_lowering::restoreNCNNLayout;
using tosa_lowering::transposeOrFoldConstant;

class ConvertSoftmax final : public OpConversionPattern<SoftmaxOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    SoftmaxOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Value input = adaptor.getInput();
    input =
      applyLowPrecisionBoundary(rewriter, operation.getLoc(), operation, input);
    auto type = cast<RankedTensorType>(input.getType());
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    auto sourceAxis = static_cast<int64_t>(operation.getAxis());
    if (sourceAxis < 0) {
      sourceAxis += sourceType.getRank();
    }
    const uint32_t axis = convertAxis(sourceAxis, sourceType.getRank());
    SmallVector<int64_t> reducedShape(type.getShape());
    reducedShape[axis] = 1;
    auto reducedType =
      RankedTensorType::get(reducedShape, type.getElementType());
    Value maximum = rewriter.create<tosa::ReduceMaxOp>(
      operation.getLoc(), reducedType, input, axis);
    Value shifted =
      rewriter.create<tosa::SubOp>(operation.getLoc(), type, input, maximum);
    Value exponent =
      rewriter.create<tosa::ExpOp>(operation.getLoc(), type, shifted);
    Value sum = rewriter.create<tosa::ReduceSumOp>(
      operation.getLoc(), reducedType, exponent, axis);
    Value reciprocal =
      rewriter.create<tosa::ReciprocalOp>(operation.getLoc(), reducedType, sum);
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), type, exponent, reciprocal, shift);
    result = applyLowPrecisionBoundary(
      rewriter, operation.getLoc(), operation, result);
    rewriter.replaceOp(operation, result);
    return success();
  }
};

class ConvertLayerNorm final : public OpConversionPattern<LayerNormOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    LayerNormOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto sourceType = cast<RankedTensorType>(operation.getInput().getType());
    Value input = restoreNCNNLayout(
      rewriter, operation.getLoc(), adaptor.getInput(), sourceType);
    const int64_t axis = sourceType.getRank() - 1;
    SmallVector<int64_t> reducedShape(sourceType.getShape());
    reducedShape[axis] = 1;
    auto reducedType =
      RankedTensorType::get(reducedShape, sourceType.getElementType());
    Value sum = rewriter.create<tosa::ReduceSumOp>(
      operation.getLoc(), reducedType, input, axis);
    auto scalarType = getBroadcastScalarType(reducedType);
    Value reciprocalCount = createSplat(rewriter,
                                        operation.getLoc(),
                                        scalarType,
                                        1.0 / sourceType.getShape().back());
    Value shift = createI8Zero(rewriter, operation.getLoc());
    Value mean = rewriter.create<tosa::MulOp>(
      operation.getLoc(), reducedType, sum, reciprocalCount, shift);
    Value centered =
      rewriter.create<tosa::SubOp>(operation.getLoc(), sourceType, input, mean);
    Value squared = rewriter.create<tosa::MulOp>(
      operation.getLoc(), sourceType, centered, centered, shift);
    Value varianceSum = rewriter.create<tosa::ReduceSumOp>(
      operation.getLoc(), reducedType, squared, axis);
    Value variance = rewriter.create<tosa::MulOp>(
      operation.getLoc(), reducedType, varianceSum, reciprocalCount, shift);
    Value epsilon = createSplat(rewriter,
                                operation.getLoc(),
                                scalarType,
                                operation.getEpsilon().convertToDouble());
    Value varianceWithEpsilon = rewriter.create<tosa::AddOp>(
      operation.getLoc(), reducedType, variance, epsilon);
    Value exponent =
      createSplat(rewriter, operation.getLoc(), scalarType, -0.5);
    Value inverseStd = rewriter.create<tosa::PowOp>(
      operation.getLoc(), reducedType, varianceWithEpsilon, exponent);
    Value result = rewriter.create<tosa::MulOp>(
      operation.getLoc(), sourceType, centered, inverseStd, shift);
    if (operation.getAffine()) {
      SmallVector<int64_t> parameterShape(sourceType.getRank(), 1);
      parameterShape.back() = sourceType.getShape().back();
      auto parameterType =
        RankedTensorType::get(parameterShape, sourceType.getElementType());
      Value gamma = reshapeValue(rewriter,
                                 operation.getLoc(),
                                 adaptor.getAffineParameters()[0],
                                 parameterType);
      Value beta = reshapeValue(rewriter,
                                operation.getLoc(),
                                adaptor.getAffineParameters()[1],
                                parameterType);
      result = rewriter.create<tosa::MulOp>(
        operation.getLoc(), sourceType, result, gamma, shift);
      result = rewriter.create<tosa::AddOp>(
        operation.getLoc(), sourceType, result, beta);
    }
    rewriter.replaceOp(
      operation,
      convertNCNNLayout(rewriter, operation.getLoc(), result, sourceType));
    return success();
  }
};

class ConvertEmbed final : public OpConversionPattern<EmbedOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    EmbedOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    auto outputType = cast<RankedTensorType>(operation.getOutput().getType());
    const int64_t words = outputType.getShape()[0];
    const int64_t inputDim = operation.getInputDim();
    const int64_t numOutput = operation.getNumOutput();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    Value indices =
      rewriter.create<tosa::ClampOp>(operation.getLoc(),
                                     inputType,
                                     adaptor.getInput(),
                                     rewriter.getI32IntegerAttr(0),
                                     rewriter.getI32IntegerAttr(inputDim - 1));
    indices =
      reshapeValue(rewriter,
                   operation.getLoc(),
                   indices,
                   RankedTensorType::get({1, words}, rewriter.getI32Type()));
    Value values =
      reshapeValue(rewriter,
                   operation.getLoc(),
                   adaptor.getWeight(),
                   RankedTensorType::get({1, inputDim, numOutput},
                                         outputType.getElementType()));
    auto gatheredType =
      RankedTensorType::get({1, words, numOutput}, outputType.getElementType());
    Value result = rewriter.create<tosa::GatherOp>(
      operation.getLoc(), gatheredType, values, indices);
    if (!adaptor.getBias().empty()) {
      Value bias = reshapeValue(
        rewriter,
        operation.getLoc(),
        adaptor.getBias().front(),
        RankedTensorType::get({1, 1, numOutput}, outputType.getElementType()));
      result = rewriter.create<tosa::AddOp>(
        operation.getLoc(), gatheredType, result, bias);
    }
    rewriter.replaceOp(
      operation,
      reshapeValue(rewriter, operation.getLoc(), result, outputType));
    return success();
  }
};

void appendAttentionSegmentRecords(MultiHeadAttentionOp operation,
                                   int64_t sequence,
                                   int64_t qdim,
                                   int64_t embed,
                                   int64_t heads,
                                   int64_t headDim,
                                   bool dynamicSequence) {
  auto module = operation->getParentOfType<ModuleOp>();
  auto function = operation->getParentOfType<func::FuncOp>();
  if (!module || !function) {
    return;
  }

  int64_t ordinal = 0;
  if (auto next =
        module->getAttrOfType<IntegerAttr>(contract::kAttentionOrdinal)) {
    ordinal = next.getInt();
  }
  module->setAttr(
    contract::kAttentionOrdinal,
    IntegerAttr::get(IntegerType::get(module.getContext(), 64), ordinal + 1));
  module->setAttr(contract::kAttentionRevision,
                  StringAttr::get(module.getContext(), "attention-segment-v1"));

  contract::ModelLedger ledger = contract::ModelLedger::read(module);
  SmallVector<Attribute> records;
  if (ledger.attentionSegments) {
    records.append(ledger.attentionSegments.begin(),
                   ledger.attentionSegments.end());
  }

  std::optional<int64_t> transposeBytes;
  if (!dynamicSequence && sequence > 0 && heads > 0 && headDim > 0) {
    constexpr int64_t elementBytes = 4;
    const auto safeMultiply = [](int64_t lhs,
                                 int64_t rhs) -> std::optional<int64_t> {
      if (lhs < 0 || rhs < 0 ||
          (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
        return std::nullopt;
      }
      return lhs * rhs;
    };
    auto elements = safeMultiply(sequence, heads);
    if (elements) {
      elements = safeMultiply(*elements, headDim);
    }
    if (elements) {
      transposeBytes = safeMultiply(*elements, elementBytes);
    }
  }
  if (!dynamicSequence) {
    transposeBytes = 0;
  }

  const std::string prefix =
    function.getName().str() + "/attention#" + std::to_string(ordinal);
  auto append = [&](StringRef phase,
                    std::optional<int64_t> m,
                    std::optional<int64_t> k,
                    std::optional<int64_t> n,
                    int64_t transposeCount,
                    StringRef status,
                    StringRef reason) {
    MLIRContext* context = module.getContext();
    NamedAttrList record;
    record.set("id", StringAttr::get(context, prefix + "/" + phase.str()));
    record.set("function", StringAttr::get(context, function.getName()));
    record.set("source_operation",
               StringAttr::get(context, contract::kLayerMultiHeadAttention));
    record.set("attention_ordinal",
               IntegerAttr::get(IntegerType::get(context, 64), ordinal));
    record.set("phase", StringAttr::get(context, phase));
    record.set("heads", IntegerAttr::get(IntegerType::get(context, 64), heads));
    if (m) {
      record.set("M", IntegerAttr::get(IntegerType::get(context, 64), *m));
    }
    if (k) {
      record.set("K", IntegerAttr::get(IntegerType::get(context, 64), *k));
    }
    if (n) {
      record.set("N", IntegerAttr::get(IntegerType::get(context, 64), *n));
    }
    record.set("kernel_status", StringAttr::get(context, status));
    if (!reason.empty()) {
      record.set("reason", StringAttr::get(context, reason));
    }
    record.set("transpose_count",
               IntegerAttr::get(IntegerType::get(context, 64), transposeCount));
    record.set("copy_bytes_known", BoolAttr::get(context, false));
    record.set(
      "layout",
      StringAttr::get(context,
                      dynamicSequence ? "head_major_copy" : "sequence_major"));
    record.set(
      "layout_producer",
      StringAttr::get(
        context, dynamicSequence ? "split_heads_copy" : "projection_reshape"));
    record.set("layout_consumer",
               StringAttr::get(context,
                               phase == "score" || phase == "context"
                                 ? "indexed_attention_contraction"
                                 : "projection_or_softmax"));
    const StringRef parallelPolicy = llvm::StringSwitch<StringRef>(phase)
                                       .Case("score", "head_query_key")
                                       .Case("softmax", "head_query")
                                       .Case("context", "sequence_head_feature")
                                       .Default("sequence_feature");
    record.set("parallel_policy", StringAttr::get(context, parallelPolicy));
    record.set("tile_policy",
               StringAttr::get(context, "shape_driven_linalg_parallel"));
    if (!dynamicSequence && phase != "softmax") {
      record.set("transpose_elided_reason",
                 StringAttr::get(context, "direct_indexed_attention"));
    }
    if (phase == "softmax") {
      record.set("softmax_strategy",
                 StringAttr::get(context, "stable_two_pass"));
    }
    if (transposeBytes) {
      record.set(
        "transpose_bytes",
        IntegerAttr::get(IntegerType::get(context, 64), *transposeBytes));
      record.set("transpose_bytes_known", BoolAttr::get(context, true));
    } else {
      record.set("transpose_bytes_known", BoolAttr::get(context, false));
    }
    records.push_back(DictionaryAttr::get(context, record));
  };

  const StringRef status = dynamicSequence ? "fallback" : "selected";
  const StringRef reason = dynamicSequence ? "dynamic_sequence" : "";
  // Written as a conditional assignment rather than a ternary: gcc 14 reports
  // a false -Wmaybe-uninitialized on `std::optional` built from a ternary.
  std::optional<int64_t> seq;
  if (!dynamicSequence) {
    seq = sequence;
  }
  const int64_t layoutTransposeCount = dynamicSequence ? 1 : 0;
  append(
    "q_projection", seq, qdim, embed, layoutTransposeCount, status, reason);
  append(
    "k_projection", seq, qdim, embed, layoutTransposeCount, status, reason);
  append(
    "v_projection", seq, qdim, embed, layoutTransposeCount, status, reason);
  append("score", seq, headDim, seq, 0, status, reason);
  append("softmax",
         seq,
         seq,
         std::nullopt,
         0,
         dynamicSequence ? "fallback" : "selected",
         dynamicSequence ? "dynamic_sequence" : "");
  append("context", seq, seq, headDim, layoutTransposeCount, status, reason);
  append("output_projection", seq, embed, qdim, 0, status, reason);

  ledger.attentionSegments = ArrayAttr::get(module.getContext(), records);
  ledger.write(module);
}

class ConvertMultiHeadAttention final
  : public OpConversionPattern<MultiHeadAttentionOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

 private:
  static Value lowerDynamic(MultiHeadAttentionOp operation,
                            OpAdaptor adaptor,
                            ConversionPatternRewriter& rewriter) {
    Location location = operation.getLoc();
    MLIRContext* context = rewriter.getContext();
    Type elementType = rewriter.getF32Type();
    const int64_t embed = operation.getEmbedDim();
    const int64_t heads = operation.getNumHeads();
    const int64_t headDim = embed / heads;
    Value sequence =
      rewriter.create<tensor::DimOp>(location, adaptor.getInput(), 0);

    auto projectionType =
      RankedTensorType::get({ShapedType::kDynamic, embed}, elementType);
    auto project = [&](Value weight, Value bias) {
      Value empty = rewriter.create<tensor::EmptyOp>(
        location, projectionType, ValueRange{sequence});
      AffineExpr row = rewriter.getAffineDimExpr(0);
      AffineExpr column = rewriter.getAffineDimExpr(1);
      AffineMap biasMap = AffineMap::get(2, 0, column, context);
      AffineMap outputMap = rewriter.getMultiDimIdentityMap(2);
      SmallVector<utils::IteratorType> parallel(2,
                                                utils::IteratorType::parallel);
      Value initialized =
        rewriter
          .create<linalg::GenericOp>(location,
                                     projectionType,
                                     ValueRange{bias},
                                     ValueRange{empty},
                                     ArrayRef<AffineMap>{biasMap, outputMap},
                                     parallel,
                                     [](OpBuilder& nested,
                                        Location nestedLocation,
                                        ValueRange arguments) {
                                       nested.create<linalg::YieldOp>(
                                         nestedLocation, arguments.front());
                                     })
          .getResult(0);

      AffineExpr reduction = rewriter.getAffineDimExpr(2);
      AffineMap inputMap = AffineMap::get(3, 0, {row, reduction}, context);
      AffineMap weightMap = AffineMap::get(3, 0, {column, reduction}, context);
      AffineMap resultMap = AffineMap::get(3, 0, {row, column}, context);
      SmallVector<utils::IteratorType> iterators(2,
                                                 utils::IteratorType::parallel);
      iterators.push_back(utils::IteratorType::reduction);
      return rewriter
        .create<linalg::GenericOp>(
          location,
          projectionType,
          ValueRange{adaptor.getInput(), weight},
          ValueRange{initialized},
          ArrayRef<AffineMap>{inputMap, weightMap, resultMap},
          iterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value sum = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, sum);
          })
        .getResult(0);
    };

    Value query = project(adaptor.getQWeight(), adaptor.getQBias());
    Value key = project(adaptor.getKWeight(), adaptor.getKBias());
    Value value = project(adaptor.getVWeight(), adaptor.getVBias());

    Value scaledEmpty = rewriter.create<tensor::EmptyOp>(
      location, projectionType, ValueRange{sequence});
    const double scale = operation.getScale().convertToDouble();
    query =
      rewriter
        .create<linalg::MapOp>(
          location,
          ValueRange{query},
          scaledEmpty,
          [scale](
            OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value factor = nested.create<arith::ConstantOp>(
              nestedLocation, nested.getF32FloatAttr(scale));
            Value result = nested.create<arith::MulFOp>(
              nestedLocation, arguments.front(), factor);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        ->getResult(0);

    auto sequenceMajorType = RankedTensorType::get(
      {ShapedType::kDynamic, heads, headDim}, elementType);
    auto headMajorType = RankedTensorType::get(
      {heads, ShapedType::kDynamic, headDim}, elementType);
    SmallVector<ReassociationIndices> splitEmbedding = {{0}, {1, 2}};
    SmallVector<OpFoldResult> splitShape = {
      sequence, rewriter.getIndexAttr(heads), rewriter.getIndexAttr(headDim)};
    auto splitHeads = [&](Value projected) {
      Value sequenceMajor = rewriter.create<tensor::ExpandShapeOp>(
        location, sequenceMajorType, projected, splitEmbedding, splitShape);
      Value empty = rewriter.create<tensor::EmptyOp>(
        location, headMajorType, ValueRange{sequence});
      AffineExpr sequenceIndex = rewriter.getAffineDimExpr(0);
      AffineExpr head = rewriter.getAffineDimExpr(1);
      AffineExpr feature = rewriter.getAffineDimExpr(2);
      AffineMap inputMap =
        AffineMap::get(3, 0, {sequenceIndex, head, feature}, context);
      AffineMap outputMap =
        AffineMap::get(3, 0, {head, sequenceIndex, feature}, context);
      SmallVector<utils::IteratorType> iterators(3,
                                                 utils::IteratorType::parallel);
      return rewriter
        .create<linalg::GenericOp>(
          location,
          headMajorType,
          ValueRange{sequenceMajor},
          ValueRange{empty},
          ArrayRef<AffineMap>{inputMap, outputMap},
          iterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            nested.create<linalg::YieldOp>(nestedLocation, arguments.front());
          })
        .getResult(0);
    };
    Value queryHeads = splitHeads(query);
    Value keyHeads = splitHeads(key);
    Value valueHeads = splitHeads(value);

    auto scoresType = RankedTensorType::get(
      {heads, ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value scoresEmpty = rewriter.create<tensor::EmptyOp>(
      location, scoresType, ValueRange{sequence, sequence});
    Value zero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(0.0));
    Value initializedScores =
      rewriter.create<linalg::FillOp>(location, zero, scoresEmpty).getResult(0);
    AffineExpr head = rewriter.getAffineDimExpr(0);
    AffineExpr queryIndex = rewriter.getAffineDimExpr(1);
    AffineExpr keyIndex = rewriter.getAffineDimExpr(2);
    AffineExpr feature = rewriter.getAffineDimExpr(3);
    AffineMap queryMap =
      AffineMap::get(4, 0, {head, queryIndex, feature}, context);
    AffineMap keyMap = AffineMap::get(4, 0, {head, keyIndex, feature}, context);
    AffineMap scoresMap =
      AffineMap::get(4, 0, {head, queryIndex, keyIndex}, context);
    SmallVector<utils::IteratorType> scoreIterators(
      3, utils::IteratorType::parallel);
    scoreIterators.push_back(utils::IteratorType::reduction);
    Value scores =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{queryHeads, keyHeads},
          ValueRange{initializedScores},
          ArrayRef<AffineMap>{queryMap, keyMap, scoresMap},
          scoreIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value sum = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, sum);
          })
        .getResult(0);

    auto reducedType =
      RankedTensorType::get({heads, ShapedType::kDynamic}, elementType);
    Value reducedEmpty = rewriter.create<tensor::EmptyOp>(
      location, reducedType, ValueRange{sequence});
    Value negativeInfinity = rewriter.create<arith::ConstantOp>(
      location,
      rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value initializedMaximum =
      rewriter.create<linalg::FillOp>(location, negativeInfinity, reducedEmpty)
        .getResult(0);
    AffineExpr reductionHead = rewriter.getAffineDimExpr(0);
    AffineExpr reductionRow = rewriter.getAffineDimExpr(1);
    AffineExpr reductionColumn = rewriter.getAffineDimExpr(2);
    AffineMap softmaxInputMap = AffineMap::get(
      3, 0, {reductionHead, reductionRow, reductionColumn}, context);
    AffineMap reductionMap =
      AffineMap::get(3, 0, {reductionHead, reductionRow}, context);
    SmallVector<utils::IteratorType> reductionIterators = {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction};
    Value maximum =
      rewriter
        .create<linalg::GenericOp>(
          location,
          reducedType,
          ValueRange{scores},
          ValueRange{initializedMaximum},
          ArrayRef<AffineMap>{softmaxInputMap, reductionMap},
          reductionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::MaximumFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value exponentEmpty = rewriter.create<tensor::EmptyOp>(
      location, scoresType, ValueRange{sequence, sequence});
    AffineMap scoresIdentity = rewriter.getMultiDimIdentityMap(3);
    AffineMap maximumMap = AffineMap::get(
      3,
      0,
      {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
      context);
    SmallVector<utils::IteratorType> softmaxParallel(
      3, utils::IteratorType::parallel);
    Value exponent =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{scores, maximum},
          ValueRange{exponentEmpty},
          ArrayRef<AffineMap>{scoresIdentity, maximumMap, scoresIdentity},
          softmaxParallel,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value shifted = nested.create<arith::SubFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value result = nested.create<math::ExpOp>(nestedLocation, shifted);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value sumEmpty = rewriter.create<tensor::EmptyOp>(
      location, reducedType, ValueRange{sequence});
    Value initializedSum =
      rewriter.create<linalg::FillOp>(location, zero, sumEmpty).getResult(0);
    Value sum =
      rewriter
        .create<linalg::GenericOp>(
          location,
          reducedType,
          ValueRange{exponent},
          ValueRange{initializedSum},
          ArrayRef<AffineMap>{softmaxInputMap, reductionMap},
          reductionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::AddFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value probabilitiesEmpty = rewriter.create<tensor::EmptyOp>(
      location, scoresType, ValueRange{sequence, sequence});
    Value probabilities =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{exponent, sum},
          ValueRange{probabilitiesEmpty},
          ArrayRef<AffineMap>{scoresIdentity, maximumMap, scoresIdentity},
          softmaxParallel,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::DivFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value contextEmpty = rewriter.create<tensor::EmptyOp>(
      location, headMajorType, ValueRange{sequence});
    Value initializedContext =
      rewriter.create<linalg::FillOp>(location, zero, contextEmpty)
        .getResult(0);
    AffineExpr contextHead = rewriter.getAffineDimExpr(0);
    AffineExpr contextRow = rewriter.getAffineDimExpr(1);
    AffineExpr contextFeature = rewriter.getAffineDimExpr(2);
    AffineExpr contextReduction = rewriter.getAffineDimExpr(3);
    AffineMap probabilityMap = AffineMap::get(
      4, 0, {contextHead, contextRow, contextReduction}, context);
    AffineMap valueMap = AffineMap::get(
      4, 0, {contextHead, contextReduction, contextFeature}, context);
    AffineMap contextMap =
      AffineMap::get(4, 0, {contextHead, contextRow, contextFeature}, context);
    SmallVector<utils::IteratorType> contextIterators(
      3, utils::IteratorType::parallel);
    contextIterators.push_back(utils::IteratorType::reduction);
    Value attentionContext =
      rewriter
        .create<linalg::GenericOp>(
          location,
          headMajorType,
          ValueRange{probabilities, valueHeads},
          ValueRange{initializedContext},
          ArrayRef<AffineMap>{probabilityMap, valueMap, contextMap},
          contextIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value result = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value sequenceMajorEmpty = rewriter.create<tensor::EmptyOp>(
      location, sequenceMajorType, ValueRange{sequence});
    AffineExpr contextSequence = rewriter.getAffineDimExpr(0);
    AffineExpr contextHeadIndex = rewriter.getAffineDimExpr(1);
    AffineExpr contextFeatureIndex = rewriter.getAffineDimExpr(2);
    AffineMap headMajorMap = AffineMap::get(
      3, 0, {contextHeadIndex, contextSequence, contextFeatureIndex}, context);
    AffineMap sequenceMajorMap = AffineMap::get(
      3, 0, {contextSequence, contextHeadIndex, contextFeatureIndex}, context);
    SmallVector<utils::IteratorType> transposeIterators(
      3, utils::IteratorType::parallel);
    Value sequenceMajor =
      rewriter
        .create<linalg::GenericOp>(
          location,
          sequenceMajorType,
          ValueRange{attentionContext},
          ValueRange{sequenceMajorEmpty},
          ArrayRef<AffineMap>{headMajorMap, sequenceMajorMap},
          transposeIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            nested.create<linalg::YieldOp>(nestedLocation, arguments.front());
          })
        .getResult(0);
    Value merged = rewriter.create<tensor::CollapseShapeOp>(
      location, projectionType, sequenceMajor, splitEmbedding);

    auto outputType = cast<RankedTensorType>(operation.getType());
    Value outputEmpty = rewriter.create<tensor::EmptyOp>(
      location, outputType, ValueRange{sequence});
    AffineExpr outputRow = rewriter.getAffineDimExpr(0);
    AffineExpr outputColumn = rewriter.getAffineDimExpr(1);
    AffineMap outBiasMap = AffineMap::get(2, 0, outputColumn, context);
    AffineMap outputIdentity = rewriter.getMultiDimIdentityMap(2);
    SmallVector<utils::IteratorType> outputParallel(
      2, utils::IteratorType::parallel);
    Value initializedOutput =
      rewriter
        .create<linalg::GenericOp>(
          location,
          outputType,
          ValueRange{adaptor.getOutBias()},
          ValueRange{outputEmpty},
          ArrayRef<AffineMap>{outBiasMap, outputIdentity},
          outputParallel,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            nested.create<linalg::YieldOp>(nestedLocation, arguments.front());
          })
        .getResult(0);
    AffineExpr outputReduction = rewriter.getAffineDimExpr(2);
    AffineMap mergedMap =
      AffineMap::get(3, 0, {outputRow, outputReduction}, context);
    AffineMap outWeightMap =
      AffineMap::get(3, 0, {outputColumn, outputReduction}, context);
    AffineMap outResultMap =
      AffineMap::get(3, 0, {outputRow, outputColumn}, context);
    SmallVector<utils::IteratorType> outputIterators(
      2, utils::IteratorType::parallel);
    outputIterators.push_back(utils::IteratorType::reduction);
    return rewriter
      .create<linalg::GenericOp>(
        location,
        outputType,
        ValueRange{merged, adaptor.getOutWeight()},
        ValueRange{initializedOutput},
        ArrayRef<AffineMap>{mergedMap, outWeightMap, outResultMap},
        outputIterators,
        [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
          Value product = nested.create<arith::MulFOp>(
            nestedLocation, arguments[0], arguments[1]);
          Value result =
            nested.create<arith::AddFOp>(nestedLocation, product, arguments[2]);
          nested.create<linalg::YieldOp>(nestedLocation, result);
        })
      .getResult(0);
  }

 public:
  LogicalResult matchAndRewrite(
    MultiHeadAttentionOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Location location = operation.getLoc();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    const int64_t sequence = inputType.getShape()[0];
    const int64_t qdim = operation.getQdim();
    const int64_t embed = operation.getEmbedDim();
    const int64_t heads = operation.getNumHeads();
    const int64_t headDim = embed / heads;
    appendAttentionSegmentRecords(operation,
                                  sequence,
                                  qdim,
                                  embed,
                                  heads,
                                  headDim,
                                  ShapedType::isDynamic(sequence));
    if (ShapedType::isDynamic(sequence)) {
      rewriter.replaceOp(operation, lowerDynamic(operation, adaptor, rewriter));
      return success();
    }
    auto reshapeSequence = [&](Value input,
                               RankedTensorType outputType,
                               unsigned outputDimension,
                               unsigned inputDimension) {
      if (outputType.hasStaticShape()) {
        return reshapeValue(rewriter, location, input, outputType);
      }
      SmallVector<std::optional<unsigned>> sourceDimensions(
        outputType.getRank());
      sourceDimensions[outputDimension] = inputDimension;
      return reshapeValue(
        rewriter, location, input, outputType, sourceDimensions);
    };
    auto matrixInputType =
      RankedTensorType::get({1, sequence, qdim}, rewriter.getF32Type());
    Value matrixInput =
      reshapeSequence(adaptor.getInput(), matrixInputType, 1, 0);
    auto project = [&](Value weight, Value bias) {
      Value transposed = transposeOrFoldConstant(
        rewriter,
        location,
        weight,
        RankedTensorType::get({qdim, embed}, rewriter.getF32Type()),
        ArrayRef<int32_t>{1, 0});
      Value matrixWeight = reshapeValue(
        rewriter,
        location,
        transposed,
        RankedTensorType::get({1, qdim, embed}, rewriter.getF32Type()));
      auto type =
        RankedTensorType::get({1, sequence, embed}, rewriter.getF32Type());
      Value projected = rewriter.create<tosa::MatMulOp>(
        location, type, matrixInput, matrixWeight);
      Value matrixBias = reshapeValue(
        rewriter,
        location,
        bias,
        RankedTensorType::get({1, 1, embed}, rewriter.getF32Type()));
      return static_cast<Value>(
        rewriter.create<tosa::AddOp>(location, type, projected, matrixBias));
    };
    Value query = project(adaptor.getQWeight(), adaptor.getQBias());
    Value key = project(adaptor.getKWeight(), adaptor.getKBias());
    Value value = project(adaptor.getVWeight(), adaptor.getVBias());
    auto projectedType =
      RankedTensorType::get({1, sequence, embed}, rewriter.getF32Type());
    Value scale = createSplat(rewriter,
                              location,
                              getBroadcastScalarType(projectedType),
                              operation.getScale().convertToDouble());
    Value shift = createI8Zero(rewriter, location);
    query = rewriter.create<tosa::MulOp>(
      location, projectedType, query, scale, shift);
    auto sequenceMajorType =
      RankedTensorType::get({sequence, heads, headDim}, rewriter.getF32Type());
    auto toSequenceMajor = [&](Value projected) {
      return reshapeSequence(projected, sequenceMajorType, 0, 1);
    };
    Value querySequence = toSequenceMajor(query);
    Value keySequence = toSequenceMajor(key);
    Value valueSequence = toSequenceMajor(value);

    auto scoresType =
      RankedTensorType::get({heads, sequence, sequence}, rewriter.getF32Type());
    Value scoresEmpty =
      rewriter.create<tensor::EmptyOp>(location, scoresType, ValueRange{});
    Value zero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(0.0));
    Value initializedScores =
      rewriter.create<linalg::FillOp>(location, zero, scoresEmpty).getResult(0);
    MLIRContext* context = rewriter.getContext();
    AffineExpr head = rewriter.getAffineDimExpr(0);
    AffineExpr queryIndex = rewriter.getAffineDimExpr(1);
    AffineExpr keyIndex = rewriter.getAffineDimExpr(2);
    AffineExpr feature = rewriter.getAffineDimExpr(3);
    AffineMap queryMap =
      AffineMap::get(4, 0, {queryIndex, head, feature}, context);
    AffineMap keyMap = AffineMap::get(4, 0, {keyIndex, head, feature}, context);
    AffineMap scoresMap =
      AffineMap::get(4, 0, {head, queryIndex, keyIndex}, context);
    SmallVector<utils::IteratorType> scoreIterators(
      3, utils::IteratorType::parallel);
    scoreIterators.push_back(utils::IteratorType::reduction);
    Value scores =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{querySequence, keySequence},
          ValueRange{initializedScores},
          ArrayRef<AffineMap>{queryMap, keyMap, scoresMap},
          scoreIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value sum = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, sum);
          })
        .getResult(0);

    auto reducedType =
      RankedTensorType::get({heads, sequence}, rewriter.getF32Type());
    Value maximumEmpty =
      rewriter.create<tensor::EmptyOp>(location, reducedType, ValueRange{});
    Value negativeInfinity = rewriter.create<arith::ConstantOp>(
      location,
      rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value initializedMaximum =
      rewriter.create<linalg::FillOp>(location, negativeInfinity, maximumEmpty)
        .getResult(0);
    AffineMap softmaxInputMap =
      AffineMap::get(3, 0, {head, queryIndex, keyIndex}, context);
    AffineMap reductionMap = AffineMap::get(3, 0, {head, queryIndex}, context);
    SmallVector<utils::IteratorType> reductionIterators = {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction};
    Value maximum =
      rewriter
        .create<linalg::GenericOp>(
          location,
          reducedType,
          ValueRange{scores},
          ValueRange{initializedMaximum},
          ArrayRef<AffineMap>{softmaxInputMap, reductionMap},
          reductionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::MaximumFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value exponentEmpty =
      rewriter.create<tensor::EmptyOp>(location, scoresType, ValueRange{});
    AffineMap scoresIdentity = rewriter.getMultiDimIdentityMap(3);
    AffineMap maximumMap = AffineMap::get(3, 0, {head, queryIndex}, context);
    SmallVector<utils::IteratorType> softmaxParallel(
      3, utils::IteratorType::parallel);
    Value exponent =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{scores, maximum},
          ValueRange{exponentEmpty},
          ArrayRef<AffineMap>{scoresIdentity, maximumMap, scoresIdentity},
          softmaxParallel,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value shifted = nested.create<arith::SubFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value result = nested.create<math::ExpOp>(nestedLocation, shifted);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value sumEmpty =
      rewriter.create<tensor::EmptyOp>(location, reducedType, ValueRange{});
    Value initializedSum =
      rewriter.create<linalg::FillOp>(location, zero, sumEmpty).getResult(0);
    Value sum =
      rewriter
        .create<linalg::GenericOp>(
          location,
          reducedType,
          ValueRange{exponent},
          ValueRange{initializedSum},
          ArrayRef<AffineMap>{softmaxInputMap, reductionMap},
          reductionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::AddFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value probabilitiesEmpty =
      rewriter.create<tensor::EmptyOp>(location, scoresType, ValueRange{});
    Value probabilities =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{exponent, sum},
          ValueRange{probabilitiesEmpty},
          ArrayRef<AffineMap>{scoresIdentity, maximumMap, scoresIdentity},
          softmaxParallel,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::DivFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    auto contextType =
      RankedTensorType::get({sequence, heads, headDim}, rewriter.getF32Type());
    Value contextEmpty =
      rewriter.create<tensor::EmptyOp>(location, contextType, ValueRange{});
    Value initializedContext =
      rewriter.create<linalg::FillOp>(location, zero, contextEmpty)
        .getResult(0);
    AffineExpr contextFeature = rewriter.getAffineDimExpr(2);
    AffineExpr contextReduction = rewriter.getAffineDimExpr(3);
    AffineMap probabilityMap =
      AffineMap::get(4, 0, {head, queryIndex, contextReduction}, context);
    AffineMap valueMap =
      AffineMap::get(4, 0, {contextReduction, head, contextFeature}, context);
    AffineMap contextMap =
      AffineMap::get(4, 0, {queryIndex, head, contextFeature}, context);
    SmallVector<utils::IteratorType> contextIterators(
      3, utils::IteratorType::parallel);
    contextIterators.push_back(utils::IteratorType::reduction);
    Value sequenceMajor =
      rewriter
        .create<linalg::GenericOp>(
          location,
          contextType,
          ValueRange{probabilities, valueSequence},
          ValueRange{initializedContext},
          ArrayRef<AffineMap>{probabilityMap, valueMap, contextMap},
          contextIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value sum = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, sum);
          })
        .getResult(0);
    Value merged = reshapeSequence(sequenceMajor, projectedType, 1, 0);
    Value outWeight = transposeOrFoldConstant(
      rewriter,
      location,
      adaptor.getOutWeight(),
      RankedTensorType::get({embed, qdim}, rewriter.getF32Type()),
      ArrayRef<int32_t>{1, 0});
    Value matrixOutWeight = reshapeValue(
      rewriter,
      location,
      outWeight,
      RankedTensorType::get({1, embed, qdim}, rewriter.getF32Type()));
    auto matrixOutputType =
      RankedTensorType::get({1, sequence, qdim}, rewriter.getF32Type());
    Value result = rewriter.create<tosa::MatMulOp>(
      location, matrixOutputType, merged, matrixOutWeight);
    Value outBias =
      reshapeValue(rewriter,
                   location,
                   adaptor.getOutBias(),
                   RankedTensorType::get({1, 1, qdim}, rewriter.getF32Type()));
    result =
      rewriter.create<tosa::AddOp>(location, matrixOutputType, result, outBias);
    rewriter.replaceOp(operation, reshapeSequence(result, inputType, 0, 1));
    return success();
  }
};

}  // namespace

void populateAttentionToTosaPatterns(RewritePatternSet& patterns,
                                     const TypeConverter& typeConverter,
                                     MLIRContext* context) {
  patterns.add<ConvertSoftmax,
               ConvertLayerNorm,
               ConvertEmbed,
               ConvertMultiHeadAttention>(typeConverter, context);
}

}  // namespace mlir::ncnn
