// Scaled dot-product attention 降低。
//
// 职责
//   把 ncnn.sdpa 降低为 Q·Kᵀ / scale / softmax / ·V 的 tosa 组合。
//
// 不变量
//   * scale 来源（显式参数或 1/sqrt(d)）必须显式判定，不得隐式假设；
//   * 布局在进入本族前已定，只做必要的 restore/convert。
//
// 顺序依赖
//   * 与 AttentionToTosa 同批注册，op 名不重叠；
//   * 之后的 batch_matmul 归 StrategyNCNN / MatmulKernelNCNN。
//
// 明确不做
//   * 不做注意力融合与分段优化；
//   * 不做数值上的等价改写（如 softmax 稳定化以外的变换）。

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
using tosa_lowering::convertNCNNLayout;
using tosa_lowering::restoreNCNNLayout;

class ConvertSDPA final : public OpConversionPattern<SDPAOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    SDPAOp operation,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Location location = operation.getLoc();
    MLIRContext* context = rewriter.getContext();
    auto queryType = cast<RankedTensorType>(operation.getQuery().getType());
    auto keyType = cast<RankedTensorType>(operation.getKey().getType());
    auto valueType = cast<RankedTensorType>(operation.getValue().getType());
    Value query =
      restoreNCNNLayout(rewriter, location, adaptor.getQuery(), queryType);
    Value key =
      restoreNCNNLayout(rewriter, location, adaptor.getKey(), keyType);
    Value value =
      restoreNCNNLayout(rewriter, location, adaptor.getValue(), valueType);

    SmallVector<Value> optional(adaptor.getOptionalInputs().begin(),
                                adaptor.getOptionalInputs().end());
    unsigned optionalIndex = 0;
    Value mask;
    if (operation.getHasMask()) {
      mask = optional[optionalIndex++];
    }
    Value totalKey = key;
    Value totalValue = value;
    if (operation.getKvCache()) {
      auto pastKeyType = cast<RankedTensorType>(
        operation.getOptionalInputs()[optionalIndex].getType());
      auto pastValueType = cast<RankedTensorType>(
        operation.getOptionalInputs()[optionalIndex + 1].getType());
      Value pastKey = restoreNCNNLayout(
        rewriter, location, optional[optionalIndex], pastKeyType);
      Value pastValue = restoreNCNNLayout(
        rewriter, location, optional[optionalIndex + 1], pastValueType);
      auto updatedKeyType =
        cast<RankedTensorType>(operation.getCacheResults()[0].getType());
      auto updatedValueType =
        cast<RankedTensorType>(operation.getCacheResults()[1].getType());
      totalKey = rewriter.create<tensor::ConcatOp>(
        location, updatedKeyType, 1, ValueRange{pastKey, key});
      totalValue = rewriter.create<tensor::ConcatOp>(
        location, updatedValueType, 1, ValueRange{pastValue, value});
    }

    Value querySequence = rewriter.create<tensor::DimOp>(location, query, 1);
    Value keySequence = rewriter.create<tensor::DimOp>(location, totalKey, 1);
    Value one = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto createEmpty = [&](RankedTensorType type, ValueRange dimensions) {
      SmallVector<Value> dynamicSizes;
      for (auto [extent, dimension] :
           llvm::zip_equal(type.getShape(), dimensions)) {
        if (ShapedType::isDynamic(extent)) {
          dynamicSizes.push_back(dimension);
        }
      }
      return rewriter.create<tensor::EmptyOp>(location, type, dynamicSizes);
    };

    const int64_t queryHeads = queryType.getShape()[0];
    const int64_t headsPerGroup = queryHeads / keyType.getShape()[0];
    auto scoresType = RankedTensorType::get(
      {queryHeads,
       queryType.getShape()[1],
       cast<RankedTensorType>(totalKey.getType()).getShape()[1]},
      rewriter.getF32Type());
    Value zero = rewriter.create<arith::ConstantOp>(
      location, rewriter.getF32FloatAttr(0.0));
    Value scoresEmpty =
      createEmpty(scoresType, ValueRange{one, querySequence, keySequence});
    Value initializedScores =
      rewriter.create<linalg::FillOp>(location, zero, scoresEmpty).getResult(0);
    AffineExpr head = rewriter.getAffineDimExpr(0);
    AffineExpr queryIndex = rewriter.getAffineDimExpr(1);
    AffineExpr keyIndex = rewriter.getAffineDimExpr(2);
    AffineExpr feature = rewriter.getAffineDimExpr(3);
    AffineMap queryMap =
      AffineMap::get(4, 0, {head, queryIndex, feature}, context);
    AffineMap keyMap = AffineMap::get(
      4, 0, {head.floorDiv(headsPerGroup), keyIndex, feature}, context);
    AffineMap scoresMap =
      AffineMap::get(4, 0, {head, queryIndex, keyIndex}, context);
    SmallVector<utils::IteratorType> contractionIterators(
      3, utils::IteratorType::parallel);
    contractionIterators.push_back(utils::IteratorType::reduction);
    Value scores =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{query, totalKey},
          ValueRange{initializedScores},
          ArrayRef<AffineMap>{queryMap, keyMap, scoresMap},
          contractionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value result = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    AffineMap scoresIdentity = rewriter.getMultiDimIdentityMap(3);
    SmallVector<Value> adjustedInputs{scores};
    SmallVector<AffineMap> adjustedMaps{scoresIdentity};
    if (mask) {
      adjustedInputs.push_back(mask);
      adjustedMaps.push_back(
        AffineMap::get(3, 0, {queryIndex, keyIndex}, context));
    }
    adjustedMaps.push_back(scoresIdentity);
    SmallVector<utils::IteratorType> parallelIterators(
      3, utils::IteratorType::parallel);
    Value adjustedEmpty =
      createEmpty(scoresType, ValueRange{one, querySequence, keySequence});
    const double scale = operation.getScale().convertToDouble();
    scores =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          adjustedInputs,
          ValueRange{adjustedEmpty},
          adjustedMaps,
          parallelIterators,
          [scale, hasMask = static_cast<bool>(mask)](
            OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value factor = nested.create<arith::ConstantOp>(
              nestedLocation, nested.getF32FloatAttr(scale));
            Value result = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], factor);
            if (hasMask) {
              result = nested.create<arith::AddFOp>(
                nestedLocation, result, arguments[1]);
            }
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    auto reducedType = RankedTensorType::get(
      {queryHeads, queryType.getShape()[1]}, rewriter.getF32Type());
    AffineMap reducedMap = AffineMap::get(3, 0, {head, queryIndex}, context);
    SmallVector<utils::IteratorType> reductionIterators = {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction};
    Value reducedEmpty =
      createEmpty(reducedType, ValueRange{one, querySequence});
    Value negativeInfinity = rewriter.create<arith::ConstantOp>(
      location,
      rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    Value initializedMaximum =
      rewriter.create<linalg::FillOp>(location, negativeInfinity, reducedEmpty)
        .getResult(0);
    Value maximum =
      rewriter
        .create<linalg::GenericOp>(
          location,
          reducedType,
          ValueRange{scores},
          ValueRange{initializedMaximum},
          ArrayRef<AffineMap>{scoresIdentity, reducedMap},
          reductionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::MaximumFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value exponentEmpty =
      createEmpty(scoresType, ValueRange{one, querySequence, keySequence});
    Value exponent =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{scores, maximum},
          ValueRange{exponentEmpty},
          ArrayRef<AffineMap>{scoresIdentity, reducedMap, scoresIdentity},
          parallelIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value shifted = nested.create<arith::SubFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value result = nested.create<math::ExpOp>(nestedLocation, shifted);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value sumEmpty = createEmpty(reducedType, ValueRange{one, querySequence});
    Value initializedSum =
      rewriter.create<linalg::FillOp>(location, zero, sumEmpty).getResult(0);
    Value sum =
      rewriter
        .create<linalg::GenericOp>(
          location,
          reducedType,
          ValueRange{exponent},
          ValueRange{initializedSum},
          ArrayRef<AffineMap>{scoresIdentity, reducedMap},
          reductionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::AddFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    Value probabilitiesEmpty =
      createEmpty(scoresType, ValueRange{one, querySequence, keySequence});
    Value probabilities =
      rewriter
        .create<linalg::GenericOp>(
          location,
          scoresType,
          ValueRange{exponent, sum},
          ValueRange{probabilitiesEmpty},
          ArrayRef<AffineMap>{scoresIdentity, reducedMap, scoresIdentity},
          parallelIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value result = nested.create<arith::DivFOp>(
              nestedLocation, arguments[0], arguments[1]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    auto outputType = cast<RankedTensorType>(operation.getContext().getType());
    Value contextEmpty =
      createEmpty(outputType, ValueRange{one, querySequence, one});
    Value initializedContext =
      rewriter.create<linalg::FillOp>(location, zero, contextEmpty)
        .getResult(0);
    AffineExpr outputFeature = rewriter.getAffineDimExpr(2);
    AffineExpr reduction = rewriter.getAffineDimExpr(3);
    AffineMap probabilityMap =
      AffineMap::get(4, 0, {head, queryIndex, reduction}, context);
    AffineMap valueMap = AffineMap::get(
      4, 0, {head.floorDiv(headsPerGroup), reduction, outputFeature}, context);
    AffineMap outputMap =
      AffineMap::get(4, 0, {head, queryIndex, outputFeature}, context);
    Value result =
      rewriter
        .create<linalg::GenericOp>(
          location,
          outputType,
          ValueRange{probabilities, totalValue},
          ValueRange{initializedContext},
          ArrayRef<AffineMap>{probabilityMap, valueMap, outputMap},
          contractionIterators,
          [](OpBuilder& nested, Location nestedLocation, ValueRange arguments) {
            Value product = nested.create<arith::MulFOp>(
              nestedLocation, arguments[0], arguments[1]);
            Value result = nested.create<arith::AddFOp>(
              nestedLocation, product, arguments[2]);
            nested.create<linalg::YieldOp>(nestedLocation, result);
          })
        .getResult(0);

    SmallVector<Value> replacements;
    replacements.push_back(
      convertNCNNLayout(rewriter, location, result, outputType));
    if (operation.getKvCache()) {
      SmallVector<Value> caches{totalKey, totalValue};
      for (auto [cache, cacheResult] :
           llvm::zip_equal(caches, operation.getCacheResults())) {
        replacements.push_back(
          convertNCNNLayout(rewriter,
                            location,
                            cache,
                            cast<RankedTensorType>(cacheResult.getType())));
      }
    }
    rewriter.replaceOp(operation, replacements);
    return success();
  }
};

}  // namespace

void populateSDPAToTosaPatterns(RewritePatternSet& patterns,
                                const TypeConverter& typeConverter,
                                MLIRContext* context) {
  patterns.add<ConvertSDPA>(typeConverter, context);
}

}  // namespace mlir::ncnn
