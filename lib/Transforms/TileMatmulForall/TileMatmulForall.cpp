#include "ncnn-mlir/Transforms/TileMatmulForall/TileMatmulForall.hpp"

#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_TILEMATMULFORALLPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// A1 之后卷积计算大头以 linalg.matmul 形态存在（折叠二维域）。普通的
// 恒等逐元素消费者可与 matmul 一起沿 M/N 输出维切进同一个 scf.forall
// 网格；若多个 DPS 输入都引用同一个 matmul 结果，先把它们规范成一个
// 输入并映射回同一个 block argument。Strategy 提升产生的宽域 epilogue
// 保持为独立 2D generic：实测把它与 matmul 一起分块会严重劣化。无可融合
// 消费者的 matmul 仅做切分。收缩维 K 不在切分维度内——每个输出元素的
// 完整归约保持在单线程单 tile 内，累加顺序与串行一致（bit-exact）。
// 只处理顶层实例：嵌套实例再切会制造嵌套并行。
//
// 切分尺寸必须是该维 extent 的因子：上游对非整除尾块会生成动态尺寸
// 切片（affine.min 形态），违反静态形状契约。选不到大于 1 的因子时该
// 维保持不切。
bool isTopLevel(Operation* operation) {
  return operation->getParentOfType<scf::ForOp>() == nullptr &&
         operation->getParentOfType<scf::ForallOp>() == nullptr &&
         operation->getParentOfType<scf::ParallelOp>() == nullptr;
}

// 普通 matmul 的 sole consumer 须是恒等映射、全并行迭代、静态 2D generic，
// 且所有输入都引用 matmul 结果；strategy_lifted_epilogue 由性能守卫单独
// 保持为 2D generic，不在此处与宽域 matmul 一起分块。
template <typename MatmulOpT>
bool findSoleIdentityElementwiseConsumer(MatmulOpT matmul,
                                         linalg::GenericOp& out) {
  Value result = matmul.getResult(0);
  linalg::GenericOp generic;
  for (OpOperand& use : result.getUses()) {
    if (!generic) {
      generic = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (!generic) {
        return false;
      }
    } else if (use.getOwner() != generic.getOperation()) {
      return false;
    }
    if (!llvm::is_contained(generic.getDpsInputOperands(), &use)) {
      return false;
    }
  }
  if (!generic) {
    return false;
  }
  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      resultType.getRank() != 2 || generic.getNumDpsInputs() < 1 ||
      generic.getNumDpsInits() != 1 ||
      generic->hasAttr("ncnn.strategy_lifted_epilogue")) {
    return false;
  }
  for (Value input : generic.getDpsInputs()) {
    if (input != matmul.getResult(0)) {
      return false;
    }
  }
  if (generic.getNumDpsInputs() > 1) {
    for (NamedAttribute attribute : generic->getAttrs()) {
      StringRef name = attribute.getName().getValue();
      if (name != "indexing_maps" && name != "iterator_types" &&
          name != "operandSegmentSizes") {
        return false;
      }
    }
  }
  for (AffineMap map : generic.getIndexingMapsArray()) {
    if (!map.isIdentity()) {
      return false;
    }
  }
  for (utils::IteratorType iteratorType : generic.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  out = generic;
  return true;
}

// Only rebuild a repeated-input generic when it has no extra attributes: the
// replacement has a different operand list, so blindly copying op attributes
// could retain stale operand-dependent metadata.
linalg::GenericOp deduplicateRepeatedInputs(PatternRewriter& rewriter,
                                            linalg::GenericOp consumer,
                                            Value producerResult) {
  if (consumer.getNumDpsInputs() == 1) {
    return consumer;
  }
  rewriter.setInsertionPoint(consumer);
  auto maps = consumer.getIndexingMapsArray();
  SmallVector<AffineMap> deduplicatedMaps{maps.front(), maps.back()};
  Block& originalBlock = consumer->getRegion(0).front();
  auto deduplicated = rewriter.create<linalg::GenericOp>(
    consumer.getLoc(),
    consumer.getResultTypes(),
    ValueRange{producerResult},
    consumer.getDpsInits(),
    deduplicatedMaps,
    consumer.getIteratorTypesArray(),
    [&](OpBuilder& builder, Location location, ValueRange arguments) {
      IRMapping mapping;
      for (unsigned index = 0; index < consumer.getNumDpsInputs(); ++index) {
        mapping.map(originalBlock.getArgument(index), arguments[0]);
      }
      for (unsigned index = 0; index < consumer.getNumDpsInits(); ++index) {
        mapping.map(
          originalBlock.getArgument(consumer.getNumDpsInputs() + index),
          arguments[1 + index]);
      }
      for (Operation& operation : originalBlock.without_terminator()) {
        builder.clone(operation, mapping);
      }
      SmallVector<Value> yieldedValues;
      for (Value yieldedValue : originalBlock.getTerminator()->getOperands()) {
        yieldedValues.push_back(mapping.lookupOrDefault(yieldedValue));
      }
      builder.create<linalg::YieldOp>(location, yieldedValues);
    });
  rewriter.replaceOp(consumer, deduplicated.getResults());
  return deduplicated;
}

template <typename MatmulOpT>
class TopLevelMatmulTile : public OpRewritePattern<MatmulOpT> {
 public:
  TopLevelMatmulTile(MLIRContext* context,
                     int64_t rowTileSize,
                     int64_t columnTileSize)
    : OpRewritePattern<MatmulOpT>(context),
      rowTileSize(rowTileSize),
      columnTileSize(columnTileSize) {}

  LogicalResult matchAndRewrite(MatmulOpT matmul,
                                PatternRewriter& rewriter) const override {
    if (!isTopLevel(matmul.getOperation())) {
      return failure();
    }
    auto resultType = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getRank() != 2) {
      return failure();
    }
    const int64_t rows = resultType.getShape()[0];
    const int64_t columns = resultType.getShape()[1];
    // 切分尺寸必须是该维 extent 的因子：上游对非整除尾块会生成动态尺寸
    // 切片（affine.min 形态），违反静态形状契约。选不到大于 1 的因子时
    // 该维取 extent 本身（单次迭代等效不切）。注意尺寸向量按迭代空间位
    // 置对应 M/N，必须恒定输出两个条目。
    auto pickDivisor = [](int64_t extent, int64_t requested) -> int64_t {
      if (requested <= 0 || requested >= extent) {
        return extent;
      }
      for (int64_t candidate = requested; candidate > 1; --candidate) {
        if (extent % candidate == 0) {
          return candidate;
        }
      }
      return extent;
    };
    const int64_t rowChunk = pickDivisor(rows, rowTileSize);
    const int64_t columnChunk = pickDivisor(columns, columnTileSize);
    if (rowChunk >= rows && columnChunk >= columns) {
      // 两维都无可行切分，保持原样。
      return failure();
    }

    scf::SCFTileSizeComputationFunction sizes =
      [rowChunk, columnChunk](
        OpBuilder& builder, Operation* operation) -> SmallVector<OpFoldResult> {
      (void)operation;
      return {builder.getIndexAttr(rowChunk),
              builder.getIndexAttr(columnChunk)};
    };
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    tilingOptions.setTileSizeComputationFunction(sizes);

    // 首选：唯一恒等逐元素消费者在场时切消费者并融合 matmul 生产者，
    // 激活随分块一起并行。
    linalg::GenericOp consumer;
    bool hasConsumer = findSoleIdentityElementwiseConsumer(matmul, consumer);
    if (hasConsumer) {
      consumer =
        deduplicateRepeatedInputs(rewriter, consumer, matmul.getResult(0));
      scf::SCFTileAndFuseOptions fuseOptions;
      fuseOptions.setTilingOptions(tilingOptions);
      auto fused = scf::tileConsumerAndFuseProducersUsingSCF(
        rewriter, cast<TilingInterface>(consumer.getOperation()), fuseOptions);
      if (succeeded(fused)) {
        for (auto [value, replacement] : fused->replacements) {
          if (Operation* definition = value.getDefiningOp()) {
            rewriter.replaceOp(definition, replacement);
          } else {
            rewriter.replaceAllUsesWith(value, replacement);
          }
        }
        return success();
      }
    }

    // 回退：无消费者可融合时仅切分 matmul 本身。
    tilingOptions.setTileSizes(
      {rewriter.getIndexAttr(rowChunk), rewriter.getIndexAttr(columnChunk)});
    auto tiling = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(matmul.getOperation()), tilingOptions);
    if (failed(tiling)) {
      return failure();
    }
    rewriter.replaceOp(matmul, tiling->replacements);
    return success();
  }

 private:
  int64_t rowTileSize;
  int64_t columnTileSize;
};

class TileMatmulForallPass final
  : public impl::TileMatmulForallPassBase<TileMatmulForallPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TopLevelMatmulTile<linalg::MatmulOp>>(
      &getContext(), tileRows.getValue(), tileColumns.getValue());
    // int8 量化路径（P4）：strategy 产出的 matmul_transpose_b 同样切块，
    // 使 i8 row-dot 内核（matmul-kernel-ncnn）可以接管 forall 内形态。
    patterns.add<TopLevelMatmulTile<linalg::MatmulTransposeBOp>>(
      &getContext(), tileRows.getValue(), tileColumns.getValue());
    if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Record the outer tile contract while the linalg op is still present.
    // MatmulKernelNCNN consumes this boundary later and upgrades it to the
    // combined outer-tile plus inner-SIMD contract.
    getOperation().walk([&](scf::ForallOp forall) {
      linalg::MatmulOp matmul;
      linalg::MatmulTransposeBOp int8Matmul;
      forall.walk([&](Operation* operation) {
        if (!matmul && !int8Matmul) {
          if (auto candidate = dyn_cast<linalg::MatmulOp>(operation)) {
            matmul = candidate;
          } else if (auto candidate =
                       dyn_cast<linalg::MatmulTransposeBOp>(operation)) {
            int8Matmul = candidate;
          }
        }
      });
      if (!matmul && !int8Matmul) {
        return;
      }
      const auto output =
        matmul ? matmul.getOutputs().front() : int8Matmul.getOutputs().front();
      const auto outputType = dyn_cast<ShapedType>(output.getType());
      if (!outputType || !outputType.hasRank() || outputType.getRank() != 2) {
        return;
      }
      const auto input =
        matmul ? matmul.getInputs().front() : int8Matmul.getInputs().front();
      const auto inputType = dyn_cast<ShapedType>(input.getType());
      const bool isInt8 = int8Matmul != nullptr;
      if (!outputType.hasStaticShape() || !inputType || !inputType.hasRank() ||
          inputType.getRank() != 2 || !inputType.hasStaticShape()) {
        contract::annotateFallback(forall.getOperation(), "dynamic_shape");
        return;
      }
      const int64_t tileM = outputType.getShape()[0];
      const int64_t tileN = outputType.getShape()[1];
      const int64_t tileK =
        inputType && inputType.hasRank() && inputType.getRank() == 2
          ? inputType.getShape()[1]
          : 0;
      const bool packedF32 =
        matmul && matmul->getAttrOfType<StringAttr>(contract::kPacking) &&
        matmul->getAttrOfType<StringAttr>(contract::kPacking).getValue() ==
          "prepacked_B";
      const StringRef weightLayout = isInt8      ? StringRef("packed_nk")
                                     : packedF32 ? StringRef("panel_nk")
                                                 : StringRef("row_major_kxn");
      contract::annotateTile(forall.getOperation(),
                             isInt8      ? "int8_row_dot"
                             : packedF32 ? "f32_packed_mxn_fma"
                                         : "f32_matmul",
                             "identity",
                             weightLayout,
                             "identity",
                             tileM,
                             tileN,
                             tileK,
                             "outer_tile",
                             "bounded_panel");
      if (packedF32) {
        auto packing = matmul->getAttrOfType<StringAttr>(contract::kPacking);
        auto factor = matmul->getAttrOfType<IntegerAttr>(contract::kPackFactor);
        auto bytes = matmul->getAttrOfType<IntegerAttr>(contract::kPackBytes);
        auto unpack =
          matmul->getAttrOfType<IntegerAttr>(contract::kUnpackBytes);
        contract::annotatePacking(forall.getOperation(),
                                  packing.getValue(),
                                  factor ? factor.getInt() : 1,
                                  bytes ? bytes.getInt() : 0,
                                  unpack ? unpack.getInt() : 0);
        for (StringRef attribute : {contract::kPackSchema,
                                    contract::kPackRawBytes,
                                    contract::kPackTileK,
                                    contract::kPackRuntime,
                                    contract::kAlignment,
                                    contract::kWeightLayout}) {
          if (Attribute value = matmul->getAttr(attribute)) {
            forall->setAttr(attribute, value);
          }
        }
      } else {
        contract::annotatePacking(forall.getOperation(),
                                  isInt8 ? "prepacked_transpose_b" : "unpacked",
                                  1,
                                  0,
                                  0);
      }
    });
  }
};

}  // namespace

}  // namespace mlir::ncnn
