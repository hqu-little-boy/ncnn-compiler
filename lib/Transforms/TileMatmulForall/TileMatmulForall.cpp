#include "ncnn-mlir/Transforms/TileMatmulForall/TileMatmulForall.hpp"

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_TILEMATMULFORALLPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// A1 之后卷积计算大头以 linalg.matmul 形态存在（折叠二维域），其唯一
// 恒等逐元素消费者（strategy 方案 a 的激活提升产物）以整形 2D generic
// 紧随其后——fuse-linalg-epilogue 的视图守卫不会二次融合它。本 pass 把
// 这一对沿 M/N 输出维切进同一个 scf.forall 网格（tile 消费者并融合
// matmul 生产者），激活随之并行；无消费者的裸 matmul 仅做切分。收缩维
// K 不在切分维度内——每个输出元素的完整归约保持在单线程单 tile 内，
// 累加顺序与串行一致（bit-exact）。只处理顶层实例：嵌套实例再切会制
// 造嵌套并行。
//
// 切分尺寸必须是该维 extent 的因子：上游对非整除尾块会生成动态尺寸
// 切片（affine.min 形态），违反静态形状契约。选不到大于 1 的因子时该
// 维保持不切。
bool isTopLevel(Operation* operation) {
  return operation->getParentOfType<scf::ForOp>() == nullptr &&
         operation->getParentOfType<scf::ForallOp>() == nullptr &&
         operation->getParentOfType<scf::ParallelOp>() == nullptr;
}

// 不超过 requested 且整除 extent 的最大因子；无更大因子但 extent 为
// 偶数时取半（仍为因子），否则返回 extent 本身（该维等效不切）。
int64_t pickDivisor(int64_t extent, int64_t requested) {
  const int64_t cap = std::min(requested, extent - 1);
  for (int64_t candidate = cap; candidate > 1; --candidate) {
    if (extent % candidate == 0) {
      return candidate;
    }
  }
  // 偶数维可取半分回退（恒为因子）；奇数且无因子时该维等效不切。
  if (extent % 2 == 0 && extent / 2 >= 2) {
    return extent / 2;
  }
  return extent;
}

// strategy 方案 a 的激活形态：matmul 结果的唯一用户是恒等映射、全并行
// 迭代、单输入的静态 2D generic（outs 为独立缓冲）。
int resultTypeRank(linalg::GenericOp generic) {
  auto type = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  return type ? static_cast<int>(type.getRank()) : -1;
}

template <typename MatmulOpT>
bool findSoleIdentityElementwiseConsumer(MatmulOpT matmul,
                                         linalg::GenericOp& out) {
  auto users = matmul.getResult(0).getUsers();
  auto first = users.begin();
  if (first == users.end() || std::next(first) != users.end()) {
    return false;
  }
  auto generic = dyn_cast<linalg::GenericOp>(*first);
  if (!generic) {
    return false;
  }
  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      resultType.getRank() != 2 || generic.getNumDpsInputs() != 1 ||
      generic.getInputs()[0] != matmul.getResult(0) ||
      generic.getNumDpsInits() != 1) {
    return false;
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
      contract::annotateTile(forall.getOperation(),
                             isInt8 ? "int8_row_dot" : "f32_matmul",
                             "identity",
                             isInt8 ? "packed_nk" : "row_major_kxn",
                             "identity",
                             tileM,
                             tileN,
                             tileK,
                             "outer_tile",
                             "bounded_panel");
      contract::annotatePacking(forall.getOperation(),
                                isInt8 ? "prepacked_transpose_b" : "unpacked",
                                1,
                                0,
                                0);
    });
  }
};

}  // namespace

}  // namespace mlir::ncnn
