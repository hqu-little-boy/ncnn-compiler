#include "ncnn-mlir/Transforms/VectorizeNCNN/LowerVectorTransfersNCNN.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_LOWERVECTORTRANSFERSNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// VectorToLLVM 仅可靠下降 rank-1 / 无前导单位维的 transfer。本 pass 在
// ConvertVectorToLLVM 之前做 memref 层 transfer 规范化：
// 1. cast-away 前导单位维（NHWC 批维=1）；
// 2. 去除单位维；
// 3. 连续内存上的 n-D transfer 展平为 1-D（上限 target-bitwidth）。
class LowerVectorTransfersNCNNPass final
  : public impl::LowerVectorTransfersNCNNPassBase<
      LowerVectorTransfersNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool hasTransfer = false;
    module.walk([&](vector::TransferReadOp op) {
      if (!hasTransfer) {
        hasTransfer = true;
      }
    });
    module.walk([&](vector::TransferWriteOp op) {
      if (!hasTransfer) {
        hasTransfer = true;
      }
    });
    if (!hasTransfer) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateFlattenVectorTransferPatterns(patterns,
                                                  targetBitwidth.getValue());
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
