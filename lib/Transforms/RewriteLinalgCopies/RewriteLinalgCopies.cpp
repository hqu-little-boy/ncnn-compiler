#include "ncnn-mlir/Transforms/RewriteLinalgCopies/RewriteLinalgCopies.hpp"

#include <functional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_REWRITELINALGCOPIESPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// bufferize 产生的恒等 linalg.generic 纯拷贝（memCpyFn 的产物形态）改写
// 为显式的顺序拷贝循环。动机：其一，forall(shared_outs) 经 bufferize 后
// 每个行片的落回都是这种拷贝，若保留 linalg 形式会被多线程路径的
// convert-linalg-to-parallel-loops 在并行区域内再并行化，产生嵌套
// omp 团队；其二，不能改写为 memref.copy——其在 MemRefToLLVM 中对非同
// 布局形态会退化为对外部运行时符号 memrefCopy 的调用，与产物
// -nostdlib 链接和符号审计约束冲突；显式 load/store 循环则完全自包含。
void buildSequentialCopy(PatternRewriter& rewriter,
                         Location location,
                         Value source,
                         Value target) {
  const unsigned rank = cast<ShapedType>(source.getType()).getRank();
  SmallVector<Value> indices(rank);
  std::function<void(unsigned)> buildLevel = [&](unsigned level) {
    if (level == rank) {
      Value value = rewriter.create<memref::LoadOp>(location, source, indices);
      rewriter.create<memref::StoreOp>(location, value, target, indices);
      return;
    }
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(location, 0);
    Value extent = rewriter.create<memref::DimOp>(location, source, level);
    Value step = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto loop = rewriter.create<scf::ForOp>(location, lowerBound, extent, step);
    rewriter.setInsertionPointToStart(loop.getBody());
    indices[level] = loop.getInductionVar();
    buildLevel(level + 1);
  };
  buildLevel(0);
}

struct LinalgCopyToSequentialLoops
  : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter& rewriter) const override {
    if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1) {
      return failure();
    }
    Value source = generic.getInputs()[0];
    Value target = generic.getDpsInitOperand(0)->get();
    if (!isa<BaseMemRefType>(source.getType()) ||
        !isa<BaseMemRefType>(target.getType())) {
      return failure();
    }
    auto sourceType = cast<ShapedType>(source.getType());
    auto targetType = cast<ShapedType>(target.getType());
    if (sourceType.getShape() != targetType.getShape() ||
        sourceType.getRank() == 0) {
      return failure();
    }
    ArrayRef<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity()) {
      return failure();
    }
    Block& block = generic.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getOperand(0) != block.getArgument(0)) {
      return failure();
    }
    buildSequentialCopy(rewriter, generic.getLoc(), source, target);
    rewriter.eraseOp(generic);
    return success();
  }
};

class RewriteLinalgCopiesPass final
  : public impl::RewriteLinalgCopiesPassBase<RewriteLinalgCopiesPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgCopyToSequentialLoops>(&getContext());
    if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
