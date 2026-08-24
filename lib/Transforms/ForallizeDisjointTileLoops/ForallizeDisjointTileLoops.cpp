#include "ncnn-mlir/Transforms/ForallizeDisjointTileLoops/ForallizeDisjointTileLoops.hpp"

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_FORALLIZEDISJOINTTILELOOPSPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// 把 fuse-linalg-epilogue 产出的分块循环改写为 scf.forall(shared_outs)。
//
// 靶形由 buildStaticTiledSequence 生成：单一 tensor 型 iter_args 串联整张
// 结果，每次迭代经 insert_slice 写入列窗口 [iv*T, (iv+1)*T)，窗口互不相
// 交却被迫顺序执行。满足下列硬约束时改写才是安全的，任何一条不满足即拒
// 绝（误判等于数据竞争）：
//   1. 下界 0、步长 1、上界为编译期常量；
//   2. 体内除终结 insert_slice 外没有任何操作引用 iter_args（生成器保证
//      初始化片取自循环外捕获的原始 init，而非 acc）；
//   3. insert_slice 的偏移中恰有一维是 iv*T（T 为编译期常量），该维的
//      size 恰为 T；其余维度全为编译期常量偏移——由此各迭代写窗严格不
//      相交；
//   4. strides 全为 1。
// 改写后 body 原样克隆进 forall，insert_slice 变为 in_parallel 内的
// parallel_insert_slice；循环外紧跟的尾块残片（extent % T 部分）写窗与
// 主循环不相交，保持原位串行不受影响。
bool resolveConstantIndex(Value value, int64_t& out) {
  // arith::ConstantIndexOp 是 index 型 arith::Constant 的语法糖，
  // 统一经 ConstantOp 匹配。
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    auto attribute = static_cast<Attribute>(constant.getValue());
    if (auto integer = mlir::dyn_cast<IntegerAttr>(attribute)) {
      out = integer.getInt();
      return true;
    }
  }
  return false;
}

std::optional<int64_t> windowStride(Value offset, Value inductionVariable) {
  auto multiply = offset.getDefiningOp<arith::MulIOp>();
  if (!multiply) {
    return std::nullopt;
  }
  int64_t stride = 0;
  if (multiply.getLhs() == inductionVariable &&
      resolveConstantIndex(multiply.getRhs(), stride)) {
    return stride;
  }
  if (multiply.getRhs() == inductionVariable &&
      resolveConstantIndex(multiply.getLhs(), stride)) {
    return stride;
  }
  return std::nullopt;
}

class DisjointTileLoopRewrite : public OpRewritePattern<scf::ForOp> {
 public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter& rewriter) const override {
    if (loop.getInitArgs().size() != 1 || loop->getNumResults() != 1) {
      return failure();
    }
    auto initType = dyn_cast<RankedTensorType>(loop.getInitArgs()[0].getType());
    if (!initType || !initType.hasStaticShape()) {
      return failure();
    }
    int64_t lowerBound = 0;
    int64_t upperBound = 0;
    int64_t step = 0;
    if (!resolveConstantIndex(loop.getLowerBound(), lowerBound) ||
        lowerBound != 0 || !resolveConstantIndex(loop.getStep(), step) ||
        step != 1 || !resolveConstantIndex(loop.getUpperBound(), upperBound) ||
        upperBound <= 0) {
      return failure();
    }

    Block& body = *loop.getBody();
    if (body.getNumArguments() != 2) {
      return failure();
    }
    Value inductionVariable = body.getArgument(0);
    Value accumulator = body.getArgument(1);

    auto yield = dyn_cast<scf::YieldOp>(body.getTerminator());
    if (!yield || yield->getNumOperands() != 1) {
      return failure();
    }
    auto insert = yield.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || insert.getDest() != accumulator) {
      return failure();
    }
    Operation* insertOperation = insert.getOperation();
    Operation* lastNonTerminator = nullptr;
    for (Operation& operation : body.without_terminator()) {
      lastNonTerminator = &operation;
    }
    if (insertOperation != lastNonTerminator) {
      return failure();
    }
    // 约束 2：除终结 insert 外不得触碰 acc。
    for (Operation& operation : body.without_terminator()) {
      if (&operation == insertOperation) {
        continue;
      }
      for (Value operand : operation.getOperands()) {
        if (operand == accumulator) {
          return failure();
        }
      }
    }

    const unsigned rank = initType.getRank();
    if (insert.getSourceType().getRank() != rank) {
      return failure();
    }
    SmallVector<OpFoldResult> offsets = insert.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = insert.getMixedSizes();
    SmallVector<OpFoldResult> strides = insert.getMixedStrides();
    int64_t windowDimension = -1;
    int64_t windowExtent = -1;
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      int64_t strideValue = 0;
      if (!resolveMixedConstant(strides[dimension], strideValue) ||
          strideValue != 1) {
        return failure();
      }
      if (offsets[dimension].is<Value>()) {
        auto offsetValue = offsets[dimension].get<Value>();
        if (int64_t constant = 0; resolveConstantIndex(offsetValue, constant)) {
          continue;
        }
        if (auto stride = windowStride(offsetValue, inductionVariable)) {
          if (windowDimension >= 0) {
            return failure();
          }
          windowDimension = dimension;
          windowExtent = *stride;
          continue;
        }
        return failure();
      }
      // 属性形态偏移视作常量。
    }
    if (windowDimension < 0) {
      return failure();
    }
    int64_t windowSize = 0;
    if (!resolveMixedConstant(sizes[windowDimension], windowSize) ||
        windowSize != windowExtent) {
      return failure();
    }

    // ---- 改写 ----
    Location location = loop.getLoc();
    auto forall = rewriter.create<scf::ForallOp>(
      location,
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(upperBound)},
      ValueRange{loop.getInitArgs()[0]},
      std::nullopt);
    Block& forallBody = forall.getRegion().front();
    IRMapping mapping;
    mapping.map(inductionVariable, forallBody.getArgument(0));
    mapping.map(accumulator, forallBody.getArgument(1));

    rewriter.setInsertionPointToStart(&forallBody);
    for (Operation& operation : body.without_terminator()) {
      if (&operation == insertOperation) {
        continue;
      }
      rewriter.clone(operation, mapping);
    }
    // 补克隆与操作数映射须在切入 in_parallel 区域之前完成，保证定义先
    // 于使用。
    Value source = ensureMapped(rewriter, mapping, insert.getSource());
    SmallVector<OpFoldResult> mappedOffsets;
    SmallVector<OpFoldResult> mappedSizes;
    SmallVector<OpFoldResult> mappedStrides;
    for (OpFoldResult mixed : offsets) {
      mappedOffsets.push_back(mapMixed(rewriter, mapping, mixed));
    }
    for (OpFoldResult mixed : sizes) {
      mappedSizes.push_back(mapMixed(rewriter, mapping, mixed));
    }
    for (OpFoldResult mixed : strides) {
      mappedStrides.push_back(mapMixed(rewriter, mapping, mixed));
    }

    scf::InParallelOp inParallel = forall.getTerminator();
    rewriter.setInsertionPointToEnd(&inParallel.getRegion().front());
    rewriter.create<tensor::ParallelInsertSliceOp>(location,
                                                   source,
                                                   forallBody.getArgument(1),
                                                   mappedOffsets,
                                                   mappedSizes,
                                                   mappedStrides);

    rewriter.replaceOp(loop, forall.getResults());
    return success();
  }

 private:
  static bool resolveMixedConstant(OpFoldResult mixed, int64_t& out) {
    if (!mixed.is<Attribute>()) {
      return resolveConstantIndex(mixed.get<Value>(), out);
    }
    if (auto integer = mlir::dyn_cast<IntegerAttr>(mixed.get<Attribute>())) {
      out = integer.getInt();
      return true;
    }
    return false;
  }

  static OpFoldResult mapMixed(PatternRewriter& rewriter,
                               IRMapping& mapping,
                               OpFoldResult mixed) {
    if (mixed.is<Value>()) {
      return ensureMapped(rewriter, mapping, mixed.get<Value>());
    }
    return mixed;
  }

  // offset 等处引用的定义链可能只被终结 insert 使用（未随体克隆），
  // 在此按 SSA 依赖递归补克隆。
  static Value ensureMapped(PatternRewriter& rewriter,
                            IRMapping& mapping,
                            Value value) {
    if (mapping.contains(value)) {
      return mapping.lookup(value);
    }
    Operation* definition = value.getDefiningOp();
    if (!definition) {
      return value;
    }
    for (Value operand : definition->getOperands()) {
      ensureMapped(rewriter, mapping, operand);
    }
    Operation* cloned = rewriter.clone(*definition, mapping);
    return cloned->getResult(0) ? cloned->getResult(0) : value;
  }
};

class ForallizeDisjointTileLoopsPass final
  : public impl::ForallizeDisjointTileLoopsPassBase<
      ForallizeDisjointTileLoopsPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<DisjointTileLoopRewrite>(&getContext());
    if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
