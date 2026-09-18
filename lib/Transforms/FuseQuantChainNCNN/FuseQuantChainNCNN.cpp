#include "ncnn-mlir/Transforms/FuseQuantChainNCNN/FuseQuantChainNCNN.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_FUSEQUANTCHAINNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// int8 量化卷积的 requant/dequant 尾部以逐 op 全量 pass 形式存在：
//   sitofp map → scale mulf → bias addf → [activation] → requantize
//   quantize（f32→i8，round-half-away + clamp）。
// 每个 op 都是混合元素类型的独立 generic（i32→f32、f32→f32、f32→i8），
// 行级向量化按"输入元素类型一致"的门槛全部拒收，热路径退化为标量循环。
// 本 pass 把单用户、全并行的逐元素 producer 内联进 consumer，链条收拢为
// 单一 generic：ins 保留各自仿射映射（恒等主值 + 常量广播 scale/bias），
// body 依原序拼接（producer 语句在前、consumer 语句在后）——op 顺序与
// 常量值均不变，数值与链式形态逐位一致。融合终止于非逐元素 producer
// （matmul/conv 的 i32 累加器、多用户值、含非 arith/math body 的 op）。

namespace {

// 全并行、单结果张量、body 纯 arith/math 且不读 outs 的逐元素 op
// （linalg.generic 或恒等映射退化的 linalg.map）。
bool isElementwiseChainOp(Operation* operation) {
  Region* region = nullptr;
  unsigned numInputs = 0;
  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    if (generic.getNumResults() != 1 || !generic.hasPureTensorSemantics() ||
        generic.getNumDpsInits() != 1) {
      return false;
    }
    for (utils::IteratorType iteratorType : generic.getIteratorTypesArray()) {
      if (iteratorType != utils::IteratorType::parallel) {
        return false;
      }
    }
    region = &generic->getRegion(0);
    numInputs = generic.getNumDpsInputs();
  } else if (auto map = dyn_cast<linalg::MapOp>(operation)) {
    if (map.getNumResults() != 1 || !map.hasPureTensorSemantics() ||
        map.getNumDpsInits() != 1) {
      return false;
    }
    region = &map->getRegion(0);
    // Unlike generic, map has input block arguments only (no outs argument).
    numInputs = map.getNumDpsInputs();
  } else {
    return false;
  }
  if (!region->hasOneBlock()) {
    return false;
  }
  Block& block = region->front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    return false;
  }
  // generic 的 body 不读 outs 才能安全丢弃其 init；map 没有 outs 块参。
  if (numInputs < block.getNumArguments() &&
      !block.getArgument(numInputs).use_empty()) {
    return false;
  }
  for (Operation& statement : block.without_terminator()) {
    const StringRef dialect = statement.getName().getDialect()->getNamespace();
    if ((dialect != "arith" && dialect != "math") ||
        !isMemoryEffectFree(&statement) || statement.getNumRegions() != 0) {
      return false;
    }
  }
  return true;
}

bool hasStaticTensorOperands(linalg::LinalgOp operation) {
  for (Value operand : operation->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type || !type.hasStaticShape() || type.getEncoding()) {
      return false;
    }
  }
  return true;
}

}  // namespace

class FuseElementwiseProducer : public OpRewritePattern<linalg::GenericOp> {
 public:
  explicit FuseElementwiseProducer(MLIRContext* context)
    : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp consumer,
                                PatternRewriter& rewriter) const override {
    if (!isElementwiseChainOp(consumer)) {
      return failure();
    }
    auto consumerType =
      dyn_cast<RankedTensorType>(consumer.getResult(0).getType());
    if (!consumerType || !consumerType.hasStaticShape()) {
      return failure();
    }

    for (OpOperand& operand : consumer->getOpOperands()) {
      // DPS 布局 [inputs..., inits...]：跳过 init 位，只融合输入位。
      if (operand.getOperandNumber() >=
          static_cast<unsigned>(consumer.getNumDpsInputs())) {
        continue;
      }
      auto producerResult = dyn_cast<OpResult>(operand.get());
      if (!producerResult || !producerResult.hasOneUse()) {
        continue;
      }
      Operation* definition = producerResult.getOwner();
      if (!isElementwiseChainOp(definition)) {
        continue;
      }
      auto producer = cast<linalg::LinalgOp>(definition);
      auto producerType =
        dyn_cast<RankedTensorType>(producer->getResult(0).getType());
      if (!producerType || producerType.getShape() != consumerType.getShape()) {
        continue;
      }
      // 流动值的映射必须逐元素恒等（链条形态）；广播参数映射随 producer
      // 原样上提。
      AffineMap flowingMap =
        consumer.getIndexingMapsArray()[operand.getOperandNumber()];
      if (!flowingMap.isIdentity() ||
          flowingMap.getNumDims() != consumerType.getRank()) {
        continue;
      }

      auto producerMaps = producer.getIndexingMapsArray();
      if (!producerMaps.back().isIdentity() ||
          producerMaps.back().getNumDims() != consumerType.getRank()) {
        continue;
      }

      if (succeeded(fuseProducer(consumer, operand, producer, rewriter))) {
        return success();
      }
    }
    return failure();
  }

 private:
  // 新操作数布局：[consumer inputs（去掉流动位）, producer inputs,
  // consumer inits]；producer 的 init 张量被丢弃（body 不读它）。
  static LogicalResult fuseProducer(linalg::GenericOp consumer,
                                    OpOperand& operand,
                                    linalg::LinalgOp producer,
                                    PatternRewriter& rewriter) {
    Block& consumerBlock = consumer->getRegion(0).front();
    Block& producerBlock = producer->getRegion(0).front();
    const auto flowingIndex = operand.getOperandNumber();
    Value flowingArg = consumerBlock.getArgument(flowingIndex);

    SmallVector<Value> newInputs;
    SmallVector<AffineMap> newMaps;
    for (unsigned index = 0; index < consumer.getNumDpsInputs(); ++index) {
      if (index == flowingIndex) {
        continue;
      }
      newInputs.push_back(consumer.getDpsInputs()[index]);
      newMaps.push_back(consumer.getIndexingMapsArray()[index]);
    }
    for (unsigned index = 0; index < producer.getNumDpsInputs(); ++index) {
      newInputs.push_back(producer.getDpsInputs()[index]);
      newMaps.push_back(producer.getIndexingMapsArray()[index]);
    }
    SmallVector<Value> newInits(consumer.getDpsInits());
    for (unsigned initIndex = 0; initIndex < newInits.size(); ++initIndex) {
      newMaps.push_back(
        consumer
          .getIndexingMapsArray()[consumer.getNumDpsInputs() + initIndex]);
    }

    auto resultType = cast<RankedTensorType>(consumer.getResult(0).getType());
    auto newGeneric = rewriter.create<linalg::GenericOp>(
      consumer.getLoc(),
      TypeRange{resultType},
      newInputs,
      newInits,
      newMaps,
      SmallVector<utils::IteratorType>(consumer.getIteratorTypesArray()));

    SmallVector<Type> argTypes;
    for (Value input : newInputs) {
      argTypes.push_back(
        cast<RankedTensorType>(input.getType()).getElementType());
    }
    for (Value init : newInits) {
      argTypes.push_back(
        cast<RankedTensorType>(init.getType()).getElementType());
    }
    Region& newRegion = newGeneric->getRegion(0);
    Block& newBlock =
      newRegion.empty() ? newRegion.emplaceBlock() : newRegion.front();
    if (newBlock.getNumArguments() == 0) {
      for (Type type : argTypes) {
        newBlock.addArgument(type, consumer.getLoc());
      }
    }
    // 克隆目标固定在新 body 末尾：producer 语句在前、consumer 语句在后，
    // 克隆结果直接成为新 body 的内容（insertion point 决定落点）。
    rewriter.setInsertionPointToEnd(&newBlock);

    // producer 输入块参 → 新块参（紧随缩减后的 consumer 输入之后）；
    // producer 的 outs 参数不映射（body 不读它）。
    IRMapping producerMapping;
    const unsigned producerInputBase = consumer.getNumDpsInputs() - 1;
    for (unsigned index = 0; index < producer.getNumDpsInputs(); ++index) {
      producerMapping.map(producerBlock.getArgument(index),
                          newBlock.getArgument(producerInputBase + index));
    }
    for (Operation& statement : producerBlock.without_terminator()) {
      rewriter.clone(statement, producerMapping);
    }
    auto producerTerminator =
      cast<linalg::YieldOp>(producerBlock.getTerminator());
    // 逐元素 body 允许直接 yield 捕获自区域外的值（如函数标量）；这类值
    // 不在 mapping 中，必须走 lookupOrDefault 保住原定义。捕获值支配
    // producer，因而也支配新块，直接引用是合法的。
    Value producerYield =
      producerMapping.lookupOrDefault(producerTerminator.getValues().front());

    // consumer 块参 → 新块参（流动输入被移除后，后续参数整体前移一位）；
    // 流动块参替换为 producer 的克隆结果。
    IRMapping consumerMapping;
    unsigned newArgIndex = 0;
    for (unsigned index = 0; index < consumer.getNumDpsInputs(); ++index) {
      if (index == flowingIndex) {
        continue;  // 流动槽位不复存在，不占位。
      }
      consumerMapping.map(consumerBlock.getArgument(index),
                          newBlock.getArgument(newArgIndex));
      ++newArgIndex;
    }
    for (unsigned initIndex = 0; initIndex < consumer.getNumDpsInits();
         ++initIndex) {
      consumerMapping.map(
        consumerBlock.getArgument(consumer.getNumDpsInputs() + initIndex),
        newBlock.getArgument(newInputs.size() + initIndex));
    }
    consumerMapping.map(flowingArg, producerYield);

    for (Operation& statement : consumerBlock.without_terminator()) {
      rewriter.clone(statement, consumerMapping);
    }
    auto consumerTerminator =
      cast<linalg::YieldOp>(consumerBlock.getTerminator());
    SmallVector<Value> newResults;
    for (Value yielded : consumerTerminator.getValues()) {
      newResults.push_back(consumerMapping.lookupOrDefault(yielded));
    }
    rewriter.setInsertionPointToEnd(&newBlock);
    rewriter.create<linalg::YieldOp>(consumer.getLoc(), newResults);

    rewriter.replaceAllUsesWith(consumer.getResult(0), newGeneric.getResult(0));
    rewriter.eraseOp(consumer);
    return success();
  }
};

// P16: normalize only a map that the existing generic-consumer fusion can
// actually absorb a producer into. Keep standalone maps and unsupported chains
// intact; in particular, do not look through tensor casts/views or memrefs.
class NormalizeMapConsumer : public OpRewritePattern<linalg::MapOp> {
 public:
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp consumer,
                                PatternRewriter& rewriter) const override {
    if (!isElementwiseChainOp(consumer) || !hasStaticTensorOperands(consumer)) {
      return failure();
    }
    auto resultType = cast<RankedTensorType>(consumer->getResult(0).getType());
    bool hasProducer = false;
    for (Value input : consumer.getDpsInputs()) {
      auto result = dyn_cast<OpResult>(input);
      if (!result || !result.hasOneUse() ||
          !isElementwiseChainOp(result.getOwner())) {
        continue;
      }
      auto producer = cast<linalg::LinalgOp>(result.getOwner());
      if (!hasStaticTensorOperands(producer) ||
          cast<RankedTensorType>(result.getType()).getShape() !=
            resultType.getShape()) {
        continue;
      }
      // Full-rank identity output is essential: all-parallel alone also
      // admits permutations and does not justify copying the input maps.
      auto maps = producer.getIndexingMapsArray();
      if (!maps.back().isIdentity() ||
          maps.back().getNumDims() != resultType.getRank()) {
        continue;
      }
      hasProducer = true;
      break;
    }
    if (!hasProducer) {
      return failure();
    }

    auto generic = rewriter.create<linalg::GenericOp>(
      consumer.getLoc(),
      TypeRange{resultType},
      consumer.getDpsInputs(),
      consumer.getDpsInits(),
      SmallVector<AffineMap>(consumer.getIndexingMapsArray()),
      SmallVector<utils::IteratorType>(resultType.getRank(),
                                       utils::IteratorType::parallel),
      [&](OpBuilder& builder, Location, ValueRange arguments) {
        IRMapping mapping;
        Block& body = consumer->getRegion(0).front();
        for (unsigned index = 0; index < body.getNumArguments(); ++index) {
          mapping.map(body.getArgument(index), arguments[index]);
        }
        // Clone every cast and the yield verbatim, with no cast cancellation
        // or reordering (e.g. f32 -> i8 -> f32 is deliberately lossy).
        for (Operation& statement : body) {
          builder.clone(statement, mapping);
        }
      });
    rewriter.replaceOp(consumer, generic.getResults());
    return success();
  }
};

class FuseQuantChainNCNNPass final
  : public impl::FuseQuantChainNCNNPassBase<FuseQuantChainNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseElementwiseProducer>(&getContext());
    auto castChain =
      getOperation()->getAttrOfType<BoolAttr>("ncnn.int8_cast_chain");
    if (castChain && castChain.getValue()) {
      patterns.add<NormalizeMapConsumer>(&getContext());
    }
    if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
