#include "ncnn-mlir/Transforms/FuseQuantChainNCNN/FuseQuantChainNCNN.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
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
  } else if (isa<linalg::MapOp>(operation)) {
    region = &operation->getRegion(0);
    numInputs = 1;
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
  // map 的块参为 [in, out]；body 不读 outs 才能安全丢弃其 init。
  if (numInputs < block.getNumArguments() &&
      !block.getArgument(numInputs).use_empty()) {
    return false;
  }
  for (Operation& statement : block.without_terminator()) {
    const StringRef dialect = statement.getName().getDialect()->getNamespace();
    if (dialect != "arith" && dialect != "math") {
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
    Value producerYield =
      producerMapping.lookup(producerTerminator.getValues().front());

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
      newResults.push_back(consumerMapping.lookup(yielded));
    }
    rewriter.setInsertionPointToEnd(&newBlock);
    rewriter.create<linalg::YieldOp>(consumer.getLoc(), newResults);

    rewriter.replaceAllUsesWith(consumer.getResult(0), newGeneric.getResult(0));
    rewriter.eraseOp(consumer);
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
    if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
