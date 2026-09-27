// InstrumentSiteAnalysis.cpp：站点分析。
//
// 职责
//   在插桩之前回答两个问题：
//     1. 每个 operation 的稳定 profile_id 是什么（fusion_site_id 优先，
//        否则 stableHash64(函数名/算子种类#序号)）；
//     2. 哪些静态分配能完整记账成 materialized 读 / 写事件。
//
// 不变量
//   * 只有「整条 alias 链都是整缓冲视图 + Linalg 全映射访问」的分配才算
//     完整，否则宁可不出事件（字节证据不完整会误导归因）；
//   * 写者必须唯一、读写必须同块且写在前；
//   * 分析本身不改 IR。
//
// 顺序依赖
//   * 必须在内核化之后（站点要落在最终形态上）；
//   * 必须在任何 instrumentXxx 之前（后者只读本分析结果）。
//
// 明确不做
//   * 不插入任何调用；
//   * 不猜 source-op——溯源不明就不记（见 recoverSource 的 ambiguous）。

#include <algorithm>
#include <map>
#include <set>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/StableHash.hpp"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn::sites {

SiteAnalysis analyzeProfileSites(ModuleOp module, func::FuncOp function) {
  SiteAnalysis analysis;
  std::map<std::string, unsigned> operation_ordinals;
  function.walk([&](Operation* operation) {
    const std::string kind = operation->getName().getStringRef().str();
    const std::string ordinal_key = function.getName().str() + "/" + kind;
    const unsigned ordinal = operation_ordinals[ordinal_key]++;
    const auto fusionProfileId =
      operation->getAttrOfType<IntegerAttr>(contract::kFusionSiteId);
    analysis.operationIds.emplace(
      operation,
      fusionProfileId
        ? static_cast<std::uint64_t>(fusionProfileId.getInt())
        : ncnn_mlir::stableHash64(ordinal_key + "#" + std::to_string(ordinal)));
  });

  std::map<Operation*, SmallVector<Operation*>> producers;
  std::map<Operation*, SmallVector<Operation*>> consumers;
  std::map<Operation*, std::map<Operation*, unsigned>> producerUses;
  std::map<Operation*, std::map<Operation*, unsigned>> consumerUses;
  std::map<Operation*, std::map<Operation*, bool>> producerCoverage;
  std::map<Operation*, std::map<Operation*, bool>> consumerCoverage;
  std::map<Operation*, SmallVector<Value>> allocationAliases;
  std::map<Operation*, std::set<OpOperand*>> trackedLinalgUses;
  auto recordLinalgAccess =
    [&](linalg::LinalgOp linalgOp, OpOperand* operand, bool isProducer) {
      SmallVector<Value> aliases;
      auto allocation = traceWholeAllocation(operand->get(), aliases);
      if (!allocation) {
        return;
      }
      Operation* root = *allocation;
      for (Value alias : aliases) {
        if (!llvm::is_contained(allocationAliases[root], alias)) {
          allocationAliases[root].push_back(alias);
        }
      }
      trackedLinalgUses[root].insert(operand);

      Operation* operation = linalgOp.getOperation();
      auto& accesses = isProducer ? producers[root] : consumers[root];
      if (!llvm::is_contained(accesses, operation)) {
        accesses.push_back(operation);
      }
      auto& counts = isProducer ? producerUses[root] : consumerUses[root];
      ++counts[operation];
      auto& coverage =
        isProducer ? producerCoverage[root] : consumerCoverage[root];
      auto [it, inserted] =
        coverage.try_emplace(operation, hasFullAccessMap(linalgOp, operand));
      if (!inserted) {
        it->second &= hasFullAccessMap(linalgOp, operand);
      }
    };
  function.walk([&](linalg::LinalgOp linalgOp) {
    for (int64_t i = 0; i < linalgOp.getNumDpsInits(); ++i) {
      recordLinalgAccess(linalgOp, linalgOp.getDpsInitOperand(i), true);
    }
    for (OpOperand* operand : linalgOp.getDpsInputOperands()) {
      recordLinalgAccess(linalgOp, operand, false);
    }
  });

  // Prove the entire allocation alias chain consists only of whole-buffer
  // views, complete Linalg accesses, and deallocation. Any escape or
  // partial view makes the byte evidence incomplete and is rejected.
  std::map<Operation*, bool> completeAliasUses;
  for (const auto& [allocation, initialAliases] : allocationAliases) {
    SmallVector<Value> aliases(initialAliases.begin(), initialAliases.end());
    bool complete = true;
    for (std::size_t i = 0; i < aliases.size() && complete; ++i) {
      for (OpOperand& use : aliases[i].getUses()) {
        Operation* user = use.getOwner();
        if (isa<memref::DeallocOp>(user)) {
          continue;
        }
        if (use.getOperandNumber() == 0 && isWholeBufferView(*user) &&
            user->getNumResults() == 1) {
          Value result = user->getResult(0);
          if (!llvm::is_contained(aliases, result)) {
            aliases.push_back(result);
          }
          continue;
        }
        auto tracked = trackedLinalgUses.find(allocation);
        if (tracked != trackedLinalgUses.end() &&
            tracked->second.contains(&use)) {
          continue;
        }
        complete = false;
        break;
      }
    }
    completeAliasUses[allocation] = complete;
  }

  // Bracket only static allocations with one full-map Linalg writer and
  // full-map readers that follow it in the same block. Events report the
  // complete logical allocation footprint, not scalar load/store traffic.
  for (const auto& [allocation, writeOps] : producers) {
    if (!completeAliasUses[allocation] || writeOps.size() != 1) {
      continue;
    }
    Operation* writer = writeOps.front();
    if (producerUses[allocation][writer] != 1 ||
        !producerCoverage[allocation][writer]) {
      continue;
    }
    auto allocOp = dyn_cast<memref::AllocOp>(allocation);
    if (!allocOp) {
      continue;
    }
    const std::int64_t bytes = staticByteSize(allocOp.getType());
    if (bytes <= 0) {
      continue;
    }

    SmallVector<Operation*> readOps;
    bool hasCompleteConsumers = true;
    if (auto it = consumers.find(allocation); it != consumers.end()) {
      for (Operation* consumer : it->second) {
        if (consumer == writer) {
          continue;
        }
        if (consumerUses[allocation][consumer] != 1 ||
            !consumerCoverage[allocation][consumer] ||
            consumer->getBlock() != writer->getBlock() ||
            !writer->isBeforeInBlock(consumer)) {
          hasCompleteConsumers = false;
          break;
        }
        readOps.push_back(consumer);
      }
    }
    if (readOps.empty() || !hasCompleteConsumers) {
      continue;
    }

    const std::uint64_t id = analysis.operationIds.at(allocation);
    analysis.materializedWrites[writer].push_back(MaterializedEvent{
      .id = id,
      .bytes = bytes,
      .expectedReaders = static_cast<std::int64_t>(readOps.size()),
    });
    for (Operation* consumer : readOps) {
      analysis.materializedReads[consumer].push_back(
        MaterializedEvent{.id = id, .bytes = bytes, .expectedReaders = 0});
    }
  }
  return analysis;
}

}  // namespace mlir::ncnn::sites
