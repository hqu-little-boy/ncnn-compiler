// InstrumentMaterializedSites：materialized 读写事件插桩。
//
// 职责
//   两套机制，都打到同一个 runtime 入口 __ncnn_profile_materialized：
//   1. InstrumentNCNNMaterializedSitesPass —— 顶层 Linalg 生产者→消费者
//      的完整物化（张量结果单 use、全映射访问）；
//   2. instrumentMaterializedReadSites / WriteSites —— 主 pass 分析出的
//      「静态分配 + 单一全映射写者 + 全映射读者」事件。
//
// 不变量
//   * kind=0 是写、kind=1 是读；
//   * 写事件插在算子**之后**，读事件插在**之前**（写完成才算一次物化，
//     读之前报事件才能被算进读者的窗口）；
//   * expectedReaders 报读者个数（读事件自身为 0），供归因层校验完整性。
//
// 插入点
//   自行设定；写侧的 after-op 倒序效应见 InstrumentNCNNSites.hpp。
//
// 明确不做
//   * 不推断部分物化（半映射访问一律不出事件）；
//   * 不统计标量流量。

#include <algorithm>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/StableHash.hpp"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn {

using sites::emitCall;
using sites::emitConstant;
using sites::hasFullAccessMap;
using sites::MaterializedEvent;
using sites::SiteAnalysis;
using sites::SiteContext;
using sites::staticByteSize;

namespace sites {

void instrumentMaterializedReadSites(const SiteContext& context,
                                     Operation* operation,
                                     const SiteAnalysis& analysis) {
  auto it = analysis.materializedReads.find(operation);
  if (it == analysis.materializedReads.end()) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  rewriter.setInsertionPoint(operation);
  for (const MaterializedEvent& event : it->second) {
    Value eventId = emitConstant(
      rewriter, operation->getLoc(), static_cast<std::int64_t>(event.id));
    Value kind = emitConstant(rewriter, operation->getLoc(), 1);
    Value bytes = emitConstant(rewriter, operation->getLoc(), event.bytes);
    Value expectedReaders =
      emitConstant(rewriter, operation->getLoc(), event.expectedReaders);
    emitCall(rewriter,
             operation->getLoc(),
             context.materialized,
             {eventId, kind, bytes, expectedReaders});
  }
}

void instrumentMaterializedWriteSites(const SiteContext& context,
                                      Operation* operation,
                                      const SiteAnalysis& analysis) {
  auto it = analysis.materializedWrites.find(operation);
  if (it == analysis.materializedWrites.end()) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  rewriter.setInsertionPointAfter(operation);
  for (const MaterializedEvent& event : it->second) {
    Value eventId = emitConstant(
      rewriter, operation->getLoc(), static_cast<std::int64_t>(event.id));
    Value kind = emitConstant(rewriter, operation->getLoc(), 0);
    Value bytes = emitConstant(rewriter, operation->getLoc(), event.bytes);
    Value expectedReaders =
      emitConstant(rewriter, operation->getLoc(), event.expectedReaders);
    emitCall(rewriter,
             operation->getLoc(),
             context.materialized,
             {eventId, kind, bytes, expectedReaders});
  }
}

}  // namespace sites

namespace {

class InstrumentNCNNMaterializedSitesPass final
  : public PassWrapper<InstrumentNCNNMaterializedSitesPass,
                       OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final {
    return "instrument-ncnn-materialized-sites";
  }

  StringRef getDescription() const final {
    return "Bracket complete static Linalg producer-consumer materializations";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(contract::kProfileMaterializedSitesInstrumented)) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    auto materialized = sites::declareRuntime(
      rewriter, module, "__ncnn_profile_materialized", {i64, i64, i64, i64});

    struct Site {
      Operation* writer;
      Operation* reader;
      std::uint64_t id;
      std::int64_t bytes;
    };
    SmallVector<Site> sites;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function->hasAttr(contract::kProfileRuntime)) {
        continue;
      }
      std::size_t ordinal = 0;
      function.walk([&](linalg::LinalgOp writer) {
        Operation* writerOp = writer.getOperation();
        if (writerOp->getParentOp() != function.getOperation()) {
          return;
        }
        const int64_t resultCount =
          std::min<int64_t>(writer->getNumResults(), writer.getNumDpsInits());
        for (int64_t resultNumber = 0; resultNumber < resultCount;
             ++resultNumber) {
          Value result = writer->getResult(resultNumber);
          auto resultType = dyn_cast<RankedTensorType>(result.getType());
          const std::int64_t bytes =
            resultType ? staticByteSize(resultType) : -1;
          if (bytes <= 0 || !result.hasOneUse() ||
              !hasFullAccessMap(writer,
                                writer.getDpsInitOperand(resultNumber))) {
            continue;
          }

          OpOperand* use = &*result.getUses().begin();
          auto reader = dyn_cast<linalg::LinalgOp>(use->getOwner());
          if (!reader ||
              !llvm::is_contained(reader.getDpsInputOperands(), use) ||
              !hasFullAccessMap(reader, use) ||
              reader->getParentOp() != function.getOperation() ||
              reader->getBlock() != writerOp->getBlock() ||
              !writerOp->isBeforeInBlock(reader)) {
            continue;
          }

          const std::string siteName = function.getName().str() +
                                       "/materialized#" +
                                       std::to_string(ordinal++);
          sites.push_back(Site{
            .writer = writerOp,
            .reader = reader.getOperation(),
            .id = ncnn_mlir::stableHash64(siteName),
            .bytes = bytes,
          });
        }
      });
    }

    for (const Site& site : sites) {
      rewriter.setInsertionPointAfter(site.writer);
      Value writeId = emitConstant(
        rewriter, site.writer->getLoc(), static_cast<std::int64_t>(site.id));
      Value writeKind = emitConstant(rewriter, site.writer->getLoc(), 0);
      Value writeBytes =
        emitConstant(rewriter, site.writer->getLoc(), site.bytes);
      Value expectedReaders = emitConstant(rewriter, site.writer->getLoc(), 1);
      emitCall(rewriter,
               site.writer->getLoc(),
               materialized,
               {writeId, writeKind, writeBytes, expectedReaders});

      rewriter.setInsertionPoint(site.reader);
      Value readId = emitConstant(
        rewriter, site.reader->getLoc(), static_cast<std::int64_t>(site.id));
      Value readKind = emitConstant(rewriter, site.reader->getLoc(), 1);
      Value readBytes =
        emitConstant(rewriter, site.reader->getLoc(), site.bytes);
      Value noExpectedReaders =
        emitConstant(rewriter, site.reader->getLoc(), 0);
      emitCall(rewriter,
               site.reader->getLoc(),
               materialized,
               {readId, readKind, readBytes, noExpectedReaders});
    }
    module->setAttr(contract::kProfileMaterializedSitesInstrumented,
                    UnitAttr::get(module.getContext()));
  }
};

}  // namespace

std::unique_ptr<Pass> createInstrumentNCNNMaterializedSitesPass() {
  return std::make_unique<InstrumentNCNNMaterializedSitesPass>();
}

}  // namespace mlir::ncnn
