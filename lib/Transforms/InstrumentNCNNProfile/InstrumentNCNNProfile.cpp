// InstrumentNCNNProfile：诊断执行 profile 的总编排。
//
// 职责
//   只做编排：声明 runtime 入口 → 逐函数分析站点 → 生成候选 →
//   按事件类别依次调用各插桩器。类别逻辑不写在这里，
//   分别落在 Instrument{Allocation,Copy,Operation,Materialized}Sites.cpp。
//
// 不变量
//   * 插桩是 opt-in（NCNN_NATIVE_INT8_PROFILE 等开关关闭时全不插）；
//   * profile_id 由 stableHash64 派生——换哈希算法即 identity 变化；
//   * 站点 source-op 溯源必须唯一；来源不明的 worker 站点宁可不插；
//   * 计时边界不得跨 worker join（P27/P28 的归因 bug 即源于此）。
//
// 顺序依赖
//   * 必须在全部改写与内核化之后（站点要落在最终形态上）；
//   * 必须在 EmitModelPlan 之前（plan 记录的 profile_id 要与插桩同源）。
//
// 明确不做
//   * 不改变计算语义（只加旁路调用）；
//   * 不引入未定义符号白名单之外的依赖（profile_allowed 约束）；
//   * 不做数值采样——只出整数计数与时钟。
//
// 插入点纪律（改这里会改产物 IR）
//   下面 candidates 循环里各类插桩器的**调用顺序**决定 IR 顺序；每个
//   插桩器自行设定插入点。after-op 的倒序效应见 InstrumentNCNNSites.hpp。

#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNProfile.hpp"

#include <map>
#include <optional>
#include <set>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_INSTRUMENTNCNNPROFILEPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class InstrumentNCNNProfilePass final
  : public impl::InstrumentNCNNProfilePassBase<InstrumentNCNNProfilePass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithDialect,
                    func::FuncDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    omp::OpenMPDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(contract::kProfileInstrumented)) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    const SmallVector<Type> oneId{i64};
    const SmallVector<Type> twoIds{i64, i64};
    const SmallVector<Type> threeIds{i64, i64, i64};
    const SmallVector<Type> fourIds{i64, i64, i64, i64};
    sites::SiteContext context{
      .rewriter = rewriter,
      .module = module,
      .begin =
        sites::declareRuntime(rewriter, module, sites::kBeginName, twoIds),
      .end = sites::declareRuntime(rewriter, module, sites::kEndName, oneId),
      .workerBegin =
        sites::declareRuntime(rewriter, module, sites::kWorkerBeginName, oneId),
      .workerEnd =
        sites::declareRuntime(rewriter, module, sites::kWorkerEndName, oneId),
      .alloc =
        sites::declareRuntime(rewriter, module, sites::kAllocName, twoIds),
      .dealloc =
        sites::declareRuntime(rewriter, module, sites::kDeallocName, oneId),
      .copy = sites::declareRuntime(rewriter, module, sites::kCopyName, twoIds),
      .movement =
        sites::declareRuntime(rewriter, module, sites::kMovementName, threeIds),
      .materialized = sites::declareRuntime(
        rewriter, module, sites::kMaterializedName, fourIds),
      .flush = sites::declareRuntime(rewriter, module, sites::kFlushName, {}),
    };

    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr(contract::kProfileRuntime)) {
        functions.push_back(function);
      }
    }

    for (func::FuncOp function : functions) {
      sites::SiteAnalysis analysis =
        sites::analyzeProfileSites(module, function);

      SmallVector<Operation*> candidates;
      function.walk([&](Operation* operation) {
        // Nested memory events need no region wrapping. Keep duration timers
        // at function scope: worker TLS spans cannot be subtracted from the
        // parent parallel span, and timing individual lanes is too expensive.
        const auto materializedReads =
          analysis.materializedReads.contains(operation);
        const auto materializedWrites =
          analysis.materializedWrites.contains(operation);
        const bool fusionBracketed =
          module->hasAttr(contract::kProfileFusionSitesInstrumented) &&
          operation->hasAttr(contract::kFusionSiteId);
        if (isa<memref::AllocOp, memref::CopyOp>(operation) ||
            operation->hasAttr(contract::kCopyContract) || materializedReads ||
            materializedWrites ||
            ((operation->getParentOp() == function.getOperation() &&
              sites::isProfileCandidate(*operation) && !fusionBracketed) ||
             sites::isDirectParallelWorkerCandidate(*operation))) {
          candidates.push_back(operation);
        }
      });

      std::optional<std::uint64_t> root_id;
      if (function->hasAttr(contract::kEntryPoint)) {
        if (auto root = analysis.operationIds.find(function.getOperation());
            root != analysis.operationIds.end()) {
          root_id = root->second;
        }
      }
      if (root_id) {
        sites::instrumentFunctionEntry(context, function, *root_id);
      }

      for (Operation* operation : candidates) {
        const std::uint64_t id = analysis.operationIds.at(operation);
        sites::instrumentMaterializedReadSites(context, operation, analysis);
        const auto category = sites::eventCategoryFor(*operation);
        const bool workerTimed =
          sites::isDirectParallelWorkerCandidate(*operation);
        const bool timed = operation->getParentOp() == function.getOperation();
        // idValue 在 before-op 处只建一次，四类插桩器共享（见
        // InstrumentNCNNSites.hpp）。
        rewriter.setInsertionPoint(operation);
        Value idValue = sites::emitConstant(
          rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
        sites::instrumentOperationTimerBegin(
          context, operation, idValue, category, workerTimed, timed);
        sites::instrumentAllocationSite(context, operation, idValue);
        sites::instrumentCopySites(context, operation, idValue);
        sites::instrumentMovementSite(context, operation, idValue);
        sites::instrumentMaterializedWriteSites(context, operation, analysis);
        sites::instrumentOperationTimerEnd(
          context, operation, id, workerTimed, timed);
      }

      sites::instrumentDeallocationSites(context, function, analysis);
      if (root_id) {
        sites::instrumentFunctionExit(context, function, *root_id);
      }
    }
    module->setAttr(contract::kProfileInstrumented,
                    UnitAttr::get(module.getContext()));
  }
};

}  // namespace

}  // namespace mlir::ncnn
