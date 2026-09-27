// InstrumentFusionSites：融合站点计时插桩。
//
// 职责
//   对带 ncnn.fusion_site_id 的算子，在其前后插
//   __ncnn_profile_event_begin(id, category=FusionSite) / _end(id)，
//   让归因层能把融合链的时间算到具体站点上。
//
// 不变量
//   * 只包 function 内的站点（顶层算子，不跨 region）；
//   * profile_id 直接用 ncnn.fusion_site_id，不另派生——保证与 plan 里
//     记录的 id 同源；
//   * 幂等：打过 ncnn.profile_fusion_sites_instrumented 就整 pass 跳过。
//
// 顺序依赖
//   * 必须在跨算子融合改写之后、canonicalization 之前（站点要落在融合后的
//     形态上，但不能被后续折叠吃掉）。
//
// 明确不做
//   * 不改写融合本身（那是 FuseLinalgEpilogue / Strategy 的事）；
//   * 不给融合站点报字节数。

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn {

namespace {

class InstrumentNCNNFusionSitesPass final
  : public PassWrapper<InstrumentNCNNFusionSitesPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "instrument-ncnn-fusion-sites"; }

  StringRef getDescription() const final {
    return "Bracket selected fusion sites before canonicalization";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(contract::kProfileFusionSitesInstrumented)) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    auto begin = sites::declareRuntime(
      rewriter, module, "__ncnn_profile_event_begin", {i64, i64});
    auto end = sites::declareRuntime(
      rewriter, module, "__ncnn_profile_event_end", {i64});
    struct FusionSite {
      Operation* operation;
      std::int64_t profileId;
    };
    SmallVector<FusionSite> fusionSites;
    module.walk([&](Operation* operation) {
      auto profileId =
        operation->getAttrOfType<IntegerAttr>(contract::kFusionSiteId);
      if (profileId && operation->getParentOfType<func::FuncOp>()) {
        fusionSites.push_back(
          FusionSite{.operation = operation, .profileId = profileId.getInt()});
      }
    });

    for (const FusionSite& site : fusionSites) {
      rewriter.setInsertionPoint(site.operation);
      Value eventId =
        sites::emitConstant(rewriter, site.operation->getLoc(), site.profileId);
      Value category = sites::emitConstant(
        rewriter,
        site.operation->getLoc(),
        static_cast<std::int64_t>(sites::EventCategory::FusionSite));
      sites::emitCall(
        rewriter, site.operation->getLoc(), begin, {eventId, category});
      rewriter.setInsertionPointAfter(site.operation);
      Value endId =
        sites::emitConstant(rewriter, site.operation->getLoc(), site.profileId);
      sites::emitCall(rewriter, site.operation->getLoc(), end, {endId});
    }
    module->setAttr(contract::kProfileFusionSitesInstrumented,
                    UnitAttr::get(module.getContext()));
  }
};

}  // namespace

std::unique_ptr<Pass> createInstrumentNCNNFusionSitesPass() {
  return std::make_unique<InstrumentNCNNFusionSitesPass>();
}

}  // namespace mlir::ncnn
