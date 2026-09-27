// InstrumentOperationSites：算子计时、搬运事件与函数边界插桩。
//
// 职责
//   * 算子时长：函数级算子用 __ncnn_profile_event_begin/end（带类别）；
//     parallel 内的粗粒度 worker 站点用 __ncnn_profile_worker_event_begin/end
//     （无类别——类别属于顶层计时的语义）；
//   * 布局搬运：memref.transpose 的 __ncnn_profile_movement(id, kind=0,
//   bytes)；
//   * 函数边界：入口函数体首插 root begin；每个 return 前插 root end + flush。
//
// 不变量
//   * 计时边界不得跨 worker join（P27/P28 的归因 bug 即源于此）；
//   * 时长计时只在函数作用域，嵌套内存事件不套时长；
//   * root begin 必须在该函数任何站点之前、root end+flush 必须在最后——
//     两者分属函数的头尾，不能合成一次调用。
//
// 插入点
//   TimerBegin / MovementSite / FunctionEntry 自行设 before-op（或函数体首）；
//   TimerEnd / FunctionExit 自行设 after-op（或 return 前）。
//   见 InstrumentNCNNSites.hpp 关于 after-op 倒序效应的说明。
//
// 明确不做
//   * 不做数值采样；
//   * 不给 worker 站点标类别。

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn::sites {

EventCategory eventCategoryFor(Operation& operation) {
  if (isa<memref::AllocOp>(operation)) {
    return EventCategory::Allocation;
  }
  if (isa<memref::CopyOp>(operation)) {
    return EventCategory::Copy;
  }
  if (isa<memref::TransposeOp>(operation)) {
    return EventCategory::Transpose;
  }
  if (isa<scf::ForallOp, scf::ParallelOp, omp::ParallelOp>(operation)) {
    return EventCategory::Parallel;
  }
  return EventCategory::Operation;
}

void instrumentOperationTimerBegin(const SiteContext& context,
                                   Operation* operation,
                                   Value idValue,
                                   EventCategory category,
                                   bool workerTimed,
                                   bool timed) {
  if (!workerTimed && !timed) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  rewriter.setInsertionPoint(operation);
  if (workerTimed) {
    emitCall(rewriter, operation->getLoc(), context.workerBegin, {idValue});
    return;
  }
  Value categoryValue = emitConstant(
    rewriter, operation->getLoc(), static_cast<std::int64_t>(category));
  emitCall(
    rewriter, operation->getLoc(), context.begin, {idValue, categoryValue});
}

void instrumentMovementSite(const SiteContext& context,
                            Operation* operation,
                            Value idValue) {
  auto transposeOp = dyn_cast<memref::TransposeOp>(operation);
  if (!transposeOp) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  // 历史上这里继承调用方插入点；现改为显式 before-op。与绝大多数实际
  // 路径一致（transpose 不会同时带 copy_contract），10 模型逐字节对照无差异。
  rewriter.setInsertionPoint(operation);
  const auto resultType =
    dyn_cast<MemRefType>(transposeOp.getResult().getType());
  const std::int64_t bytes = resultType ? staticByteSize(resultType) : -1;
  Value kind = emitConstant(rewriter, operation->getLoc(), 0);
  Value byteValue = emitConstant(rewriter, operation->getLoc(), bytes);
  emitCall(rewriter,
           operation->getLoc(),
           context.movement,
           {idValue, kind, byteValue});
}

void instrumentOperationTimerEnd(const SiteContext& context,
                                 Operation* operation,
                                 std::uint64_t id,
                                 bool workerTimed,
                                 bool timed) {
  if (!workerTimed && !timed) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  rewriter.setInsertionPointAfter(operation);
  Value endId =
    emitConstant(rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
  emitCall(rewriter,
           operation->getLoc(),
           workerTimed ? context.workerEnd : context.end,
           {endId});
}

void instrumentFunctionEntry(const SiteContext& context,
                             func::FuncOp function,
                             std::uint64_t rootId) {
  if (function.getBody().empty()) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  rewriter.setInsertionPointToStart(&function.getBody().front());
  Value idValue = emitConstant(
    rewriter, function.getLoc(), static_cast<std::int64_t>(rootId));
  Value categoryValue =
    emitConstant(rewriter,
                 function.getLoc(),
                 static_cast<std::int64_t>(EventCategory::Operation));
  emitCall(
    rewriter, function.getLoc(), context.begin, {idValue, categoryValue});
}

void instrumentFunctionExit(const SiteContext& context,
                            func::FuncOp function,
                            std::uint64_t rootId) {
  SmallVector<func::ReturnOp> returns;
  function.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  IRRewriter& rewriter = context.rewriter;
  for (func::ReturnOp returnOp : returns) {
    rewriter.setInsertionPoint(returnOp);
    Value idValue = emitConstant(
      rewriter, returnOp.getLoc(), static_cast<std::int64_t>(rootId));
    emitCall(rewriter, returnOp.getLoc(), context.end, {idValue});
    emitCall(rewriter, returnOp.getLoc(), context.flush, {});
  }
}

}  // namespace mlir::ncnn::sites
