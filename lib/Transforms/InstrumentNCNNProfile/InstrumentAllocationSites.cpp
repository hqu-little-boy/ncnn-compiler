// InstrumentAllocationSites：分配与释放事件插桩。
//
// 职责
//   在 memref.alloc 前插入 __ncnn_profile_alloc(id, bytes)；
//   在 memref.dealloc 前插入 __ncnn_profile_dealloc(id)。
//
// 不变量
//   * 释放事件的 id 归属到**原始分配**，不是 dealloc 自身的序号——否则
//     「谁分配的这块」在报告里断链；
//   * 归属要穿过视图 / scf.for 的 iter_arg 与 result 转发，穿不透就退回
//     dealloc 自己的 id（宁可重复计数，不可错配）；
//   * 释放没有时长，**不**挂进算子计时器——否则分配字节会被算进算术时间。
//
// 顺序依赖
//   * 必须在内核化之后（alloc 要落在最终形态上）；
//   * 与 CopySites / OperationSites 共用同一次 candidates 遍历，插入点顺序
//     决定 IR 顺序，见 InstrumentNCNNSites.hpp。
//
// 明确不做
//   * 不做生命周期分析（那是 ReuseWorkspaceSlots 的事）；
//   * 不统计标量 load/store 流量，只报整块分配足迹。

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn::sites {

void instrumentAllocationSite(const SiteContext& context,
                              Operation* operation,
                              Value idValue) {
  auto allocOp = dyn_cast<memref::AllocOp>(operation);
  if (!allocOp) {
    return;
  }
  IRRewriter& rewriter = context.rewriter;
  rewriter.setInsertionPoint(operation);
  const std::int64_t bytes = staticByteSize(allocOp.getType());
  Value byteValue = emitConstant(rewriter, operation->getLoc(), bytes);
  emitCall(rewriter, operation->getLoc(), context.alloc, {idValue, byteValue});
}

namespace {

// Follow the SSA forwarding performed by scf.for in addition to view-like
// definitions. A loop result (or an iter arg used by a nested dealloc) still
// denotes the original allocation.
Operation* resolveAllocation(Value value) {
  while (true) {
    if (auto allocOp = value.getDefiningOp<memref::AllocOp>()) {
      return allocOp.getOperation();
    }
    if (Operation* defining = value.getDefiningOp()) {
      if (isViewLike(*defining) && defining->getNumOperands() != 0) {
        value = defining->getOperand(0);
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(defining)) {
        auto result = dyn_cast<OpResult>(value);
        if (!result || result.getResultNumber() >= forOp.getInitArgs().size()) {
          return nullptr;
        }
        value = forOp.getInitArgs()[result.getResultNumber()];
        continue;
      }
      return nullptr;
    }
    auto blockArgument = dyn_cast<BlockArgument>(value);
    if (!blockArgument) {
      return nullptr;
    }
    auto forOp = dyn_cast<scf::ForOp>(blockArgument.getOwner()->getParentOp());
    const unsigned argumentNumber = blockArgument.getArgNumber();
    if (!forOp || argumentNumber == 0 ||
        argumentNumber - 1 >= forOp.getInitArgs().size()) {
      return nullptr;
    }
    value = forOp.getInitArgs()[argumentNumber - 1];
  }
}

}  // namespace

void instrumentDeallocationSites(const SiteContext& context,
                                 func::FuncOp function,
                                 const SiteAnalysis& analysis) {
  SmallVector<Operation*> deallocations;
  function.walk([&](memref::DeallocOp operation) {
    deallocations.push_back(operation.getOperation());
  });
  IRRewriter& rewriter = context.rewriter;
  for (Operation* operation : deallocations) {
    std::uint64_t id = analysis.operationIds.at(operation);
    if (operation->getNumOperands() == 1) {
      if (Operation* allocation = resolveAllocation(operation->getOperand(0))) {
        id = analysis.operationIds.at(allocation);
      }
    }
    rewriter.setInsertionPoint(operation);
    Value idValue = emitConstant(
      rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
    emitCall(rewriter, operation->getLoc(), context.dealloc, {idValue});
  }
}

}  // namespace mlir::ncnn::sites
