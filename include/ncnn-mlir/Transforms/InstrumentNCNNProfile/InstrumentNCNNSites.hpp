// InstrumentNCNNSites：插桩器共享契约。
//
// 职责
//   放 profile runtime 的私有 ABI（事件类别、入口符号名、已声明的 FuncOp）、
//   各类插桩器都要用的积木（常量 / 调用 / 字节尺寸 / 候选判定），
//   以及「按事件类别」的插桩器入口声明。
//
// 不变量
//   * 插桩是 opt-in：开关关闭时各 pass 全不插；
//   * profile_id 由 stableHash64 派生——换哈希算法即 identity 变化；
//   * 站点 source-op 溯源必须唯一；来源不明的 worker 站点宁可不插；
//   * 计时边界不得跨 worker join。
//
// 插入点契约（改动这里会改产物 IR）
//   每个 instrumentXxx 在发射前自行设定插入点，调用顺序决定 IR 顺序。
//   有一个刻意保留的既有行为：setInsertionPointAfter(operation) 之后创建的
//   多个调用，最终 IR 里是**倒序**（后创建的更靠近 op）。因此
//   OperationTimerEnd / MaterializedWriteSites / CopySites 各自重新设定
//   after-op 插入点，倒序效应必须逐字保留，不要"顺手整理"。
//   例外是 instrumentMovementSite：历史上它继承调用方插入点；现在显式设为
//   before-op（与绝大多数实际路径一致），10 模型逐字节对照已证无差异。
//
// 顺序依赖
//   * 必须在全部改写与内核化之后（站点要落在最终形态上）；
//   * 必须在 EmitModelPlan 之前（plan 记录的 profile_id 要与插桩同源）。
//
// 明确不做
//   * 不改变计算语义（只加旁路调用）；
//   * 不引入未定义符号白名单之外的依赖（profile_allowed 约束）；
//   * 不做数值采样——只出整数计数与时钟。
#ifndef NCNN_MLIR_TRANSFORMS_INSTRUMENTNCNNPROFILE_INSTRUMENTNCNNSITES_HPP
#define NCNN_MLIR_TRANSFORMS_INSTRUMENTNCNNPROFILE_INSTRUMENTNCNNSITES_HPP

#include <cstdint>
#include <map>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

namespace mlir::ncnn::sites {


// profile runtime 的私有 ABI 符号名。与 lib/ProfileRuntime/profile_runtime.c
// 的导出必须一字不差——生成的模块靠名字调用它们。
inline constexpr llvm::StringLiteral kBeginName = "__ncnn_profile_event_begin";
inline constexpr llvm::StringLiteral kEndName = "__ncnn_profile_event_end";
inline constexpr llvm::StringLiteral kWorkerBeginName =
  "__ncnn_profile_worker_event_begin";
inline constexpr llvm::StringLiteral kWorkerEndName =
  "__ncnn_profile_worker_event_end";
inline constexpr llvm::StringLiteral kAllocName = "__ncnn_profile_alloc";
inline constexpr llvm::StringLiteral kDeallocName = "__ncnn_profile_dealloc";
inline constexpr llvm::StringLiteral kCopyName = "__ncnn_profile_copy";
inline constexpr llvm::StringLiteral kMovementName = "__ncnn_profile_movement";
inline constexpr llvm::StringLiteral kMaterializedName =
  "__ncnn_profile_materialized";
inline constexpr llvm::StringLiteral kFlushName = "__ncnn_profile_flush";

// Categories are part of the private profile ABI.  They intentionally remain
// numeric so the generated module does not depend on a compiler-side enum.
enum class EventCategory : std::int64_t {
  Operation = 0,
  Allocation = 1,
  Deallocation = 2,
  Copy = 3,
  Parallel = 4,
  Transpose = 5,
  Pack = 6,
  Unpack = 7,
  MaterializedWrite = 8,
  MaterializedRead = 9,
  FusionSite = 10,
  WorkerOperation = 11,
};

struct MaterializedEvent {
  std::uint64_t id;
  std::int64_t bytes;
  std::int64_t expectedReaders;
};

// 一次插桩所需的运行期入口（declare 之后）与目标模块。
struct SiteContext {
  IRRewriter& rewriter;
  ModuleOp module;
  func::FuncOp begin;
  func::FuncOp end;
  func::FuncOp workerBegin;
  func::FuncOp workerEnd;
  func::FuncOp alloc;
  func::FuncOp dealloc;
  func::FuncOp copy;
  func::FuncOp movement;
  func::FuncOp materialized;
  func::FuncOp flush;
};

// 一个函数的站点分析结果：稳定 id、以及可完整记账的 materialized 读写事件。
struct SiteAnalysis {
  std::map<Operation*, std::uint64_t> operationIds;
  std::map<Operation*, llvm::SmallVector<MaterializedEvent>> materializedWrites;
  std::map<Operation*, llvm::SmallVector<MaterializedEvent>> materializedReads;
};

// ── 积木 ──────────────────────────────────────────────────────────────
func::FuncOp declareRuntime(IRRewriter& rewriter,
                            ModuleOp module,
                            llvm::StringRef name,
                            llvm::ArrayRef<Type> argumentTypes);
Value emitConstant(IRRewriter& rewriter, Location location, std::int64_t value);
void emitCall(IRRewriter& rewriter,
              Location location,
              func::FuncOp callee,
              llvm::ArrayRef<Value> arguments);
std::int64_t staticByteSize(ShapedType type);
bool isProfileCandidate(Operation& operation);
bool isDirectParallelWorkerCandidate(Operation& operation);
bool isViewLike(Operation& operation);
bool isStaticIdentityMemRef(MemRefType type);
bool isWholeBufferView(Operation& operation);
std::optional<Operation*> traceWholeAllocation(
  Value value, llvm::SmallVectorImpl<Value>& aliases);
bool hasFullAccessMap(linalg::LinalgOp operation, OpOperand* operand);

// ── 分析 ──────────────────────────────────────────────────────────────
SiteAnalysis analyzeProfileSites(ModuleOp module, func::FuncOp function);

// ── 各类插桩器（各自 TU）────────────────────────────────────────────────
// idValue 由编排循环在 before-op 处**建一次**再传给各插桩器共享：原实现
// 就是这么做的，各自建一份会多出 arith.constant 重复、改产物 IR。
void instrumentMaterializedReadSites(const SiteContext& context,
                                     Operation* operation,
                                     const SiteAnalysis& analysis);
void instrumentOperationTimerBegin(const SiteContext& context,
                                   Operation* operation,
                                   Value idValue,
                                   EventCategory category,
                                   bool workerTimed,
                                   bool timed);
void instrumentAllocationSite(const SiteContext& context,
                              Operation* operation,
                              Value idValue);
void instrumentCopySites(const SiteContext& context,
                         Operation* operation,
                         Value idValue);
void instrumentMovementSite(const SiteContext& context,
                            Operation* operation,
                            Value idValue);
void instrumentMaterializedWriteSites(const SiteContext& context,
                                      Operation* operation,
                                      const SiteAnalysis& analysis);
void instrumentOperationTimerEnd(const SiteContext& context,
                                 Operation* operation,
                                 std::uint64_t id,
                                 bool workerTimed,
                                 bool timed);
void instrumentDeallocationSites(const SiteContext& context,
                                 func::FuncOp function,
                                 const SiteAnalysis& analysis);
EventCategory eventCategoryFor(Operation& operation);
void instrumentFunctionEntry(const SiteContext& context,
                             func::FuncOp function,
                             std::uint64_t rootId);
void instrumentFunctionExit(const SiteContext& context,
                            func::FuncOp function,
                            std::uint64_t rootId);

}  // namespace mlir::ncnn::sites

#endif  // NCNN_MLIR_TRANSFORMS_INSTRUMENTNCNNPROFILE_INSTRUMENTNCNNSITES_HPP
