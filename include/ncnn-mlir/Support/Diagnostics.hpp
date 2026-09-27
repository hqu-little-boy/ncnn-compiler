// Diagnostics：ncnn 侧的诊断分级。
//
// 背景
//   全库原先 emitWarning 零调用——该降级的情况（预算拒绝、策略回退、
//   worker 站点溯源不明）只能静默或直接报错，用户看不到「为什么没走更快
//   的路径」。本模块补上中间档，并提供 CI 收紧用的升级开关。
//
// 三级
//   1. 静默  —— 常态路径（布局不匹配、动态形状、非常量 RHS），不报；
//   2. 警告  —— 本模块的 emitNcnnWarning，「想走但没走成」的少数路径；
//   3. 错误  —— 升级开关打开时，警告改发 error 并让 pass 失败。
//
// 去重
//   同一 reason 一次编译只报一次，重复抑制并在首条里注明。理由：
//   逐点报警会把 stderr 冲垮（一个小模型就有上百个 matmul）。计数没有丢，
//   plan JSON 里 conv_fallback_reasons / 各 rejected 属性本来就逐个记账。
//
// 开关怎么下沉
//   ncnn-compile 不跑 pass，它逐个 exec ncnn-mlir-opt 子进程。诊断是子进程
//   打的，所以 --warnings-as-errors 用环境变量 NCNN_WARNINGS_AS_ERRORS
//   下沉（ExecuteAndWait 的 env 传 nullopt = 继承）。选环境变量而不是
//   pipeline option：这是**诊断策略**不是代码生成参数，不进 identity 串、
//   不碰 Passes.td / pipeline 组装，零产物风险。
//
// 明确不做
//   * 不为常态拒绝路径报警（会变成噪音）；
//   * 不改变任何产物字节——诊断只走 stderr。
#ifndef NCNN_MLIR_SUPPORT_DIAGNOSTICS_HPP
#define NCNN_MLIR_SUPPORT_DIAGNOSTICS_HPP

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ncnn {

// 进程内开关。默认取环境变量 NCNN_WARNINGS_AS_ERRORS（非空且非 "0" 为开）。
void setWarningsAsErrors(bool enabled);
bool warningsAsErrors();

// 记账 fallback 并发一条诊断。用它替换裸 contract::annotateFallback，
// 这样「想走但没走成」在 stderr 可见（reason 同时是去重键与 plan 字段值）。
void emitNcnnFallbackWarning(Operation* operation, llvm::StringRef reason);

// 发出一条 ncnn 警告：`warning: ncnn: <reason>[: <detail>]`。
// 同一 reason 进程内只发一次。升级开启时改为 `error:` 并记入失败标志。
void emitNcnnWarning(Operation* operation,
                     llvm::StringRef reason,
                     llvm::StringRef detail = {});

// 取走并清零「是否出现过被升级的警告」。pass 在 runOnOperation 末尾调用，
// 为真则 signalPassFailure()。
bool consumeDiagnosticFailure();

}  // namespace mlir::ncnn

#endif  // NCNN_MLIR_SUPPORT_DIAGNOSTICS_HPP
