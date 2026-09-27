// Diagnostics 实现。设计与纪律见 ncnn-mlir/Support/Diagnostics.hpp。

#include "ncnn-mlir/Support/Diagnostics.hpp"

#include <cstdlib>
#include <string>

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {
namespace {

bool detectWarningsAsErrors() {
  const char* value = std::getenv("NCNN_WARNINGS_AS_ERRORS");
  return value != nullptr && *value != '\0' && std::string(value) != "0";
}

bool g_warnings_as_errors = detectWarningsAsErrors();
bool g_diagnostic_failure = false;
llvm::StringSet<> g_reported_reasons;

std::string formatMessage(llvm::StringRef reason, llvm::StringRef detail) {
  std::string message = "ncnn: ";
  message += reason;
  if (!detail.empty()) {
    message += ": ";
    message += detail;
  }
  return message;
}

}  // namespace

void setWarningsAsErrors(bool enabled) {
  g_warnings_as_errors = enabled;
}

bool warningsAsErrors() {
  return g_warnings_as_errors;
}

void emitNcnnWarning(Operation* operation,
                     llvm::StringRef reason,
                     llvm::StringRef detail) {
  if (!g_reported_reasons.insert(reason).second) {
    return;
  }
  std::string message = formatMessage(reason, detail);
  // 按 location 发而不是 op->emitWarning：后者会让 SourceMgr 多打一行
  // "see current operation:" 并把整个算子（含 20 个属性）整段吐出来。
  // 上下文改由 detail 承载（调用方传算子名），一行就够。
  if (g_warnings_as_errors) {
    g_diagnostic_failure = true;
    (void)mlir::emitError(operation->getLoc(), message);
    return;
  }
  (void)mlir::emitWarning(operation->getLoc(), message);
}

void emitNcnnFallbackWarning(Operation* operation, llvm::StringRef reason) {
  contract::annotateFallback(operation, reason);
  emitNcnnWarning(operation, reason, operation->getName().getStringRef());
}

bool consumeDiagnosticFailure() {
  const bool failure = g_diagnostic_failure;
  g_diagnostic_failure = false;
  return failure;
}

}  // namespace mlir::ncnn
