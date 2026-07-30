// ncnn-opt：解析含 ncnn 方言的 MLIR，校验后原样打印。
// 用于方言的 round-trip 测试（parse -> verify -> print）。

#include <string>
#include <utility>

#include "RegisterNCNNDialects.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

int main(int argc, char** argv) {
  llvm::InitLLVM initializer(argc, argv);

  llvm::cl::opt<std::string> g_input(
    llvm::cl::Positional,
    llvm::cl::desc("<input .mlir file, '-' for stdin>"),
    llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "ncnn-mlir-opt\n");

  mlir::DialectRegistry registry;
  ncnn_mlir::register_all_dialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string errorMessage;
  auto input = mlir::openInputFile(g_input, &errorMessage);
  if (input == nullptr) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);

  mlir::ParserConfig config(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
    mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, config);
  if (!module) {
    llvm::errs() << "error: failed to parse input\n";
    return 1;
  }
  if (mlir::failed(mlir::verify(*module))) {
    llvm::errs() << "error: module verification failed\n";
    return 1;
  }

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
