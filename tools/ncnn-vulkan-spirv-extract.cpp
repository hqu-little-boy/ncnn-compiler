#include <cstring>
#include <fstream>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

namespace {

llvm::cl::opt<std::string> g_input(llvm::cl::Positional,
                                   llvm::cl::desc("GPU binary MLIR"),
                                   llvm::cl::Required);
llvm::cl::opt<std::string> g_output("o",
                                    llvm::cl::desc("Output SPIR-V file"),
                                    llvm::cl::Required);
llvm::cl::opt<std::string> g_binary_symbol(
  "binary-symbol",
  llvm::cl::desc("gpu.binary symbol to extract"),
  llvm::cl::Required);

constexpr unsigned char kSpirvMagic[] = {0x03, 0x02, 0x23, 0x07};

}  // namespace

int main(int argc, char** argv) {
  llvm::InitLLVM init_llvm(argc, argv);
  llvm::cl::ParseCommandLineOptions(
    argc, argv, "Extract embedded Vulkan SPIR-V\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(g_input, &context);
  if (!module) {
    llvm::errs() << "ncnn-vulkan-spirv-extract: failed to parse " << g_input
                 << '\n';
    return 1;
  }

  std::string shader;
  unsigned binary_count = 0;
  bool ambiguous = false;
  module->walk([&](mlir::gpu::BinaryOp binary) {
    if (binary.getSymName() != g_binary_symbol) {
      return;
    }
    ++binary_count;
    for (mlir::Attribute attribute : binary.getObjects()) {
      auto object = mlir::dyn_cast<mlir::gpu::ObjectAttr>(attribute);
      if (!object) {
        continue;
      }
      llvm::StringRef bytes = object.getObject().getValue();
      if (bytes.size() < sizeof(kSpirvMagic) ||
          bytes.size() % sizeof(std::uint32_t) != 0 ||
          std::memcmp(bytes.data(), kSpirvMagic, sizeof(kSpirvMagic)) != 0) {
        continue;
      }
      if (!shader.empty()) {
        ambiguous = true;
        return;
      }
      shader.assign(bytes.data(), bytes.size());
    }
  });
  if (binary_count != 1 || shader.empty() || ambiguous) {
    llvm::errs() << "ncnn-vulkan-spirv-extract: expected exactly one "
                 << g_binary_symbol
                 << " GPU binary containing a SPIR-V object; found "
                 << binary_count << '\n';
    return 1;
  }

  std::ofstream output(g_output, std::ios::binary | std::ios::trunc);
  if (!output || !output.write(shader.data(),
                               static_cast<std::streamsize>(shader.size()))) {
    llvm::errs() << "ncnn-vulkan-spirv-extract: failed to write " << g_output
                 << '\n';
    return 1;
  }
  llvm::outs() << "extracted " << shader.size()
               << " SPIR-V bytes from gpu.binary @" << g_binary_symbol << '\n';
  return 0;
}
