// ncnn-mlir-driver：ncnn 编译器端到端驱动的入口。
//
// 当前阶段（M1）流水线：
//   .param/.bin ──ncnn_graph::Graph::load──▶ 原始计算图
//              ──ncnn_frontend::import_graph──▶ 类型化前端 IR
//              ──ncnn_frontend::verify_graph──▶ 校验
//              ──dump──▶ 文本产物
//
// 参数解析用 LLVM 自带的 llvm::cl（mlir-opt 同款 CommandLine 库），
// 后续接入 MLIR pass 管线时可平滑复用同一套 option 基础设施。

#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "ncnn_frontend/importer.hpp"
#include "ncnn_frontend/verifier.hpp"
#include "ncnn_graph/graph.hpp"

namespace {

// 编译器驱动自己的 option 归到这个 category；链接 libLLVM 会注册海量
// codegen option，HideUnrelatedOptions 让 --help 只显示我们这一组。
llvm::cl::OptionCategory g_driver_category("ncnn-mlir-driver options");

llvm::cl::opt<std::string> g_input_path(
  llvm::cl::Positional,
  llvm::cl::desc("<input .param file>"),
  llvm::cl::Required,
  llvm::cl::cat(g_driver_category));

llvm::cl::opt<std::string> g_bin_path(
  "bin",
  llvm::cl::desc("Weight file (.bin). Defaults to <input> with .param replaced "
                 "by .bin"),
  llvm::cl::value_desc("path"),
  llvm::cl::init(""),
  llvm::cl::cat(g_driver_category));

llvm::cl::opt<std::string> g_output_path(
  "o",
  llvm::cl::desc("Output file. '-' writes to stdout (default)"),
  llvm::cl::value_desc("path"),
  llvm::cl::init("-"),
  llvm::cl::cat(g_driver_category));

// 产物阶段。后续 tosa/linalg/llvm/library 等阶段接入这同一个枚举即可。
enum class EmitStage { ParsedGraph, NcnnIr };

llvm::cl::opt<EmitStage> g_emit_stage(
  "emit",
  llvm::cl::desc("Select the stage to emit:"),
  llvm::cl::init(EmitStage::NcnnIr),
  llvm::cl::values(
    clEnumValN(EmitStage::ParsedGraph,
               "parsed-graph",
               "Raw parsed ncnn graph (param + bound weights)"),
    clEnumValN(EmitStage::NcnnIr,
               "ncnn-ir",
               "Typed ncnn frontend IR (default)")),
  llvm::cl::cat(g_driver_category));

llvm::cl::opt<bool> g_verify(
  "verify",
  llvm::cl::desc("Run the IR verifier on the typed ncnn IR (default: on)"),
  llvm::cl::init(true),
  llvm::cl::cat(g_driver_category));

// 由 .param 输入路径推导默认 .bin 路径：末尾 .param 换成 .bin，否则追加 .bin。
std::string derive_bin_path(std::string_view param_path) {
  constexpr std::string_view kParamSuffix = ".param";
  std::string result(param_path);
  if (result.ends_with(kParamSuffix)) {
    result.resize(result.size() - kParamSuffix.size());
  }
  result += ".bin";
  return result;
}

// 把产物写到 -o 指定的文件或 stdout。返回是否成功。
bool write_output(std::string_view output_path, std::string_view content) {
  if (output_path == "-") {
    std::cout << content;
    std::cout.flush();
    return static_cast<bool>(std::cout);
  }
  std::ofstream file(std::string(output_path), std::ios::binary);
  if (!file) {
    llvm::errs() << "error: cannot open output file '" << output_path << "'\n";
    return false;
  }
  file << content;
  file.flush();
  if (!file) {
    llvm::errs() << "error: failed while writing '" << output_path << "'\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(g_driver_category);
  llvm::cl::ParseCommandLineOptions(
    argc,
    argv,
    "ncnn-mlir-driver -- compile ncnn .param/.bin toward MLIR/native code\n");

  const std::string bin_path =
    g_bin_path.empty() ? derive_bin_path(g_input_path) : g_bin_path.getValue();

  auto decoded = ncnn_graph::Graph::load(g_input_path, bin_path);
  if (!decoded.has_value()) {
    llvm::errs() << "error: failed to load model: " << decoded.error() << "\n";
    return 1;
  }

  if (g_emit_stage == EmitStage::ParsedGraph) {
    if (!write_output(g_output_path, decoded->dump())) {
      return 1;
    }
    return 0;
  }

  auto imported = ncnn_frontend::import_graph(*decoded);
  if (!imported.has_value()) {
    llvm::errs() << "error: failed to import graph: "
                 << imported.error().to_string() << "\n";
    return 1;
  }

  if (g_verify) {
    auto verified = ncnn_frontend::verify_graph(*imported);
    if (!verified.has_value()) {
      llvm::errs() << "error: IR verification failed: " << verified.error()
                   << "\n";
      return 1;
    }
  }

  if (!write_output(g_output_path, imported->dump())) {
    return 1;
  }
  return 0;
}
