// ncnn-mlir-driver：ncnn 编译器端到端驱动的入口。
//
// 流水线：
//   .param/.bin ──ncnn_graph::Graph::load──▶ 原始计算图（parsed-graph）
//              ──ncnn_importer::import_graph──▶ ncnn 方言 MLIR 模块
//              ──print──▶ ncnn.model 形式的 MLIR 文本产物
//
// 参数解析用 LLVM 自带的 llvm::cl（mlir-opt 同款 CommandLine 库），后续接入
// tosa/linalg/llvm 下降阶段时可平滑复用同一套 option 基础设施。

#include <charconv>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"
#include "ncnn-mlir/Graph/graph.hpp"
#include "ncnn-mlir/Importer/NCNNImporter.hpp"
#include "ncnn-mlir/Support/Precision.hpp"

namespace {

// 编译器驱动自己的 option 归到这个 category；链接 libLLVM 会注册海量
// codegen option，HideUnrelatedOptions 让 --help 只显示我们这一组。
llvm::cl::OptionCategory g_driver_category("ncnn-mlir-driver options");

llvm::cl::opt<std::string> g_input_path(llvm::cl::Positional,
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

llvm::cl::list<std::string> g_input_shapes(
  "input-shape",
  llvm::cl::desc("Input shape override as CxHxW; '?' is a dynamic extent, and "
                 "'*' requests dynamic rank 1..4"),
  llvm::cl::value_desc("shape|*"),
  llvm::cl::ZeroOrMore,
  llvm::cl::cat(g_driver_category));

llvm::cl::list<std::string> g_input_dim_constraints(
  "input-dim-constraint",
  llvm::cl::desc("Dynamic input dimension constraint as "
                 "INPUT:DIM:min=N,multiple=N"),
  llvm::cl::value_desc("constraint"),
  llvm::cl::ZeroOrMore,
  llvm::cl::cat(g_driver_category));

llvm::cl::opt<std::string> g_precision(
  "precision",
  llvm::cl::desc("Precision policy: auto, f32, fp16, bf16, or int8"),
  llvm::cl::init("auto"),
  llvm::cl::cat(g_driver_category));
llvm::cl::opt<std::string> g_fp16_accumulator(
  "fp16-accumulator",
  llvm::cl::desc("FP16 convolution accumulator: f16 or f32"),
  llvm::cl::init("f16"),
  llvm::cl::cat(g_driver_category));
llvm::cl::opt<bool> g_allow_fallback(
  "allow-fallback",
  llvm::cl::desc("Allow unsupported FP16 arithmetic to use FP32 accumulation"),
  llvm::cl::cat(g_driver_category));

llvm::cl::opt<std::string> g_target_triple("target-triple",
                                           llvm::cl::init(""),
                                           llvm::cl::cat(g_driver_category));
llvm::cl::opt<std::string> g_march("march",
                                   llvm::cl::init(""),
                                   llvm::cl::cat(g_driver_category));
llvm::cl::opt<std::string> g_mcpu("mcpu",
                                  llvm::cl::init(""),
                                  llvm::cl::cat(g_driver_category));
llvm::cl::list<std::string> g_target_features(
  "target-feature",
  llvm::cl::desc("Target feature used for precision capability checks"),
  llvm::cl::ZeroOrMore,
  llvm::cl::cat(g_driver_category));

// 产物阶段。后续 tosa/linalg/llvm/library 等阶段接入这同一个枚举即可。
enum class EmitStage { ParsedGraph, Mlir };

llvm::cl::opt<EmitStage> g_emit_stage(
  "emit",
  llvm::cl::desc("Select the stage to emit:"),
  llvm::cl::init(EmitStage::Mlir),
  llvm::cl::values(clEnumValN(EmitStage::ParsedGraph,
                              "parsed-graph",
                              "Raw parsed ncnn graph (param + bound weights)"),
                   clEnumValN(EmitStage::Mlir,
                              "mlir",
                              "MLIR module in the ncnn dialect (default)")),
  llvm::cl::cat(g_driver_category));

llvm::cl::opt<bool> g_verify(
  "verify",
  llvm::cl::desc("Re-verify the imported MLIR module (default: on)"),
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

std::optional<std::vector<std::int64_t>> parse_input_shape(
  std::string_view text) {
  if (text.empty()) {
    return std::vector<std::int64_t>{};
  }
  std::vector<std::int64_t> shape;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    std::size_t end = text.find_first_of("xX", begin);
    std::string_view token = text.substr(begin, end - begin);
    std::int64_t value = 0;
    if (token == "?") {
      value = ncnn_importer::kDynamicExtent;
    } else {
      auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value);
      if (token.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != token.data() + token.size() || value <= 0) {
        return std::nullopt;
      }
    }
    shape.push_back(value);
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  if (shape.size() != 3) {
    return std::nullopt;
  }
  return shape;
}

std::optional<ncnn_importer::InputDimConstraint> parse_input_dim_constraint(
  std::string_view text) {
  auto parseInteger = [](std::string_view token, auto& value) {
    auto parsed =
      std::from_chars(token.data(), token.data() + token.size(), value);
    return !token.empty() && parsed.ec == std::errc{} &&
           parsed.ptr == token.data() + token.size();
  };
  const std::size_t firstColon = text.find(':');
  const std::size_t secondColon = text.find(':', firstColon + 1);
  if (firstColon == std::string_view::npos ||
      secondColon == std::string_view::npos) {
    return std::nullopt;
  }
  ncnn_importer::InputDimConstraint result{};
  if (!parseInteger(text.substr(0, firstColon), result.input) ||
      !parseInteger(text.substr(firstColon + 1, secondColon - firstColon - 1),
                    result.dimension)) {
    return std::nullopt;
  }
  bool sawMinimum = false;
  bool sawMultiple = false;
  std::string_view fields = text.substr(secondColon + 1);
  while (!fields.empty()) {
    const std::size_t comma = fields.find(',');
    const std::string_view field = fields.substr(0, comma);
    const std::size_t equals = field.find('=');
    if (equals == std::string_view::npos) {
      return std::nullopt;
    }
    const std::string_view name = field.substr(0, equals);
    const std::string_view value = field.substr(equals + 1);
    if (name == "min" && !sawMinimum) {
      sawMinimum = parseInteger(value, result.minimum);
      if (!sawMinimum || result.minimum <= 0) {
        return std::nullopt;
      }
    } else if (name == "multiple" && !sawMultiple) {
      sawMultiple = parseInteger(value, result.multiple_of);
      if (!sawMultiple || result.multiple_of <= 0) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    if (comma == std::string_view::npos) {
      break;
    }
    fields.remove_prefix(comma + 1);
  }
  return sawMinimum && sawMultiple
           ? std::optional<ncnn_importer::InputDimConstraint>(result)
           : std::nullopt;
}

}  // namespace

int main(int argc, char** argv) {
  llvm::cl::HideUnrelatedOptions(g_driver_category);
  llvm::cl::ParseCommandLineOptions(
    argc,
    argv,
    "ncnn-mlir-driver -- compile ncnn .param/.bin toward MLIR/native code\n");

  auto precision = ncnn_mlir::parse_precision_mode(g_precision);
  if (!precision) {
    llvm::errs() << "error: " << precision.error() << "\n";
    return 1;
  }
  auto accumulator =
    ncnn_mlir::parse_fp16_accumulator_mode(g_fp16_accumulator);
  if (!accumulator) {
    llvm::errs() << "error: " << accumulator.error() << "\n";
    return 1;
  }
  ncnn_mlir::TargetSpec target{
    .triple = g_target_triple,
    .march = g_march,
    .mcpu = g_mcpu,
    .features = {g_target_features.begin(), g_target_features.end()}};
  auto policy = ncnn_mlir::resolve_precision_policy(
    *precision, *accumulator, g_allow_fallback, target);
  if (!policy) {
    llvm::errs() << "error: " << policy.error() << "\n";
    return 1;
  }
  if (policy->used_fallback) {
    llvm::errs() << "warning: FP16 arithmetic unavailable; using FP32 "
                    "accumulation because --allow-fallback was specified\n";
  }

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

  mlir::DialectRegistry registry;
  registry.insert<mlir::ncnn::NCNNDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  ncnn_importer::ImportOptions import_options;
  import_options.precision = *policy;
  for (const std::string& input_shape : g_input_shapes) {
    if (input_shape == "*") {
      if (g_input_shapes.size() != 1) {
        llvm::errs() << "error: dynamic-rank '*' must be the only "
                        "--input-shape\n";
        return 1;
      }
      import_options.dynamic_rank = true;
      continue;
    }
    auto parsed_shape = parse_input_shape(input_shape);
    if (!parsed_shape || parsed_shape->empty()) {
      llvm::errs() << "error: --input-shape must be CxHxW; each extent must "
                      "be positive or '?'\n";
      return 1;
    }
    import_options.input_shapes.push_back(std::move(*parsed_shape));
  }
  for (const std::string& constraint : g_input_dim_constraints) {
    auto parsed = parse_input_dim_constraint(constraint);
    if (!parsed) {
      llvm::errs() << "error: --input-dim-constraint must be "
                      "INPUT:DIM:min=N,multiple=N with positive values\n";
      return 1;
    }
    import_options.input_dim_constraints.push_back(*parsed);
  }
  auto imported =
    ncnn_importer::import_graph(*decoded, context, import_options);
  if (!imported.has_value()) {
    llvm::errs() << "error: failed to import graph: "
                 << imported.error().to_string() << "\n";
    return 1;
  }

  if (g_verify) {
    if (mlir::failed(mlir::verify(imported->get().getOperation()))) {
      llvm::errs() << "error: MLIR module verification failed\n";
      return 1;
    }
  }

  std::string rendered;
  llvm::raw_string_ostream stream(rendered);
  imported->get().print(stream);
  stream.flush();
  rendered += "\n";
  if (!write_output(g_output_path, rendered)) {
    return 1;
  }
  return 0;
}
