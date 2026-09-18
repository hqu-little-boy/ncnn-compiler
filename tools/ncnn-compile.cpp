#include <algorithm>
#include <cctype>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileSystem/UniqueID.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "ncnn-mlir/Support/Int8Target.hpp"
#include "ncnn-mlir/Support/Precision.hpp"
#include "ncnn-mlir/Support/ShapeProgram.hpp"
#include "ncnn-mlir/Support/TargetVectorInfo.hpp"

namespace {

namespace fs = std::filesystem;

int g_executable_anchor;

llvm::cl::OptionCategory g_category("ncnn-compile options");

llvm::cl::opt<std::string> g_input(llvm::cl::Positional,
                                   llvm::cl::desc("<input .param file>"),
                                   llvm::cl::init(""),
                                   llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_param("param",
                                   llvm::cl::init(""),
                                   llvm::cl::Hidden,
                                   llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_bin(
  "bin",
  llvm::cl::desc("Weight file (defaults to <input>.bin)"),
  llvm::cl::value_desc("path"),
  llvm::cl::init(""),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_model_name(
  "model-name",
  llvm::cl::desc("Exported model function name"),
  llvm::cl::value_desc("name"),
  llvm::cl::init(""),
  llvm::cl::cat(g_category));
llvm::cl::list<std::string> g_input_shapes(
  "input-shape",
  llvm::cl::desc("Input shape override as CxHxW; '?' is a dynamic extent. "
                 "Repeat once per Input with omitted dimensions"),
  llvm::cl::value_desc("CxHxW"),
  llvm::cl::ZeroOrMore,
  llvm::cl::cat(g_category));
llvm::cl::list<std::string> g_input_dim_constraints(
  "input-dim-constraint",
  llvm::cl::desc("Dynamic input dimension constraint as "
                 "INPUT:DIM:min=N,multiple=N; repeat as needed"),
  llvm::cl::value_desc("constraint"),
  llvm::cl::ZeroOrMore,
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_precision(
  "precision",
  llvm::cl::desc("Precision policy: auto, f32, fp16, bf16, or int8"),
  llvm::cl::init("auto"),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_fp16_accumulator(
  "fp16-accumulator",
  llvm::cl::desc("FP16 convolution accumulator: f16 or f32"),
  llvm::cl::init("f16"),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_allow_fallback(
  "allow-fallback",
  llvm::cl::desc("Allow unsupported FP16 arithmetic to use FP32 accumulation"),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_output_dir("output-dir",
                                        llvm::cl::desc("Output directory"),
                                        llvm::cl::value_desc("path"),
                                        llvm::cl::init(""),
                                        llvm::cl::cat(g_category));
llvm::cl::alias g_output_alias("o",
                               llvm::cl::aliasopt(g_output_dir),
                               llvm::cl::cat(g_category));
llvm::cl::list<std::string> g_emit(
  "emit",
  llvm::cl::desc(
    "Keep compiler stages (repeat or comma-separate): "
    "ncnn,tosa,linalg,memref,capi,llvm,llvm-ir,object,assembly,all"),
  llvm::cl::CommaSeparated,
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_optimization(
  "O",
  llvm::cl::desc("Optimization level (0, 1, 2, or 3)"),
  llvm::cl::Prefix,
  llvm::cl::init("3"),
  llvm::cl::cat(g_category));
llvm::cl::opt<unsigned> g_threads(
  "threads",
  llvm::cl::desc(
    "OpenMP worker threads (0 uses all runtime-available CPUs; 1 disables)"),
  llvm::cl::init(0),
  llvm::cl::cat(g_category));
llvm::cl::opt<unsigned> g_vector_width(
  "vector-width",
  llvm::cl::desc("Preferred SIMD vector width in bits (0 disables preference)"),
  llvm::cl::init(256),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_vector_mode(
  "vector-mode",
  llvm::cl::desc(
    "MLIR-level vectorization mode: off, auto, fixed-width, or scalable"),
  llvm::cl::init("off"),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_int8_kernel(
  "int8-kernel",
  llvm::cl::desc(
    "INT8 row-dot kernel policy: portable (default), auto (currently keeps "
    "portable pending performance validation), or vnni (explicit opt-in)"),
  llvm::cl::init("portable"),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_int8_depthwise(
  "int8-depthwise",
  llvm::cl::desc("Enable static INT8 depthwise SIMD"),
  llvm::cl::init(false),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_int8_cast_chain(
  "int8-cast-chain",
  llvm::cl::desc("Fuse proven static cast map consumers"),
  llvm::cl::init(false),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_conv_strategy(
  "conv-strategy",
  llvm::cl::desc(
    "Convolution operator-shape strategy (A1): auto, gemm, conv, winograd"),
  llvm::cl::init("auto"),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_vector_math(
  "vector-math",
  llvm::cl::desc(
    "Vector math backend for transcendental calls: auto, libmvec, sleef, "
    "or none"),
  llvm::cl::init("auto"),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_sleef_path(
  "sleef-path",
  llvm::cl::desc("Directory holding vendored SLEEF static archives "
                 "(libsleefdispatch.a + libsleef.a)"),
  llvm::cl::init(""),
  llvm::cl::Hidden,
  llvm::cl::cat(g_category));
llvm::cl::opt<int64_t> g_conv_gemm_l2_bytes(
  "conv-gemm-l2-bytes",
  llvm::cl::desc(
    "L2 cache byte budget for the convolution prefer-GEMM heuristic "
    "(ncnn prefer_sgemm)"),
  llvm::cl::init(524288),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_target_triple(
  "target-triple",
  llvm::cl::desc("Target triple (64-bit Linux ELF only)"),
  llvm::cl::init(""),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_march("march",
                                   llvm::cl::init(""),
                                   llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_mcpu("mcpu",
                                  llvm::cl::init(""),
                                  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_mtune("mtune",
                                   llvm::cl::init(""),
                                   llvm::cl::cat(g_category));
llvm::cl::list<std::string> g_target_features(
  "target-feature",
  llvm::cl::desc("Clang target feature"),
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_sysroot("sysroot",
                                     llvm::cl::init(""),
                                     llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_debug("g",
                            llvm::cl::desc("Generate debug information"),
                            llvm::cl::cat(g_category));
llvm::cl::alias g_debug_alias("debug-info",
                              llvm::cl::aliasopt(g_debug),
                              llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_verbose("v",
                              llvm::cl::desc("Print executed commands"),
                              llvm::cl::cat(g_category));
llvm::cl::list<std::string> g_clang_args(
  "clang-arg",
  llvm::cl::desc("Additional clang compile argument"),
  llvm::cl::cat(g_category));
llvm::cl::list<std::string> g_linker_args(
  "linker-arg",
  llvm::cl::desc("Additional clang link argument"),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_emit_manifest(
  "emit-manifest",
  llvm::cl::desc("Emit the JSON ABI manifest"),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_emit_execution_plan(
  "emit-execution-plan",
  llvm::cl::desc("Emit a deterministic JSON execution-plan manifest"),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_verify_execution(
  "verify-execution",
  llvm::cl::desc("Build and run an ABI smoke harness"),
  llvm::cl::cat(g_category));
llvm::cl::opt<bool> g_profile(
  "profile",
  llvm::cl::desc("Instrument the generated library for diagnostic profiling"),
  llvm::cl::cat(g_category));

llvm::cl::opt<std::string> g_driver("driver",
                                    llvm::cl::init(""),
                                    llvm::cl::Hidden,
                                    llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_opt("opt",
                                 llvm::cl::init(""),
                                 llvm::cl::Hidden,
                                 llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_translate("translate",
                                       llvm::cl::init(""),
                                       llvm::cl::Hidden,
                                       llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_clang("clang",
                                   llvm::cl::init(""),
                                   llvm::cl::Hidden,
                                   llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_nm("nm",
                                llvm::cl::init(""),
                                llvm::cl::Hidden,
                                llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_readelf("readelf",
                                     llvm::cl::init(""),
                                     llvm::cl::Hidden,
                                     llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_llvm_as("llvm-as",
                                     llvm::cl::init(""),
                                     llvm::cl::Hidden,
                                     llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_expected_undefined("expected-undefined",
                                                llvm::cl::init(""),
                                                llvm::cl::Hidden,
                                                llvm::cl::cat(g_category));

struct Argument {
  struct DimensionConstraint {
    std::uint32_t dimension;
    std::int64_t minimum;
    std::int64_t multiple_of;
  };

  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> maximum_shape;
  std::string element_type;
  std::uint32_t dynamic_dim_mask;
  bool shape_depends_on_data;
  std::int32_t shape_source_input;
  std::int32_t shape_program_version;
  bool dynamic_rank;
  std::uint32_t rank_min;
  std::uint32_t rank_max;
  std::vector<DimensionConstraint> dimension_constraints;
  std::vector<std::vector<std::int64_t>> shape_program;
};

struct Manifest {
  struct InputDimensionRelation {
    std::uint32_t lhs_input;
    std::uint32_t lhs_dimension;
    std::uint32_t rhs_input;
    std::uint32_t rhs_dimension;
    std::int64_t offset;
  };

  struct PrecisionPolicy {
    std::string storage;
    std::string complex_math;
    std::string complex_accumulator;
    std::optional<std::string> fp16_accumulator;
    bool fallback;
  };

  struct Target {
    std::string triple;
    std::string cpu;
    std::string march;
    std::string tune;
    std::vector<std::string> features;
    std::string execution_profile;
  };

  std::string function;
  std::vector<Argument> inputs;
  std::vector<Argument> outputs;
  std::vector<InputDimensionRelation> input_dimension_relations;
  std::optional<PrecisionPolicy> precision_policy;
  std::optional<Target> target;
  // 多线程产物是否依赖 OpenMP 运行时（libomp 探测失败回退 threads=1 时
  // 为 false；探测成功为 true）。旧 manifest 无此字段。
  std::optional<bool> openmp;
  // 实际生效的向量数学后端（none/libmvec/sleef）。旧 manifest 无此字段。
  std::optional<std::string> vector_math;
};

class ScopedDirectory {
 public:
  explicit ScopedDirectory(fs::path path)
    : path_(std::move(path)), remove_(true) {}
  ~ScopedDirectory() {
    if (!remove_) {
      return;
    }
    std::error_code error;
    fs::remove_all(path_, error);
    if (error) {
      llvm::errs() << "ncnn-compile: warning: cannot clean up '"
                   << path_.string() << "': " << error.message() << '\n';
    }
  }
  const fs::path& path() const { return path_; }
  void release() { remove_ = false; }

 private:
  fs::path path_;
  bool remove_;
};

// 定位 vendored SLEEF 静态档案（单档案，dispatcher 内置运行时 ISA 自分
// 发）。顺序：--sleef-path > 可执行文件相对 ../lib > 构建期注入的构建树
// 路径。
bool locate_sleef_archive(fs::path& archive) {
  std::vector<fs::path> candidates;
  if (!g_sleef_path.empty()) {
    candidates.emplace_back(g_sleef_path.getValue());
  }
  std::error_code error;
  const fs::path executable = fs::read_symlink("/proc/self/exe", error);
  if (!error) {
    candidates.push_back(executable.parent_path() / ".." / "lib");
    candidates.push_back(executable.parent_path());
  }
#ifdef NCNN_SLEEF_ARCHIVE_DIR
  candidates.emplace_back(NCNN_SLEEF_ARCHIVE_DIR);
#endif
  for (const fs::path& directory : candidates) {
    const fs::path candidate = directory / "libsleef.a";
    if (fs::is_regular_file(candidate)) {
      archive = candidate;
      return true;
    }
  }
  return false;
}

int fail(llvm::Twine message) {
  llvm::errs() << "ncnn-compile: error: " << message << '\n';
  return 1;
}

std::string derive_bin_path(std::string path) {
  constexpr std::string_view suffix = ".param";
  if (path.ends_with(suffix)) {
    path.resize(path.size() - suffix.size());
  }
  return path + ".bin";
}

std::string c_identifier(std::string_view name) {
  std::string result;
  result.reserve(name.size());
  bool previous_was_replacement = false;
  for (unsigned char character : name) {
    const bool ascii_alphanumeric = (character >= 'a' && character <= 'z') ||
                                    (character >= 'A' && character <= 'Z') ||
                                    (character >= '0' && character <= '9');
    if (ascii_alphanumeric || character == '_') {
      result += static_cast<char>(character);
      previous_was_replacement = false;
    } else if (!previous_was_replacement) {
      result += '_';
      previous_was_replacement = true;
    }
  }
  static const std::set<std::string> keywords = {"alignas",
                                                 "alignof",
                                                 "auto",
                                                 "bool",
                                                 "break",
                                                 "case",
                                                 "char",
                                                 "const",
                                                 "constexpr",
                                                 "continue",
                                                 "default",
                                                 "do",
                                                 "double",
                                                 "else",
                                                 "enum",
                                                 "extern",
                                                 "false",
                                                 "float",
                                                 "for",
                                                 "goto",
                                                 "if",
                                                 "inline",
                                                 "int",
                                                 "long",
                                                 "main",
                                                 "nullptr",
                                                 "register",
                                                 "restrict",
                                                 "return",
                                                 "short",
                                                 "signed",
                                                 "sizeof",
                                                 "static",
                                                 "static_assert",
                                                 "struct",
                                                 "switch",
                                                 "thread_local",
                                                 "true",
                                                 "typedef",
                                                 "typeof",
                                                 "typeof_unqual",
                                                 "union",
                                                 "unsigned",
                                                 "void",
                                                 "volatile",
                                                 "while",
                                                 "_Alignas",
                                                 "_Alignof",
                                                 "_Atomic",
                                                 "_BitInt",
                                                 "_Bool",
                                                 "_Complex",
                                                 "_Decimal128",
                                                 "_Decimal32",
                                                 "_Decimal64",
                                                 "_Generic",
                                                 "_Imaginary",
                                                 "_Noreturn",
                                                 "_Static_assert",
                                                 "_Thread_local"};
  if (result.empty() || (result[0] >= '0' && result[0] <= '9') ||
      result[0] == '_' || keywords.contains(result)) {
    result.insert(0, "ncnn_");
  }
  return result;
}

using ToolResult = std::expected<std::optional<std::string>, std::string>;

[[nodiscard]] ToolResult find_tool(
  const fs::path& executable_dir,
  std::string_view explicit_path,
  std::initializer_list<std::string_view> nearby_names,
  std::initializer_list<std::string_view> path_names) {
  auto canonicalize = [](const fs::path& path) -> ToolResult {
    std::error_code error;
    fs::path canonical = fs::weakly_canonical(path, error);
    if (error) {
      return std::unexpected(
        std::format("cannot canonicalize compiler tool "
                    "'{}': {}",
                    path.string(),
                    error.message()));
    }
    return canonical.string();
  };
  if (!explicit_path.empty()) {
    return canonicalize(explicit_path);
  }
  for (std::string_view name : nearby_names) {
    for (const fs::path& directory :
         {executable_dir, executable_dir / "../bin"}) {
      fs::path candidate = directory / name;
      std::error_code error;
      if (fs::is_regular_file(candidate, error)) {
        return canonicalize(candidate);
      }
      if (error && error != std::errc::no_such_file_or_directory) {
        return std::unexpected(
          std::format("cannot inspect compiler tool "
                      "'{}': {}",
                      candidate.string(),
                      error.message()));
      }
    }
  }
  for (std::string_view name : path_names) {
    auto program = llvm::sys::findProgramByName(name);
    if (program) {
      return canonicalize(*program);
    }
  }
  return std::nullopt;
}

void print_command(const std::vector<std::string>& command) {
  for (std::size_t index = 0; index < command.size(); ++index) {
    if (index != 0) {
      llvm::errs() << ' ';
    }
    llvm::sys::printArg(llvm::errs(), command[index], true);
  }
  llvm::errs() << '\n';
}

int run(const std::vector<std::string>& command,
        std::optional<fs::path> stdout_path = std::nullopt) {
  if (g_verbose) {
    print_command(command);
  }
  std::vector<llvm::StringRef> arguments;
  arguments.reserve(command.size());
  for (const std::string& argument : command) {
    arguments.emplace_back(argument);
  }
  std::vector<std::optional<llvm::StringRef>> redirects;
  std::string stdout_storage;
  if (stdout_path) {
    stdout_storage = stdout_path->string();
    redirects = {std::nullopt, llvm::StringRef(stdout_storage), std::nullopt};
  }
  std::string error;
  bool execution_failed = false;
  int status = llvm::sys::ExecuteAndWait(command.front(),
                                         arguments,
                                         std::nullopt,
                                         redirects,
                                         0,
                                         0,
                                         &error,
                                         &execution_failed);
  if (execution_failed || status < 0) {
    llvm::errs() << "ncnn-compile: error: cannot execute '" << command.front()
                 << "': " << error << '\n';
    return status == 0 ? 1 : std::max(1, -status);
  }
  if (status != 0) {
    llvm::errs() << "ncnn-compile: error: command failed with exit code "
                 << status << ": ";
    print_command(command);
  }
  return status;
}

[[nodiscard]] std::expected<void, std::string> write_file(
  const fs::path& path, std::string_view contents) {
  std::error_code error;
  llvm::raw_fd_ostream output(path.string(), error);
  if (error) {
    return std::unexpected(
      std::format("cannot write '{}': {}", path.string(), error.message()));
  }
  output << contents;
  output.flush();
  output.close();
  if (output.has_error()) {
    return std::unexpected(
      std::format("cannot finish writing '{}'", path.string()));
  }
  return {};
}

[[nodiscard]] std::expected<Manifest, std::string> read_manifest(
  const fs::path& path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    return std::unexpected(std::format("cannot read ABI manifest '{}': {}",
                                       path.string(),
                                       buffer.getError().message()));
  }
  auto value = llvm::json::parse((*buffer)->getBuffer());
  if (!value) {
    return std::unexpected(std::format("invalid ABI manifest '{}': {}",
                                       path.string(),
                                       llvm::toString(value.takeError())));
  }
  auto* object = value->getAsObject();
  if (object == nullptr) {
    return std::unexpected("ABI manifest must be a JSON object");
  }
  auto function = object->getString("function");
  if (!function) {
    return std::unexpected("ABI manifest has no function name");
  }
  Manifest manifest{.function = std::string(*function),
                    .inputs = {},
                    .outputs = {},
                    .input_dimension_relations = {},
                    .precision_policy = std::nullopt,
                    .target = std::nullopt};
  if (auto* policy = object->getObject("precision_policy")) {
    auto storage = policy->getString("storage");
    auto complex_math = policy->getString("complex_math");
    auto complex_accumulator = policy->getString("complex_accumulator");
    auto fallback = policy->getBoolean("fallback");
    if (!storage || !complex_math || !complex_accumulator || !fallback) {
      return std::unexpected("ABI manifest has an invalid precision policy");
    }
    auto fp16_accumulator = policy->getString("fp16_accumulator");
    manifest.precision_policy = Manifest::PrecisionPolicy{
      .storage = std::string(*storage),
      .complex_math = std::string(*complex_math),
      .complex_accumulator = std::string(*complex_accumulator),
      .fp16_accumulator = fp16_accumulator ? std::optional<std::string>(
                                               std::string(*fp16_accumulator))
                                           : std::nullopt,
      .fallback = *fallback};
  }
  auto parse_arguments = [&](llvm::StringRef key,
                             std::vector<Argument>& destination) {
    auto* array = object->getArray(key);
    if (array == nullptr) {
      return false;
    }
    for (llvm::json::Value& item : *array) {
      auto* argument_object = item.getAsObject();
      if (argument_object == nullptr) {
        return false;
      }
      auto name = argument_object->getString("name");
      auto* shape = argument_object->getArray("shape");
      auto element_type = argument_object->getString("element_type");
      auto dynamic_dim_mask = argument_object->getInteger("dynamic_dim_mask");
      if (!name || shape == nullptr || !element_type || !dynamic_dim_mask ||
          !std::in_range<std::uint32_t>(*dynamic_dim_mask)) {
        return false;
      }
      Argument argument{
        .name = std::string(*name),
        .shape = {},
        .maximum_shape = {},
        .element_type = std::string(*element_type),
        .dynamic_dim_mask = static_cast<std::uint32_t>(*dynamic_dim_mask),
        .shape_depends_on_data =
          argument_object->getBoolean("shape_depends_on_data").value_or(false),
        .shape_source_input = static_cast<std::int32_t>(
          argument_object->getInteger("shape_source_input").value_or(-1)),
        .shape_program_version = static_cast<std::int32_t>(
          argument_object->getInteger("shape_program_version").value_or(1)),
        .dynamic_rank =
          argument_object->getBoolean("dynamic_rank").value_or(false),
        .rank_min = static_cast<std::uint32_t>(
          argument_object->getInteger("rank_min").value_or(0)),
        .rank_max = static_cast<std::uint32_t>(
          argument_object->getInteger("rank_max").value_or(0)),
        .dimension_constraints = {},
        .shape_program = {}};
      for (llvm::json::Value& dimension : *shape) {
        auto integer = dimension.getAsInteger();
        if (!integer || *integer < -1) {
          return false;
        }
        argument.shape.push_back(*integer);
      }
      if (auto* maximum_shape = argument_object->getArray("maximum_shape")) {
        for (llvm::json::Value& dimension : *maximum_shape) {
          auto integer = dimension.getAsInteger();
          if (!integer || *integer <= 0) {
            return false;
          }
          argument.maximum_shape.push_back(*integer);
        }
      }
      if (auto* constraints =
            argument_object->getArray("dimension_constraints")) {
        std::set<std::uint32_t> constrainedDimensions;
        for (llvm::json::Value& constraintValue : *constraints) {
          auto* constraint = constraintValue.getAsObject();
          if (constraint == nullptr) {
            return false;
          }
          auto dimension = constraint->getInteger("dimension");
          auto minimum = constraint->getInteger("minimum");
          auto multiple = constraint->getInteger("multiple_of");
          if (!dimension || !minimum || !multiple ||
              !std::in_range<std::uint32_t>(*dimension) || *minimum <= 0 ||
              *multiple <= 0 ||
              !constrainedDimensions
                 .insert(static_cast<std::uint32_t>(*dimension))
                 .second) {
            return false;
          }
          argument.dimension_constraints.push_back(
            {.dimension = static_cast<std::uint32_t>(*dimension),
             .minimum = *minimum,
             .multiple_of = *multiple});
        }
      }
      if (auto* programs = argument_object->getArray("shape_program")) {
        for (llvm::json::Value& programValue : *programs) {
          auto* program = programValue.getAsArray();
          if (program == nullptr) {
            return false;
          }
          std::vector<std::int64_t> instructions;
          for (llvm::json::Value& instruction : *program) {
            auto integer = instruction.getAsInteger();
            if (!integer) {
              return false;
            }
            instructions.push_back(*integer);
          }
          argument.shape_program.push_back(std::move(instructions));
        }
      }
      if (argument.shape.size() > 32) {
        return false;
      }
      std::uint32_t expected_mask = 0;
      for (std::size_t index = 0; index < argument.shape.size(); ++index) {
        if (argument.shape[index] == -1) {
          expected_mask |= UINT32_C(1) << index;
        }
      }
      if (argument.dynamic_dim_mask != expected_mask) {
        return false;
      }
      if (std::ranges::any_of(
            argument.dimension_constraints,
            [&](const Argument::DimensionConstraint& constraint) {
              return constraint.dimension >= argument.shape.size() ||
                     argument.shape[constraint.dimension] != -1;
            })) {
        return false;
      }
      if (argument.shape_source_input < -1) {
        return false;
      }
      if (argument.shape_program_version != 1 &&
          argument.shape_program_version != 2) {
        return false;
      }
      if (argument.dynamic_rank &&
          (!argument.shape.empty() || argument.dynamic_dim_mask != 0 ||
           argument.rank_min != 1 || argument.rank_max != 4)) {
        return false;
      }
      destination.push_back(std::move(argument));
    }
    return true;
  };
  if (!parse_arguments("inputs", manifest.inputs) ||
      !parse_arguments("outputs", manifest.outputs)) {
    return std::unexpected("ABI manifest has invalid inputs or outputs");
  }
  if (auto* relations = object->getArray("input_dimension_relations")) {
    for (llvm::json::Value& value : *relations) {
      auto* relation = value.getAsObject();
      if (relation == nullptr) {
        return std::unexpected(
          "ABI manifest has an invalid input dimension relation");
      }
      auto lhs_input = relation->getInteger("lhs_input");
      auto lhs_dimension = relation->getInteger("lhs_dimension");
      auto rhs_input = relation->getInteger("rhs_input");
      auto rhs_dimension = relation->getInteger("rhs_dimension");
      auto offset = relation->getInteger("offset");
      if (!lhs_input || !lhs_dimension || !rhs_input || !rhs_dimension ||
          !offset || !std::in_range<std::uint32_t>(*lhs_input) ||
          !std::in_range<std::uint32_t>(*lhs_dimension) ||
          !std::in_range<std::uint32_t>(*rhs_input) ||
          !std::in_range<std::uint32_t>(*rhs_dimension) ||
          std::cmp_greater_equal(*lhs_input, manifest.inputs.size()) ||
          std::cmp_greater_equal(*rhs_input, manifest.inputs.size()) ||
          std::cmp_greater_equal(*lhs_dimension,
                                 manifest.inputs[*lhs_input].shape.size()) ||
          std::cmp_greater_equal(*rhs_dimension,
                                 manifest.inputs[*rhs_input].shape.size())) {
        return std::unexpected(
          "ABI manifest has an invalid input dimension relation");
      }
      manifest.input_dimension_relations.push_back(
        {.lhs_input = static_cast<std::uint32_t>(*lhs_input),
         .lhs_dimension = static_cast<std::uint32_t>(*lhs_dimension),
         .rhs_input = static_cast<std::uint32_t>(*rhs_input),
         .rhs_dimension = static_cast<std::uint32_t>(*rhs_dimension),
         .offset = *offset});
    }
  }
  for (const Argument& output : manifest.outputs) {
    if (output.shape_source_input >= 0 &&
        static_cast<std::size_t>(output.shape_source_input) >=
          manifest.inputs.size()) {
      return std::unexpected(
        "ABI manifest output has an invalid input shape "
        "source");
    }
    if (output.shape_depends_on_data &&
        (output.maximum_shape.size() != output.shape.size() ||
         output.dynamic_dim_mask == 0)) {
      return std::unexpected(
        "ABI manifest data-dependent output has no finite maximum shape");
    }
    if ((output.dynamic_rank || output.dynamic_dim_mask != 0) &&
        output.shape_program_version == 1 && output.shape_source_input < 0 &&
        !output.shape_depends_on_data) {
      return std::unexpected(
        "ABI manifest dynamic output has no input shape source");
    }
    if (output.shape_program_version == 2 && output.shape_source_input >= 0) {
      return std::unexpected(
        "ABI manifest V2 output must not have an input shape source");
    }
    if (!output.shape_depends_on_data && output.dynamic_dim_mask != 0 &&
        output.shape_program.size() != output.shape.size()) {
      return std::unexpected(
        "ABI manifest dynamic output has an invalid shape program");
    }
    if (output.shape_program_version == 2) {
      llvm::SmallVector<unsigned> input_ranks;
      for (const Argument& input : manifest.inputs) {
        input_ranks.push_back(input.shape.size());
      }
      for (const auto& program : output.shape_program) {
        auto expression = mlir::ncnn::ShapeExpr::deserialize(program);
        if (!expression || !expression->validateInputRanks(input_ranks)) {
          return std::unexpected(
            "ABI manifest dynamic output has an invalid V2 shape program");
        }
      }
    }
  }
  manifest.function = c_identifier(manifest.function);
  std::set<std::string> argument_names;
  for (Argument* argument : [&] {
         std::vector<Argument*> result;
         for (Argument& input : manifest.inputs) {
           result.push_back(&input);
         }
         for (Argument& output : manifest.outputs) {
           result.push_back(&output);
         }
         return result;
       }()) {
    const std::string original = argument->name;
    argument->name = c_identifier(original);
    if (!argument_names.insert(argument->name).second) {
      return std::unexpected(std::format(
        "ABI manifest argument '{}' duplicates sanitized C identifier '{}'",
        original,
        argument->name));
    }
  }
  return manifest;
}

[[nodiscard]] std::string hex_encode(std::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() * 2);
  for (const unsigned char character : value) {
    result.push_back(digits[character >> 4]);
    result.push_back(digits[character & 0x0f]);
  }
  return result;
}

[[nodiscard]] std::expected<std::string, std::string> read_execution_plan_field(
  const fs::path& path, llvm::StringRef field) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    return std::unexpected(std::format("cannot read execution plan '{}': {}",
                                       path.string(),
                                       buffer.getError().message()));
  }
  auto value = llvm::json::parse((*buffer)->getBuffer());
  if (!value) {
    return std::unexpected(std::format("invalid execution plan '{}': {}",
                                       path.string(),
                                       llvm::toString(value.takeError())));
  }
  auto* object = value->getAsObject();
  if (object == nullptr) {
    return std::unexpected("execution plan must be a JSON object");
  }
  auto text = object->getString(field);
  if (!text || text->empty()) {
    return std::unexpected("execution plan has no " + field.str());
  }
  return text->str();
}

[[nodiscard]] std::expected<std::size_t, std::string> element_count(
  const Argument& argument) {
  const auto& shape =
    argument.shape_depends_on_data ? argument.maximum_shape : argument.shape;
  for (std::int64_t dimension : shape) {
    if (dimension < 0) {
      return std::unexpected(std::format(
        "argument '{}' has a dynamic element count", argument.name));
    }
    if (static_cast<std::uint64_t>(dimension) >
        std::numeric_limits<std::size_t>::max()) {
      return std::unexpected(
        std::format("argument '{}' has a dimension that does not fit size_t",
                    argument.name));
    }
    if (dimension == 0) {
      return 0;
    }
  }
  std::size_t count = 1;
  for (std::int64_t dimension : shape) {
    const auto size = static_cast<std::size_t>(dimension);
    if (count > std::numeric_limits<std::size_t>::max() / size) {
      return std::unexpected(std::format(
        "argument '{}' element count overflows size_t", argument.name));
    }
    count *= size;
  }
  return count;
}

[[nodiscard]] std::expected<void, std::string> write_manifest(
  const fs::path& path, const Manifest& manifest) {
  auto arguments = [](const std::vector<Argument>& source) {
    llvm::json::Array result;
    for (const Argument& argument : source) {
      llvm::json::Array shape;
      for (std::int64_t dimension : argument.shape) {
        shape.push_back(dimension);
      }
      llvm::json::Object object;
      object["name"] = argument.name;
      object["shape"] = std::move(shape);
      object["element_type"] = argument.element_type;
      object["dynamic_dim_mask"] = argument.dynamic_dim_mask;
      if (!argument.dimension_constraints.empty()) {
        llvm::json::Array constraints;
        for (const Argument::DimensionConstraint& constraint :
             argument.dimension_constraints) {
          llvm::json::Object constraintObject;
          constraintObject["dimension"] = constraint.dimension;
          constraintObject["minimum"] = constraint.minimum;
          constraintObject["multiple_of"] = constraint.multiple_of;
          constraints.push_back(std::move(constraintObject));
        }
        object["dimension_constraints"] = std::move(constraints);
      }
      if (!argument.maximum_shape.empty()) {
        llvm::json::Array maximum_shape;
        for (std::int64_t dimension : argument.maximum_shape) {
          maximum_shape.push_back(dimension);
        }
        object["maximum_shape"] = std::move(maximum_shape);
      }
      if (argument.dynamic_rank) {
        object["dynamic_rank"] = true;
        object["rank_min"] = argument.rank_min;
        object["rank_max"] = argument.rank_max;
      }
      if (argument.shape_depends_on_data) {
        object["shape_depends_on_data"] = true;
      }
      if (argument.shape_source_input >= 0) {
        object["shape_source_input"] = argument.shape_source_input;
      }
      if (argument.shape_program_version == 2) {
        object["shape_program_version"] = 2;
      }
      if (!argument.shape_program.empty()) {
        llvm::json::Array programs;
        for (const auto& program : argument.shape_program) {
          llvm::json::Array instructions;
          for (std::int64_t instruction : program) {
            instructions.push_back(instruction);
          }
          programs.push_back(std::move(instructions));
        }
        object["shape_program"] = std::move(programs);
      }
      result.push_back(std::move(object));
    }
    return result;
  };
  llvm::json::Object object;
  object["function"] = manifest.function;
  object["inputs"] = arguments(manifest.inputs);
  object["outputs"] = arguments(manifest.outputs);
  if (!manifest.input_dimension_relations.empty()) {
    llvm::json::Array relations;
    for (const Manifest::InputDimensionRelation& relation :
         manifest.input_dimension_relations) {
      llvm::json::Object relationObject;
      relationObject["lhs_input"] = relation.lhs_input;
      relationObject["lhs_dimension"] = relation.lhs_dimension;
      relationObject["rhs_input"] = relation.rhs_input;
      relationObject["rhs_dimension"] = relation.rhs_dimension;
      relationObject["offset"] = relation.offset;
      relations.push_back(std::move(relationObject));
    }
    object["input_dimension_relations"] = std::move(relations);
  }
  if (manifest.precision_policy) {
    llvm::json::Object policy;
    policy["storage"] = manifest.precision_policy->storage;
    policy["complex_math"] = manifest.precision_policy->complex_math;
    policy["complex_accumulator"] =
      manifest.precision_policy->complex_accumulator;
    if (manifest.precision_policy->fp16_accumulator) {
      policy["fp16_accumulator"] = *manifest.precision_policy->fp16_accumulator;
    }
    policy["fallback"] = manifest.precision_policy->fallback;
    object["precision_policy"] = std::move(policy);
  }
  if (manifest.target) {
    llvm::json::Array features;
    for (const std::string& feature : manifest.target->features) {
      features.push_back(feature);
    }
    llvm::json::Object target;
    target["triple"] = manifest.target->triple;
    target["cpu"] = manifest.target->cpu;
    target["march"] = manifest.target->march;
    target["tune"] = manifest.target->tune;
    target["features"] = std::move(features);
    target["execution_profile"] = manifest.target->execution_profile;
    object["target"] = std::move(target);
  }
  if (manifest.openmp) {
    object["openmp"] = *manifest.openmp;
  }
  if (manifest.vector_math) {
    object["vector_math"] = *manifest.vector_math;
  }
  return write_file(
    path, llvm::formatv("{0:2}\n", llvm::json::Value(std::move(object))).str());
}

std::optional<std::string_view> c_type(const Argument& argument) {
  static const std::map<std::string, std::string_view> types = {
    {"f16", "ncnn_float16_t"},
    {"bf16", "ncnn_bfloat16_t"},
    {"f32", "float"},
    {"f64", "double"},
    {"i8", "int8_t"},
    {"i16", "int16_t"},
    {"i32", "int32_t"},
    {"i64", "int64_t"},
    {"ui8", "uint8_t"},
    {"ui16", "uint16_t"},
    {"ui32", "uint32_t"},
    {"ui64", "uint64_t"}};
  auto found = types.find(argument.element_type);
  return found == types.end() ? std::nullopt
                              : std::optional<std::string_view>(found->second);
}

std::string macro_name(const Manifest& manifest,
                       const Argument& argument,
                       std::string_view suffix) {
  std::string result =
    std::format("{}_{}_{}", manifest.function, argument.name, suffix);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return result;
}

[[nodiscard]] std::expected<void, std::string> write_header(
  const fs::path& path, const Manifest& manifest) {
  std::string guard = std::format("NCNN_{}_H", manifest.function);
  std::ranges::transform(guard, guard.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  std::string contents = std::format(
    "#ifndef {}\n#define {}\n\n#include <stdint.h>\n\n"
    "#define NCNN_DYNAMIC_DIM INT64_C(-1)\n"
    "#define NCNN_MAX_RANK 4\n"
    "#define NCNN_STATUS_SUCCESS 0\n"
    "#define NCNN_STATUS_NULL_POINTER 1\n"
    "#define NCNN_STATUS_INVALID_SHAPE 2\n"
    "#define NCNN_STATUS_CONSTRAINT_VIOLATION 3\n"
    "#define NCNN_STATUS_SHAPE_ARITHMETIC_OVERFLOW 4\n"
    "#define NCNN_STATUS_OUTPUT_CAPACITY_INSUFFICIENT 5\n\n"
    "typedef uint16_t ncnn_float16_t;\n"
    "typedef uint16_t ncnn_bfloat16_t;\n\n",
    guard,
    guard);
  contents += std::format(
    "#define {}_INPUT_COUNT {}\n"
    "#define {}_OUTPUT_COUNT {}\n\n",
    [&] {
      std::string name = manifest.function;
      std::ranges::transform(name, name.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
      return name;
    }(),
    manifest.inputs.size(),
    [&] {
      std::string name = manifest.function;
      std::ranges::transform(name, name.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
      });
      return name;
    }(),
    manifest.outputs.size());
  std::string functionMacro = manifest.function;
  std::ranges::transform(
    functionMacro, functionMacro.begin(), [](unsigned char character) {
      return static_cast<char>(std::toupper(character));
    });
  if (!manifest.input_dimension_relations.empty()) {
    contents += std::format("#define {}_INPUT_DIM_RELATION_COUNT {}\n",
                            functionMacro,
                            manifest.input_dimension_relations.size());
    for (auto [index, relation] :
         llvm::enumerate(manifest.input_dimension_relations)) {
      const std::string prefix =
        std::format("{}_INPUT_DIM_RELATION{}", functionMacro, index + 1);
      contents += std::format(
        "#define {}_LHS_INPUT {}\n#define {}_LHS_DIMENSION {}\n"
        "#define {}_RHS_INPUT {}\n#define {}_RHS_DIMENSION {}\n"
        "#define {}_OFFSET INT64_C({})\n",
        prefix,
        relation.lhs_input,
        prefix,
        relation.lhs_dimension,
        prefix,
        relation.rhs_input,
        prefix,
        relation.rhs_dimension,
        prefix,
        relation.offset);
    }
    contents += "\n";
  }
  std::set<std::string> macros;
  for (const Argument* argument : [&] {
         std::vector<const Argument*> result;
         for (const Argument& input : manifest.inputs) {
           result.push_back(&input);
         }
         for (const Argument& output : manifest.outputs) {
           result.push_back(&output);
         }
         return result;
       }()) {
    if (argument->dynamic_rank) {
      contents += std::format("#define {} {}\n#define {} {}\n",
                              macro_name(manifest, *argument, "rank_min"),
                              argument->rank_min,
                              macro_name(manifest, *argument, "rank_max"),
                              argument->rank_max);
      if (std::ranges::any_of(manifest.outputs, [&](const Argument& output) {
            return &output == argument;
          })) {
        contents +=
          std::format("#define {} {}\n",
                      macro_name(manifest, *argument, "shape_depends_on_data"),
                      argument->shape_depends_on_data ? 1 : 0);
      }
      contents += "\n";
      continue;
    }
    const std::string rank_macro = macro_name(manifest, *argument, "rank");
    if (!macros.insert(rank_macro).second) {
      return std::unexpected(
        std::format("argument '{}' duplicates generated ABI macro '{}'",
                    argument->name,
                    rank_macro));
    }
    contents +=
      std::format("#define {} {}\n", rank_macro, argument->shape.size());
    for (auto [index, dimension] : llvm::enumerate(argument->shape)) {
      contents += std::format(
        "#define {} {}\n",
        macro_name(manifest, *argument, std::format("dim{}", index)),
        dimension < 0 ? "NCNN_DYNAMIC_DIM"
                      : std::format("INT64_C({})", dimension));
    }
    contents += std::format("#define {} UINT32_C(0x{:x})\n",
                            macro_name(manifest, *argument, "dynamic_dim_mask"),
                            argument->dynamic_dim_mask);
    for (const Argument::DimensionConstraint& constraint :
         argument->dimension_constraints) {
      contents += std::format(
        "#define {} INT64_C({})\n#define {} INT64_C({})\n",
        macro_name(manifest,
                   *argument,
                   std::format("dim{}_minimum", constraint.dimension)),
        constraint.minimum,
        macro_name(manifest,
                   *argument,
                   std::format("dim{}_multiple_of", constraint.dimension)),
        constraint.multiple_of);
    }
    if (argument->dynamic_dim_mask == 0) {
      auto count = element_count(*argument);
      if (!count) {
        return std::unexpected(count.error());
      }
      contents += std::format("#define {} UINT64_C({})\n",
                              macro_name(manifest, *argument, "elements"),
                              *count);
    }
    if (argument->shape_depends_on_data) {
      for (auto [index, dimension] : llvm::enumerate(argument->maximum_shape)) {
        contents += std::format(
          "#define {} INT64_C({})\n",
          macro_name(manifest, *argument, std::format("max_dim{}", index)),
          dimension);
      }
      auto count = element_count(*argument);
      if (!count) {
        return std::unexpected(count.error());
      }
      contents += std::format("#define {} UINT64_C({})\n",
                              macro_name(manifest, *argument, "max_elements"),
                              *count);
    }
    if (std::ranges::any_of(manifest.outputs, [&](const Argument& output) {
          return &output == argument;
        })) {
      contents +=
        std::format("#define {} {}\n",
                    macro_name(manifest, *argument, "shape_depends_on_data"),
                    argument->shape_depends_on_data ? 1 : 0);
    }
    contents += "\n";
  }
  contents +=
    std::format("\n#ifdef __cplusplus\nextern \"C\" {{\n#endif\n\nint {}(",
                manifest.function);
  bool first = true;
  for (const Argument& input : manifest.inputs) {
    auto type = c_type(input);
    if (!type) {
      return std::unexpected(
        std::format("argument '{}' has unsupported element type '{}'",
                    input.name,
                    input.element_type));
    }
    contents +=
      std::format("{}const {} *{}", first ? "" : ", ", *type, input.name);
    first = false;
    if (input.dynamic_rank) {
      contents += std::format(
        ", const int64_t *{}_shape, uint32_t {}_rank", input.name, input.name);
    } else if (input.dynamic_dim_mask != 0) {
      contents += std::format(", const int64_t {}_shape[{}]",
                              input.name,
                              macro_name(manifest, input, "rank"));
    }
  }
  for (const Argument& output : manifest.outputs) {
    auto type = c_type(output);
    if (!type) {
      return std::unexpected(
        std::format("argument '{}' has unsupported element type '{}'",
                    output.name,
                    output.element_type));
    }
    contents += std::format("{}{} *{}", first ? "" : ", ", *type, output.name);
    first = false;
  }
  for (const Argument& output : manifest.outputs) {
    if (output.shape_depends_on_data ||
        (!output.dynamic_rank && output.dynamic_dim_mask == 0)) {
      continue;
    }
    contents += std::format(", uint64_t {}_capacity", output.name);
  }
  for (const Argument& output : manifest.outputs) {
    if (!output.shape_depends_on_data) {
      continue;
    }
    contents += std::format(
      ", int64_t *{}_shape, "
      "uint32_t {}_shape_capacity, "
      "uint32_t *{}_rank",
      output.name,
      output.name,
      output.name);
  }
  contents += ");\n";
  const bool has_dynamic_output =
    std::ranges::any_of(manifest.outputs, [](const Argument& output) {
      return (output.dynamic_rank || output.dynamic_dim_mask != 0) &&
             !output.shape_depends_on_data;
    });
  if (has_dynamic_output) {
    contents += std::format("\nint {}_infer_output_shapes(", manifest.function);
    first = true;
    for (const Argument& input : manifest.inputs) {
      if (input.dynamic_rank) {
        contents += std::format("{}const int64_t *{}_shape, uint32_t {}_rank",
                                first ? "" : ", ",
                                input.name,
                                input.name);
        first = false;
        continue;
      }
      if (input.dynamic_dim_mask == 0) {
        continue;
      }
      contents += std::format("{}const int64_t {}_shape[{}]",
                              first ? "" : ", ",
                              input.name,
                              macro_name(manifest, input, "rank"));
      first = false;
    }
    for (const Argument& output : manifest.outputs) {
      if (output.shape_depends_on_data) {
        continue;
      }
      if (output.dynamic_rank) {
        contents += std::format(
          "{}int64_t *{}_shape, uint32_t {}_shape_capacity, uint32_t "
          "*{}_rank",
          first ? "" : ", ",
          output.name,
          output.name,
          output.name);
        first = false;
        continue;
      }
      if (output.dynamic_dim_mask == 0) {
        continue;
      }
      contents += std::format("{}int64_t {}_shape[{}]",
                              first ? "" : ", ",
                              output.name,
                              macro_name(manifest, output, "rank"));
      first = false;
    }
    contents += ");\n";
  }
  contents +=
    std::format("\n#ifdef __cplusplus\n}}\n#endif\n\n#endif  // {}\n", guard);
  return write_file(path, contents);
}

std::set<std::string> symbols(std::string_view output) {
  static const std::regex pattern(R"(\b[UTW]\s+(\S+))");
  std::set<std::string> result;
  std::string text(output);
  for (auto match = std::sregex_iterator(text.begin(), text.end(), pattern);
       match != std::sregex_iterator();
       ++match) {
    std::string symbol = (*match)[1];
    if (std::size_t version = symbol.find('@'); version != std::string::npos) {
      symbol.resize(version);
    }
    result.insert(std::move(symbol));
  }
  return result;
}

[[nodiscard]] std::expected<std::string, std::string> read_text(
  const fs::path& path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    return std::unexpected(std::format(
      "cannot read '{}': {}", path.string(), buffer.getError().message()));
  }
  return (*buffer)->getBuffer().str();
}

bool is_generated_output(const fs::path& path, std::string_view model_name) {
  const std::string name = path.filename().string();
  const std::set<std::string> fixed = {"model.ncnn.mlir",
                                       "model.tosa.mlir",
                                       "model.linalg.mlir",
                                       "model.memref.mlir",
                                       "model.capi.mlir",
                                       "model.llvm.mlir",
                                       "model.ll",
                                       "model.o",
                                       "model.s"};
  return fixed.contains(name) ||
         name == "lib" + std::string(model_name) + ".so" ||
         name == std::string(model_name) + ".h" ||
         name == std::string(model_name) + ".json" ||
         name == std::string(model_name) + ".plan.json";
}

struct OutputDirectoryState {
  bool exists;
  std::optional<llvm::sys::fs::UniqueID> identity;
};

[[nodiscard]] std::expected<OutputDirectoryState, std::string>
validate_output_directory(const fs::path& output_dir,
                          std::string_view model_name) {
  std::error_code error;
  const bool exists = fs::exists(output_dir, error);
  if (error) {
    return std::unexpected(std::format("cannot inspect output path '{}': {}",
                                       output_dir.string(),
                                       error.message()));
  }
  if (!exists) {
    return OutputDirectoryState{.exists = false, .identity = std::nullopt};
  }
  if (!fs::is_directory(output_dir, error)) {
    if (error) {
      return std::unexpected(
        std::format("cannot inspect output directory '{}': {}",
                    output_dir.string(),
                    error.message()));
    }
    return std::unexpected(
      std::format("output path is not a directory: {}", output_dir.string()));
  }
  llvm::sys::fs::UniqueID identity;
  if (std::error_code identity_error =
        llvm::sys::fs::getUniqueID(output_dir.string(), identity)) {
    return std::unexpected(
      std::format("cannot identify output directory '{}': {}",
                  output_dir.string(),
                  identity_error.message()));
  }
  fs::directory_iterator iterator(output_dir, error);
  if (error) {
    return std::unexpected(
      std::format("cannot iterate output directory '{}': {}",
                  output_dir.string(),
                  error.message()));
  }
  const fs::directory_iterator end;
  while (iterator != end) {
    error.clear();
    const bool is_symlink = iterator->is_symlink(error);
    if (error) {
      return std::unexpected(std::format("cannot inspect output entry '{}': {}",
                                         iterator->path().string(),
                                         error.message()));
    }
    error.clear();
    const bool is_regular_file = iterator->is_regular_file(error);
    if (error) {
      return std::unexpected(std::format("cannot inspect output entry '{}': {}",
                                         iterator->path().string(),
                                         error.message()));
    }
    if (is_symlink || !is_regular_file ||
        !is_generated_output(iterator->path(), model_name)) {
      return std::unexpected(std::format(
        "output directory contains a file not owned by ncnn-compile: {}",
        iterator->path().filename().string()));
    }
    iterator.increment(error);
    if (error) {
      return std::unexpected(
        std::format("cannot continue iterating output directory '{}': {}",
                    output_dir.string(),
                    error.message()));
    }
  }
  return OutputDirectoryState{.exists = true, .identity = identity};
}

[[nodiscard]] std::expected<fs::path, std::string> unique_sibling_path(
  const fs::path& output_dir, std::string_view purpose, bool create) {
  fs::path parent = output_dir.parent_path();
  if (parent.empty()) {
    parent = ".";
  }
  const std::string stem = output_dir.filename().string() + ".ncnn-compile-" +
                           std::string(purpose) + "-" +
                           std::to_string(llvm::sys::Process::getProcessId());
  for (unsigned attempt = 0; attempt < 1000; ++attempt) {
    fs::path candidate = parent / (stem + "-" + std::to_string(attempt));
    std::error_code error;
    if (create) {
      if (fs::create_directory(candidate, error)) {
        return candidate;
      }
      if (!error) {
        continue;
      }
      if (error != std::errc::file_exists) {
        return std::unexpected(
          std::format("cannot create replacement directory '{}': {}",
                      candidate.string(),
                      error.message()));
      }
    } else if (!fs::exists(candidate, error)) {
      if (error) {
        return std::unexpected(
          std::format("cannot inspect backup path '{}': {}",
                      candidate.string(),
                      error.message()));
      }
      return candidate;
    } else if (error) {
      return std::unexpected(std::format("cannot inspect backup path '{}': {}",
                                         candidate.string(),
                                         error.message()));
    }
  }
  return std::unexpected(
    std::format("cannot reserve a sibling path for output directory '{}'",
                output_dir.string()));
}

[[nodiscard]] std::expected<void, std::string> write_harness(
  const fs::path& path, std::string_view header, const Manifest& manifest) {
  std::string code = std::format(
    "#include <math.h>\n#include <stddef.h>\n#include <stdio.h>\n"
    "#include <stdlib.h>\n#include \"{}\"\n\n"
    "static size_t element_count(const int64_t *shape, size_t rank) {{\n"
    "  size_t count = 1;\n"
    "  for (size_t i = 0; i < rank; ++i) count *= (size_t)shape[i];\n"
    "  return count;\n"
    "}}\n\nint main(void) {{\n",
    header);

  const bool dynamic_rank =
    manifest.inputs.size() == 1 && manifest.outputs.size() == 1 &&
    manifest.inputs[0].dynamic_rank && manifest.outputs[0].dynamic_rank;
  if (dynamic_rank) {
    auto input_type = c_type(manifest.inputs[0]);
    auto output_type = c_type(manifest.outputs[0]);
    if (!input_type || !output_type) {
      return std::unexpected("unsupported dynamic-rank harness element type");
    }
    code += std::format(
      "  int64_t input1_shape[NCNN_MAX_RANK] = {{2, 2, 2, 2}};\n"
      "  int64_t output1_shape[NCNN_MAX_RANK] = {{0}};\n"
      "  uint32_t output1_rank = 0;\n"
      "  {} input1[16] = {{0}};\n"
      "  {} output1[16] = {{0}};\n"
      "  for (uint32_t i = 0; i < 16; ++i) input1[i] = ({})i + 1;\n"
      "  for (uint32_t rank = 1; rank <= NCNN_MAX_RANK; ++rank) {{\n"
      "    if ({}_infer_output_shapes(input1_shape, rank, output1_shape, "
      "NCNN_MAX_RANK, &output1_rank) != 0) return 6;\n"
      "    if (output1_rank != rank) return 7;\n"
      "    for (uint32_t i = 0; i < rank; ++i) if (output1_shape[i] != 2) "
      "return 8;\n"
      "    if ({}(input1, input1_shape, rank, output1, 16) != 0) return 4;\n"
      "    if ({}(input1, input1_shape, rank, output1, 16) != 0) return 9;\n"
      "    size_t count = element_count(input1_shape, rank);\n"
      "    for (size_t i = 0; i < count; ++i) if (output1[i] != input1[i]) "
      "return 13;\n"
      "  }}\n"
      "  if ({}(input1, input1_shape, 0, output1, 16) == 0) return 10;\n"
      "  if ({}(input1, input1_shape, 5, output1, 16) == 0) return 11;\n"
      "  if ({}_infer_output_shapes(input1_shape, 4, output1_shape, 3, "
      "&output1_rank) == 0) return 12;\n"
      "  if ({}(NULL, input1_shape, 1, output1, 16) == 0) return 3;\n"
      "  if ({}(input1, NULL, 1, output1, 16) == 0) return 3;\n"
      "  if ({}(input1, input1_shape, 1, NULL, 16) == 0) return 3;\n"
      "  if ({}(input1, input1_shape, 4, output1, 15) != 5) return 14;\n"
      "  return 0;\n}}\n",
      *input_type,
      *output_type,
      *input_type,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function,
      manifest.function);
    return write_file(path, code);
  }

  for (const Argument& input : manifest.inputs) {
    if (input.dynamic_dim_mask != 0) {
      code += std::format(
        "  int64_t {}_shape[{}] = {{", input.name, input.shape.size());
      for (std::size_t index = 0; index < input.shape.size(); ++index) {
        int64_t dynamicExtent = 2;
        auto constraint =
          std::ranges::find(input.dimension_constraints,
                            static_cast<std::uint32_t>(index),
                            &Argument::DimensionConstraint::dimension);
        if (constraint != input.dimension_constraints.end()) {
          dynamicExtent = constraint->minimum;
          const int64_t remainder = dynamicExtent % constraint->multiple_of;
          if (remainder != 0) {
            dynamicExtent += constraint->multiple_of - remainder;
          }
        }
        code += std::format(
          "{}{}",
          index == 0 ? "" : ", ",
          input.shape[index] < 0 ? dynamicExtent : input.shape[index]);
      }
      code += "};\n";
      code += std::format("  size_t {}_count = element_count({}_shape, {});\n",
                          input.name,
                          input.name,
                          input.shape.size());
    } else {
      auto count = element_count(input);
      if (!count) {
        return std::unexpected(count.error());
      }
      code += std::format("  size_t {}_count = {};\n", input.name, *count);
    }
    auto type = c_type(input);
    if (!type) {
      return std::unexpected("unsupported harness input element type");
    }
    code += std::format("  {} *{} = calloc({}_count, sizeof({}));\n",
                        *type,
                        input.name,
                        input.name,
                        *type);
    code += std::format(
      "  if (!{}) {{ fprintf(stderr, \"ABI verification failed: cannot "
      "allocate {}\\n\"); return 2; }}\n",
      input.name,
      input.name);
  }

  const bool has_dynamic_output =
    std::ranges::any_of(manifest.outputs, [](const Argument& output) {
      return (output.dynamic_rank || output.dynamic_dim_mask != 0) &&
             !output.shape_depends_on_data;
    });
  std::string shape_call = manifest.function + "_infer_output_shapes(";
  bool first_shape_argument = true;
  for (const Argument& input : manifest.inputs) {
    if (input.dynamic_dim_mask != 0) {
      shape_call +=
        std::format("{}{}_shape", first_shape_argument ? "" : ", ", input.name);
      first_shape_argument = false;
    }
  }
  for (const Argument& output : manifest.outputs) {
    if (output.dynamic_dim_mask != 0 && !output.shape_depends_on_data) {
      code += std::format(
        "  int64_t {}_shape[{}] = {{0}};\n", output.name, output.shape.size());
      shape_call += std::format(
        "{}{}_shape", first_shape_argument ? "" : ", ", output.name);
      first_shape_argument = false;
    }
  }
  shape_call += ")";
  if (has_dynamic_output) {
    code += std::format("  if ({} != 0) return 6;\n", shape_call);
  }

  for (const Argument& output : manifest.outputs) {
    if (output.dynamic_dim_mask != 0 && !output.shape_depends_on_data) {
      code += std::format("  size_t {}_count = element_count({}_shape, {});\n",
                          output.name,
                          output.name,
                          output.shape.size());
      code += std::format(
        "  uint64_t {}_capacity = {}_count;\n", output.name, output.name);
    } else {
      auto count = element_count(output);
      if (!count) {
        return std::unexpected(count.error());
      }
      code += std::format("  size_t {}_count = {};\n", output.name, *count);
    }
    auto type = c_type(output);
    if (!type) {
      return std::unexpected("unsupported harness output element type");
    }
    code += std::format("  {} *{} = calloc({}_count, sizeof({}));\n",
                        *type,
                        output.name,
                        output.name,
                        *type);
    code += std::format("  if (!{}) return 2;\n", output.name);
  }

  std::vector<std::string> call_arguments;
  for (const Argument& input : manifest.inputs) {
    call_arguments.push_back(input.name);
    if (input.dynamic_dim_mask != 0) {
      call_arguments.push_back(input.name + "_shape");
    }
  }
  for (const Argument& output : manifest.outputs) {
    call_arguments.push_back(output.name);
  }
  for (const Argument& output : manifest.outputs) {
    if (output.shape_depends_on_data || output.dynamic_dim_mask == 0) {
      continue;
    }
    call_arguments.push_back(output.name + "_capacity");
  }
  for (const Argument& output : manifest.outputs) {
    if (output.shape_depends_on_data) {
      code += std::format(
        "  int64_t {}_shape[{}] = {{0}};\n", output.name, output.shape.size());
      call_arguments.push_back(output.name + "_shape");
      call_arguments.push_back(std::to_string(output.shape.size()));
      code += std::format("  uint32_t {}_rank = 0;\n", output.name);
      call_arguments.push_back("&" + output.name + "_rank");
    }
  }
  auto call = [&](std::optional<std::size_t> null_index) {
    std::string result = manifest.function + "(";
    for (std::size_t index = 0; index < call_arguments.size(); ++index) {
      result += index == 0 ? "" : ", ";
      result += null_index == index ? "NULL" : call_arguments[index];
    }
    return result + ")";
  };
  code += std::format("  int model_status = {};\n", call(std::nullopt));
  code +=
    "  if (model_status != 0) { fprintf(stderr, \"ABI verification "
    "failed: model returned %d\\n\", model_status); return 4; }\n";
  for (const Argument& output : manifest.outputs) {
    code += std::format(
      "  for (size_t i = 0; i < {}_count; ++i) if (!isfinite({}[i])) {{ "
      "fprintf(stderr, \"ABI verification failed: output {}[%zu] is not "
      "finite\\n\", i); return 5; }}\n",
      output.name,
      output.name,
      output.name);
  }
  for (std::size_t index = 0; index < call_arguments.size(); ++index) {
    if (call_arguments[index].ends_with("_capacity")) {
      continue;
    }
    code += std::format(
      "  if ({} == 0) {{ fprintf(stderr, \"ABI verification failed: NULL "
      "argument {} was accepted\\n\"); return 3; }}\n",
      call(index),
      call_arguments[index]);
  }
  for (const Argument& output : manifest.outputs) {
    if (output.shape_depends_on_data || output.dynamic_dim_mask == 0) {
      continue;
    }
    std::vector<std::string> insufficientArguments = call_arguments;
    auto capacity =
      std::ranges::find(insufficientArguments, output.name + "_capacity");
    *capacity = output.name + "_capacity - 1";
    std::string insufficientCall = manifest.function + "(";
    for (std::size_t index = 0; index < insufficientArguments.size(); ++index) {
      insufficientCall += index == 0 ? "" : ", ";
      insufficientCall += insufficientArguments[index];
    }
    insufficientCall += ")";
    code += std::format(
      "  if ({} != NCNN_STATUS_OUTPUT_CAPACITY_INSUFFICIENT) return 14;\n",
      insufficientCall);
  }
  for (const Argument& input : manifest.inputs) {
    code += std::format("  free({});\n", input.name);
  }
  for (const Argument& output : manifest.outputs) {
    code += std::format("  free({});\n", output.name);
  }
  code += "  return 0;\n}\n";
  return write_file(path, code);
}

struct ClangTargetArguments {
  std::vector<std::string> target;
  std::vector<std::string> isa;
};

ClangTargetArguments build_clang_target_arguments(
  const std::string& effective_target_triple) {
  ClangTargetArguments result;
  result.target.push_back("--target=" + effective_target_triple);
  if (!g_sysroot.empty()) {
    result.target.push_back("--sysroot=" + g_sysroot);
  }
  if (!g_march.empty()) {
    result.isa.push_back("-march=" + g_march);
  }
  if (!g_mcpu.empty()) {
    result.isa.push_back("-mcpu=" + g_mcpu);
  }
  for (const std::string& feature : g_target_features) {
    result.isa.insert(result.isa.end(),
                      {"-Xclang", "-target-feature", "-Xclang", feature});
  }
  return result;
}

// Probe only dedicated target arguments; opaque Clang options are rejected.
[[nodiscard]] std::expected<std::string, std::string> resolve_int8_target(
  const fs::path& staging_path,
  const std::string& clang_path,
  const std::string& effective_target_triple,
  const std::vector<std::string>& target_args,
  const std::vector<std::string>& isa_args,
  unsigned effective_threads,
  bool vector_active,
  bool vector_scalable) {
  std::string resolved_int8_target = "portable";
  if (g_int8_kernel != "portable" && g_int8_kernel != "auto" &&
      g_int8_kernel != "vnni") {
    return std::unexpected("--int8-kernel must be one of portable, auto, vnni");
  }
  if (g_int8_kernel != "portable") {
    // Opaque driver options (including response/config files, -Xclang and
    // macro definitions) cannot be proven equivalent for C probing and IR
    // codegen. Reject rather than maintain an incomplete target-flag denylist.
    if (!g_clang_args.empty()) {
      return std::unexpected(
        "--clang-arg cannot be combined with --int8-kernel=" +
        g_int8_kernel.getValue() +
        "; use --target-triple/--march/--mcpu/--target-feature so the INT8 "
        "probe and final codegen use the same target arguments");
    }
    if (g_march == "native" || g_mcpu == "native") {
      return std::unexpected(
        "--march=native/--mcpu=native cannot be combined with "
        "--int8-kernel=" +
        g_int8_kernel.getValue() +
        "; name the CPU or features explicitly so the probed capability is "
        "reproducible on any machine");
    }
    const llvm::StringRef triple_for_probe(effective_target_triple);
    const bool probe_is_x86_64 =
      triple_for_probe.contains("x86_64") || triple_for_probe.contains("amd64");
    if (!probe_is_x86_64) {
      if (g_int8_kernel == "vnni") {
        return std::unexpected(
          "--int8-kernel=vnni requires an x86-64 target triple, got '" +
          effective_target_triple + "'");
      }
      llvm::errs() << "ncnn-compile: info: --int8-kernel=auto keeps portable "
                      "row-dot for non-x86-64 target '"
                   << effective_target_triple << "'\n";
    } else {
      const fs::path macro_probe = staging_path / "int8_target_probe.c";
      const fs::path macro_capture = staging_path / "int8_target_macros.txt";
      if (auto written = write_file(macro_probe, ""); !written) {
        return std::unexpected(written.error());
      }
      std::vector<std::string> macro_command{
        clang_path, "-E", "-dM", macro_probe.string()};
      // 与最终 compile 完全同源的目标参数（见上方 isa_args/target_args）。
      macro_command.insert(
        macro_command.end(), target_args.begin(), target_args.end());
      macro_command.insert(
        macro_command.end(), isa_args.begin(), isa_args.end());
      if (int status = run(macro_command, macro_capture)) {
        return std::unexpected("INT8 target capability probe failed");
      }
      auto macro_text = read_text(macro_capture);
      if (!macro_text) {
        return std::unexpected(macro_text.error());
      }
      const ncnn_mlir::Int8DotTarget probed =
        ncnn_mlir::resolve_int8_dot_target(*macro_text);
      if (probed == ncnn_mlir::Int8DotTarget::Portable) {
        if (g_int8_kernel == "vnni") {
          return std::unexpected(
            "--int8-kernel=vnni requires AVX2 plus AVX-VNNI or "
            "AVX512-VNNI in the final target arguments (march/mcpu/"
            "target-feature), which the probe did not report for " +
            effective_target_triple);
        }
        llvm::errs() << "ncnn-compile: info: no INT8 dot-product ISA for the "
                        "target; keeping portable row-dot\n";
      } else {
        resolved_int8_target =
          std::string(ncnn_mlir::int8_dot_target_name(probed));
        if (g_int8_kernel == "auto") {
          // auto 刻意保持 portable（性能验收未达，见 P16 计划 §3）：
          // 探测到的能力只进入 identity 与计划，不升级 kernel policy。
          llvm::errs() << "ncnn-compile: info: target reports "
                       << resolved_int8_target
                       << "; --int8-kernel=auto keeps portable row-dot "
                          "pending performance validation\n";
        }
      }
    }
  }
  // 显式 vnni 必须有现代向量尾：threads!=1（OpenMP 路径）或张量级固定宽
  // 向量化。纯串行 legacy 路径没有向量下降（内核会被静默跳过），而强行
  // 打开 vector-tail 又会让 pass 的逐元素清理发射串行路径无法下降的
  // scf.parallel（实测 relu 夹具在 VerifyNoSCFForall 失败）。拒绝该组合
  // 而不是静默降级 portable。
  if (g_int8_kernel == "vnni" && resolved_int8_target != "portable" &&
      effective_threads == 1 && (!vector_active || vector_scalable)) {
    return std::unexpected(
      "--int8-kernel=vnni requires OpenMP worker threads (default "
      "--threads=0) or fixed-width MLIR vectorization "
      "(--vector-mode=fixed-width); the serial legacy path cannot lower "
      "the VNNI kernel");
  }
  // depthwise opt-in 走 VectorizeNCNN 固定宽向量发射器：scalable 或未启用
  // MLIR 级向量化时该 pass 不会改写，显式 opt-in 静默失效不可接受。
  if (g_int8_depthwise && (vector_scalable || !vector_active)) {
    return std::unexpected(
      "--int8-depthwise requires fixed-width MLIR vectorization; pass "
      "--vector-mode=fixed-width (or auto on a fixed-width target). "
      "Scalable vectors are not supported by the INT8 depthwise path");
  }

  return resolved_int8_target;
}

std::string build_codegen_identity(std::string_view target_triple,
                                   std::string_view resolved_int8_target,
                                   unsigned effective_threads,
                                   std::string_view resolved_vector_math,
                                   std::string_view vector_math_abi,
                                   unsigned vector_math_lanes) {
  std::string result =
    "target=" + std::string(target_triple) + "|march=" + g_march +
    "|mcpu=" + g_mcpu + "|mtune=" + g_mtune + "|precision=" + g_precision +
    "|fp16-accumulator=" + g_fp16_accumulator +
    "|allow-fallback=" + std::to_string(g_allow_fallback.getValue()) +
    "|optimization=" + g_optimization +
    "|vector-width=" + std::to_string(g_vector_width) +
    "|vector-mode=" + g_vector_mode +
    "|vector-math=" + std::string(resolved_vector_math) +
    "|vector-math-abi=" + std::string(vector_math_abi) +
    "|vector-math-lanes=" + std::to_string(vector_math_lanes) +
    "|sysroot=" + g_sysroot + "|conv-strategy=" + g_conv_strategy +
    "|conv-gemm-l2-bytes=" + std::to_string(g_conv_gemm_l2_bytes) +
    "|int8-kernel=" + g_int8_kernel.getValue() +
    "|int8-target=" + std::string(resolved_int8_target) +
    "|int8-depthwise=" + std::to_string(g_int8_depthwise.getValue()) +
    "|int8-cast-chain=" + std::to_string(g_int8_cast_chain.getValue()) +
    "|threads=" + std::to_string(effective_threads);
  for (const std::string& feature : g_target_features) {
    result += "|target-feature=" + feature;
  }
  for (const std::string& argument : g_clang_args) {
    result += "|clang-arg=" + argument;
  }
  for (const std::string& argument : g_linker_args) {
    result += "|linker-arg=" + argument;
  }
  return result;
}

std::vector<std::string> normalize_arguments(int argc, char** argv) {
  std::vector<std::string> result;
  result.reserve(argc);
  result.emplace_back(argv[0]);
  const std::set<std::string_view> passthrough = {
    "--target-feature", "--clang-arg", "--linker-arg"};
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if (passthrough.contains(argument) && index + 1 < argc &&
        std::string_view(argv[index + 1]).starts_with('-')) {
      result.push_back(argument + "=" + argv[++index]);
    } else {
      result.push_back(std::move(argument));
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> normalized = normalize_arguments(argc, argv);
  std::vector<const char*> normalized_argv;
  normalized_argv.reserve(normalized.size());
  for (const std::string& argument : normalized) {
    normalized_argv.push_back(argument.c_str());
  }
  llvm::cl::HideUnrelatedOptions(g_category);
  llvm::cl::ParseCommandLineOptions(
    static_cast<int>(normalized_argv.size()),
    normalized_argv.data(),
    "Compile an ncnn .param/.bin model into a C header and shared library\n");

  if (!g_input.empty() && !g_param.empty()) {
    return fail("use either positional input or --param, not both");
  }
  const std::string param_path = g_param.empty() ? g_input : g_param;
  if (param_path.empty()) {
    return fail("an input .param file is required");
  }
  const std::string bin_path =
    g_bin.empty() ? derive_bin_path(param_path) : g_bin;
  std::error_code error;
  if (!fs::is_regular_file(param_path, error)) {
    return fail(llvm::Twine("cannot use input file '") + param_path +
                "': " + (error ? error.message() : "not a regular file"));
  }
  error.clear();
  if (!fs::is_regular_file(bin_path, error)) {
    return fail(llvm::Twine("cannot use weight file '") + bin_path +
                "': " + (error ? error.message() : "not a regular file"));
  }
  if (g_optimization != "0" && g_optimization != "1" && g_optimization != "2" &&
      g_optimization != "3") {
    return fail("-O must be one of -O0, -O1, -O2, or -O3");
  }
  if (g_vector_width != 0 && g_vector_width % 64 != 0) {
    return fail("--vector-width must be 0 or a multiple of 64 bits");
  }
  if (!g_target_triple.empty() &&
      (g_march == "native" || g_mcpu == "native" || g_mtune == "native")) {
    return fail(
      "native CPU selection cannot be combined with an explicit "
      "--target-triple; name the CPU explicitly for cross targets");
  }
  if (!g_target_triple.empty()) {
    std::string triple = g_target_triple;
    std::ranges::transform(
      triple, triple.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string architecture = triple.substr(0, triple.find('-'));
    if (!triple.contains("linux")) {
      return fail("--target-triple currently supports Linux ELF targets only");
    }
    if (!architecture.contains("64") && architecture != "s390x") {
      return fail("--target-triple currently supports 64-bit targets only");
    }
  }

  const std::string model_name = c_identifier(
    g_model_name.empty() ? fs::path(param_path).stem().string() : g_model_name);
  fs::path output_dir =
    (g_output_dir.empty() ? fs::path(model_name)
                          : fs::path(g_output_dir.getValue()))
      .lexically_normal();
  while (output_dir.filename().empty() &&
         output_dir != output_dir.root_path()) {
    output_dir = output_dir.parent_path();
  }
  if (output_dir.filename().empty() || output_dir.filename() == "." ||
      output_dir.filename() == "..") {
    return fail("output directory must name a non-root directory");
  }
  auto output_exists = validate_output_directory(output_dir, model_name);
  if (!output_exists) {
    return fail(output_exists.error());
  }

  const std::set<std::string> valid_stages = {"ncnn",
                                              "tosa",
                                              "linalg",
                                              "memref",
                                              "capi",
                                              "llvm",
                                              "llvm-ir",
                                              "object",
                                              "assembly",
                                              "all"};
  std::set<std::string> emitted;
  for (const std::string& stage : g_emit) {
    if (!valid_stages.contains(stage)) {
      return fail(llvm::Twine("invalid --emit stage: ") + stage);
    }
    emitted.insert(stage);
  }
  if (emitted.contains("all")) {
    emitted = {"ncnn",
               "tosa",
               "linalg",
               "memref",
               "capi",
               "llvm",
               "llvm-ir",
               "object",
               "assembly"};
  }

  const fs::path executable =
    llvm::sys::fs::getMainExecutable(argv[0], &g_executable_anchor);
  const fs::path executable_dir = executable.parent_path();
  auto driver = find_tool(
    executable_dir, g_driver, {"ncnn-mlir-driver"}, {"ncnn-mlir-driver"});
  auto opt =
    find_tool(executable_dir, g_opt, {"ncnn-mlir-opt"}, {"ncnn-mlir-opt"});
  auto translate =
    find_tool(executable_dir, g_translate, {}, {"mlir-translate-21"});
  auto clang = find_tool(executable_dir, g_clang, {}, {"clang-21"});
  auto nm = find_tool(executable_dir, g_nm, {}, {"llvm-nm-21"});
  auto readelf = find_tool(executable_dir, g_readelf, {}, {"llvm-readelf-21"});
  auto llvm_as = find_tool(executable_dir, g_llvm_as, {}, {"llvm-as-21"});
  for (const ToolResult* tool :
       {&driver, &opt, &translate, &clang, &nm, &readelf, &llvm_as}) {
    if (!*tool) {
      return fail(tool->error());
    }
  }
  if (!*driver || !*opt || !*translate || !*clang || !*nm || !*readelf ||
      !*llvm_as) {
    return fail(
      "required compiler tool not found; use -v and verify PATH or "
      "the installed toolchain");
  }
  const std::string& driver_path = **driver;
  const std::string& opt_path = **opt;
  const std::string& translate_path = **translate;
  const std::string& clang_path = **clang;
  const std::string& nm_path = **nm;
  const std::string& readelf_path = **readelf;

  llvm::SmallString<256> staging_storage;
  if (llvm::sys::fs::createUniqueDirectory("ncnn-compile", staging_storage)) {
    return fail("cannot create staging directory");
  }
  ScopedDirectory staging(fs::path(staging_storage.str().str()));
  std::string effective_target_triple = g_target_triple;
  if (effective_target_triple.empty()) {
    const fs::path target_capture = staging.path() / "target-triple.txt";
    if (int status = run({clang_path, "-dumpmachine"}, target_capture)) {
      return status;
    }
    auto target_text = read_text(target_capture);
    if (!target_text) {
      return fail(target_text.error());
    }
    effective_target_triple = llvm::StringRef(*target_text).trim().str();
    if (effective_target_triple.empty()) {
      return fail("clang returned an empty native target triple");
    }
  }

  // The probe and final backend share one ordered target/ISA argument list.
  const auto clang_target =
    build_clang_target_arguments(effective_target_triple);
  const auto& target_args = clang_target.target;
  const auto& isa_args = clang_target.isa;
  std::vector<std::string> codegen_args;

  // libomp 探测：多线程产物链接 -lomp；sysroot 缺少 OpenMP 运行时（部分
  // RISC-V 裸环境）要到链接阶段才失败。先行以最小探针验证 -lomp 可解析，
  // 失败则等价回退 --threads=1 并在 manifest 标注 openmp=false。
  unsigned effective_threads = g_threads;
  if (g_threads != 1) {
    const fs::path probe_source = staging.path() / "omp_probe.c";
    const fs::path probe_library = staging.path() / "libomp_probe.so";
    constexpr std::string_view probe_text =
      "extern int __kmpc_global_thread_num(void *);\n"
      "int ncnn_omp_probe(void) {\n"
      "  return __kmpc_global_thread_num((void *)0);\n"
      "}\n";
    auto probe_written = write_file(probe_source, probe_text);
    bool omp_available = probe_written.has_value();
    if (omp_available) {
      std::vector<std::string> probe_command{clang_path,
                                             "-shared",
                                             "-fPIC",
                                             probe_source.string(),
                                             "-o",
                                             probe_library.string(),
                                             "-lomp"};
      probe_command.insert(
        probe_command.end(), target_args.begin(), target_args.end());
      omp_available =
        run(probe_command) == 0 && fs::is_regular_file(probe_library);
    }
    if (!omp_available) {
      llvm::errs() << "ncnn-compile: warning: OpenMP runtime (-lomp) not "
                      "available for the target; falling back to threads=1\n";
      effective_threads = 1;
    }
  }

  // 解析 MLIR 级向量化模式：off 保持历史行为；auto 由目标推导；
  // fixed-width/scalable 显式指定（lanes 可经 --vector-width 覆盖）。
  const llvm::SmallVector<llvm::StringRef, 8> target_feature_refs(
    g_target_features.begin(), g_target_features.end());
  const ncnn_mlir::TargetVectorInfo vector_info =
    ncnn_mlir::TargetVectorInfo::resolve(
      effective_target_triple, g_march, target_feature_refs);
  unsigned vector_lanes = 0;
  bool vector_scalable = false;
  bool vector_active = false;
  if (g_conv_strategy != "auto" && g_conv_strategy != "gemm" &&
      g_conv_strategy != "conv" && g_conv_strategy != "winograd") {
    return fail("--conv-strategy must be one of auto, gemm, conv, winograd");
  }
  if (g_vector_mode == "off") {
    // 历史行为：不做 MLIR 级向量化。
  } else if (g_vector_mode == "auto") {
    switch (vector_info.mode) {
      case ncnn_mlir::TargetVectorInfo::Mode::FixedWidth:
        vector_lanes = vector_info.lanes;
        break;
      case ncnn_mlir::TargetVectorInfo::Mode::Scalable:
        vector_lanes = vector_info.lanes;
        vector_scalable = true;
        break;
      case ncnn_mlir::TargetVectorInfo::Mode::Scalar:
        break;
    }
  } else if (g_vector_mode == "fixed-width") {
    vector_lanes =
      vector_info.mode == ncnn_mlir::TargetVectorInfo::Mode::FixedWidth
        ? vector_info.lanes
        : 4;
  } else if (g_vector_mode == "scalable") {
    vector_scalable = true;
    vector_lanes =
      vector_info.mode == ncnn_mlir::TargetVectorInfo::Mode::Scalable
        ? vector_info.lanes
        : 4;
  } else {
    return fail(
      "--vector-mode must be one of off, auto, fixed-width, "
      "scalable");
  }
  if (g_vector_mode == "fixed-width" || g_vector_mode == "scalable") {
    if (g_vector_width != 0 && g_vector_width % 32 == 0 &&
        g_vector_width / 32 > vector_lanes) {
      vector_lanes = g_vector_width / 32;
    }
  }
  vector_active = vector_lanes != 0;

  auto int8_target = resolve_int8_target(staging.path(),
                                         clang_path,
                                         effective_target_triple,
                                         target_args,
                                         isa_args,
                                         effective_threads,
                                         vector_active,
                                         vector_scalable);
  if (!int8_target) {
    return fail(int8_target.error());
  }
  const std::string& resolved_int8_target = *int8_target;

  // 解析向量数学后端：auto 按目标探测 libmvec，缺失时静默降级 vendored
  // SLEEF 静态档案，再退回 none；显式指定而不可用时报错退出。部署环境
  // 由 --sysroot 声明并据此重编，编译器不补偿构建机与部署机的 libc 差异。
  std::string resolved_vector_math;
  std::string vector_math_abi;
  unsigned vector_math_lanes = 0;
  bool uses_libmvec = false;
  fs::path sleef_archive;
  {
    if (g_vector_math != "auto" && g_vector_math != "libmvec" &&
        g_vector_math != "sleef" && g_vector_math != "none") {
      return fail("--vector-math must be one of auto, libmvec, sleef, none");
    }
    const llvm::StringRef backend_triple(effective_target_triple);
    const bool backend_arm64 = backend_triple.contains("aarch64");
    const bool backend_x86 =
      backend_triple.contains("x86_64") || backend_triple.contains("amd64") ||
      backend_triple.contains("i686") || backend_triple.contains("i386");
    auto hint_present = [&](std::string_view hint) {
      for (const std::string& feature : g_target_features) {
        if (llvm::StringRef(feature).contains(hint)) {
          return true;
        }
      }
      return llvm::StringRef(g_march).contains(hint);
    };
    // libmvec 变体绑定编译期 ISA：显式 feature/march 提示优先，缺省取基线。
    // 变体宽度跟随 ISA 寄存器而非 --vector-width；--march=native 不展开
    // feature，按基线变体探测。
    std::string libmvec_abi;
    unsigned libmvec_lanes = 0;
    if (!vector_scalable && backend_x86) {
      if (hint_present("avx512")) {
        libmvec_abi = "_ZGVeN16";
        libmvec_lanes = 16;
      } else if (hint_present("avx2") || hint_present("-v3")) {
        libmvec_abi = "_ZGVdN8";
        libmvec_lanes = 8;
      } else if (hint_present("avx")) {
        libmvec_abi = "_ZGVcN8";
        libmvec_lanes = 8;
      } else {
        libmvec_abi = "_ZGVbN4";
        libmvec_lanes = 4;
      }
    } else if (!vector_scalable && backend_arm64) {
      libmvec_abi = "_ZGVnN4";
      libmvec_lanes = 4;
    }

    // 探针引用后端可能发射的全部五个符号：任一缺失即判定不可用，避免
    // -z defs 在产物链接期才暴露。链接以 -Wl,-z,defs + -lm 强制解析——
    // glibc 的 libm linker script 会按需带入 libmvec，musl/旧 glibc 则
    // 直接失败。
    auto probe_libmvec = [&]() -> bool {
      if (libmvec_abi.empty()) {
        return false;
      }
      const unsigned probe_bytes = libmvec_lanes * 4;
      const fs::path probe_source = staging.path() / "libmvec_probe.c";
      const fs::path probe_library = staging.path() / "liblibmvec_probe.so";
      std::string probe_text = "typedef float vsf __attribute__((vector_size(" +
                               std::to_string(probe_bytes) +
                               ")));\n"
                               "extern vsf " +
                               libmvec_abi +
                               "v_expf(vsf);\n"
                               "extern vsf " +
                               libmvec_abi +
                               "v_tanhf(vsf);\n"
                               "extern vsf " +
                               libmvec_abi +
                               "v_erff(vsf);\n"
                               "extern vsf " +
                               libmvec_abi +
                               "v_logf(vsf);\n"
                               "extern vsf " +
                               libmvec_abi +
                               "vv_powf(vsf, vsf);\n"
                               "int ncnn_libmvec_probe(vsf x) {\n"
                               "  vsf r = " +
                               libmvec_abi +
                               "v_expf(x);\n"
                               "  r += " +
                               libmvec_abi +
                               "v_tanhf(x);\n"
                               "  r += " +
                               libmvec_abi +
                               "v_erff(x);\n"
                               "  r += " +
                               libmvec_abi +
                               "v_logf(x);\n"
                               "  r += " +
                               libmvec_abi +
                               "vv_powf(x, x);\n"
                               "  return (int)r[0];\n"
                               "}\n";
      auto written = write_file(probe_source, probe_text);
      bool available = written.has_value();
      if (available) {
        std::vector<std::string> command{clang_path,
                                         "-shared",
                                         "-fPIC",
                                         probe_source.string(),
                                         "-o",
                                         probe_library.string(),
                                         "-Wl,-z,defs",
                                         "-lm"};
        command.insert(command.end(), target_args.begin(), target_args.end());
        available = run(command) == 0 && fs::is_regular_file(probe_library);
      }
      return available;
    };

    if (g_vector_math == "none") {
      resolved_vector_math = "none";
    } else {
      // SLEEF 静态档案的宽度入口自带运行时 ISA 分发；scalable 目标无
      // 固定宽度入口，v1 不选 SLEEF。
      const bool sleef_ready =
        !vector_scalable && locate_sleef_archive(sleef_archive);
      if (g_vector_math == "auto") {
        if (probe_libmvec()) {
          uses_libmvec = true;
        } else if (sleef_ready) {
          llvm::errs() << "ncnn-compile: info: libmvec unavailable for the "
                          "target; using vendored SLEEF\n";
        } else {
          llvm::errs() << "ncnn-compile: warning: neither libmvec nor "
                          "vendored SLEEF is available; keeping scalar math\n";
        }
      } else if (g_vector_math == "libmvec") {
        if (!probe_libmvec()) {
          return fail(
            "--vector-math=libmvec is not available for target " +
            effective_target_triple +
            "; provide a sysroot whose libm pulls in the required _ZGV "
            "symbols or choose --vector-math=auto");
        }
        uses_libmvec = true;
      } else {  // sleef
        if (!sleef_ready) {
          return fail(
            std::string(
              "--vector-math=sleef requires the vendored SLEEF static archive "
              "(libsleef.a); point --sleef-path at its directory") +
            (vector_scalable ? "; scalable targets are not supported yet"
                             : ""));
        }
      }
      resolved_vector_math =
        uses_libmvec ? "libmvec" : (!sleef_archive.empty() ? "sleef" : "none");
    }
    if (uses_libmvec) {
      vector_math_abi = libmvec_abi;
      vector_math_lanes = libmvec_lanes;
    } else if (resolved_vector_math == "sleef") {
      // 入口名 = Sleef_<基名><宽度>_u10，如 Sleef_expf8_u10；宽度入口
      // 内部自带运行时 ISA 分发。
      const bool backend_x86 =
        backend_triple.contains("x86_64") || backend_triple.contains("amd64");
      if (backend_x86 && hint_present("avx512")) {
        vector_math_abi = "16_u10";
        vector_math_lanes = 16;
      } else if (backend_x86 && hint_present("avx2")) {
        vector_math_abi = "8_u10";
        vector_math_lanes = 8;
      } else {
        vector_math_abi = "4_u10";
        vector_math_lanes = 4;
      }
    }
  }
  const fs::path ncnn_ir = staging.path() / "model.ncnn.mlir";
  const fs::path tosa_ir = staging.path() / "model.tosa.mlir";
  const fs::path linalg_ir = staging.path() / "model.linalg.mlir";
  const fs::path memref_ir = staging.path() / "model.memref.mlir";
  const fs::path capi_ir = staging.path() / "model.capi.mlir";
  const fs::path llvm_dialect_ir = staging.path() / "model.llvm.mlir";
  const fs::path llvm_ir = staging.path() / "model.ll";
  const fs::path object = staging.path() / "model.o";
  const fs::path assembly = staging.path() / "model.s";
  const fs::path profile_object = staging.path() / "profile_runtime.o";
#ifdef NCNN_PROFILE_RUNTIME_SOURCE
  fs::path profile_runtime_source = NCNN_PROFILE_RUNTIME_SOURCE;
#else
  fs::path profile_runtime_source;
#endif
#ifdef NCNN_PROFILE_RUNTIME_RELATIVE_PATH
  if (g_profile && !fs::is_regular_file(profile_runtime_source)) {
    std::error_code executable_error;
    const fs::path executable =
      fs::read_symlink("/proc/self/exe", executable_error);
    if (!executable_error) {
      const fs::path installed_source =
        executable.parent_path() / NCNN_PROFILE_RUNTIME_RELATIVE_PATH;
      if (fs::is_regular_file(installed_source)) {
        profile_runtime_source = installed_source;
      }
    }
  }
#endif
  const fs::path manifest_path = staging.path() / (model_name + ".json");
  const fs::path execution_plan_path =
    staging.path() / (model_name + ".plan.json");
  const fs::path header = staging.path() / (model_name + ".h");
  const fs::path exports = staging.path() / "exports.map";
  const fs::path library = staging.path() / ("lib" + model_name + ".so");
  // A diagnostic profile is only joinable when its matching static plan is
  // published alongside the library, so profiling implicitly emits the plan.
  const bool emit_execution_plan = g_emit_execution_plan || g_profile;
  const std::string codegen_identity =
    build_codegen_identity(effective_target_triple,
                           resolved_int8_target,
                           effective_threads,
                           resolved_vector_math,
                           vector_math_abi,
                           vector_math_lanes);
  const std::string codegen_identity_transport = hex_encode(codegen_identity);

  std::vector<std::string> driver_command{
    driver_path, param_path, "--bin", bin_path, "-o", ncnn_ir.string()};
  driver_command.push_back("--precision=" + g_precision);
  driver_command.push_back("--fp16-accumulator=" + g_fp16_accumulator);
  if (g_allow_fallback) {
    driver_command.emplace_back("--allow-fallback");
  }
  driver_command.push_back("--target-triple=" + effective_target_triple);
  if (!g_march.empty()) {
    driver_command.push_back("--march=" + g_march);
  }
  if (!g_mcpu.empty()) {
    driver_command.push_back("--mcpu=" + g_mcpu);
  }
  for (const std::string& feature : g_target_features) {
    driver_command.push_back("--target-feature=" + feature);
  }
  for (const std::string& input_shape : g_input_shapes) {
    driver_command.push_back("--input-shape=" + input_shape);
  }
  for (const std::string& constraint : g_input_dim_constraints) {
    driver_command.push_back("--input-dim-constraint=" + constraint);
  }
  if (int status = run(driver_command)) {
    return status;
  }
  if (int status = run({opt_path,
                        "--ncnn-to-tosa-pipeline",
                        ncnn_ir.string(),
                        "-o",
                        tosa_ir.string()})) {
    return status;
  }
  std::string tosa_linalg_pipeline_option = "--ncnn-tosa-to-linalg-pipeline";
  if (g_conv_strategy != "auto" || g_conv_gemm_l2_bytes != 524288 ||
      g_int8_cast_chain) {
    tosa_linalg_pipeline_option +=
      "=conv-strategy=" + g_conv_strategy +
      " conv-gemm-l2-bytes=" + std::to_string(g_conv_gemm_l2_bytes) +
      " int8-cast-chain=" + (g_int8_cast_chain ? "true" : "false");
  }
  if (int status = run({opt_path,
                        tosa_linalg_pipeline_option,
                        tosa_ir.string(),
                        "-o",
                        linalg_ir.string()})) {
    return status;
  }
  std::string linalg_pipeline_option = "--ncnn-linalg-to-memref-pipeline";
  {
    // 现代向量尾 = OpenMP 路径或张量级行向量化。串行 legacy affine
    // 路径（threads=1 且未启用 vector-mode）仍由 affine-super-vectorize
    // 负责，本组改写不得抢占。显式 vnni 在 threads=1 时已在选项校验中
    // 要求 vector-mode，因此这里 vector_tail 恒覆盖 vnni 场景。
    const bool vector_tail = g_threads != 1 || vector_active;
    llvm::SmallVector<std::string> linalgOptions;
    if (vector_active) {
      linalgOptions.push_back("vector-lanes=" + std::to_string(vector_lanes));
      linalgOptions.push_back(vector_scalable ? "vector-scalable=true"
                                              : "vector-scalable=false");
    }
    if (vector_tail) {
      linalgOptions.push_back("vector-tail=true");
    }
    linalgOptions.push_back("int8-kernel=" + g_int8_kernel.getValue());
    linalgOptions.push_back("int8-target=" + resolved_int8_target);
    linalgOptions.push_back(std::string("int8-depthwise=") +
                            (g_int8_depthwise ? "true" : "false"));
    if (g_profile) {
      linalgOptions.push_back("profile-instrumentation=true");
    }
    if (emit_execution_plan) {
      linalgOptions.push_back("execution-plan-path=" +
                              execution_plan_path.string());
      linalgOptions.push_back("execution-plan-model=" + model_name);
      linalgOptions.push_back("execution-plan-target-triple=" +
                              effective_target_triple);
      linalgOptions.push_back("execution-plan-threads=" +
                              std::to_string(effective_threads));
      linalgOptions.push_back("execution-plan-codegen-identity=" +
                              codegen_identity_transport);
    }
    if (!linalgOptions.empty()) {
      std::string joined;
      for (const std::string& item : linalgOptions) {
        if (!joined.empty()) {
          joined += ' ';
        }
        joined += item;
      }
      linalg_pipeline_option += "=" + joined;
    }
  }
  if (int status = run({opt_path,
                        linalg_pipeline_option,
                        linalg_ir.string(),
                        "-o",
                        memref_ir.string()})) {
    return status;
  }
  const std::string capi_option =
    "--generate-ncnn-c-api=export-name=" + model_name +
    " manifest-path=" + manifest_path.string();
  if (int status = run(
        {opt_path, capi_option, memref_ir.string(), "-o", capi_ir.string()})) {
    return status;
  }
  std::string execution_plan_hash;
  std::string execution_plan_revision;
  if (g_profile) {
    auto hash = read_execution_plan_field(execution_plan_path, "plan_hash");
    if (!hash) {
      return fail(hash.error());
    }
    execution_plan_hash = *hash;
    auto revision =
      read_execution_plan_field(execution_plan_path, "plan_revision");
    if (!revision) {
      return fail(revision.error());
    }
    execution_plan_revision = *revision;
  }
  std::string llvm_pipeline = "--ncnn-memref-to-llvm-pipeline=";
  const bool uses_openmp = effective_threads != 1;
  const bool uses_sleef = resolved_vector_math == "sleef";
  llvm_pipeline += "threads=" + std::to_string(effective_threads);
  if (!vector_active) {
    llvm_pipeline += " vector-size=" + std::to_string(g_vector_width / 32);
  } else {
    llvm_pipeline += " vector-lowering=true";
  }
  if (resolved_vector_math != "none") {
    llvm_pipeline += " vector-math=" + resolved_vector_math;
    if (!vector_math_abi.empty()) {
      llvm_pipeline += " vector-math-abi=" + vector_math_abi;
    }
    llvm_pipeline += " vector-math-lanes=" + std::to_string(vector_math_lanes);
  }
  if (int status = run({opt_path,
                        llvm_pipeline,
                        capi_ir.string(),
                        "-o",
                        llvm_dialect_ir.string()})) {
    return status;
  }
  if (int status = run({translate_path,
                        "--mlir-to-llvmir",
                        llvm_dialect_ir.string(),
                        "-o",
                        llvm_ir.string()})) {
    return status;
  }
  const fs::path llvm_bitcode = staging.path() / "model.bc";
  if (int status =
        run({**llvm_as, llvm_ir.string(), "-o", llvm_bitcode.string()})) {
    return status;
  }

  codegen_args.insert(codegen_args.end(), isa_args.begin(), isa_args.end());
  if (!g_mtune.empty()) {
    codegen_args.push_back("-mtune=" + g_mtune);
  }
  const llvm::StringRef effective_triple_ref(effective_target_triple);
  const bool target_is_x86 = effective_triple_ref.contains("x86_64") ||
                             effective_triple_ref.contains("amd64") ||
                             effective_triple_ref.contains("i686") ||
                             effective_triple_ref.contains("i386");
  if (g_vector_width != 0 && target_is_x86) {
    codegen_args.push_back("-mprefer-vector-width=" +
                           std::to_string(g_vector_width));
  }
  if (g_debug) {
    codegen_args.emplace_back("-g");
  }
  const std::string optimization = "-O" + g_optimization;
  std::vector<std::string> compile = {
    clang_path, "-x", "ir", "-fPIC", optimization};
  compile.insert(compile.end(), target_args.begin(), target_args.end());
  compile.insert(compile.end(), codegen_args.begin(), codegen_args.end());
  compile.insert(compile.end(), g_clang_args.begin(), g_clang_args.end());
  compile.insert(compile.end(),
                 {"-c", llvm_bitcode.string(), "-o", object.string()});
  if (int status = run(compile)) {
    return status;
  }
  if (g_profile) {
    if (profile_runtime_source.empty() ||
        !fs::is_regular_file(profile_runtime_source)) {
      return fail("diagnostic profile runtime source is unavailable");
    }
    std::vector<std::string> profile_compile{
      clang_path,
      "-std=c11",
      "-fPIC",
      "-O2",
      "-DNCNN_PROFILE_DEFAULT_MODEL=\"" + model_name + "\"",
      "-DNCNN_PROFILE_DEFAULT_TARGET=\"" + effective_target_triple + "\"",
      "-DNCNN_PROFILE_DEFAULT_THREADS=\"" + std::to_string(effective_threads) +
        "\"",
      "-DNCNN_PROFILE_DEFAULT_PLAN_HASH=\"" + execution_plan_hash + "\"",
      "-DNCNN_PROFILE_DEFAULT_BUILD_IDENTITY=\"" + execution_plan_hash + "\"",
      "-DNCNN_PROFILE_DEFAULT_PLAN_REVISION=\"" + execution_plan_revision +
        "\""};
    profile_compile.insert(
      profile_compile.end(), target_args.begin(), target_args.end());
    profile_compile.insert(
      profile_compile.end(), codegen_args.begin(), codegen_args.end());
    profile_compile.insert(
      profile_compile.end(),
      {"-c", profile_runtime_source.string(), "-o", profile_object.string()});
    if (int status = run(profile_compile)) {
      return status;
    }
  }
  std::vector<std::string> assemble = {clang_path, "-x", "ir", optimization};
  assemble.insert(assemble.end(), target_args.begin(), target_args.end());
  assemble.insert(assemble.end(), codegen_args.begin(), codegen_args.end());
  assemble.insert(assemble.end(), g_clang_args.begin(), g_clang_args.end());
  assemble.insert(assemble.end(),
                  {"-S", llvm_bitcode.string(), "-o", assembly.string()});
  if (int status = run(assemble)) {
    return status;
  }

  auto manifest = read_manifest(manifest_path);
  if (!manifest) {
    return fail(manifest.error());
  }
  if (manifest->function != model_name) {
    return fail(llvm::Twine("ABI manifest function '") + manifest->function +
                "' does not match requested model name '" + model_name + "'");
  }
  auto precision = ncnn_mlir::parse_precision_mode(g_precision);
  auto accumulator = ncnn_mlir::parse_fp16_accumulator_mode(g_fp16_accumulator);
  if (!precision || !accumulator) {
    return fail(!precision ? precision.error() : accumulator.error());
  }
  ncnn_mlir::TargetSpec target_spec{
    .triple = effective_target_triple,
    .march = g_march,
    .mcpu = g_mcpu,
    .features = {g_target_features.begin(), g_target_features.end()}};
  auto policy = ncnn_mlir::resolve_precision_policy(
    *precision, *accumulator, g_allow_fallback, target_spec);
  if (!policy) {
    return fail(policy.error());
  }
  manifest->target = Manifest::Target{
    .triple = effective_target_triple,
    .cpu = g_mcpu,
    .march = g_march,
    .tune = g_mtune,
    .features = {g_target_features.begin(), g_target_features.end()},
    .execution_profile =
      ncnn_mlir::precision_execution_profile(*policy, target_spec)};
  manifest->openmp = uses_openmp;
  manifest->vector_math = resolved_vector_math;
  auto manifest_result = write_manifest(manifest_path, *manifest);
  if (!manifest_result) {
    return fail(manifest_result.error());
  }
  auto header_result = write_header(header, *manifest);
  if (!header_result) {
    return fail(header_result.error());
  }
  const bool has_dynamic_output =
    std::ranges::any_of(manifest->outputs, [](const Argument& output) {
      return (output.dynamic_rank || output.dynamic_dim_mask != 0) &&
             !output.shape_depends_on_data;
    });
  std::string exported_symbols = std::format("{{\n  global: {};", model_name);
  if (has_dynamic_output) {
    exported_symbols += std::format(" {}_infer_output_shapes;", model_name);
  }
  exported_symbols += "\n  local: *;\n};\n";
  auto exports_result = write_file(exports, exported_symbols);
  if (!exports_result) {
    return fail(exports_result.error());
  }
  bool uses_address_sanitizer = false;
  bool uses_undefined_sanitizer = false;
  for (const std::string& argument : g_linker_args) {
    constexpr std::string_view prefix = "-fsanitize=";
    if (!argument.starts_with(prefix)) {
      continue;
    }
    std::string_view values(argument);
    values.remove_prefix(prefix.size());
    while (!values.empty()) {
      const std::size_t separator = values.find(',');
      const std::string_view value = values.substr(0, separator);
      uses_address_sanitizer |= value == "address";
      uses_undefined_sanitizer |= value == "undefined";
      if (separator == std::string_view::npos) {
        break;
      }
      values.remove_prefix(separator + 1);
    }
  }
  const bool uses_sanitizer =
    uses_address_sanitizer || uses_undefined_sanitizer;
  const fs::path builtins_capture = staging.path() / "compiler-rt.txt";
  std::vector<std::string> builtins_command{
    clang_path, "--rtlib=compiler-rt", "--print-libgcc-file-name"};
  builtins_command.insert(
    builtins_command.end(), target_args.begin(), target_args.end());
  if (int status = run(builtins_command, builtins_capture)) {
    return status;
  }
  auto builtins_text = read_text(builtins_capture);
  if (!builtins_text) {
    return fail(builtins_text.error());
  }
  std::string builtins_path = *builtins_text;
  while (!builtins_path.empty() &&
         std::isspace(static_cast<unsigned char>(builtins_path.back()))) {
    builtins_path.pop_back();
  }
  if (!fs::is_regular_file(builtins_path)) {
    return fail(llvm::Twine("clang compiler builtins archive not found: ") +
                builtins_path);
  }
  std::vector<std::string> link = {
    clang_path, "-shared", "-nostdlib", optimization};
  link.insert(link.end(), target_args.begin(), target_args.end());
  link.push_back(object.string());
  if (g_profile) {
    link.push_back(profile_object.string());
  }
  link.push_back(builtins_path);
  if (uses_sleef) {
    // SLEEF 静态档案在目标对象之后、系统库之前：符号由 version script
    // 保持 local，导出面不变。
    link.push_back(sleef_archive.string());
  }
  link.insert(link.end(), g_linker_args.begin(), g_linker_args.end());
  if (uses_openmp) {
    link.emplace_back("-lomp");
  }
  if (!uses_sanitizer) {
    link.insert(link.end(), {"-Wl,-z,defs", "-Wl,--no-undefined"});
  }
  link.insert(link.end(),
              {"-Wl,--build-id=none",
               "-Wl,--version-script=" + exports.string(),
               "-lc",
               "-lm",
               "-o",
               library.string()});
  if (int status = run(link)) {
    return status;
  }

  fs::path capture_path = staging.path() / "undefined.txt";
  if (int status = run({nm_path, "-D", "--undefined-only", library.string()},
                       capture_path)) {
    return status;
  }
  auto text = read_text(capture_path);
  if (!text) {
    return fail(text.error());
  }
  const std::set<std::string> undefined = symbols(*text);
  const std::set<std::string> common_allowed = {"ceilf",
                                                "erfcf",
                                                "erff",
                                                "expf",
                                                "floorf",
                                                "free",
                                                "malloc",
                                                "memcpy",
                                                "memset",
                                                "powf",
                                                "tanhf"};
  const std::set<std::string> profile_allowed = {"clock_gettime",
                                                 "fclose",
                                                 "fopen",
                                                 "fputc",
                                                 "fputs",
                                                 "fprintf",
                                                 "fwrite",
                                                 "getenv",
                                                 "__tls_get_addr"};
  const auto is_allowed_undefined = [&](const std::string& symbol) {
    // SLEEF 静态档案的分发器运行需要这两个 libc 例程（计时与对齐分配）。
    static const std::set<std::string> sleef_allowed = {"clock_gettime",
                                                        "posix_memalign"};
    return common_allowed.contains(symbol) ||
           (g_profile && profile_allowed.contains(symbol)) ||
           (uses_libmvec && symbol.starts_with("_ZGV")) ||
           (uses_sleef && sleef_allowed.contains(symbol)) ||
           (uses_openmp && symbol.starts_with("__kmpc_")) ||
           (uses_address_sanitizer && symbol.starts_with("__asan_")) ||
           (uses_undefined_sanitizer && symbol.starts_with("__ubsan_")) ||
           (uses_sanitizer && symbol.starts_with("__sanitizer_"));
  };
  if (!std::ranges::all_of(undefined, is_allowed_undefined)) {
    std::string unexpected;
    for (const std::string& symbol : undefined) {
      if (!is_allowed_undefined(symbol)) {
        unexpected += (unexpected.empty() ? "" : ", ") + symbol;
      }
    }
    return fail("shared library contains unexpected undefined symbols: " +
                unexpected);
  }
  if (!g_expected_undefined.empty()) {
    std::set<std::string> expected;
    std::string value = g_expected_undefined;
    for (std::size_t begin = 0, end; begin <= value.size(); begin = end + 1) {
      end = value.find(',', begin);
      expected.insert(value.substr(begin, end - begin));
      if (end == std::string::npos) {
        break;
      }
    }
    if (undefined != expected) {
      return fail("undefined symbols do not match --expected-undefined");
    }
  }
  capture_path = staging.path() / "defined.txt";
  if (int status = run({nm_path, "-D", "--defined-only", library.string()},
                       capture_path)) {
    return status;
  }
  text = read_text(capture_path);
  if (!text) {
    return fail(text.error());
  }
  std::set<std::string> expected_exports = {model_name};
  if (has_dynamic_output) {
    expected_exports.insert(model_name + "_infer_output_shapes");
  }
  if (symbols(*text) != expected_exports) {
    return fail(
      "shared library exports symbols outside the model ABI entry points");
  }
  capture_path = staging.path() / "needed.txt";
  if (int status =
        run({readelf_path, "--needed-libs", library.string()}, capture_path)) {
    return status;
  }
  text = read_text(capture_path);
  if (!text) {
    return fail(text.error());
  }
  static const std::regex needed_pattern(
    R"(^\s*(lib[^\s]+\.so(?:\.\d+)*)\s*$)",
    std::regex_constants::ECMAScript | std::regex_constants::multiline);
  bool has_address_runtime = false;
  bool has_undefined_runtime = false;
  for (auto match =
         std::sregex_iterator(text->begin(), text->end(), needed_pattern);
       match != std::sregex_iterator();
       ++match) {
    const std::string name = (*match)[1];
    has_address_runtime |=
      name.starts_with("libasan.so") || name.starts_with("libclang_rt.asan-");
    has_undefined_runtime |= name.starts_with("libubsan.so") ||
                             name.starts_with("libclang_rt.ubsan_standalone-");
    const bool sanitizer_runtime =
      (uses_address_sanitizer && (name.starts_with("libasan.so") ||
                                  name.starts_with("libclang_rt.asan-"))) ||
      (uses_undefined_sanitizer &&
       (name.starts_with("libubsan.so") ||
        name.starts_with("libclang_rt.ubsan_standalone-")));
    const bool openmp_runtime = uses_openmp && name.starts_with("libomp.so");
    const bool libmvec_runtime = uses_libmvec && name.starts_with("libmvec.so");
    if (!name.starts_with("libc.so") && !name.starts_with("libm.so") &&
        !openmp_runtime && !sanitizer_runtime && !libmvec_runtime) {
      return fail(
        std::format("shared library has an unexpected dependency: {}", name));
    }
  }
  const bool has_address_symbols = std::ranges::any_of(
    undefined,
    [](const std::string& symbol) { return symbol.starts_with("__asan_"); });
  const bool has_undefined_symbols = std::ranges::any_of(
    undefined,
    [](const std::string& symbol) { return symbol.starts_with("__ubsan_"); });
  if ((has_address_symbols && !has_address_runtime) ||
      (has_undefined_symbols && !has_undefined_runtime)) {
    return fail("shared library lacks a required sanitizer runtime dependency");
  }
  capture_path = staging.path() / "symbols.txt";
  if (int status = run({nm_path, "-D", library.string()}, capture_path)) {
    return status;
  }
  text = read_text(capture_path);
  if (!text) {
    return fail(text.error());
  }
  for (std::string_view forbidden :
       {"memrefCopy", "runner_utils", "RunnerUtils", "ncnn_runtime"}) {
    if (text->contains(forbidden)) {
      return fail("forbidden runtime symbol found in shared library");
    }
  }

  if (g_verify_execution) {
    const fs::path harness_source = staging.path() / "harness.c";
    const fs::path harness = staging.path() / "harness";
    auto harness_result =
      write_harness(harness_source, header.filename().string(), *manifest);
    if (!harness_result) {
      return fail(harness_result.error());
    }
    error.clear();
    fs::path absolute_staging = fs::absolute(staging.path(), error);
    if (error) {
      return fail(llvm::Twine("cannot make staging path absolute: ") +
                  error.message());
    }
    if (int status = run({clang_path,
                          "-std=c23",
                          harness_source.string(),
                          "-I",
                          staging.path().string(),
                          "-L",
                          staging.path().string(),
                          "-l" + model_name,
                          "-Wl,-rpath," + absolute_staging.string(),
                          "-lm",
                          "-o",
                          harness.string()})) {
      llvm::errs()
        << "ncnn-compile: error: ABI verification harness compilation failed\n";
      return status;
    }
    if (int status = run({harness.string()})) {
      llvm::errs() << "ncnn-compile: error: ABI execution verification failed "
                      "with harness exit code "
                   << status << '\n';
      return status;
    }
    llvm::errs() << "ncnn-compile: ABI execution verification passed\n";
  }

  fs::path output_parent = output_dir.parent_path();
  if (output_parent.empty()) {
    output_parent = ".";
  }
  fs::create_directories(output_parent, error);
  if (error) {
    return fail(llvm::Twine("cannot create output parent directory '") +
                output_parent.string() + "': " + error.message());
  }
  auto replacement_path = unique_sibling_path(output_dir, "replacement", true);
  if (!replacement_path) {
    return fail(replacement_path.error());
  }
  ScopedDirectory replacement(*replacement_path);
  auto publish =
    [&](const fs::path& source) -> std::expected<void, std::string> {
    error.clear();
    fs::copy_file(source,
                  replacement.path() / source.filename(),
                  fs::copy_options::none,
                  error);
    if (error) {
      return std::unexpected(std::format("cannot prepare output '{}': {}",
                                         source.filename().string(),
                                         error.message()));
    }
    return {};
  };
  if (auto result = publish(header); !result) {
    return fail(result.error());
  }
  if (auto result = publish(library); !result) {
    return fail(result.error());
  }
  if (g_emit_manifest) {
    if (auto result = publish(manifest_path); !result) {
      return fail(result.error());
    }
  }
  if (emit_execution_plan) {
    if (auto result = publish(execution_plan_path); !result) {
      return fail(result.error());
    }
  }
  const std::vector<std::pair<std::string, fs::path>> stages = {
    {"ncnn", ncnn_ir},
    {"tosa", tosa_ir},
    {"linalg", linalg_ir},
    {"memref", memref_ir},
    {"capi", capi_ir},
    {"llvm", llvm_dialect_ir},
    {"llvm-ir", llvm_ir},
    {"object", object},
    {"assembly", assembly}};
  for (const auto& [stage, path] : stages) {
    if (emitted.contains(stage)) {
      if (auto result = publish(path); !result) {
        return fail(result.error());
      }
    }
  }

  auto current_output_exists =
    validate_output_directory(output_dir, model_name);
  if (!current_output_exists) {
    return fail(current_output_exists.error());
  }
  if (current_output_exists->exists != output_exists->exists ||
      current_output_exists->identity != output_exists->identity) {
    return fail("output directory changed while compilation was in progress");
  }
  auto backup_path = unique_sibling_path(output_dir, "backup", false);
  if (!backup_path) {
    return fail(backup_path.error());
  }
  ScopedDirectory backup(*backup_path);
  if (output_exists->exists) {
    fs::rename(output_dir, backup.path(), error);
    if (error) {
      return fail(llvm::Twine("cannot move previous output directory '") +
                  output_dir.string() + "' to backup: " + error.message());
    }
    llvm::sys::fs::UniqueID backup_identity;
    if (std::error_code identity_error =
          llvm::sys::fs::getUniqueID(backup.path().string(), backup_identity);
        identity_error || backup_identity != *output_exists->identity) {
      error.clear();
      fs::rename(backup.path(), output_dir, error);
      if (error) {
        backup.release();
        return fail(
          llvm::Twine("output directory changed during publication; ") +
          "rollback failed and the moved directory remains at '" +
          backup.path().string() + "': " + error.message());
      }
      return fail("output directory changed during publication");
    }
  }
  error.clear();
  fs::rename(replacement.path(), output_dir, error);
  if (error) {
    const std::string publication_error = error.message();
    if (output_exists->exists) {
      error.clear();
      fs::rename(backup.path(), output_dir, error);
      if (error) {
        backup.release();
        return fail(llvm::Twine("cannot publish replacement output: ") +
                    publication_error +
                    "; rollback also failed; previous output remains at '" +
                    backup.path().string() + "': " + error.message());
      }
    }
    return fail(llvm::Twine("cannot publish replacement output: ") +
                publication_error);
  }
  replacement.release();
  llvm::outs() << output_dir.string() << '\n';
  return 0;
}
