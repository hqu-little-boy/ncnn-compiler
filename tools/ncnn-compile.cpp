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

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileSystem/UniqueID.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

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
                 "Repeat once per Input layer"),
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
  llvm::cl::desc("Keep MLIR stages (repeat or comma-separate): "
                 "ncnn,tosa,linalg,memref,capi,llvm,all"),
  llvm::cl::CommaSeparated,
  llvm::cl::cat(g_category));
llvm::cl::opt<std::string> g_optimization(
  "O",
  llvm::cl::desc("Optimization level (0, 1, 2, or 3)"),
  llvm::cl::Prefix,
  llvm::cl::init("3"),
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
llvm::cl::opt<bool> g_verify_execution(
  "verify-execution",
  llvm::cl::desc("Build and run an ABI smoke harness"),
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
  bool dynamic_rank;
  std::uint32_t rank_min;
  std::uint32_t rank_max;
  std::vector<DimensionConstraint> dimension_constraints;
};

struct Manifest {
  std::string function;
  std::vector<Argument> inputs;
  std::vector<Argument> outputs;
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
  Manifest manifest{
    .function = std::string(*function), .inputs = {}, .outputs = {}};
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
        .dynamic_rank =
          argument_object->getBoolean("dynamic_rank").value_or(false),
        .rank_min = static_cast<std::uint32_t>(
          argument_object->getInteger("rank_min").value_or(0)),
        .rank_max = static_cast<std::uint32_t>(
          argument_object->getInteger("rank_max").value_or(0)),
        .dimension_constraints = {}};
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
        output.shape_source_input < 0 && !output.shape_depends_on_data) {
      return std::unexpected(
        "ABI manifest dynamic output has no input shape source");
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
      result.push_back(std::move(object));
    }
    return result;
  };
  llvm::json::Object object;
  object["function"] = manifest.function;
  object["inputs"] = arguments(manifest.inputs);
  object["outputs"] = arguments(manifest.outputs);
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
    "#define NCNN_MAX_RANK 4\n\n"
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
                                       "model.llvm.mlir"};
  return fixed.contains(name) ||
         name == "lib" + std::string(model_name) + ".so" ||
         name == std::string(model_name) + ".h" ||
         name == std::string(model_name) + ".json";
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
      "    if ({}(input1, input1_shape, rank, output1) != 0) return 4;\n"
      "    if ({}(input1, input1_shape, rank, output1) != 0) return 9;\n"
      "    size_t count = element_count(input1_shape, rank);\n"
      "    for (size_t i = 0; i < count; ++i) if (output1[i] != input1[i]) "
      "return 13;\n"
      "  }}\n"
      "  if ({}(input1, input1_shape, 0, output1) == 0) return 10;\n"
      "  if ({}(input1, input1_shape, 5, output1) == 0) return 11;\n"
      "  if ({}_infer_output_shapes(input1_shape, 4, output1_shape, 3, "
      "&output1_rank) == 0) return 12;\n"
      "  if ({}(NULL, input1_shape, 1, output1) == 0) return 3;\n"
      "  if ({}(input1, NULL, 1, output1) == 0) return 3;\n"
      "  if ({}(input1, input1_shape, 1, NULL) == 0) return 3;\n"
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
    code += std::format(
      "  if ({} == 0) {{ fprintf(stderr, \"ABI verification failed: NULL "
      "argument {} was accepted\\n\"); return 3; }}\n",
      call(index),
      call_arguments[index]);
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

  const std::set<std::string> valid_stages = {
    "ncnn", "tosa", "linalg", "memref", "capi", "llvm", "all"};
  std::set<std::string> emitted;
  for (const std::string& stage : g_emit) {
    if (!valid_stages.contains(stage)) {
      return fail(llvm::Twine("invalid --emit stage: ") + stage);
    }
    emitted.insert(stage);
  }
  if (emitted.contains("all")) {
    emitted = {"ncnn", "tosa", "linalg", "memref", "capi", "llvm"};
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
  for (const ToolResult* tool :
       {&driver, &opt, &translate, &clang, &nm, &readelf}) {
    if (!*tool) {
      return fail(tool->error());
    }
  }
  if (!*driver || !*opt || !*translate || !*clang || !*nm || !*readelf) {
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
  const fs::path ncnn_ir = staging.path() / "model.ncnn.mlir";
  const fs::path tosa_ir = staging.path() / "model.tosa.mlir";
  const fs::path linalg_ir = staging.path() / "model.linalg.mlir";
  const fs::path memref_ir = staging.path() / "model.memref.mlir";
  const fs::path capi_ir = staging.path() / "model.capi.mlir";
  const fs::path llvm_dialect_ir = staging.path() / "model.llvm.mlir";
  const fs::path llvm_ir = staging.path() / "model.ll";
  const fs::path object = staging.path() / "model.o";
  const fs::path manifest_path = staging.path() / (model_name + ".json");
  const fs::path header = staging.path() / (model_name + ".h");
  const fs::path exports = staging.path() / "exports.map";
  const fs::path library = staging.path() / ("lib" + model_name + ".so");

  std::vector<std::string> driver_command{
    driver_path, param_path, "--bin", bin_path, "-o", ncnn_ir.string()};
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
  if (int status = run({opt_path,
                        "--ncnn-tosa-to-linalg-pipeline",
                        tosa_ir.string(),
                        "-o",
                        linalg_ir.string()})) {
    return status;
  }
  if (int status = run({opt_path,
                        "--ncnn-linalg-to-memref-pipeline",
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
  if (int status = run({opt_path,
                        "--ncnn-memref-to-llvm-pipeline",
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

  std::vector<std::string> target_args;
  std::vector<std::string> codegen_args;
  if (!g_target_triple.empty()) {
    target_args.push_back("--target=" + g_target_triple);
  }
  if (!g_sysroot.empty()) {
    target_args.push_back("--sysroot=" + g_sysroot);
  }
  if (!g_march.empty()) {
    codegen_args.push_back("-march=" + g_march);
  }
  if (!g_mcpu.empty()) {
    codegen_args.push_back("-mcpu=" + g_mcpu);
  }
  if (!g_mtune.empty()) {
    codegen_args.push_back("-mtune=" + g_mtune);
  }
  for (const std::string& feature : g_target_features) {
    codegen_args.insert(codegen_args.end(),
                        {"-Xclang", "-target-feature", "-Xclang", feature});
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
                 {"-c", llvm_ir.string(), "-o", object.string()});
  if (int status = run(compile)) {
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
  std::vector<std::string> link = {
    clang_path, "-shared", "-nostdlib", optimization};
  link.insert(link.end(), target_args.begin(), target_args.end());
  link.push_back(object.string());
  link.insert(link.end(), g_linker_args.begin(), g_linker_args.end());
  link.insert(link.end(),
              {"-Wl,-z,defs",
               "-Wl,--no-undefined",
               "-Wl,--build-id=none",
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
  const std::set<std::string> allowed = {
    "erfcf", "erff", "expf", "free", "malloc", "memcpy", "memset", "powf"};
  if (!std::ranges::includes(allowed, undefined)) {
    std::string unexpected;
    for (const std::string& symbol : undefined) {
      if (!allowed.contains(symbol)) {
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
  for (auto match =
         std::sregex_iterator(text->begin(), text->end(), needed_pattern);
       match != std::sregex_iterator();
       ++match) {
    const std::string name = (*match)[1];
    if (!name.starts_with("libc.so") && !name.starts_with("libm.so")) {
      return fail(
        std::format("shared library has an unexpected dependency: {}", name));
    }
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
  const std::vector<std::pair<std::string, fs::path>> stages = {
    {"ncnn", ncnn_ir},
    {"tosa", tosa_ir},
    {"linalg", linalg_ir},
    {"memref", memref_ir},
    {"capi", capi_ir},
    {"llvm", llvm_dialect_ir}};
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
