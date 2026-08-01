#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <ranges>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
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
  std::string name;
  std::vector<std::int64_t> shape;
};

struct Manifest {
  std::string function;
  std::vector<Argument> inputs;
  std::vector<Argument> outputs;
};

class StagingDirectory {
 public:
  explicit StagingDirectory(fs::path path) : path_(std::move(path)) {}
  ~StagingDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }
  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
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
  for (unsigned char character : name) {
    result += std::isalnum(character) || character == '_' ? character : '_';
  }
  static const std::set<std::string> keywords = {
    "auto",      "break",          "case",         "char",     "const",
    "continue",  "default",        "do",           "double",   "else",
    "enum",      "extern",         "float",        "for",      "goto",
    "if",        "inline",         "int",          "long",     "register",
    "restrict",  "return",         "short",        "signed",   "sizeof",
    "static",    "struct",         "switch",       "typedef",  "union",
    "unsigned",  "void",           "volatile",     "while",    "_Alignas",
    "_Alignof",  "_Atomic",        "_Bool",        "_Complex", "_Generic",
    "_Noreturn", "_Static_assert", "_Thread_local"};
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result[0])) ||
      result[0] == '_' || keywords.contains(result)) {
    result.insert(0, "ncnn_");
  }
  return result;
}

std::optional<std::string> find_tool(
  const fs::path& executable_dir,
  std::string_view explicit_path,
  std::initializer_list<std::string_view> nearby_names,
  std::initializer_list<std::string_view> path_names) {
  if (!explicit_path.empty()) {
    return std::string(explicit_path);
  }
  for (std::string_view name : nearby_names) {
    for (const fs::path& directory :
         {executable_dir, executable_dir / "../bin"}) {
      fs::path candidate = directory / name;
      std::error_code error;
      if (fs::is_regular_file(candidate, error)) {
        return fs::weakly_canonical(candidate, error).string();
      }
    }
  }
  for (std::string_view name : path_names) {
    auto program = llvm::sys::findProgramByName(name);
    if (program) {
      return *program;
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

bool write_file(const fs::path& path, std::string_view contents) {
  std::error_code error;
  llvm::raw_fd_ostream output(path.string(), error);
  if (error) {
    llvm::errs() << "ncnn-compile: error: cannot write '" << path.string()
                 << "': " << error.message() << '\n';
    return false;
  }
  output << contents;
  return true;
}

std::optional<Manifest> read_manifest(const fs::path& path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    fail(llvm::Twine("cannot read ABI manifest: ") +
         buffer.getError().message());
    return std::nullopt;
  }
  auto value = llvm::json::parse((*buffer)->getBuffer());
  if (!value) {
    fail(llvm::Twine("invalid ABI manifest: ") +
         llvm::toString(value.takeError()));
    return std::nullopt;
  }
  auto* object = value->getAsObject();
  if (object == nullptr) {
    fail("ABI manifest must be a JSON object");
    return std::nullopt;
  }
  auto function = object->getString("function");
  if (!function) {
    fail("ABI manifest has no function name");
    return std::nullopt;
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
      if (!name || shape == nullptr) {
        return false;
      }
      Argument argument{.name = std::string(*name), .shape = {}};
      for (llvm::json::Value& dimension : *shape) {
        auto integer = dimension.getAsInteger();
        if (!integer || *integer < 0) {
          return false;
        }
        argument.shape.push_back(*integer);
      }
      destination.push_back(std::move(argument));
    }
    return true;
  };
  if (!parse_arguments("inputs", manifest.inputs) ||
      !parse_arguments("outputs", manifest.outputs)) {
    fail("ABI manifest has invalid inputs or outputs");
    return std::nullopt;
  }
  return manifest;
}

std::uint64_t element_count(const Argument& argument) {
  std::uint64_t count = 1;
  for (std::int64_t dimension : argument.shape) {
    count *= static_cast<std::uint64_t>(dimension);
  }
  return count;
}

bool write_header(const fs::path& path, const Manifest& manifest) {
  std::string guard = "NCNN_" + manifest.function + "_H";
  std::ranges::transform(guard, guard.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  std::string contents = "#ifndef " + guard + "\n#define " + guard + "\n\n";
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
    std::string macro = manifest.function + "_" + argument->name + "_elements";
    std::ranges::transform(
      macro, macro.begin(), [](unsigned char c) { return std::toupper(c); });
    contents += "#define " + macro + " " +
                std::to_string(element_count(*argument)) + "\n";
  }
  contents += "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\nint " +
              manifest.function + "(";
  bool first = true;
  for (const Argument& input : manifest.inputs) {
    contents += (first ? "" : ", ") + std::string("const float *") + input.name;
    first = false;
  }
  for (const Argument& output : manifest.outputs) {
    contents += (first ? "" : ", ") + std::string("float *") + output.name;
    first = false;
  }
  contents +=
    ");\n\n#ifdef __cplusplus\n}\n#endif\n\n#endif  // " + guard + "\n";
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

std::optional<std::string> read_text(const fs::path& path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer) {
    fail(llvm::Twine("cannot read '") + path.string() +
         "': " + buffer.getError().message());
    return std::nullopt;
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
                                       "exports.map",
                                       "harness",
                                       "harness.c",
                                       "libncnn_model.so",
                                       "ncnn_model.h",
                                       "ncnn_model.json"};
  return name.starts_with(".ncnn-compile-") || fixed.contains(name) ||
         name == "lib" + std::string(model_name) + ".so" ||
         name == std::string(model_name) + ".h" ||
         name == std::string(model_name) + ".json";
}

bool write_harness(const fs::path& path,
                   std::string_view header,
                   const Manifest& manifest) {
  std::vector<const Argument*> arguments;
  for (const Argument& input : manifest.inputs) {
    arguments.push_back(&input);
  }
  for (const Argument& output : manifest.outputs) {
    arguments.push_back(&output);
  }
  std::string code =
    "#include <math.h>\n#include <stddef.h>\n#include <stdlib.h>\n"
    "#include \"" +
    std::string(header) + "\"\n\nint main(void) {\n";
  for (const Argument* argument : arguments) {
    code += "  float *" + argument->name + " = calloc(" +
            std::to_string(element_count(*argument)) + ", sizeof(float));\n";
    code += "  if (!" + argument->name + ") return 2;\n";
  }
  auto call = [&](std::optional<std::size_t> null_index) {
    std::string result = manifest.function + "(";
    for (std::size_t index = 0; index < arguments.size(); ++index) {
      if (index != 0) {
        result += ", ";
      }
      result += null_index == index ? "NULL" : arguments[index]->name;
    }
    return result + ")";
  };
  code += "  if (" + call(std::nullopt) + " != 0) return 4;\n";
  for (const Argument& output : manifest.outputs) {
    code += "  for (size_t i = 0; i < " +
            std::to_string(element_count(output)) + "; ++i) if (!isfinite(" +
            output.name + "[i])) return 5;\n";
  }
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    code += "  if (" + call(index) + " == 0) return 3;\n";
  }
  for (const Argument* argument : arguments) {
    code += "  free(" + argument->name + ");\n";
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
  if (!fs::is_regular_file(param_path)) {
    return fail(llvm::Twine("input file does not exist: ") + param_path);
  }
  if (!fs::is_regular_file(bin_path)) {
    return fail(llvm::Twine("weight file does not exist: ") + bin_path);
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
  const fs::path output_dir = g_output_dir.empty()
                                ? fs::path(model_name)
                                : fs::path(g_output_dir.getValue());
  std::error_code error;
  fs::create_directories(output_dir, error);
  if (error) {
    return fail(llvm::Twine("cannot create output directory: ") +
                error.message());
  }
  for (const fs::directory_entry& entry : fs::directory_iterator(output_dir)) {
    if (!is_generated_output(entry.path(), model_name)) {
      return fail(llvm::Twine("output directory contains a file not owned by "
                              "ncnn-compile: ") +
                  entry.path().filename().string());
    }
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
  auto translate = find_tool(
    executable_dir, g_translate, {}, {"mlir-translate-21", "mlir-translate"});
  auto clang = find_tool(executable_dir, g_clang, {}, {"clang-21", "clang"});
  auto nm = find_tool(executable_dir, g_nm, {}, {"llvm-nm-21", "llvm-nm"});
  auto readelf = find_tool(
    executable_dir, g_readelf, {}, {"llvm-readelf-21", "llvm-readelf"});
  if (!driver || !opt || !translate || !clang || !nm || !readelf) {
    return fail(
      "required compiler tool not found; use -v and verify PATH or "
      "the installed toolchain");
  }

  llvm::SmallString<256> staging_storage;
  if (llvm::sys::fs::createUniqueDirectory("ncnn-compile", staging_storage)) {
    return fail("cannot create staging directory");
  }
  StagingDirectory staging(fs::path(staging_storage.str().str()));
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

  if (int status =
        run({*driver, param_path, "--bin", bin_path, "-o", ncnn_ir.string()})) {
    return status;
  }
  if (int status = run({*opt,
                        "--ncnn-to-tosa-pipeline",
                        ncnn_ir.string(),
                        "-o",
                        tosa_ir.string()})) {
    return status;
  }
  if (int status = run({*opt,
                        "--ncnn-tosa-to-linalg-pipeline",
                        tosa_ir.string(),
                        "-o",
                        linalg_ir.string()})) {
    return status;
  }
  if (int status = run({*opt,
                        "--ncnn-linalg-to-memref-pipeline",
                        linalg_ir.string(),
                        "-o",
                        memref_ir.string()})) {
    return status;
  }
  const std::string capi_option =
    "--generate-ncnn-c-api=export-name=" + model_name +
    " manifest-path=" + manifest_path.string();
  if (int status =
        run({*opt, capi_option, memref_ir.string(), "-o", capi_ir.string()})) {
    return status;
  }
  if (int status = run({*opt,
                        "--ncnn-memref-to-llvm-pipeline",
                        capi_ir.string(),
                        "-o",
                        llvm_dialect_ir.string()})) {
    return status;
  }
  if (int status = run({*translate,
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
    *clang, "-x", "ir", "-fPIC", optimization};
  compile.insert(compile.end(), target_args.begin(), target_args.end());
  compile.insert(compile.end(), codegen_args.begin(), codegen_args.end());
  compile.insert(compile.end(), g_clang_args.begin(), g_clang_args.end());
  compile.insert(compile.end(),
                 {"-c", llvm_ir.string(), "-o", object.string()});
  if (int status = run(compile)) {
    return status;
  }

  auto manifest = read_manifest(manifest_path);
  if (!manifest || !write_header(header, *manifest) ||
      !write_file(exports,
                  "{\n  global: " + model_name + ";\n  local: *;\n};\n")) {
    return 1;
  }
  std::vector<std::string> link = {
    *clang, "-shared", "-nostdlib", optimization};
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
  if (int status =
        run({*nm, "-D", "--undefined-only", library.string()}, capture_path)) {
    return status;
  }
  auto text = read_text(capture_path);
  if (!text) {
    return 1;
  }
  const std::set<std::string> undefined = symbols(*text);
  const std::set<std::string> allowed = {
    "expf", "free", "malloc", "memcpy", "memset"};
  if (!std::ranges::includes(allowed, undefined)) {
    return fail("shared library contains unexpected undefined symbols");
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
  if (int status =
        run({*nm, "-D", "--defined-only", library.string()}, capture_path)) {
    return status;
  }
  text = read_text(capture_path);
  if (!text || symbols(*text) != std::set<std::string>{model_name}) {
    return fail(
      "shared library exports symbols other than the model entry point");
  }
  capture_path = staging.path() / "needed.txt";
  if (int status =
        run({*readelf, "--needed-libs", library.string()}, capture_path)) {
    return status;
  }
  text = read_text(capture_path);
  if (!text) {
    return 1;
  }
  static const std::regex needed_pattern(R"(lib[^\s]+\.so(?:\.\d+)*)");
  for (auto match =
         std::sregex_iterator(text->begin(), text->end(), needed_pattern);
       match != std::sregex_iterator();
       ++match) {
    const std::string name = (*match)[0];
    if (!name.starts_with("libc.so") && !name.starts_with("libm.so")) {
      return fail("shared library has an unexpected dependency: " + name);
    }
  }
  capture_path = staging.path() / "symbols.txt";
  if (int status = run({*nm, "-D", library.string()}, capture_path)) {
    return status;
  }
  text = read_text(capture_path);
  if (!text) {
    return 1;
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
    if (!write_harness(harness_source, header.filename().string(), *manifest)) {
      return 1;
    }
    if (int status = run({*clang,
                          "-std=c23",
                          harness_source.string(),
                          "-I",
                          staging.path().string(),
                          "-L",
                          staging.path().string(),
                          "-l" + model_name,
                          "-Wl,-rpath," + fs::absolute(staging.path()).string(),
                          "-lm",
                          "-o",
                          harness.string()})) {
      return status;
    }
    if (int status = run({harness.string()})) {
      return status;
    }
  }

  for (const fs::directory_entry& entry : fs::directory_iterator(output_dir)) {
    fs::remove_all(entry.path(), error);
    if (error) {
      return fail(llvm::Twine("cannot remove stale output '") +
                  entry.path().string() + "': " + error.message());
    }
  }
  auto publish = [&](const fs::path& source) {
    error.clear();
    fs::copy_file(source,
                  output_dir / source.filename(),
                  fs::copy_options::overwrite_existing,
                  error);
    return !error;
  };
  if (!publish(header) || !publish(library)) {
    return fail("cannot publish output files");
  }
  if (g_emit_manifest && !publish(manifest_path)) {
    return fail("cannot publish ABI manifest");
  }
  const std::vector<std::pair<std::string, fs::path>> stages = {
    {"ncnn", ncnn_ir},
    {"tosa", tosa_ir},
    {"linalg", linalg_ir},
    {"memref", memref_ir},
    {"capi", capi_ir},
    {"llvm", llvm_dialect_ir}};
  for (const auto& [stage, path] : stages) {
    if (emitted.contains(stage) && !publish(path)) {
      return fail("cannot publish MLIR stage");
    }
  }
  llvm::outs() << output_dir.string() << '\n';
  return 0;
}
