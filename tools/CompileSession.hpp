#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem/UniqueID.h"
#include "llvm/Support/raw_ostream.h"

namespace ncnn_compile {

namespace fs = std::filesystem;

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
  // Default-constructed sessions hold no directory: the destructor is a
  // no-op until move-assignment installs the real staging directory.
  ScopedDirectory() : remove_(false) {}
  ScopedDirectory& operator=(ScopedDirectory&& other) {
    path_ = std::move(other.path_);
    remove_ = std::exchange(other.remove_, false);
    return *this;
  }
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

using ToolResult = std::expected<std::optional<std::string>, std::string>;

struct OutputDirectoryState {
  bool exists;
  std::optional<llvm::sys::fs::UniqueID> identity;
};

struct ClangTargetArguments {
  std::vector<std::string> target;
  std::vector<std::string> isa;
};

struct TuningSettings {
  std::string profile;
  std::string status = "stable";
  std::string fallbackReason;
  int64_t matmulMRows = 4;
  int64_t matmulAccColumns = 16;
  std::string matmulPacking = "auto";
  unsigned rowChunkLanes = 8;
  int64_t matmulI8Rows = 2;
  int64_t matmulI8AccColumns = 4;
};

// Declaration only.  Member names deliberately match the locals of the original
// main() so the extracted bodies are verbatim copies.
class CompileSession final {
 public:
  int run(int argc, char** argv);  // orchestration only

 private:
  int parseArguments(int argc, char** argv);
  int resolveTools(char** argv);
  int resolveTarget();
  int declareArtifacts();
  int runPipeline();
  int emitABI();
  int linkAndAudit();
  int verifyExecution();
  int publishOutputs();

  // State shared across the extracted phases; names match the original
  // main() locals so the moved bodies stay verbatim.
  std::error_code error;
  std::string param_path;
  std::string bin_path;
  std::string model_name;
  fs::path output_dir;
  std::expected<OutputDirectoryState, std::string> output_exists;
  std::set<std::string> emitted;
  ToolResult driver;
  ToolResult opt;
  ToolResult translate;
  ToolResult clang;
  ToolResult nm;
  ToolResult readelf;
  ToolResult llvm_as;
  std::string driver_path;
  std::string opt_path;
  std::string translate_path;
  std::string clang_path;
  std::string nm_path;
  std::string readelf_path;
  llvm::SmallString<256> staging_storage;
  ScopedDirectory staging;
  std::string effective_target_triple;
  ClangTargetArguments clang_target;
  std::vector<std::string> target_args;
  std::vector<std::string> isa_args;
  std::vector<std::string> codegen_args;
  unsigned effective_threads;
  unsigned vector_lanes;
  bool vector_scalable;
  bool vector_active;
  TuningSettings tuning;
  std::string resolved_int8_target;
  std::string resolved_vector_math;
  std::string vector_math_abi;
  unsigned vector_math_lanes;
  bool uses_libmvec;
  fs::path sleef_archive;
  fs::path ncnn_ir;
  fs::path tosa_ir;
  fs::path linalg_ir;
  fs::path memref_ir;
  fs::path capi_ir;
  fs::path llvm_dialect_ir;
  fs::path llvm_ir;
  fs::path object;
  fs::path assembly;
  fs::path profile_object;
  fs::path profile_writer_object;
  fs::path profile_runtime_source;
  fs::path profile_writer_source;
  fs::path manifest_path;
  fs::path execution_plan_path;
  fs::path header;
  fs::path exports;
  fs::path library;
  bool emit_execution_plan;
  std::string codegen_identity;
  std::string codegen_identity_transport;
  std::string execution_plan_hash;
  std::string execution_plan_revision;
  std::string execution_attribution_revision;
  bool uses_openmp;
  bool uses_sleef;
  fs::path llvm_bitcode;
  std::string optimization;
  std::expected<Manifest, std::string> manifest;
  bool has_dynamic_output;
  bool uses_address_sanitizer;
  bool uses_undefined_sanitizer;
  bool uses_sanitizer;
  fs::path capture_path;
  std::expected<std::string, std::string> text;
  std::set<std::string> undefined;
};

}  // namespace ncnn_compile
