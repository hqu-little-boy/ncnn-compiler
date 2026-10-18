#include "VulkanProduct.hpp"
#include "vulkan_shader_fixture.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Support/VulkanHostProgram.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace {

using ncnn_compile::VulkanLifecycleSource;
using ncnn_compile::VulkanModelPackage;
using ncnn_mlir::vulkan::DescriptorBinding;
using ncnn_mlir::vulkan::KernelExecution;
using ncnn_mlir::vulkan::ResourceAccess;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;
using ncnn_mlir::vulkan::VulkanModelShader;
using ncnn_mlir::vulkan::VulkanModelWeight;

constexpr std::string_view kModelName = "lifecycle_fixture";
constexpr std::string_view kHostTriple = "x86_64-linux-gnu";
constexpr std::uint64_t kElementCount = 17;

VulkanModelPackage make_package(bool with_weight) {
  std::vector<TensorResource> resources{
    TensorResource{.id = 7,
                   .kind = ResourceKind::Input,
                   .name = "input",
                   .shape = {kElementCount},
                   .logical_layout = "linear",
                   .physical_layout = "linear",
                   .host_layout = "linear",
                   .layout_transform = std::nullopt,
                   .weight_sha256 = "",
                   .byte_size = kElementCount * sizeof(float),
                   .first_kernel = 0,
                   .last_kernel = 0},
    TensorResource{.id = 41,
                   .kind = ResourceKind::Output,
                   .name = "output",
                   .shape = {kElementCount},
                   .logical_layout = "linear",
                   .physical_layout = "linear",
                   .host_layout = "linear",
                   .layout_transform = std::nullopt,
                   .weight_sha256 = "",
                   .byte_size = kElementCount * sizeof(float),
                   .first_kernel = 0,
                   .last_kernel = 0},
  };
  std::vector<DescriptorBinding> bindings{
    DescriptorBinding{
      .set = 0, .binding = 0, .resource_id = 7, .access = ResourceAccess::Read},
  };
  std::vector<std::uint32_t> reads{7};
  std::vector<VulkanModelWeight> weights;
  if (with_weight) {
    constexpr std::uint32_t kWeightId = 103;
    constexpr std::size_t kWeightElements = 4;
    resources.push_back(TensorResource{
      .id = kWeightId,
      .kind = ResourceKind::Weight,
      .name = "relu_scale",
      .shape = {kWeightElements},
      .logical_layout = "linear",
      .physical_layout = "linear",
      .host_layout = "",
      .layout_transform = std::nullopt,
      .weight_sha256 = "",
      .byte_size = kWeightElements * sizeof(float),
      .first_kernel = 0,
      .last_kernel = 0,
    });
    bindings.push_back(DescriptorBinding{.set = 0,
                                         .binding = 1,
                                         .resource_id = kWeightId,
                                         .access = ResourceAccess::Read});
    reads.push_back(kWeightId);
    weights.push_back(VulkanModelWeight{
      .resource_id = kWeightId,
      .bytes = {0x00,
                0x00,
                0x80,
                0x3f,
                0x00,
                0x00,
                0x80,
                0x3f,
                0x00,
                0x00,
                0x80,
                0x3f,
                0x00,
                0x00,
                0x80,
                0x3f},
    });
    auto weight_hash = ncnn_mlir::vulkan::vulkanWeightSHA256(
      resources.back(), weights.back().bytes);
    if (!weight_hash) {
      throw std::runtime_error("cannot hash lifecycle weight fixture: " +
                               weight_hash.error());
    }
    resources.back().weight_sha256 = *weight_hash;
  }
  bindings.push_back(
    DescriptorBinding{.set = 0,
                      .binding = static_cast<std::uint32_t>(bindings.size()),
                      .resource_id = 41,
                      .access = ResourceAccess::Write});
  const KernelExecution kernel{
    .id = 29,
    .operation = "relu",
    .entry_point = "relu",
    .local_size = {64, 1, 1},
    .reads = std::move(reads),
    .writes = {41},
    .bindings = std::move(bindings),
    .push_constants = {},
  };
  VulkanExecutionPlan plan{
    .schema_version = ncnn_mlir::vulkan::kExecutionPlanSchemaVersion,
    .model = std::string(kModelName),
    .build_identity = "lifecycle-fixture-source",
    .host_triple = std::string(kHostTriple),
    .device_target = "vulkan1.1-spirv1.0",
    .resources = std::move(resources),
    .kernels = {kernel},
    .inputs = {7},
    .outputs = {41},
    .plan_hash = "",
  };
  auto shader = ncnn_mlir::vulkan::test::makeTestVulkanShader(
    plan, plan.kernels.front(), "lifecycle_fixture_vulkan_k29");
  if (!shader) {
    throw std::runtime_error("cannot build checked Vulkan shader fixture: " +
                             shader.error());
  }
  auto plan_json = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  if (!plan_json) {
    throw std::runtime_error("cannot serialize lifecycle execution plan: " +
                             plan_json.error());
  }
  const std::array<VulkanModelShader, 1> shaders{std::move(*shader)};
  auto package = ncnn_compile::makeVulkanModelPackage(
    kModelName, kHostTriple, *plan_json, shaders, weights, "", "");
  if (!package) {
    throw std::runtime_error("cannot build checked Vulkan package: " +
                             package.error());
  }
  return std::move(*package);
}

void write_file(const std::filesystem::path& path, std::string_view text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output ||
      !output.write(text.data(), static_cast<std::streamsize>(text.size()))) {
    throw std::runtime_error("cannot write generated lifecycle file: " +
                             path.string());
  }
}

std::string encode_spirv(std::span<const std::uint32_t> words) {
  std::string bytes;
  if (words.size() > bytes.max_size() / sizeof(std::uint32_t)) {
    throw std::runtime_error("SPIR-V fixture size overflows string capacity");
  }
  bytes.reserve(words.size() * sizeof(std::uint32_t));
  for (const std::uint32_t word : words) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<char>((word >> shift) & 0xffU));
    }
  }
  return bytes;
}

struct Options final {
  std::string kind;
  std::filesystem::path output_dir;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + std::string(argument));
    }
    if (argument == "--kind") {
      options.kind = argv[++index];
    } else if (argument == "--output-dir") {
      options.output_dir = argv[++index];
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argument));
    }
  }
  if ((options.kind != "relu" && options.kind != "weighted") ||
      options.output_dir.empty()) {
    throw std::runtime_error(
      "usage: vulkan_lifecycle_fixture_emitter --kind relu|weighted "
      "--output-dir PATH");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::filesystem::create_directories(options.output_dir);
    VulkanModelPackage package = make_package(options.kind == "weighted");
    auto generated = ncnn_compile::makeVulkanLifecycleSource(package);
    if (!generated) {
      throw std::runtime_error("cannot generate lifecycle source: " +
                               generated.error());
    }
    auto host_program_json =
      ncnn_mlir::vulkan::serializeVulkanHostProgram(package.host_program);
    if (!host_program_json) {
      throw std::runtime_error("cannot serialize checked host program: " +
                               host_program_json.error());
    }
    write_file(options.output_dir / "candidate_source.cpp", generated->source);
    write_file(options.output_dir / "lifecycle_fixture_vulkan.h",
               generated->header);
    write_file(options.output_dir / "host-program-identity.txt",
               generated->host_program_identity + "\n");
    write_file(options.output_dir / "shader_0.spv",
               encode_spirv(package.shaders.front().words));
    write_file(options.output_dir / "plan.json", package.plan_json);
    write_file(options.output_dir / "host_program.json", *host_program_json);
    write_file(options.output_dir / "package_manifest.json",
               package.manifest_json);
  } catch (const std::exception& error) {
    std::cerr << "vulkan_lifecycle_fixture_emitter: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
