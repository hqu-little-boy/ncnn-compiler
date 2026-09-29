#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ncnn_mlir::vulkan {

inline constexpr std::uint32_t kExecutionPlanSchemaVersion = 1;
inline constexpr std::uint64_t kMinimumStorageBufferRange =
  128ULL * 1024ULL * 1024ULL;

enum class ResourceKind : std::uint8_t { Input, Output, Intermediate, Weight };
enum class ResourceAccess : std::uint8_t { Read, Write };

struct LayoutTransform final {
  std::string from_layout;
  std::string to_layout;
  std::string operation;
};

struct TensorResource final {
  std::uint32_t id;
  ResourceKind kind;
  std::string name;
  std::vector<std::uint64_t> shape;
  std::string logical_layout;
  std::string physical_layout;
  std::string host_layout;
  std::optional<LayoutTransform> layout_transform;
  std::string weight_sha256;
  std::uint64_t byte_size;
  std::uint32_t first_kernel;
  std::uint32_t last_kernel;
};

struct DescriptorBinding final {
  std::uint32_t set;
  std::uint32_t binding;
  std::uint32_t resource_id;
  ResourceAccess access;
};

struct PushConstantBlock final {
  std::uint32_t offset;
  std::vector<std::uint8_t> bytes;
};

struct KernelExecution final {
  std::uint32_t id;
  std::string operation;
  std::string entry_point;
  std::array<std::uint32_t, 3> local_size;
  std::vector<std::uint32_t> reads;
  std::vector<std::uint32_t> writes;
  std::vector<DescriptorBinding> bindings;
  std::vector<PushConstantBlock> push_constants;
};

struct VulkanExecutionPlan final {
  std::uint32_t schema_version;
  std::string model;
  std::string build_identity;
  std::string host_triple;
  std::string device_target;
  std::vector<TensorResource> resources;
  std::vector<KernelExecution> kernels;
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> outputs;
  std::string plan_hash;
};

[[nodiscard]] std::expected<void, std::string> verifyExecutionPlan(
  const VulkanExecutionPlan& plan);
[[nodiscard]] std::expected<std::string, std::string> serializeExecutionPlan(
  const VulkanExecutionPlan& plan);
[[nodiscard]] std::expected<VulkanExecutionPlan, std::string>
deserializeExecutionPlan(std::string_view json);

}  // namespace ncnn_mlir::vulkan
