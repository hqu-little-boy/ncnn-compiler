#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/StableHash.hpp"

namespace ncnn_mlir::vulkan {
namespace {

using JsonArray = llvm::json::Array;
using JsonObject = llvm::json::Object;
using JsonValue = llvm::json::Value;

constexpr std::string_view kPlanKind{
  mlir::ncnn::contract::kVulkanExecutionPlanKind.data(),
  mlir::ncnn::contract::kVulkanExecutionPlanKind.size()};
constexpr std::uint64_t kF32ByteWidth = 4;
constexpr std::uint32_t kMinimumPushConstantBytes = 128;
constexpr std::array<std::uint32_t, 3> kMinimumLocalSize = {128, 128, 64};

std::string_view toString(ResourceKind kind) {
  switch (kind) {
    case ResourceKind::Input:
      return "input";
    case ResourceKind::Output:
      return "output";
    case ResourceKind::Intermediate:
      return "intermediate";
    case ResourceKind::Weight:
      return "weight";
  }
  return {};
}

std::string_view toString(ResourceAccess access) {
  switch (access) {
    case ResourceAccess::Read:
      return "read";
    case ResourceAccess::Write:
      return "write";
  }
  return {};
}

std::expected<ResourceKind, std::string> parseResourceKind(
  std::string_view value) {
  if (value == "input") {
    return ResourceKind::Input;
  }
  if (value == "output") {
    return ResourceKind::Output;
  }
  if (value == "intermediate") {
    return ResourceKind::Intermediate;
  }
  if (value == "weight") {
    return ResourceKind::Weight;
  }
  return std::unexpected("resource.kind is unknown");
}

std::expected<ResourceAccess, std::string> parseResourceAccess(
  std::string_view value) {
  if (value == "read") {
    return ResourceAccess::Read;
  }
  if (value == "write") {
    return ResourceAccess::Write;
  }
  return std::unexpected("descriptor.access must be read or write");
}

std::expected<std::uint64_t, std::string> checkedByteSize(
  const std::vector<std::uint64_t>& shape) {
  if (shape.empty() || shape.size() > 4) {
    return std::unexpected("resource shape rank must be between 1 and 4");
  }
  std::uint64_t elements = 1;
  for (std::uint64_t extent : shape) {
    if (extent == 0 ||
        extent > std::numeric_limits<std::uint64_t>::max() / elements) {
      return std::unexpected(
        "resource element count is zero or overflows uint64");
    }
    elements *= extent;
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / kF32ByteWidth) {
    return std::unexpected("resource byte size overflows uint64");
  }
  const std::uint64_t bytes = elements * kF32ByteWidth;
  if (bytes > kMinimumStorageBufferRange) {
    return std::unexpected(
      "resource exceeds the Vulkan 1.1 minimum storage-buffer range");
  }
  return bytes;
}

bool isSha256(std::string_view digest) {
  if (digest.size() != 64) {
    return false;
  }
  return std::ranges::all_of(digest, [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  });
}

std::expected<void, std::string> requireOnlyFields(
  const JsonObject& object, std::initializer_list<std::string_view> allowed) {
  for (const auto& item : object) {
    const llvm::StringRef key = item.first;
    const std::string_view keyView(key.data(), key.size());
    if (std::ranges::find(allowed, keyView) == allowed.end()) {
      return std::unexpected(
        std::format("unexpected JSON field '{}'", keyView));
    }
  }
  return {};
}

std::expected<std::string, std::string> requiredString(const JsonObject& object,
                                                       llvm::StringRef key) {
  auto value = object.getString(key);
  if (!value) {
    return std::unexpected(
      std::format("missing or invalid string field '{}'", key.str()));
  }
  return value->str();
}

std::expected<std::uint32_t, std::string> requiredU32(const JsonObject& object,
                                                      llvm::StringRef key) {
  auto value = object.getInteger(key);
  if (!value || *value < 0 ||
      std::cmp_greater(*value, std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected(
      std::format("missing or invalid uint32 field '{}'", key.str()));
  }
  return static_cast<std::uint32_t>(*value);
}

std::expected<std::uint64_t, std::string> requiredU64(const JsonObject& object,
                                                      llvm::StringRef key) {
  auto value = object.getInteger(key);
  if (!value || *value < 0) {
    return std::unexpected(
      std::format("missing or invalid uint64 field '{}'", key.str()));
  }
  return static_cast<std::uint64_t>(*value);
}

std::expected<const JsonArray*, std::string> requiredArray(
  const JsonObject& object, llvm::StringRef key) {
  const JsonValue* value = object.get(key);
  if (!value) {
    return std::unexpected(std::format("missing array field '{}'", key.str()));
  }
  const JsonArray* array = value->getAsArray();
  if (!array) {
    return std::unexpected(
      std::format("field '{}' must be an array", key.str()));
  }
  return array;
}

std::expected<std::optional<LayoutTransform>, std::string>
requiredLayoutTransform(const JsonObject& resource) {
  const JsonValue* value = resource.get("layout_transform");
  if (!value) {
    return std::unexpected("missing resource.layout_transform field");
  }
  if (value->getAsNull()) {
    return std::optional<LayoutTransform>();
  }
  const JsonObject* object = value->getAsObject();
  if (!object) {
    return std::unexpected(
      "resource.layout_transform must be null or an object");
  }
  if (auto fields =
        requireOnlyFields(*object, {"from_layout", "to_layout", "operation"});
      !fields) {
    return std::unexpected(fields.error());
  }
  auto from = requiredString(*object, "from_layout");
  auto to = requiredString(*object, "to_layout");
  auto operation = requiredString(*object, "operation");
  if (!from || !to || !operation) {
    return std::unexpected("layout_transform has a missing or invalid field");
  }
  return std::optional<LayoutTransform>(LayoutTransform{
    .from_layout = std::move(*from),
    .to_layout = std::move(*to),
    .operation = std::move(*operation),
  });
}

std::expected<std::uint32_t, std::string> valueU32(const JsonValue& value,
                                                   std::string_view field) {
  auto integer = value.getAsInteger();
  if (!integer || *integer < 0 ||
      std::cmp_greater(*integer, std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected(std::format("'{}' must be a uint32", field));
  }
  return static_cast<std::uint32_t>(*integer);
}

std::expected<std::uint64_t, std::string> valueU64(const JsonValue& value,
                                                   std::string_view field) {
  auto integer = value.getAsInteger();
  if (!integer || *integer < 0) {
    return std::unexpected(std::format("'{}' must be a uint64", field));
  }
  return static_cast<std::uint64_t>(*integer);
}

std::expected<std::vector<std::uint32_t>, std::string> parseU32Array(
  const JsonArray& array, std::string_view field) {
  std::vector<std::uint32_t> values;
  values.reserve(array.size());
  for (const JsonValue& value : array) {
    auto parsed = valueU32(value, field);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    values.push_back(*parsed);
  }
  return values;
}

std::expected<std::vector<std::uint64_t>, std::string> parseU64Array(
  const JsonArray& array, std::string_view field) {
  std::vector<std::uint64_t> values;
  values.reserve(array.size());
  for (const JsonValue& value : array) {
    auto parsed = valueU64(value, field);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    values.push_back(*parsed);
  }
  return values;
}

std::string bytesToHex(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (std::uint8_t byte : bytes) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

std::expected<std::vector<std::uint8_t>, std::string> parseHexBytes(
  std::string_view text) {
  if (text.size() % 2 != 0) {
    return std::unexpected("push-constant data must have an even hex length");
  }
  std::vector<std::uint8_t> result;
  result.reserve(text.size() / 2);
  auto digit = [](char value) -> int {
    if (value >= '0' && value <= '9') {
      return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
      return value - 'a' + 10;
    }
    return -1;
  };
  for (std::size_t index = 0; index < text.size(); index += 2) {
    const int high = digit(text[index]);
    const int low = digit(text[index + 1]);
    if (high < 0 || low < 0) {
      return std::unexpected(
        "push-constant data must use lowercase hexadecimal");
    }
    result.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return result;
}

JsonValue toJson(const VulkanExecutionPlan& plan, bool includeHash) {
  JsonObject root;
  root["kind"] = std::string(kPlanKind);
  root["schema_version"] = static_cast<std::int64_t>(plan.schema_version);
  root["model"] = plan.model;
  root["build_identity"] = plan.build_identity;
  root["host_triple"] = plan.host_triple;
  root["device_target"] = plan.device_target;

  JsonArray resources;
  for (const TensorResource& resource : plan.resources) {
    JsonArray shape;
    for (std::uint64_t extent : resource.shape) {
      shape.push_back(static_cast<std::int64_t>(extent));
    }
    JsonObject object;
    object["id"] = static_cast<std::int64_t>(resource.id);
    object["kind"] = std::string(toString(resource.kind));
    object["name"] = resource.name;
    object["shape"] = std::move(shape);
    object["element_type"] = "f32";
    object["logical_layout"] = resource.logical_layout;
    object["physical_layout"] = resource.physical_layout;
    object["host_layout"] = resource.host_layout;
    if (resource.layout_transform) {
      JsonObject transform;
      transform["from_layout"] = resource.layout_transform->from_layout;
      transform["to_layout"] = resource.layout_transform->to_layout;
      transform["operation"] = resource.layout_transform->operation;
      object["layout_transform"] = std::move(transform);
    } else {
      object["layout_transform"] = nullptr;
    }
    object["weight_sha256"] = resource.weight_sha256;
    object["byte_size"] = static_cast<std::int64_t>(resource.byte_size);
    object["first_kernel"] = static_cast<std::int64_t>(resource.first_kernel);
    object["last_kernel"] = static_cast<std::int64_t>(resource.last_kernel);
    resources.push_back(std::move(object));
  }
  root["resources"] = std::move(resources);

  JsonArray kernels;
  for (const KernelExecution& kernel : plan.kernels) {
    JsonArray localSize;
    for (std::uint32_t value : kernel.local_size) {
      localSize.push_back(static_cast<std::int64_t>(value));
    }
    JsonArray reads;
    for (std::uint32_t resource : kernel.reads) {
      reads.push_back(static_cast<std::int64_t>(resource));
    }
    JsonArray writes;
    for (std::uint32_t resource : kernel.writes) {
      writes.push_back(static_cast<std::int64_t>(resource));
    }
    JsonArray bindings;
    for (const DescriptorBinding& binding : kernel.bindings) {
      JsonObject object;
      object["set"] = static_cast<std::int64_t>(binding.set);
      object["binding"] = static_cast<std::int64_t>(binding.binding);
      object["resource_id"] = static_cast<std::int64_t>(binding.resource_id);
      object["access"] = std::string(toString(binding.access));
      bindings.push_back(std::move(object));
    }
    JsonArray pushConstants;
    for (const PushConstantBlock& block : kernel.push_constants) {
      JsonObject object;
      object["offset"] = static_cast<std::int64_t>(block.offset);
      object["data_hex"] = bytesToHex(block.bytes);
      pushConstants.push_back(std::move(object));
    }
    JsonObject object;
    object["id"] = static_cast<std::int64_t>(kernel.id);
    object["operation"] = kernel.operation;
    object["entry_point"] = kernel.entry_point;
    object["local_size"] = std::move(localSize);
    object["reads"] = std::move(reads);
    object["writes"] = std::move(writes);
    object["bindings"] = std::move(bindings);
    object["push_constants"] = std::move(pushConstants);
    kernels.push_back(std::move(object));
  }
  root["kernels"] = std::move(kernels);

  JsonArray inputs;
  for (std::uint32_t resource : plan.inputs) {
    inputs.push_back(static_cast<std::int64_t>(resource));
  }
  root["inputs"] = std::move(inputs);
  JsonArray outputs;
  for (std::uint32_t resource : plan.outputs) {
    outputs.push_back(static_cast<std::int64_t>(resource));
  }
  root["outputs"] = std::move(outputs);
  if (includeHash) {
    root["plan_hash"] = plan.plan_hash;
  }
  JsonValue result = std::move(root);
  return result;
}

std::string renderJson(JsonValue value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvm::json::OStream jsonStream(stream, 2);
  jsonStream.value(value);
  stream << '\n';
  return result;
}

std::string calculatePlanHash(const VulkanExecutionPlan& plan) {
  const std::string canonical = renderJson(toJson(plan, false));
  return std::format("{:016x}", ncnn_mlir::stableHash64(canonical));
}

std::expected<const JsonObject*, std::string> requiredObject(
  const JsonValue& value, std::string_view field) {
  const JsonObject* object = value.getAsObject();
  if (!object) {
    return std::unexpected(std::format("'{}' must be an object", field));
  }
  return object;
}

std::expected<TensorResource, std::string> parseResource(
  const JsonValue& value) {
  auto objectResult = requiredObject(value, "resource");
  if (!objectResult) {
    return std::unexpected(objectResult.error());
  }
  const JsonObject& object = **objectResult;
  if (auto fields = requireOnlyFields(object,
                                      {"id",
                                       "kind",
                                       "name",
                                       "shape",
                                       "element_type",
                                       "logical_layout",
                                       "physical_layout",
                                       "host_layout",
                                       "layout_transform",
                                       "weight_sha256",
                                       "byte_size",
                                       "first_kernel",
                                       "last_kernel"});
      !fields) {
    return std::unexpected(fields.error());
  }
  auto id = requiredU32(object, "id");
  auto kindText = requiredString(object, "kind");
  auto name = requiredString(object, "name");
  auto shapeArray = requiredArray(object, "shape");
  auto elementType = requiredString(object, "element_type");
  auto logicalLayout = requiredString(object, "logical_layout");
  auto physicalLayout = requiredString(object, "physical_layout");
  auto hostLayout = requiredString(object, "host_layout");
  auto transform = requiredLayoutTransform(object);
  auto weightHash = requiredString(object, "weight_sha256");
  auto byteSize = requiredU64(object, "byte_size");
  auto firstKernel = requiredU32(object, "first_kernel");
  auto lastKernel = requiredU32(object, "last_kernel");
  if (!id || !kindText || !name || !shapeArray || !elementType ||
      !logicalLayout || !physicalLayout || !hostLayout || !transform ||
      !weightHash || !byteSize || !firstKernel || !lastKernel) {
    return std::unexpected("resource has a missing or invalid field");
  }
  auto kind = parseResourceKind(*kindText);
  auto shape = parseU64Array(**shapeArray, "resource.shape[]");
  if (!kind || !shape || *elementType != "f32") {
    return std::unexpected(!kind    ? kind.error()
                           : !shape ? shape.error()
                                    : "resource.element_type must be f32");
  }
  return TensorResource{.id = *id,
                        .kind = *kind,
                        .name = std::move(*name),
                        .shape = std::move(*shape),
                        .logical_layout = std::move(*logicalLayout),
                        .physical_layout = std::move(*physicalLayout),
                        .host_layout = std::move(*hostLayout),
                        .layout_transform = std::move(*transform),
                        .weight_sha256 = std::move(*weightHash),
                        .byte_size = *byteSize,
                        .first_kernel = *firstKernel,
                        .last_kernel = *lastKernel};
}

std::expected<DescriptorBinding, std::string> parseBinding(
  const JsonValue& value) {
  auto objectResult = requiredObject(value, "binding");
  if (!objectResult) {
    return std::unexpected(objectResult.error());
  }
  const JsonObject& object = **objectResult;
  if (auto fields =
        requireOnlyFields(object, {"set", "binding", "resource_id", "access"});
      !fields) {
    return std::unexpected(fields.error());
  }
  auto set = requiredU32(object, "set");
  auto binding = requiredU32(object, "binding");
  auto resource = requiredU32(object, "resource_id");
  auto accessText = requiredString(object, "access");
  if (!set || !binding || !resource || !accessText) {
    return std::unexpected("descriptor binding has a missing or invalid field");
  }
  auto access = parseResourceAccess(*accessText);
  if (!access) {
    return std::unexpected(access.error());
  }
  return DescriptorBinding{.set = *set,
                           .binding = *binding,
                           .resource_id = *resource,
                           .access = *access};
}

std::expected<PushConstantBlock, std::string> parsePushConstant(
  const JsonValue& value) {
  auto objectResult = requiredObject(value, "push constant");
  if (!objectResult) {
    return std::unexpected(objectResult.error());
  }
  const JsonObject& object = **objectResult;
  if (auto fields = requireOnlyFields(object, {"offset", "data_hex"});
      !fields) {
    return std::unexpected(fields.error());
  }
  auto offset = requiredU32(object, "offset");
  auto data = requiredString(object, "data_hex");
  if (!offset || !data) {
    return std::unexpected("push constant has a missing or invalid field");
  }
  auto bytes = parseHexBytes(*data);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  return PushConstantBlock{.offset = *offset, .bytes = std::move(*bytes)};
}

std::expected<KernelExecution, std::string> parseKernel(
  const JsonValue& value) {
  auto objectResult = requiredObject(value, "kernel");
  if (!objectResult) {
    return std::unexpected(objectResult.error());
  }
  const JsonObject& object = **objectResult;
  if (auto fields = requireOnlyFields(object,
                                      {"id",
                                       "operation",
                                       "entry_point",
                                       "local_size",
                                       "reads",
                                       "writes",
                                       "bindings",
                                       "push_constants"});
      !fields) {
    return std::unexpected(fields.error());
  }
  auto id = requiredU32(object, "id");
  auto operation = requiredString(object, "operation");
  auto entryPoint = requiredString(object, "entry_point");
  auto localSizeArray = requiredArray(object, "local_size");
  auto readArray = requiredArray(object, "reads");
  auto writeArray = requiredArray(object, "writes");
  auto bindingArray = requiredArray(object, "bindings");
  auto pushArray = requiredArray(object, "push_constants");
  if (!id || !operation || !entryPoint || !localSizeArray || !readArray ||
      !writeArray || !bindingArray || !pushArray) {
    return std::unexpected("kernel has a missing or invalid field");
  }
  auto localSize = parseU32Array(**localSizeArray, "kernel.local_size[]");
  auto reads = parseU32Array(**readArray, "kernel.reads[]");
  auto writes = parseU32Array(**writeArray, "kernel.writes[]");
  if (!localSize || localSize->size() != 3 || !reads || !writes) {
    return std::unexpected("kernel has an invalid local size or resource list");
  }
  std::vector<DescriptorBinding> bindings;
  bindings.reserve((**bindingArray).size());
  for (const JsonValue& bindingValue : **bindingArray) {
    auto binding = parseBinding(bindingValue);
    if (!binding) {
      return std::unexpected(binding.error());
    }
    bindings.push_back(std::move(*binding));
  }
  std::vector<PushConstantBlock> pushConstants;
  pushConstants.reserve((**pushArray).size());
  for (const JsonValue& pushValue : **pushArray) {
    auto block = parsePushConstant(pushValue);
    if (!block) {
      return std::unexpected(block.error());
    }
    pushConstants.push_back(std::move(*block));
  }
  return KernelExecution{
    .id = *id,
    .operation = std::move(*operation),
    .entry_point = std::move(*entryPoint),
    .local_size = {(*localSize)[0], (*localSize)[1], (*localSize)[2]},
    .reads = std::move(*reads),
    .writes = std::move(*writes),
    .bindings = std::move(bindings),
    .push_constants = std::move(pushConstants)};
}

}  // namespace

std::expected<void, std::string> verifyExecutionPlan(
  const VulkanExecutionPlan& plan) {
  if (plan.schema_version != kExecutionPlanSchemaVersion) {
    return std::unexpected("unsupported Vulkan execution-plan schema_version");
  }
  if (plan.model.empty() || plan.build_identity.empty() ||
      plan.host_triple.empty() || plan.device_target.empty()) {
    return std::unexpected(
      "plan model/build/host/device identity must be nonempty");
  }
  if (plan.resources.empty() || plan.kernels.empty() || plan.inputs.empty() ||
      plan.outputs.empty()) {
    return std::unexpected(
      "plan requires resources, kernels, inputs, and outputs");
  }

  std::map<std::uint32_t, const TensorResource*> resources;
  for (const TensorResource& resource : plan.resources) {
    if (!resources.emplace(resource.id, &resource).second) {
      return std::unexpected("resource IDs must be unique");
    }
    if (toString(resource.kind).empty()) {
      return std::unexpected("resource kind is invalid");
    }
    if (resource.name.empty() || resource.logical_layout.empty() ||
        resource.physical_layout.empty()) {
      return std::unexpected("resource name and layouts must be nonempty");
    }
    auto expectedBytes = checkedByteSize(resource.shape);
    if (!expectedBytes) {
      return std::unexpected(expectedBytes.error());
    }
    if (resource.byte_size != *expectedBytes) {
      return std::unexpected("resource byte_size does not match f32 shape");
    }
    if (resource.first_kernel > resource.last_kernel ||
        resource.last_kernel >= plan.kernels.size()) {
      return std::unexpected(
        "resource lifetime is outside the ordered kernel list");
    }
    if (resource.kind == ResourceKind::Weight) {
      if (!isSha256(resource.weight_sha256)) {
        return std::unexpected(
          "weight resource requires a lowercase SHA-256 identity");
      }
    } else if (!resource.weight_sha256.empty()) {
      return std::unexpected("only weight resources may carry weight_sha256");
    }
    const bool hostVisible = resource.kind == ResourceKind::Input ||
                             resource.kind == ResourceKind::Output;
    if (hostVisible && resource.host_layout.empty()) {
      return std::unexpected("public input/output requires a host layout");
    }
    if (!hostVisible && !resource.host_layout.empty()) {
      return std::unexpected(
        "device-only resource must not declare a host layout");
    }
    if (!hostVisible && resource.layout_transform) {
      return std::unexpected(
        "device-only resource must not declare a host/device layout transform");
    }
    if (hostVisible && resource.host_layout != resource.physical_layout) {
      const std::string_view expectedFrom = resource.kind == ResourceKind::Input
                                              ? resource.host_layout
                                              : resource.physical_layout;
      const std::string_view expectedTo = resource.kind == ResourceKind::Input
                                            ? resource.physical_layout
                                            : resource.host_layout;
      if (!resource.layout_transform ||
          resource.layout_transform->from_layout != expectedFrom ||
          resource.layout_transform->to_layout != expectedTo ||
          resource.layout_transform->operation.empty()) {
        return std::unexpected(
          "host/device layout mismatch requires a matching explicit transform");
      }
    } else if (resource.layout_transform &&
               (resource.layout_transform->from_layout !=
                  resource.host_layout ||
                resource.layout_transform->to_layout !=
                  resource.physical_layout ||
                resource.layout_transform->operation != "identity")) {
      return std::unexpected(
        "layout_transform must be an identity mapping when layouts agree");
    }
  }

  auto verifyPublicResources =
    [&](const std::vector<std::uint32_t>& ids,
        ResourceKind expectedKind,
        std::string_view field) -> std::expected<void, std::string> {
    std::set<std::uint32_t> unique;
    for (std::uint32_t id : ids) {
      auto resource = resources.find(id);
      if (resource == resources.end()) {
        return std::unexpected(
          std::format("{} references an unknown resource", field));
      }
      if (resource->second->kind != expectedKind) {
        return std::unexpected(
          std::format("{} references the wrong resource kind", field));
      }
      if (!unique.insert(id).second) {
        return std::unexpected(
          std::format("{} contains duplicate resource IDs", field));
      }
    }
    return {};
  };
  if (auto valid =
        verifyPublicResources(plan.inputs, ResourceKind::Input, "inputs");
      !valid) {
    return valid;
  }
  if (auto valid =
        verifyPublicResources(plan.outputs, ResourceKind::Output, "outputs");
      !valid) {
    return valid;
  }
  for (const TensorResource& resource : plan.resources) {
    if (resource.kind == ResourceKind::Input &&
        std::ranges::find(plan.inputs, resource.id) == plan.inputs.end()) {
      return std::unexpected(
        "every input resource must appear in public inputs");
    }
    if (resource.kind == ResourceKind::Output &&
        std::ranges::find(plan.outputs, resource.id) == plan.outputs.end()) {
      return std::unexpected(
        "every output resource must appear in public outputs");
    }
  }

  std::map<std::uint32_t, std::size_t> resourceFirstTouch;
  std::map<std::uint32_t, std::size_t> resourceLastTouch;
  std::set<std::uint32_t> initialized;
  for (std::uint32_t id : plan.inputs) {
    initialized.insert(id);
  }
  for (const TensorResource& resource : plan.resources) {
    if (resource.kind == ResourceKind::Weight) {
      initialized.insert(resource.id);
    }
  }

  std::set<std::uint32_t> kernelIds;
  for (std::size_t index = 0; index < plan.kernels.size(); ++index) {
    const KernelExecution& kernel = plan.kernels[index];
    if (!kernelIds.insert(kernel.id).second) {
      return std::unexpected("kernel IDs must be unique");
    }
    if (kernel.operation.empty() || kernel.entry_point.empty()) {
      return std::unexpected(
        "kernel operation and entry_point must be nonempty");
    }
    std::uint64_t workgroupInvocations = 1;
    for (std::size_t axis = 0; axis < kernel.local_size.size(); ++axis) {
      const std::uint32_t size = kernel.local_size[axis];
      if (size == 0 || size > kMinimumLocalSize[axis] ||
          workgroupInvocations > 128 / size) {
        return std::unexpected(
          "kernel local_size exceeds Vulkan 1.1 minimum workgroup limits");
      }
      workgroupInvocations *= size;
    }
    if (kernel.reads.empty() || kernel.writes.empty()) {
      return std::unexpected("kernel requires at least one read and one write");
    }
    std::set<std::uint32_t> readIds(kernel.reads.begin(), kernel.reads.end());
    std::set<std::uint32_t> writeIds(kernel.writes.begin(),
                                     kernel.writes.end());
    if (readIds.size() != kernel.reads.size() ||
        writeIds.size() != kernel.writes.size()) {
      return std::unexpected("kernel read/write resource IDs must be unique");
    }
    for (std::uint32_t id : readIds) {
      if (!resources.contains(id)) {
        return std::unexpected("kernel reads an unknown resource");
      }
      if (writeIds.contains(id)) {
        return std::unexpected("in-place kernel resources are not supported");
      }
      if (!initialized.contains(id)) {
        return std::unexpected(
          "kernel reads a resource before a preceding write");
      }
      resourceFirstTouch.try_emplace(id, index);
      resourceLastTouch[id] = index;
    }
    for (std::uint32_t id : writeIds) {
      auto resource = resources.find(id);
      if (resource == resources.end()) {
        return std::unexpected("kernel writes an unknown resource");
      }
      if (resource->second->kind == ResourceKind::Input ||
          resource->second->kind == ResourceKind::Weight) {
        return std::unexpected(
          "kernel may not write an input or weight resource");
      }
      resourceFirstTouch.try_emplace(id, index);
      resourceLastTouch[id] = index;
    }

    std::set<std::pair<std::uint32_t, std::uint32_t>> descriptorIds;
    std::set<std::pair<std::uint32_t, ResourceAccess>> describedResources;
    for (const DescriptorBinding& binding : kernel.bindings) {
      if (toString(binding.access).empty()) {
        return std::unexpected("descriptor access is invalid");
      }
      if (!descriptorIds.emplace(binding.set, binding.binding).second) {
        return std::unexpected(
          "descriptor set/binding pairs must be unique per kernel");
      }
      if (!resources.contains(binding.resource_id)) {
        return std::unexpected(
          "descriptor binding references an unknown resource");
      }
      const bool validAccess = binding.access == ResourceAccess::Read
                                 ? readIds.contains(binding.resource_id)
                                 : writeIds.contains(binding.resource_id);
      if (!validAccess ||
          !describedResources.emplace(binding.resource_id, binding.access)
             .second) {
        return std::unexpected(
          "descriptor access does not match kernel reads/writes");
      }
    }
    if (describedResources.size() != readIds.size() + writeIds.size()) {
      return std::unexpected(
        "every kernel resource must have one descriptor binding");
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> constantRanges;
    for (const PushConstantBlock& block : kernel.push_constants) {
      if (block.bytes.empty() || block.offset % 4 != 0 ||
          block.bytes.size() % 4 != 0 ||
          block.bytes.size() > kMinimumPushConstantBytes ||
          block.offset > kMinimumPushConstantBytes - block.bytes.size()) {
        return std::unexpected(
          "push constants must be aligned and fit 128 bytes");
      }
      const std::uint32_t begin = block.offset;
      const std::uint32_t end =
        begin + static_cast<std::uint32_t>(block.bytes.size());
      for (const auto& range : constantRanges) {
        if (begin < range.second && range.first < end) {
          return std::unexpected("push-constant ranges must not overlap");
        }
      }
      constantRanges.emplace_back(begin, end);
    }
    initialized.insert(writeIds.begin(), writeIds.end());
  }

  for (const TensorResource& resource : plan.resources) {
    auto first = resourceFirstTouch.find(resource.id);
    auto last = resourceLastTouch.find(resource.id);
    if (first == resourceFirstTouch.end() || last == resourceLastTouch.end()) {
      return std::unexpected(
        "every resource must be used by at least one kernel");
    }
    if (resource.first_kernel != first->second ||
        resource.last_kernel != last->second) {
      return std::unexpected(
        "resource lifetime does not cover its kernel uses exactly");
    }
    if (resource.kind == ResourceKind::Output &&
        !initialized.contains(resource.id)) {
      return std::unexpected("public output has no preceding producer");
    }
  }

  if (!plan.plan_hash.empty() && plan.plan_hash != calculatePlanHash(plan)) {
    return std::unexpected("plan_hash does not match canonical plan contents");
  }
  return {};
}

std::expected<std::string, std::string> serializeExecutionPlan(
  const VulkanExecutionPlan& plan) {
  if (auto valid = verifyExecutionPlan(plan); !valid) {
    return std::unexpected(valid.error());
  }
  VulkanExecutionPlan serializedPlan = plan;
  serializedPlan.plan_hash = calculatePlanHash(plan);
  return renderJson(toJson(serializedPlan, true));
}

std::expected<VulkanExecutionPlan, std::string> deserializeExecutionPlan(
  std::string_view json) {
  auto parsed = llvm::json::parse(llvm::StringRef(json.data(), json.size()));
  if (!parsed) {
    return std::unexpected(llvm::toString(parsed.takeError()));
  }
  auto objectResult = requiredObject(*parsed, "plan");
  if (!objectResult) {
    return std::unexpected(objectResult.error());
  }
  const JsonObject& root = **objectResult;
  if (auto fields = requireOnlyFields(root,
                                      {"kind",
                                       "schema_version",
                                       "model",
                                       "build_identity",
                                       "host_triple",
                                       "device_target",
                                       "resources",
                                       "kernels",
                                       "inputs",
                                       "outputs",
                                       "plan_hash"});
      !fields) {
    return std::unexpected(fields.error());
  }
  auto kind = requiredString(root, "kind");
  auto version = requiredU32(root, "schema_version");
  auto model = requiredString(root, "model");
  auto buildIdentity = requiredString(root, "build_identity");
  auto hostTriple = requiredString(root, "host_triple");
  auto deviceTarget = requiredString(root, "device_target");
  auto resourceArray = requiredArray(root, "resources");
  auto kernelArray = requiredArray(root, "kernels");
  auto inputArray = requiredArray(root, "inputs");
  auto outputArray = requiredArray(root, "outputs");
  auto planHash = requiredString(root, "plan_hash");
  if (!kind || *kind != kPlanKind || !version || !model || !buildIdentity ||
      !hostTriple || !deviceTarget || !resourceArray || !kernelArray ||
      !inputArray || !outputArray || !planHash || planHash->empty()) {
    return std::unexpected("plan has a missing or invalid top-level field");
  }
  VulkanExecutionPlan result{.schema_version = *version,
                             .model = std::move(*model),
                             .build_identity = std::move(*buildIdentity),
                             .host_triple = std::move(*hostTriple),
                             .device_target = std::move(*deviceTarget),
                             .resources = {},
                             .kernels = {},
                             .inputs = {},
                             .outputs = {},
                             .plan_hash = std::move(*planHash)};
  result.resources.reserve((**resourceArray).size());
  for (const JsonValue& resourceValue : **resourceArray) {
    auto resource = parseResource(resourceValue);
    if (!resource) {
      return std::unexpected(resource.error());
    }
    result.resources.push_back(std::move(*resource));
  }
  result.kernels.reserve((**kernelArray).size());
  for (const JsonValue& kernelValue : **kernelArray) {
    auto kernel = parseKernel(kernelValue);
    if (!kernel) {
      return std::unexpected(kernel.error());
    }
    result.kernels.push_back(std::move(*kernel));
  }
  auto inputs = parseU32Array(**inputArray, "inputs[]");
  auto outputs = parseU32Array(**outputArray, "outputs[]");
  if (!inputs || !outputs) {
    return std::unexpected(!inputs ? inputs.error() : outputs.error());
  }
  result.inputs = std::move(*inputs);
  result.outputs = std::move(*outputs);
  if (auto valid = verifyExecutionPlan(result); !valid) {
    return std::unexpected(valid.error());
  }
  return result;
}

}  // namespace ncnn_mlir::vulkan
