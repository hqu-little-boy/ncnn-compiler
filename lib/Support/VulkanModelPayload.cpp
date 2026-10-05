#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "ncnn-mlir/Support/VulkanDocumentKinds.hpp"

namespace ncnn_mlir::vulkan {
namespace {

using JsonArray = llvm::json::Array;
using JsonObject = llvm::json::Object;
using JsonValue = llvm::json::Value;

constexpr std::uint32_t kPayloadSchemaVersion = 1;
constexpr std::size_t kF32ByteWidth = sizeof(float);

std::string sha256Hex(std::span<const std::uint8_t> bytes) {
  llvm::SHA256 hash;
  hash.update(
    llvm::StringRef(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
  const std::array<std::uint8_t, 32> digest = hash.final();
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (std::uint8_t byte : digest) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

std::expected<void, std::string> requireOnlyFields(
  const JsonObject& object, std::initializer_list<std::string_view> allowed) {
  for (const auto& item : object) {
    const llvm::StringRef key = item.first;
    const std::string_view key_view(key.data(), key.size());
    if (std::ranges::find(allowed, key_view) == allowed.end()) {
      return std::unexpected(
        std::format("unexpected Vulkan weight sidecar field '{}'", key_view));
    }
  }
  return {};
}

std::expected<std::uint32_t, std::string> requiredU32(const JsonObject& object,
                                                      llvm::StringRef key) {
  const auto value = object.getInteger(key);
  if (!value || *value < 0 ||
      std::cmp_greater(*value, std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected(
      std::format("missing or invalid uint32 field '{}'", key.str()));
  }
  return static_cast<std::uint32_t>(*value);
}

std::expected<std::size_t, std::string> elementCount(
  std::span<const std::uint64_t> shape) {
  if (shape.empty()) {
    return std::unexpected("weight logical shape must not be empty");
  }
  std::size_t count = 1;
  for (const std::uint64_t extent : shape) {
    if (extent == 0 ||
        std::cmp_greater(extent, std::numeric_limits<std::size_t>::max()) ||
        count > std::numeric_limits<std::size_t>::max() /
                  static_cast<std::size_t>(extent)) {
      return std::unexpected("weight shape is zero-sized or overflows size_t");
    }
    count *= static_cast<std::size_t>(extent);
  }
  return count;
}

std::expected<std::vector<std::uint8_t>, std::string> logicalWeightBytes(
  const TensorResource& resource, std::span<const std::uint8_t> bytes) {
  if (resource.kind != ResourceKind::Weight) {
    return std::unexpected("logical weight hash requires a weight resource");
  }
  if (bytes.size() != resource.byte_size || bytes.size() % kF32ByteWidth != 0) {
    return std::unexpected(std::format(
      "weight resource {} payload size differs from its FP32 byte_size",
      resource.id));
  }
  auto count = elementCount(resource.shape);
  if (!count) {
    return std::unexpected(
      std::format("weight resource {} has an invalid logical shape: {}",
                  resource.id,
                  count.error()));
  }
  if (*count > std::numeric_limits<std::size_t>::max() / kF32ByteWidth ||
      *count * kF32ByteWidth != bytes.size()) {
    return std::unexpected(std::format(
      "weight resource {} shape does not match its payload size", resource.id));
  }
  if (resource.logical_layout == resource.physical_layout) {
    return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
  }
  if (resource.logical_layout != "OIHW" || resource.shape.size() != 4 ||
      (resource.physical_layout != "OHWI" &&
       resource.physical_layout != "HWCF")) {
    return std::unexpected(std::format(
      "weight resource {} has an unsupported logical/physical layout {} -> {}",
      resource.id,
      resource.physical_layout,
      resource.logical_layout));
  }

  const auto output_channels = static_cast<std::size_t>(resource.shape[0]);
  const auto input_channels = static_cast<std::size_t>(resource.shape[1]);
  const auto kernel_height = static_cast<std::size_t>(resource.shape[2]);
  const auto kernel_width = static_cast<std::size_t>(resource.shape[3]);
  std::vector<std::uint8_t> logical(bytes.size());
  for (std::size_t output = 0; output < output_channels; ++output) {
    for (std::size_t input = 0; input < input_channels; ++input) {
      for (std::size_t y = 0; y < kernel_height; ++y) {
        for (std::size_t x = 0; x < kernel_width; ++x) {
          const std::size_t logical_index =
            (((output * input_channels + input) * kernel_height + y) *
             kernel_width) +
            x;
          std::size_t physical_index = 0;
          if (resource.physical_layout == "OHWI") {
            physical_index =
              (((output * kernel_height + y) * kernel_width + x) *
               input_channels) +
              input;
          } else {
            physical_index =
              (((y * kernel_width + x) * input_channels + input) *
               output_channels) +
              output;
          }
          const std::size_t logical_offset = logical_index * kF32ByteWidth;
          const std::size_t physical_offset = physical_index * kF32ByteWidth;
          std::copy_n(bytes.begin() + physical_offset,
                      kF32ByteWidth,
                      logical.begin() + logical_offset);
        }
      }
    }
  }
  return logical;
}

std::string hexEncode(std::span<const std::uint8_t> bytes) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (const std::uint8_t byte : bytes) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

std::expected<std::vector<std::uint8_t>, std::string> hexDecode(
  std::string_view value) {
  if (value.empty() || value.size() % 2 != 0) {
    return std::unexpected("weight bytes must be nonempty lowercase hex");
  }
  if (value.size() / 2 > std::vector<std::uint8_t>{}.max_size()) {
    return std::unexpected("weight payload exceeds host vector size limit");
  }
  std::vector<std::uint8_t> result;
  result.reserve(value.size() / 2);
  const auto decode = [](char character) -> int {
    if (character >= '0' && character <= '9') {
      return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
      return character - 'a' + 10;
    }
    return -1;
  };
  for (std::size_t index = 0; index < value.size(); index += 2) {
    const int high = decode(value[index]);
    const int low = decode(value[index + 1]);
    if (high < 0 || low < 0) {
      return std::unexpected("weight bytes must be lowercase hexadecimal");
    }
    result.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  if (result.size() % kF32ByteWidth != 0) {
    return std::unexpected("weight payload size is not a multiple of FP32");
  }
  return result;
}

}  // namespace

std::expected<std::string, std::string> serializeVulkanModelWeights(
  std::span<const VulkanModelWeight> weights) {
  std::vector<const VulkanModelWeight*> sorted;
  sorted.reserve(weights.size());
  std::set<std::uint32_t> ids;
  for (const VulkanModelWeight& weight : weights) {
    if (weight.bytes.empty() || weight.bytes.size() % kF32ByteWidth != 0) {
      return std::unexpected(
        std::format("weight resource {} has an empty or non-FP32 payload",
                    weight.resource_id));
    }
    if (weight.bytes.size() > std::string{}.max_size() / 2) {
      return std::unexpected(
        std::format("weight resource {} exceeds JSON hex encoding size limit",
                    weight.resource_id));
    }
    if (!ids.insert(weight.resource_id).second) {
      return std::unexpected(
        std::format("weight sidecar contains duplicate resource ID {}",
                    weight.resource_id));
    }
    sorted.push_back(&weight);
  }
  std::ranges::sort(sorted, {}, [](const VulkanModelWeight* weight) {
    return weight->resource_id;
  });

  JsonArray payloads;
  for (const VulkanModelWeight* weight : sorted) {
    JsonObject object;
    object["resource_id"] = static_cast<std::int64_t>(weight->resource_id);
    object["bytes"] = hexEncode(weight->bytes);
    payloads.push_back(std::move(object));
  }
  JsonObject root;
  root["kind"] = mlir::ncnn::contract::kVulkanWeightPayloadKind.str();
  root["schema_version"] = static_cast<std::int64_t>(kPayloadSchemaVersion);
  root["weights"] = std::move(payloads);
  return llvm::formatv("{0:2}\n", JsonValue(std::move(root))).str();
}

std::expected<std::vector<VulkanModelWeight>, std::string>
deserializeVulkanModelWeights(std::string_view json) {
  auto parsed = llvm::json::parse(llvm::StringRef(json.data(), json.size()));
  if (!parsed) {
    return std::unexpected("cannot parse Vulkan weight sidecar JSON: " +
                           llvm::toString(parsed.takeError()));
  }
  const JsonObject* root = parsed->getAsObject();
  if (root == nullptr) {
    return std::unexpected("Vulkan weight sidecar must be a JSON object");
  }
  if (auto fields =
        requireOnlyFields(*root, {"kind", "schema_version", "weights"});
      !fields) {
    return std::unexpected(fields.error());
  }
  const auto kind = root->getString("kind");
  const auto version = root->getInteger("schema_version");
  const JsonValue* weights_value = root->get("weights");
  const JsonArray* weight_array =
    weights_value == nullptr ? nullptr : weights_value->getAsArray();
  if (!kind || *kind != mlir::ncnn::contract::kVulkanWeightPayloadKind ||
      !version || !std::cmp_equal(*version, kPayloadSchemaVersion) ||
      weight_array == nullptr) {
    return std::unexpected(
      "Vulkan weight sidecar kind, schema_version, or weights is invalid");
  }

  std::vector<VulkanModelWeight> result;
  result.reserve(weight_array->size());
  std::set<std::uint32_t> ids;
  for (const JsonValue& value : *weight_array) {
    const JsonObject* object = value.getAsObject();
    if (object == nullptr) {
      return std::unexpected("weight sidecar entries must be objects");
    }
    if (auto fields = requireOnlyFields(*object, {"resource_id", "bytes"});
        !fields) {
      return std::unexpected(fields.error());
    }
    auto resource_id = requiredU32(*object, "resource_id");
    const auto bytes_string = object->getString("bytes");
    if (!resource_id || !bytes_string) {
      return std::unexpected("weight sidecar entry is missing required fields");
    }
    if (!ids.insert(*resource_id).second) {
      return std::unexpected(std::format(
        "weight sidecar contains duplicate resource ID {}", *resource_id));
    }
    auto bytes = hexDecode(bytes_string->str());
    if (!bytes) {
      return std::unexpected(
        std::format("weight resource {}: {}", *resource_id, bytes.error()));
    }
    result.push_back(VulkanModelWeight{.resource_id = *resource_id,
                                       .bytes = std::move(*bytes)});
  }
  return result;
}

std::expected<std::string, std::string> vulkanWeightSHA256(
  const TensorResource& resource, std::span<const std::uint8_t> bytes) {
  if (resource.logical_layout == resource.physical_layout) {
    if (resource.kind != ResourceKind::Weight) {
      return std::unexpected("logical weight hash requires a weight resource");
    }
    if (bytes.size() != resource.byte_size ||
        bytes.size() % kF32ByteWidth != 0) {
      return std::unexpected(std::format(
        "weight resource {} payload size differs from its FP32 byte_size",
        resource.id));
    }
    auto count = elementCount(resource.shape);
    if (!count) {
      return std::unexpected(
        std::format("weight resource {} has an invalid logical shape: {}",
                    resource.id,
                    count.error()));
    }
    if (*count > std::numeric_limits<std::size_t>::max() / kF32ByteWidth ||
        *count * kF32ByteWidth != bytes.size()) {
      return std::unexpected(
        std::format("weight resource {} shape does not match its payload size",
                    resource.id));
    }
    return sha256Hex(bytes);
  }
  auto logical = logicalWeightBytes(resource, bytes);
  if (!logical) {
    return std::unexpected(logical.error());
  }
  return sha256Hex(*logical);
}

std::expected<void, std::string> verifyVulkanModelWeights(
  const VulkanExecutionPlan& plan, std::span<const VulkanModelWeight> weights) {
  if (auto verified_plan = verifyExecutionPlan(plan); !verified_plan) {
    return std::unexpected("invalid Vulkan execution plan: " +
                           verified_plan.error());
  }
  std::map<std::uint32_t, const TensorResource*> expected;
  for (const TensorResource& resource : plan.resources) {
    if (resource.kind == ResourceKind::Weight) {
      expected.emplace(resource.id, &resource);
    }
  }
  if (weights.size() != expected.size()) {
    return std::unexpected(
      std::format("weight payload count {} does not match plan weight count {}",
                  weights.size(),
                  expected.size()));
  }
  std::set<std::uint32_t> found;
  for (const VulkanModelWeight& weight : weights) {
    const auto resource = expected.find(weight.resource_id);
    if (resource == expected.end()) {
      return std::unexpected(
        std::format("weight payload resource ID {} is not a plan weight",
                    weight.resource_id));
    }
    if (!found.insert(weight.resource_id).second) {
      return std::unexpected(
        std::format("weight payload contains duplicate resource ID {}",
                    weight.resource_id));
    }
    if (weight.bytes.size() != resource->second->byte_size) {
      return std::unexpected(std::format(
        "weight resource {} payload size {} differs from expected {}",
        weight.resource_id,
        weight.bytes.size(),
        resource->second->byte_size));
    }
    auto digest = vulkanWeightSHA256(*resource->second, weight.bytes);
    if (!digest) {
      return std::unexpected(digest.error());
    }
    if (*digest != resource->second->weight_sha256) {
      return std::unexpected(
        std::format("weight resource {} logical SHA-256 differs from the plan",
                    weight.resource_id));
    }
  }
  return {};
}

}  // namespace ncnn_mlir::vulkan
