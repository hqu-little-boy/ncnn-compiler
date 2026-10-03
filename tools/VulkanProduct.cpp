#include "VulkanProduct.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Vulkan/ModelABI.h"

namespace ncnn_compile {
namespace {

using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;

constexpr std::array<unsigned char, 4> kSpirvMagic = {0x03, 0x02, 0x23, 0x07};

std::string sha256Hex(std::string_view bytes) {
  llvm::SHA256 hash;
  hash.update(llvm::StringRef(bytes.data(), bytes.size()));
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

std::string uppercaseIdentifier(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(std::toupper(character));
  });
  return result;
}

void replaceAll(std::string& value,
                std::string_view needle,
                std::string_view replacement) {
  std::size_t position = 0;
  while ((position = value.find(needle, position)) != std::string::npos) {
    value.replace(position, needle.size(), replacement);
    position += replacement.size();
  }
}

const TensorResource* findResource(const VulkanExecutionPlan& plan,
                                   std::uint32_t resource_id) {
  const auto found =
    std::ranges::find(plan.resources, resource_id, &TensorResource::id);
  return found == plan.resources.end() ? nullptr : &*found;
}

std::expected<std::vector<std::uint32_t>, std::string> decodeSpirv(
  std::string_view bytes) {
  if (bytes.size() < kSpirvMagic.size() ||
      bytes.size() % sizeof(std::uint32_t) != 0) {
    return std::unexpected("SPIR-V payload has an invalid byte length");
  }
  for (std::size_t index = 0; index < kSpirvMagic.size(); ++index) {
    if (static_cast<unsigned char>(bytes[index]) != kSpirvMagic[index]) {
      return std::unexpected("SPIR-V payload has an invalid magic number");
    }
  }
  std::vector<std::uint32_t> words;
  words.reserve(bytes.size() / sizeof(std::uint32_t));
  for (std::size_t offset = 0; offset < bytes.size();
       offset += sizeof(std::uint32_t)) {
    const auto byte = [&bytes](std::size_t index) {
      return static_cast<std::uint32_t>(
        static_cast<unsigned char>(bytes[index]));
    };
    words.push_back(byte(offset) | (byte(offset + 1) << 8) |
                    (byte(offset + 2) << 16) | (byte(offset + 3) << 24));
  }
  return words;
}

std::string byteInitializer(std::string_view bytes) {
  std::string result;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    if (index != 0) {
      result += ", ";
    }
    result += std::format("0x{:02x}", static_cast<unsigned char>(bytes[index]));
  }
  return result;
}

std::string wordInitializer(std::span<const std::uint32_t> words) {
  std::string result;
  for (std::size_t index = 0; index < words.size(); ++index) {
    if (index != 0) {
      result += ", ";
    }
    result += std::format("0x{:08x}U", words[index]);
  }
  return result;
}

std::string makeHeader(std::string_view model_name,
                       const TensorResource& input,
                       const TensorResource& output) {
  const std::string macro = uppercaseIdentifier(model_name) + "_VULKAN";
  const std::string context = std::string(model_name) + "_vk_context";
  const std::uint64_t input_elements = input.byte_size / sizeof(float);
  const std::uint64_t output_elements = output.byte_size / sizeof(float);
  std::string result = std::format(
    "#ifndef {0}_H\n#define {0}_H\n\n"
    "#include <stddef.h>\n#include <stdint.h>\n"
    "#include \"ncnn-mlir/Vulkan/ModelABI.h\"\n\n"
    "#define {1}_INPUT_COUNT UINT32_C(1)\n"
    "#define {1}_OUTPUT_COUNT UINT32_C(1)\n"
    "#define {1}_INPUT1_RANK UINT32_C({2})\n"
    "#define {1}_INPUT1_ELEMENTS UINT64_C({3})\n"
    "#define {1}_OUTPUT1_RANK UINT32_C({4})\n"
    "#define {1}_OUTPUT1_ELEMENTS UINT64_C({5})\n",
    macro,
    macro,
    input.shape.size(),
    input_elements,
    output.shape.size(),
    output_elements);
  for (std::size_t index = 0; index < input.shape.size(); ++index) {
    result += std::format("#define {0}_INPUT1_DIM{1} UINT64_C({2})\n",
                          macro,
                          index,
                          input.shape[index]);
  }
  for (std::size_t index = 0; index < output.shape.size(); ++index) {
    result += std::format("#define {0}_OUTPUT1_DIM{1} UINT64_C({2})\n",
                          macro,
                          index,
                          output.shape[index]);
  }
  result += std::format(
    "\ntypedef struct {0} {0};\n\n"
    "#ifdef __cplusplus\nextern \"C\" {{\n#endif\n"
    "ncnn_vk_status {1}_vk_create(const ncnn_vk_options *options,\n"
    "    {0} **context, char *diagnostic, size_t diagnostic_capacity);\n"
    "ncnn_vk_status {1}_vk_run({0} *context, const float *input1,\n"
    "    size_t input1_elements, float *output1, size_t output1_elements,\n"
    "    char *diagnostic, size_t diagnostic_capacity);\n"
    "ncnn_vk_status {1}_vk_destroy({0} *context);\n"
    "#ifdef __cplusplus\n}}\n#endif\n\n#endif  // {2}_H\n",
    context,
    model_name,
    macro);
  return result;
}

std::string makeWrapperSource(std::string_view model_name,
                              std::string_view header_name,
                              std::string_view plan_json,
                              std::span<const std::uint32_t> shader_words,
                              const VulkanExecutionPlan& plan) {
  // clang-format off
  constexpr std::string_view kTemplate = R"cpp(#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <span>
#include <string_view>
#include <utility>

#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"
#include "ncnn-mlir/Vulkan/VulkanRuntime.hpp"
#include "@HEADER@"

          namespace {

          using ncnn_mlir::vulkan::VulkanExecutionPlan;
          using ncnn_mlir::vulkan::VulkanInputTensor;
          using ncnn_mlir::vulkan::VulkanKernelBinary;
          using ncnn_mlir::vulkan::VulkanOutputTensor;
          using ncnn_mlir::vulkan::VulkanResourcePayload;
          using ncnn_mlir::vulkan::VulkanRuntimeContext;
          using ncnn_mlir::vulkan::VulkanRuntimeOptions;

          constexpr unsigned char kPlanJsonBytes[] = {@PLAN_BYTES@};
          constexpr std::uint32_t kShaderWords[] = {@SHADER_WORDS@};
          constexpr std::uint32_t kKernelId = @KERNEL_ID@;
          constexpr std::uint32_t kInputResourceId = @INPUT_RESOURCE_ID@;
          constexpr std::uint32_t kOutputResourceId = @OUTPUT_RESOURCE_ID@;

          ncnn_vk_status report(ncnn_vk_status status,
                                char* diagnostic,
                                size_t diagnostic_capacity,
                                std::string_view message) noexcept {
            if (diagnostic_capacity > 0 && diagnostic != nullptr) {
              const std::size_t count = std::min(message.size(), diagnostic_capacity - 1);
              std::copy_n(message.data(), count, diagnostic);
              diagnostic[count] = '\0';
            }
            return status;
          }

          ncnn_vk_status validateDiagnostic(
            char* diagnostic, size_t diagnostic_capacity) noexcept {
            if (diagnostic_capacity > 0 && diagnostic == nullptr) {
              return NCNN_VK_STATUS_INVALID_ARGUMENT;
            }
            if (diagnostic_capacity > 0) {
              diagnostic[0] = '\0';
            }
            return NCNN_VK_STATUS_SUCCESS;
          }

          }  // namespace

struct @CONTEXT@ {
  explicit @CONTEXT@(VulkanRuntimeContext value)
    : runtime(std::move(value)) {}
  VulkanRuntimeContext runtime;
};

          extern "C" ncnn_vk_status @CREATE@(const ncnn_vk_options* options,
                                              @CONTEXT@** context,
                                              char* diagnostic,
                                              size_t diagnostic_capacity) {
            if (context == nullptr) {
              return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                            diagnostic,
                            diagnostic_capacity,
                            "missing context output");
            }
            *context = nullptr;
            const ncnn_vk_status diagnostic_status =
              validateDiagnostic(diagnostic, diagnostic_capacity);
            if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
              return diagnostic_status;
            }
            try {
              if (options == nullptr ||
                  options->struct_size < NCNN_VK_OPTIONS_V1_SIZE ||
                  options->abi_version != NCNN_VK_ABI_VERSION ||
                  options->flags != 0) {
                return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                              diagnostic,
                              diagnostic_capacity,
                              "invalid Vulkan options");
              }
              const std::string_view plan_json(
                reinterpret_cast<const char*>(kPlanJsonBytes),
                sizeof(kPlanJsonBytes));
              auto plan = ncnn_mlir::vulkan::deserializeExecutionPlan(plan_json);
              if (!plan) {
                return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                              diagnostic,
                              diagnostic_capacity,
                              plan.error());
              }
              const std::array<VulkanKernelBinary, 1> kernels{
                VulkanKernelBinary{.kernel_id = kKernelId,
                                   .words = kShaderWords}};
              const std::array<VulkanResourcePayload, 0> resources{};
              VulkanRuntimeOptions runtime_options =
                ncnn_mlir::vulkan::makeVulkanRuntimeOptions();
              runtime_options.device_index = options->device_index;
              auto runtime = VulkanRuntimeContext::create(
                *plan, kernels, resources, runtime_options);
              if (!runtime) {
                return report(runtime.error().status,
                              diagnostic,
                              diagnostic_capacity,
                              runtime.error().diagnostic);
              }
              auto created = std::make_unique<@CONTEXT@>(std::move(*runtime));
              *context = created.release();
              return NCNN_VK_STATUS_SUCCESS;
            } catch (const std::bad_alloc&) {
              return report(
                NCNN_VK_STATUS_OUT_OF_MEMORY,
                diagnostic,
                diagnostic_capacity,
                "host allocation failed while creating Vulkan context");
            } catch (const std::exception& error) {
              return report(NCNN_VK_STATUS_API_ERROR,
                            diagnostic,
                            diagnostic_capacity,
                            error.what());
            } catch (...) {
              return report(NCNN_VK_STATUS_API_ERROR,
                            diagnostic,
                            diagnostic_capacity,
                            "unknown error while creating Vulkan context");
            }
          }

          extern "C" ncnn_vk_status @RUN@(@CONTEXT@* context,
                                           const float* input1,
                                           size_t input1_elements,
                                           float* output1,
                                           size_t output1_elements,
                                           char* diagnostic,
                                           size_t diagnostic_capacity) {
            const ncnn_vk_status diagnostic_status =
              validateDiagnostic(diagnostic, diagnostic_capacity);
            if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
              return diagnostic_status;
            }
            if (context == nullptr || input1 == nullptr || output1 == nullptr) {
              return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                            diagnostic,
                            diagnostic_capacity,
                            "invalid run argument");
            }
            if (input1_elements != @INPUT_ELEMENTS@ ||
                output1_elements != @OUTPUT_ELEMENTS@) {
              return report(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
                            diagnostic,
                            diagnostic_capacity,
                            "element count does not match static shape");
            }
            try {
              const std::span<const float> input(input1, input1_elements);
              const std::span<float> output(output1, output1_elements);
              const std::array<VulkanInputTensor, 1> inputs{
                VulkanInputTensor{.resource_id = kInputResourceId,
                                  .bytes = std::as_bytes(input)}};
              const std::array<VulkanOutputTensor, 1> outputs{
                VulkanOutputTensor{.resource_id = kOutputResourceId,
                                   .bytes = std::as_writable_bytes(output)}};
              auto result = context->runtime.run(inputs, outputs);
              if (!result) {
                return report(result.error().status,
                              diagnostic,
                              diagnostic_capacity,
                              result.error().diagnostic);
              }
              return NCNN_VK_STATUS_SUCCESS;
            } catch (const std::bad_alloc&) {
              return report(NCNN_VK_STATUS_OUT_OF_MEMORY,
                            diagnostic,
                            diagnostic_capacity,
                            "host allocation failed during Vulkan run");
            } catch (const std::exception& error) {
              return report(NCNN_VK_STATUS_API_ERROR,
                            diagnostic,
                            diagnostic_capacity,
                            error.what());
            } catch (...) {
              return report(NCNN_VK_STATUS_API_ERROR,
                            diagnostic,
                            diagnostic_capacity,
                            "unknown Vulkan run error");
            }
          }

          extern "C" ncnn_vk_status @DESTROY@(@CONTEXT@* context) {
            const std::unique_ptr<@CONTEXT@> owned(context);
            if (!owned) {
              return NCNN_VK_STATUS_SUCCESS;
            }
            try {
              auto result = owned->runtime.destroy();
              return result ? NCNN_VK_STATUS_SUCCESS : result.error().status;
            } catch (...) {
              return NCNN_VK_STATUS_API_ERROR;
            }
          }
  )cpp";
  // clang-format on
  const TensorResource* input = findResource(plan, plan.inputs.front());
  const TensorResource* output = findResource(plan, plan.outputs.front());
  std::string source(kTemplate);
  replaceAll(source, "@HEADER@", header_name);
  replaceAll(source, "@PLAN_BYTES@", byteInitializer(plan_json));
  replaceAll(source, "@SHADER_WORDS@", wordInitializer(shader_words));
  replaceAll(source, "@KERNEL_ID@", std::to_string(plan.kernels.front().id));
  replaceAll(source, "@INPUT_RESOURCE_ID@", std::to_string(input->id));
  replaceAll(source, "@OUTPUT_RESOURCE_ID@", std::to_string(output->id));
  replaceAll(source, "@CONTEXT@", std::string(model_name) + "_vk_context");
  replaceAll(source, "@CREATE@", std::string(model_name) + "_vk_create");
  replaceAll(source, "@RUN@", std::string(model_name) + "_vk_run");
  replaceAll(source, "@DESTROY@", std::string(model_name) + "_vk_destroy");
  replaceAll(source,
             "@INPUT_ELEMENTS@",
             std::to_string(input->byte_size / sizeof(float)));
  replaceAll(source,
             "@OUTPUT_ELEMENTS@",
             std::to_string(output->byte_size / sizeof(float)));
  return source;
}

llvm::json::Object makeResourceJson(const TensorResource& resource) {
  llvm::json::Array shape;
  for (std::uint64_t extent : resource.shape) {
    shape.push_back(static_cast<std::int64_t>(extent));
  }
  llvm::json::Object object;
  object["id"] = static_cast<std::int64_t>(resource.id);
  object["name"] = resource.name;
  object["element_type"] = "f32";
  object["shape"] = std::move(shape);
  object["byte_size"] = static_cast<std::int64_t>(resource.byte_size);
  object["host_layout"] = resource.host_layout;
  object["logical_layout"] = resource.logical_layout;
  object["physical_layout"] = resource.physical_layout;
  return object;
}

std::string makeManifest(const VulkanExecutionPlan& plan,
                         std::string_view model_name,
                         std::string_view binary_symbol,
                         std::string_view build_identity,
                         std::string_view plan_hash,
                         std::string_view plan_sha256,
                         std::string_view shader_sha256,
                         std::size_t shader_size,
                         std::string_view param_sha256,
                         std::string_view bin_sha256) {
  llvm::json::Array inputs;
  llvm::json::Array outputs;
  for (std::uint32_t id : plan.inputs) {
    const TensorResource* resource = findResource(plan, id);
    inputs.push_back(makeResourceJson(*resource));
  }
  for (std::uint32_t id : plan.outputs) {
    const TensorResource* resource = findResource(plan, id);
    outputs.push_back(makeResourceJson(*resource));
  }
  llvm::json::Array kernels;
  for (const auto& kernel : plan.kernels) {
    llvm::json::Array local_size;
    for (std::uint32_t extent : kernel.local_size) {
      local_size.push_back(static_cast<std::int64_t>(extent));
    }
    llvm::json::Object object;
    object["id"] = static_cast<std::int64_t>(kernel.id);
    object["operation"] = kernel.operation;
    object["entry_point"] = kernel.entry_point;
    object["local_size"] = std::move(local_size);
    kernels.push_back(std::move(object));
  }
  llvm::json::Object manifest;
  manifest["schema_version"] = 1;
  manifest["kind"] = mlir::ncnn::contract::kVulkanModelManifestKind.str();
  manifest["backend"] = "vulkan";
  manifest["abi_version"] = static_cast<std::int64_t>(NCNN_VK_ABI_VERSION);
  manifest["model"] = std::string(model_name);
  manifest["host_triple"] = plan.host_triple;
  manifest["device_target"] = plan.device_target;
  manifest["build_identity"] = std::string(build_identity);
  manifest["plan_hash"] = std::string(plan_hash);
  manifest["plan_sha256"] = std::string(plan_sha256);
  manifest["shader_sha256"] = std::string(shader_sha256);
  manifest["shader_size_bytes"] = static_cast<std::int64_t>(shader_size);
  manifest["binary_symbol"] = std::string(binary_symbol);
  manifest["model_param_sha256"] = std::string(param_sha256);
  manifest["model_bin_sha256"] = std::string(bin_sha256);
  manifest["inputs"] = std::move(inputs);
  manifest["outputs"] = std::move(outputs);
  manifest["kernels"] = std::move(kernels);
  manifest["dependencies"] =
    llvm::json::Array{"libLLVM.so.21.1", "libvulkan.so.1"};
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
}

}  // namespace

std::expected<void, std::string> validateVulkanSpirvInterface(
  const VulkanExecutionPlan& plan,
  std::string_view binary_ir,
  std::string_view disassembly) {
  if (plan.kernels.size() != 1) {
    return std::unexpected("Vulkan product v1 requires exactly one kernel");
  }
  const auto& kernel = plan.kernels.front();
  const std::string binary_symbol = plan.model + "_vulkan";
  const std::string binary_declaration = "gpu.binary @" + binary_symbol;
  const std::size_t binary_position = binary_ir.find(binary_declaration);
  if (binary_position == std::string_view::npos ||
      binary_ir.find(binary_declaration,
                     binary_position + binary_declaration.size()) !=
        std::string_view::npos) {
    return std::unexpected(
      "GPU binary symbol does not match the execution plan");
  }
  for (std::string_view residual : {"gpu.module",
                                    "gpu.func",
                                    "memref.",
                                    "func.func",
                                    "ncnn.",
                                    "tosa.",
                                    "linalg.",
                                    "tensor.",
                                    "unrealized_conversion_cast"}) {
    if (binary_ir.find(residual) != std::string_view::npos) {
      return std::unexpected(std::format(
        "serialized GPU module retains forbidden IR: {}", residual));
    }
  }
  if (disassembly.find("; Version: 1.0") == std::string_view::npos ||
      disassembly.find(
        "OpExtension \"SPV_KHR_storage_buffer_storage_class\"") ==
        std::string_view::npos) {
    return std::unexpected("SPIR-V target is not Vulkan 1.1 / SPIR-V 1.0");
  }
  const std::regex entry_point_pattern(std::format(
    R"(OpEntryPoint\s+GLCompute\s+%\S+\s+\"{}\")", kernel.entry_point));
  if (!std::regex_search(
        disassembly.begin(), disassembly.end(), entry_point_pattern)) {
    return std::unexpected("SPIR-V entry point does not match the plan");
  }
  const std::regex local_size_pattern(
    std::format(R"(OpExecutionMode\s+%\S+\s+LocalSize\s+{}\s+{}\s+{})",
                kernel.local_size[0],
                kernel.local_size[1],
                kernel.local_size[2]));
  if (!std::regex_search(
        disassembly.begin(), disassembly.end(), local_size_pattern)) {
    return std::unexpected("SPIR-V workgroup size does not match the plan");
  }

  std::map<std::string, std::uint32_t> descriptor_bindings;
  std::map<std::string, std::uint32_t> descriptor_sets;
  const std::regex binding_pattern(
    R"(OpDecorate\s+(%\S+)\s+Binding\s+([0-9]+))");
  const std::regex set_pattern(
    R"(OpDecorate\s+(%\S+)\s+DescriptorSet\s+([0-9]+))");
  for (std::cregex_iterator
         it(disassembly.begin(), disassembly.end(), binding_pattern),
       end;
       it != end;
       ++it) {
    descriptor_bindings[(*it)[1].str()] =
      static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
  }
  for (std::cregex_iterator
         it(disassembly.begin(), disassembly.end(), set_pattern),
       end;
       it != end;
       ++it) {
    descriptor_sets[(*it)[1].str()] =
      static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
  }
  std::set<std::pair<std::uint32_t, std::uint32_t>> actual_bindings;
  for (const auto& [target, binding] : descriptor_bindings) {
    auto set = descriptor_sets.find(target);
    if (set == descriptor_sets.end()) {
      return std::unexpected("SPIR-V descriptor is missing its descriptor set");
    }
    actual_bindings.emplace(set->second, binding);
  }
  std::set<std::pair<std::uint32_t, std::uint32_t>> planned_bindings;
  for (const auto& binding : kernel.bindings) {
    planned_bindings.emplace(binding.set, binding.binding);
  }
  if (actual_bindings.size() != descriptor_bindings.size() ||
      actual_bindings != planned_bindings) {
    return std::unexpected(
      "SPIR-V descriptor set/bindings differ from the plan");
  }
  return {};
}

std::expected<VulkanModelPackage, std::string> makeVulkanModelPackage(
  std::string_view model_name,
  std::string_view host_triple,
  std::string_view plan_json,
  std::string_view spirv_bytes,
  std::string_view param_bytes,
  std::string_view bin_bytes) {
  auto parsed = ncnn_mlir::vulkan::deserializeExecutionPlan(plan_json);
  if (!parsed) {
    return std::unexpected("invalid Vulkan execution plan: " + parsed.error());
  }
  const auto isX86Linux = [](std::string_view triple) {
    const std::size_t separator = triple.find('-');
    return triple.substr(0, separator) == "x86_64" &&
           triple.find("linux") != std::string_view::npos;
  };
  if (parsed->device_target != "vulkan1.1-spirv1.0" ||
      !isX86Linux(parsed->host_triple) || !isX86Linux(host_triple)) {
    return std::unexpected(
      "Vulkan product v1 requires an x86-64 Linux host "
      "and Vulkan 1.1 / SPIR-V 1.0 device target");
  }
  if (parsed->kernels.size() != 1 || parsed->inputs.size() != 1 ||
      parsed->outputs.size() != 1 ||
      parsed->kernels.front().operation != "relu" ||
      parsed->kernels.front().entry_point != "relu" ||
      parsed->kernels.front().local_size !=
        std::array<std::uint32_t, 3>{64, 1, 1}) {
    return std::unexpected(
      "Vulkan product v1 currently supports one static "
      "FP32 ReLU kernel with one input and one output");
  }
  const TensorResource* input = findResource(*parsed, parsed->inputs.front());
  const TensorResource* output = findResource(*parsed, parsed->outputs.front());
  if (input == nullptr || output == nullptr ||
      input->kind != ResourceKind::Input ||
      output->kind != ResourceKind::Output ||
      input->byte_size != output->byte_size || input->shape != output->shape ||
      input->byte_size % sizeof(float) != 0 ||
      input->host_layout != output->host_layout ||
      input->host_layout != "linear" || parsed->resources.size() != 2) {
    return std::unexpected(
      "Vulkan product ReLU I/O must be matching static "
      "FP32 continuous buffers without model weights");
  }
  for (const TensorResource& resource : parsed->resources) {
    if (resource.kind != ResourceKind::Input &&
        resource.kind != ResourceKind::Output) {
      return std::unexpected(
        "Vulkan product ReLU does not accept static "
        "weight or intermediate resources");
    }
  }
  auto shader_words = decodeSpirv(spirv_bytes);
  if (!shader_words) {
    return std::unexpected(shader_words.error());
  }

  const std::string binary_symbol = parsed->model + "_vulkan";
  const std::string shader_sha256 = sha256Hex(spirv_bytes);
  const std::string param_sha256 = sha256Hex(param_bytes);
  const std::string bin_sha256 = sha256Hex(bin_bytes);
  const std::string identity_input = std::format(
    "ncnn-vulkan-product-v1|model={}|host={}|param={}|bin={}|shader={}",
    model_name,
    host_triple,
    param_sha256,
    bin_sha256,
    shader_sha256);
  const std::string build_identity =
    "ncnn-vulkan-product-v1-" + sha256Hex(identity_input);
  parsed->model = model_name;
  parsed->host_triple = host_triple;
  parsed->build_identity = build_identity;
  parsed->plan_hash.clear();
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(*parsed);
  if (!serialized) {
    return std::unexpected("cannot serialize product execution plan: " +
                           serialized.error());
  }
  auto final_plan = ncnn_mlir::vulkan::deserializeExecutionPlan(*serialized);
  if (!final_plan) {
    return std::unexpected("product execution plan failed validation: " +
                           final_plan.error());
  }

  const std::string plan_sha256 = sha256Hex(*serialized);
  const std::string header = makeHeader(model_name, *input, *output);
  const std::string wrapper_source =
    makeWrapperSource(model_name,
                      std::string(model_name) + "_vulkan.h",
                      *serialized,
                      *shader_words,
                      *final_plan);
  const std::string manifest = makeManifest(*final_plan,
                                            model_name,
                                            binary_symbol,
                                            build_identity,
                                            final_plan->plan_hash,
                                            plan_sha256,
                                            shader_sha256,
                                            spirv_bytes.size(),
                                            param_sha256,
                                            bin_sha256);
  return VulkanModelPackage{.plan = std::move(*final_plan),
                            .shader_words = std::move(*shader_words),
                            .binary_symbol = binary_symbol,
                            .plan_json = std::move(*serialized),
                            .header = header,
                            .wrapper_source = wrapper_source,
                            .manifest_json = manifest,
                            .build_identity = build_identity,
                            .shader_sha256 = shader_sha256,
                            .plan_sha256 = plan_sha256};
}

std::expected<std::string, std::string> updateVulkanManifestDependencies(
  std::string_view manifest_json, std::vector<std::string> dependencies) {
  auto parsed = llvm::json::parse(
    llvm::StringRef(manifest_json.data(), manifest_json.size()));
  if (!parsed) {
    return std::unexpected("cannot parse generated Vulkan manifest JSON");
  }
  llvm::json::Object* object = parsed->getAsObject();
  if (object == nullptr) {
    return std::unexpected("generated Vulkan manifest is not a JSON object");
  }
  const auto kind = object->getString("kind");
  if (!kind || *kind != mlir::ncnn::contract::kVulkanModelManifestKind) {
    return std::unexpected("generated Vulkan manifest has an unexpected kind");
  }
  std::ranges::sort(dependencies);
  const auto unique = std::ranges::unique(dependencies);
  dependencies.erase(unique.begin(), unique.end());
  llvm::json::Array dependency_array;
  for (std::string& dependency : dependencies) {
    dependency_array.push_back(std::move(dependency));
  }
  (*object)["dependencies"] = std::move(dependency_array);
  return llvm::formatv("{0:2}\n", std::move(*parsed)).str();
}

}  // namespace ncnn_compile
