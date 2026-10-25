#include "VulkanProduct.hpp"

#include "VulkanLifecycleCodegen.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
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
using ncnn_mlir::vulkan::VulkanModelShader;
using ncnn_mlir::vulkan::VulkanModelWeight;

constexpr std::array<unsigned char, 4> kSpirvMagic = {0x03, 0x02, 0x23, 0x07};
// VA5 contract: the published model links only the generated fixed-run host
// object, the minimal device helper, and platform/Vulkan libraries.  The
// integer codegen version moves to 2 together with the host implementation
// flip; the string domain in the header pins the fixed-run source generator.
constexpr std::string_view kHostImplementation = "model-fixed-run";
constexpr std::int64_t kHostCodegenVersion = 2;

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

void appendU64(std::string& output, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void appendIdentityField(std::string& output,
                         std::string_view tag,
                         std::string_view value) {
  appendU64(output, tag.size());
  output.append(tag);
  appendU64(output, value.size());
  output.append(value);
}

// The fixed-run generated-source identity is a pure function of the verified
// host program identity and the codegen domain, so packaging can pin it in the
// manifest/build identity without generating the deployment source itself.
std::string hostSourceCodegenIdentity(std::string_view host_program_identity) {
  std::string identity;
  appendIdentityField(
    identity, "domain", "ncnn-vulkan-host-source-identity/v1");
  appendIdentityField(identity, "execution-mode", "fixed-run");
  appendIdentityField(
    identity, "codegen-version", kVulkanHostSourceCodegenVersion);
  appendIdentityField(identity, "host-program", host_program_identity);
  return sha256Hex(identity);
}

std::string semanticPlanSHA256(VulkanExecutionPlan plan) {
  // Both execution-plan hashes and the package identity are excluded from this
  // semantic digest. The fixed nonempty build identity satisfies the plan
  // verifier while making equivalent parsed JSON documents serialize alike.
  plan.build_identity = "vulkan-product-semantic-plan-v1";
  plan.plan_hash.clear();
  auto canonical = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  return canonical ? sha256Hex(*canonical) : std::string{};
}

std::expected<llvm::json::Value, std::string> parseHostProgramJson(
  std::string_view json) {
  auto parsed = llvm::json::parse(llvm::StringRef(json.data(), json.size()));
  if (!parsed) {
    return std::unexpected("cannot parse serialized Vulkan host program: " +
                           llvm::toString(parsed.takeError()));
  }
  if (!parsed->getAsObject()) {
    return std::unexpected("serialized Vulkan host program is not an object");
  }
  return std::move(*parsed);
}

std::string uppercaseIdentifier(std::string_view value) {
  std::string result(value);
  std::ranges::transform(result, result.begin(), [](unsigned char character) {
    return static_cast<char>(
      character >= 'a' && character <= 'z' ? character - 'a' + 'A' : character);
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

bool isCIdentifier(std::string_view value) {
  if (value.empty() || !((value.front() >= 'a' && value.front() <= 'z') ||
                         (value.front() >= 'A' && value.front() <= 'Z') ||
                         value.front() == '_')) {
    return false;
  }
  return std::ranges::all_of(value.substr(1), [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_';
  });
}

std::string makeGeneralHeader(std::string_view model_name,
                              const VulkanExecutionPlan& plan) {
  const std::string macro = uppercaseIdentifier(model_name) + "_VULKAN";
  const std::string context = std::string(model_name) + "_vk_context";
  std::string result = std::format(
    "#ifndef {0}_H\n#define {0}_H\n\n"
    "#include <stddef.h>\n#include <stdint.h>\n"
    "#include \"ncnn-mlir/Vulkan/ModelABI.h\"\n\n"
    "#define {1}_INPUT_COUNT UINT32_C({2})\n"
    "#define {1}_OUTPUT_COUNT UINT32_C({3})\n",
    macro,
    macro,
    plan.inputs.size(),
    plan.outputs.size());
  for (std::size_t index = 0; index < plan.inputs.size(); ++index) {
    const TensorResource* input = findResource(plan, plan.inputs[index]);
    result += std::format(
      "#define {0}_INPUT{1}_RANK UINT32_C({2})\n"
      "#define {0}_INPUT{1}_ELEMENTS UINT64_C({3})\n",
      macro,
      index + 1,
      input->shape.size(),
      input->byte_size / sizeof(float));
    for (std::size_t dimension = 0; dimension < input->shape.size();
         ++dimension) {
      result += std::format("#define {0}_INPUT{1}_DIM{2} UINT64_C({3})\n",
                            macro,
                            index + 1,
                            dimension,
                            input->shape[dimension]);
    }
  }
  for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
    const TensorResource* output = findResource(plan, plan.outputs[index]);
    result += std::format(
      "#define {0}_OUTPUT{1}_RANK UINT32_C({2})\n"
      "#define {0}_OUTPUT{1}_ELEMENTS UINT64_C({3})\n",
      macro,
      index + 1,
      output->shape.size(),
      output->byte_size / sizeof(float));
    for (std::size_t dimension = 0; dimension < output->shape.size();
         ++dimension) {
      result += std::format("#define {0}_OUTPUT{1}_DIM{2} UINT64_C({3})\n",
                            macro,
                            index + 1,
                            dimension,
                            output->shape[dimension]);
    }
  }
  result += std::format(
    "\ntypedef struct {0} {0};\n\n"
    "#ifdef __cplusplus\nextern \"C\" {{\n#endif\n"
    "ncnn_vk_status {1}_vk_create(const ncnn_vk_options *options,\n"
    "    {0} **context, char *diagnostic, size_t diagnostic_capacity);\n"
    "ncnn_vk_status {1}_vk_run({0} *context,",
    context,
    model_name);
  for (std::size_t index = 0; index < plan.inputs.size(); ++index) {
    result += std::format(
      "\n    const float *input{0}, size_t input{0}_elements,", index + 1);
  }
  for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
    result += std::format("\n    float *output{0}, size_t output{0}_elements,",
                          index + 1);
  }
  result += std::format(
    "\n    char *diagnostic, size_t diagnostic_capacity);\n"
    "ncnn_vk_status {1}_vk_destroy({0} *context);\n"
    "#ifdef __cplusplus\n}}\n#endif\n\n#endif  // {2}_H\n",
    context,
    model_name,
    macro);
  return result;
}

std::string makeGeneralWrapperSource(std::string_view model_name,
                                     std::string_view header_name,
                                     std::string_view plan_json,
                                     std::span<const VulkanModelShader> shaders,
                                     std::span<const VulkanModelWeight> weights,
                                     const VulkanExecutionPlan& plan) {
  std::string source =
    "#include <algorithm>\n#include <array>\n#include <cstddef>\n"
    "#include <cstdint>\n#include <exception>\n#include <memory>\n"
    "#include <new>\n#include <span>\n#include <string_view>\n"
    "#include <utility>\n\n"
    "#include \"ncnn-mlir/Support/VulkanExecutionPlan.hpp\"\n"
    "#include \"ncnn-mlir/Vulkan/VulkanRuntime.hpp\"\n"
    "#include \"@HEADER@\"\n\n"
    "namespace {\n"
    "using ncnn_mlir::vulkan::VulkanExecutionPlan;\n"
    "using ncnn_mlir::vulkan::VulkanInputTensor;\n"
    "using ncnn_mlir::vulkan::VulkanKernelBinary;\n"
    "using ncnn_mlir::vulkan::VulkanOutputTensor;\n"
    "using ncnn_mlir::vulkan::VulkanResourcePayload;\n"
    "using ncnn_mlir::vulkan::VulkanRuntimeContext;\n"
    "using ncnn_mlir::vulkan::VulkanRuntimeOptions;\n"
    "constexpr unsigned char kPlanJsonBytes[] = {@PLAN_BYTES@};\n";
  replaceAll(source, "@HEADER@", header_name);
  replaceAll(source, "@PLAN_BYTES@", byteInitializer(plan_json));

  for (std::size_t index = 0; index < shaders.size(); ++index) {
    source +=
      std::format("constexpr std::uint32_t kShaderWords{0}[] = {{{1}}};\n",
                  index,
                  wordInitializer(shaders[index].words));
  }
  source +=
    std::format("constexpr std::array<VulkanKernelBinary, {0}> kKernels{{{{\n",
                shaders.size());
  for (std::size_t index = 0; index < shaders.size(); ++index) {
    source += std::format(
      "  VulkanKernelBinary{{.kernel_id = {0}, .words = "
      "kShaderWords{1}}}{2}\n",
      shaders[index].kernel_id,
      index,
      index + 1 == shaders.size() ? "" : ",");
  }
  source += "}};\n";

  for (std::size_t index = 0; index < weights.size(); ++index) {
    std::string initializer;
    for (std::size_t byte = 0; byte < weights[index].bytes.size(); ++byte) {
      if (byte != 0) {
        initializer += ", ";
      }
      initializer += std::format("0x{:02x}", weights[index].bytes[byte]);
    }
    source +=
      std::format("constexpr std::uint8_t kWeightBytes{0}[] = {{{1}}};\n",
                  index,
                  initializer);
  }
  source +=
    std::format("const std::array<VulkanResourcePayload, {0}> kWeights{{{{\n",
                weights.size());
  for (std::size_t index = 0; index < weights.size(); ++index) {
    source += std::format(
      "  VulkanResourcePayload{{.resource_id = {0}, .bytes = "
      "std::as_bytes(std::span<const std::uint8_t>(kWeightBytes{1}))}}{2}\n",
      weights[index].resource_id,
      index,
      index + 1 == weights.size() ? "" : ",");
  }
  source += "}};\n";
  // clang-format off
  source += R"cpp(

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

    ncnn_vk_status validateDiagnostic(char* diagnostic,
                                      size_t diagnostic_capacity) noexcept {
      if (diagnostic_capacity > 0 && diagnostic == nullptr) {
        return NCNN_VK_STATUS_INVALID_ARGUMENT;
      }
      if (diagnostic_capacity > 0) {
        diagnostic[0] = '\0';
      }
      return NCNN_VK_STATUS_SUCCESS;
    }

    }  // namespace

    struct @CONTEXT@{explicit @CONTEXT@(VulkanRuntimeContext value) : runtime(
      std::move(value)){} VulkanRuntimeContext runtime;
    }
    ;

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
        VulkanRuntimeOptions runtime_options =
          ncnn_mlir::vulkan::makeVulkanRuntimeOptions();
        runtime_options.device_index = options->device_index;
        auto runtime = VulkanRuntimeContext::create(
          *plan, kKernels, kWeights, runtime_options);
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
        return report(NCNN_VK_STATUS_OUT_OF_MEMORY,
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

    extern "C" ncnn_vk_status @RUN@(@RUN_ARGUMENTS@) {
      const ncnn_vk_status diagnostic_status =
        validateDiagnostic(diagnostic, diagnostic_capacity);
      if (diagnostic_status != NCNN_VK_STATUS_SUCCESS) {
        return diagnostic_status;
      }
      if (context == nullptr @NULL_CHECKS@) {
        return report(NCNN_VK_STATUS_INVALID_ARGUMENT,
                      diagnostic,
                      diagnostic_capacity,
                      "invalid run argument");
      }
      if (@COUNT_CHECKS@) {
        return report(NCNN_VK_STATUS_BUFFER_SIZE_MISMATCH,
                      diagnostic,
                      diagnostic_capacity,
                      "element count does not match static shape");
      }
      try {
        const std::array<VulkanInputTensor, @INPUT_COUNT@> inputs{{@INPUTS@}};
        const std::array<VulkanOutputTensor, @OUTPUT_COUNT@> outputs{{@OUTPUTS@}};
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

  std::string run_arguments = std::string(model_name) + "_vk_context* context";
  std::string null_checks;
  std::string count_checks;
  std::string input_initializers;
  std::string output_initializers;
  for (std::size_t index = 0; index < plan.inputs.size(); ++index) {
    const std::string number = std::to_string(index + 1);
    const TensorResource* input = findResource(plan, plan.inputs[index]);
    run_arguments +=
      std::format(", const float* input{0}, size_t input{0}_elements", number);
    null_checks += std::format(" || input{0} == nullptr", number);
    if (!count_checks.empty()) {
      count_checks += " || ";
    }
    count_checks += std::format(
      "input{0}_elements != {1}", number, input->byte_size / sizeof(float));
    input_initializers += std::format(
      "      VulkanInputTensor{{.resource_id = {0}, .bytes = "
      "std::as_bytes(std::span<const float>(input{1}, "
      "input{1}_elements))}}{2}\n",
      input->id,
      number,
      index + 1 == plan.inputs.size() ? "" : ",");
  }
  for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
    const std::string number = std::to_string(index + 1);
    const TensorResource* output = findResource(plan, plan.outputs[index]);
    run_arguments +=
      std::format(", float* output{0}, size_t output{0}_elements", number);
    null_checks += std::format(" || output{0} == nullptr", number);
    if (!count_checks.empty()) {
      count_checks += " || ";
    }
    count_checks += std::format(
      "output{0}_elements != {1}", number, output->byte_size / sizeof(float));
    output_initializers += std::format(
      "      VulkanOutputTensor{{.resource_id = {0}, .bytes = "
      "std::as_writable_bytes(std::span<float>(output{1}, "
      "output{1}_elements))}}{2}\n",
      output->id,
      number,
      index + 1 == plan.outputs.size() ? "" : ",");
  }
  run_arguments += ", char* diagnostic, size_t diagnostic_capacity";
  replaceAll(source, "@CONTEXT@", std::string(model_name) + "_vk_context");
  replaceAll(source, "@CREATE@", std::string(model_name) + "_vk_create");
  replaceAll(source, "@RUN@", std::string(model_name) + "_vk_run");
  replaceAll(source, "@RUN_ARGUMENTS@", run_arguments);
  replaceAll(source, "@NULL_CHECKS@", null_checks);
  replaceAll(source, "@COUNT_CHECKS@", count_checks);
  replaceAll(source, "@INPUT_COUNT@", std::to_string(plan.inputs.size()));
  replaceAll(source, "@OUTPUT_COUNT@", std::to_string(plan.outputs.size()));
  replaceAll(source, "@INPUTS@", input_initializers);
  replaceAll(source, "@OUTPUTS@", output_initializers);
  replaceAll(source, "@DESTROY@", std::string(model_name) + "_vk_destroy");
  return source;
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
                         std::string_view bin_sha256,
                         std::string_view host_program_identity,
                         std::string_view host_codegen_identity,
                         llvm::json::Value host_program) {
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
  manifest["host_implementation"] = std::string(kHostImplementation);
  manifest["host_codegen_version"] = kHostCodegenVersion;
  manifest["host_source_codegen_version"] =
    std::string(kVulkanHostSourceCodegenVersion);
  manifest["host_codegen_identity"] = std::string(host_codegen_identity);
  manifest["host_program"] = std::move(host_program);
  manifest["host_program_identity"] = std::string(host_program_identity);
  manifest["host_program_usage"] = "compiler-only-diagnostic";
  // The deployment no longer parses this plan; the sidecar stays as build
  // evidence only.  The final dependency list is rewritten after the strict
  // link/audit step.
  manifest["execution_plan_usage"] = "build-evidence-only";
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
  manifest["dependencies"] = llvm::json::Array{"libvulkan.so.1"};
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
}

std::string makeGeneralManifest(const VulkanExecutionPlan& plan,
                                std::string_view model_name,
                                std::string_view build_identity,
                                std::string_view plan_sha256,
                                std::string_view param_sha256,
                                std::string_view bin_sha256,
                                std::span<const VulkanModelShader> shaders,
                                std::span<const std::string> shader_hashes,
                                std::span<const VulkanModelWeight> weights,
                                std::string_view host_program_identity,
                                std::string_view host_codegen_identity,
                                llvm::json::Value host_program) {
  llvm::json::Array inputs;
  llvm::json::Array outputs;
  for (std::uint32_t id : plan.inputs) {
    inputs.push_back(makeResourceJson(*findResource(plan, id)));
  }
  for (std::uint32_t id : plan.outputs) {
    outputs.push_back(makeResourceJson(*findResource(plan, id)));
  }

  llvm::json::Array kernels;
  llvm::json::Array shader_records;
  llvm::json::Array shader_hash_list;
  for (const auto& kernel : plan.kernels) {
    const auto shader =
      std::ranges::find(shaders, kernel.id, &VulkanModelShader::kernel_id);
    const std::size_t shader_index =
      static_cast<std::size_t>(shader - shaders.begin());
    llvm::json::Array local_size;
    for (std::uint32_t extent : kernel.local_size) {
      local_size.push_back(static_cast<std::int64_t>(extent));
    }
    llvm::json::Object kernel_object;
    kernel_object["id"] = static_cast<std::int64_t>(kernel.id);
    kernel_object["operation"] = kernel.operation;
    kernel_object["entry_point"] = kernel.entry_point;
    kernel_object["local_size"] = std::move(local_size);
    kernels.push_back(std::move(kernel_object));

    llvm::json::Object shader_object;
    shader_object["kernel_id"] = static_cast<std::int64_t>(kernel.id);
    shader_object["binary_symbol"] = shader->binary_symbol;
    shader_object["entry_point"] = kernel.entry_point;
    shader_object["sha256"] = shader_hashes[shader_index];
    shader_object["size_bytes"] =
      static_cast<std::int64_t>(shader->words.size() * sizeof(std::uint32_t));
    shader_records.push_back(std::move(shader_object));
    shader_hash_list.push_back(shader_hashes[shader_index]);
  }

  llvm::json::Array weight_records;
  llvm::json::Array weight_hash_list;
  for (const TensorResource& resource : plan.resources) {
    if (resource.kind != ResourceKind::Weight) {
      continue;
    }
    const auto payload =
      std::ranges::find(weights, resource.id, &VulkanModelWeight::resource_id);
    const std::string physical_hash = sha256Hex(
      std::string_view(reinterpret_cast<const char*>(payload->bytes.data()),
                       payload->bytes.size()));
    llvm::json::Object weight_object;
    weight_object["resource_id"] = static_cast<std::int64_t>(resource.id);
    weight_object["sha256"] = physical_hash;
    weight_object["size_bytes"] = static_cast<std::int64_t>(resource.byte_size);
    weight_records.push_back(std::move(weight_object));
    weight_hash_list.push_back(physical_hash);
  }

  const std::string first_symbol = shaders.front().binary_symbol;
  const std::string first_shader_hash = shader_hashes.front();
  const std::size_t first_shader_size =
    shaders.front().words.size() * sizeof(std::uint32_t);
  llvm::json::Object manifest;
  manifest["schema_version"] = 1;
  manifest["kind"] = mlir::ncnn::contract::kVulkanModelManifestKind.str();
  manifest["backend"] = "vulkan";
  manifest["host_implementation"] = std::string(kHostImplementation);
  manifest["host_codegen_version"] = kHostCodegenVersion;
  manifest["host_source_codegen_version"] =
    std::string(kVulkanHostSourceCodegenVersion);
  manifest["host_codegen_identity"] = std::string(host_codegen_identity);
  manifest["host_program"] = std::move(host_program);
  manifest["host_program_identity"] = std::string(host_program_identity);
  manifest["host_program_usage"] = "compiler-only-diagnostic";
  // The deployment no longer parses this plan; the sidecar stays as build
  // evidence only.  The final dependency list is rewritten after the strict
  // link/audit step.
  manifest["execution_plan_usage"] = "build-evidence-only";
  manifest["abi_version"] = static_cast<std::int64_t>(NCNN_VK_ABI_VERSION);
  manifest["model"] = std::string(model_name);
  manifest["host_triple"] = plan.host_triple;
  manifest["device_target"] = plan.device_target;
  manifest["build_identity"] = std::string(build_identity);
  manifest["plan_hash"] = plan.plan_hash;
  manifest["plan_sha256"] = std::string(plan_sha256);
  manifest["shader_sha256"] = first_shader_hash;
  manifest["shader_size_bytes"] = static_cast<std::int64_t>(first_shader_size);
  manifest["binary_symbol"] = first_symbol;
  manifest["shader_sha256s"] = std::move(shader_hash_list);
  manifest["shaders"] = std::move(shader_records);
  manifest["weight_sha256s"] = std::move(weight_hash_list);
  manifest["weights"] = std::move(weight_records);
  manifest["model_param_sha256"] = std::string(param_sha256);
  manifest["model_bin_sha256"] = std::string(bin_sha256);
  manifest["inputs"] = std::move(inputs);
  manifest["outputs"] = std::move(outputs);
  manifest["kernels"] = std::move(kernels);
  manifest["dependencies"] = llvm::json::Array{"libvulkan.so.1"};
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
}

std::string spirvBytes(std::span<const std::uint32_t> words) {
  std::string result;
  result.reserve(words.size() * sizeof(std::uint32_t));
  for (std::uint32_t word : words) {
    result.push_back(static_cast<char>(word & 0xff));
    result.push_back(static_cast<char>((word >> 8) & 0xff));
    result.push_back(static_cast<char>((word >> 16) & 0xff));
    result.push_back(static_cast<char>((word >> 24) & 0xff));
  }
  return result;
}

std::expected<std::uint32_t, std::string> parseU32(std::string_view value) {
  std::uint32_t parsed = 0;
  const auto [end, error] =
    std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size()) {
    return std::unexpected("SPIR-V descriptor index is out of range");
  }
  return parsed;
}

}  // namespace

std::expected<void, std::string> validateVulkanSpirvInterface(
  const VulkanExecutionPlan& plan,
  std::string_view binary_ir,
  std::string_view disassembly) {
  if (plan.kernels.size() != 1) {
    return std::unexpected(
      "legacy Vulkan product validation requires one kernel");
  }
  return validateVulkanSpirvInterface(
    plan, plan.kernels.front(), plan.model + "_vulkan", binary_ir, disassembly);
}

std::expected<void, std::string> validateVulkanSpirvInterface(
  const VulkanExecutionPlan& plan,
  const ncnn_mlir::vulkan::KernelExecution& kernel,
  std::string_view binary_symbol,
  std::string_view binary_ir,
  std::string_view disassembly) {
  const auto planned_kernel = std::ranges::find(
    plan.kernels, kernel.id, &ncnn_mlir::vulkan::KernelExecution::id);
  if (planned_kernel == plan.kernels.end() ||
      planned_kernel->entry_point != kernel.entry_point ||
      planned_kernel->local_size != kernel.local_size ||
      planned_kernel->bindings.size() != kernel.bindings.size()) {
    return std::unexpected("kernel interface does not belong to the plan");
  }
  for (std::size_t index = 0; index < kernel.bindings.size(); ++index) {
    const auto& expected = planned_kernel->bindings[index];
    const auto& actual = kernel.bindings[index];
    if (expected.set != actual.set || expected.binding != actual.binding ||
        expected.resource_id != actual.resource_id ||
        expected.access != actual.access) {
      return std::unexpected("kernel bindings do not belong to the plan");
    }
  }
  if (binary_symbol.empty()) {
    return std::unexpected("GPU binary symbol must not be empty");
  }
  const std::string binary_declaration =
    "gpu.binary @" + std::string(binary_symbol);
  std::size_t matches = 0;
  std::size_t position = 0;
  while ((position = binary_ir.find(binary_declaration, position)) !=
         std::string_view::npos) {
    position += binary_declaration.size();
    if (position == binary_ir.size() || binary_ir[position] == '{' ||
        std::isspace(static_cast<unsigned char>(binary_ir[position])) != 0) {
      ++matches;
    }
  }
  if (matches != 1) {
    return std::unexpected("GPU binary symbol does not match the kernel");
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
  const std::string disassembly_text(disassembly);
  const std::regex entry_point_pattern(
    R"(OpEntryPoint\s+GLCompute\s+(%\S+)\s+\"([^\"]+)\")");
  std::string entry_point_id;
  std::size_t entry_point_matches = 0;
  for (std::sregex_iterator it(
         disassembly_text.begin(), disassembly_text.end(), entry_point_pattern),
       end;
       it != end;
       ++it) {
    if ((*it)[2].str() == kernel.entry_point) {
      entry_point_id = (*it)[1].str();
      ++entry_point_matches;
    }
  }
  if (entry_point_matches != 1) {
    return std::unexpected("SPIR-V entry point does not match the plan");
  }
  const std::regex local_size_pattern(
    R"(OpExecutionMode\s+(%\S+)\s+LocalSize\s+([0-9]+)\s+([0-9]+)\s+([0-9]+))");
  std::size_t local_size_matches = 0;
  for (std::sregex_iterator it(
         disassembly_text.begin(), disassembly_text.end(), local_size_pattern),
       end;
       it != end;
       ++it) {
    if ((*it)[1].str() != entry_point_id) {
      continue;
    }
    ++local_size_matches;
    if ((*it)[2].str() != std::to_string(kernel.local_size[0]) ||
        (*it)[3].str() != std::to_string(kernel.local_size[1]) ||
        (*it)[4].str() != std::to_string(kernel.local_size[2])) {
      return std::unexpected("SPIR-V workgroup size does not match the plan");
    }
  }
  if (local_size_matches != 1) {
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
    auto binding = parseU32((*it)[2].str());
    if (!binding) {
      return std::unexpected(binding.error());
    }
    descriptor_bindings[(*it)[1].str()] = *binding;
  }
  for (std::cregex_iterator
         it(disassembly.begin(), disassembly.end(), set_pattern),
       end;
       it != end;
       ++it) {
    auto set = parseU32((*it)[2].str());
    if (!set) {
      return std::unexpected(set.error());
    }
    descriptor_sets[(*it)[1].str()] = *set;
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
  std::span<const VulkanModelShader> shaders,
  std::span<const VulkanModelWeight> weights,
  std::string_view param_bytes,
  std::string_view bin_bytes) {
  auto parsed = ncnn_mlir::vulkan::deserializeExecutionPlan(plan_json);
  if (!parsed) {
    return std::unexpected("invalid Vulkan execution plan: " + parsed.error());
  }
  if (!isCIdentifier(model_name)) {
    return std::unexpected("Vulkan model name must be a C identifier");
  }
  const auto isX86Linux = [](std::string_view triple) {
    const std::size_t separator = triple.find('-');
    return triple.substr(0, separator) == "x86_64" &&
           triple.find("linux") != std::string_view::npos;
  };
  if (parsed->device_target != "vulkan1.1-spirv1.0" ||
      !isX86Linux(parsed->host_triple) || !isX86Linux(host_triple)) {
    return std::unexpected(
      "Vulkan product requires an x86-64 Linux host "
      "and Vulkan 1.1 / SPIR-V 1.0 device target");
  }
  for (const TensorResource& resource : parsed->resources) {
    if (resource.byte_size == 0 || resource.byte_size % sizeof(float) != 0) {
      return std::unexpected(std::format(
        "resource {} is not a nonempty static FP32 tensor", resource.id));
    }
    if ((resource.kind == ResourceKind::Input ||
         resource.kind == ResourceKind::Output) &&
        resource.host_layout != resource.physical_layout) {
      return std::unexpected(
        std::format("public resource {} requires an unsupported host/physical "
                    "layout transform",
                    resource.id));
    }
  }
  if (auto verified =
        ncnn_mlir::vulkan::verifyVulkanModelWeights(*parsed, weights);
      !verified) {
    return std::unexpected("invalid Vulkan model weights: " + verified.error());
  }
  if (shaders.size() != parsed->kernels.size()) {
    return std::unexpected(
      std::format("shader count {} does not match plan kernel count {}",
                  shaders.size(),
                  parsed->kernels.size()));
  }
  std::vector<VulkanModelShader> ordered_shaders;
  std::vector<std::string> shader_hashes;
  ordered_shaders.reserve(parsed->kernels.size());
  shader_hashes.reserve(parsed->kernels.size());
  std::set<std::uint32_t> found_kernel_ids;
  std::set<std::string> found_symbols;
  for (const auto& kernel : parsed->kernels) {
    const auto shader =
      std::ranges::find(shaders, kernel.id, &VulkanModelShader::kernel_id);
    if (shader == shaders.end() ||
        !found_kernel_ids.insert(shader->kernel_id).second ||
        shader->binary_symbol.empty() ||
        !found_symbols.insert(shader->binary_symbol).second ||
        shader->words.size() < 5 ||
        shader->words.front() != UINT32_C(0x07230203)) {
      return std::unexpected(std::format(
        "kernel {} has a missing, duplicate, or invalid SPIR-V payload",
        kernel.id));
    }
    if (shader->words.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      return std::unexpected(std::format(
        "kernel {} SPIR-V payload size overflows size_t", kernel.id));
    }
    ordered_shaders.push_back(*shader);
    const std::string bytes = spirvBytes(shader->words);
    shader_hashes.push_back(sha256Hex(bytes));
  }

  std::vector<VulkanModelWeight> ordered_weights(weights.begin(),
                                                 weights.end());
  std::ranges::sort(ordered_weights, {}, &VulkanModelWeight::resource_id);
  parsed->model = model_name;
  parsed->host_triple = host_triple;
  parsed->plan_hash.clear();
  auto host_program = ncnn_mlir::vulkan::makeVulkanHostProgram(
    *parsed, ordered_shaders, ordered_weights);
  if (!host_program) {
    return std::unexpected("cannot build Vulkan host program: " +
                           host_program.error());
  }
  if (auto verified = ncnn_mlir::vulkan::verifyVulkanHostProgram(
        *host_program, *parsed, ordered_shaders, ordered_weights);
      !verified) {
    return std::unexpected("invalid Vulkan host program: " + verified.error());
  }
  auto host_program_json =
    ncnn_mlir::vulkan::serializeVulkanHostProgram(*host_program);
  if (!host_program_json) {
    return std::unexpected("cannot serialize Vulkan host program: " +
                           host_program_json.error());
  }
  auto host_program_value = parseHostProgramJson(*host_program_json);
  if (!host_program_value) {
    return std::unexpected(host_program_value.error());
  }

  const std::string param_sha256 = sha256Hex(param_bytes);
  const std::string bin_sha256 = sha256Hex(bin_bytes);
  const std::string plan_sha256_semantic = semanticPlanSHA256(*parsed);
  if (plan_sha256_semantic.empty()) {
    return std::unexpected("cannot calculate semantic Vulkan plan identity");
  }
  constexpr std::string_view kIdentityDomain = "ncnn-vulkan-static-plan-v3";
  std::string identity_input;
  appendIdentityField(identity_input, "domain", kIdentityDomain);
  appendIdentityField(identity_input, "model", model_name);
  appendIdentityField(identity_input, "host_triple", host_triple);
  appendIdentityField(identity_input, "param_sha256", param_sha256);
  appendIdentityField(identity_input, "bin_sha256", bin_sha256);
  appendIdentityField(
    identity_input, "semantic_plan_sha256", plan_sha256_semantic);
  appendIdentityField(
    identity_input, "host_implementation", kHostImplementation);
  appendIdentityField(identity_input,
                      "host_codegen_version",
                      std::to_string(kHostCodegenVersion));
  appendIdentityField(identity_input,
                      "host_source_codegen_version",
                      std::string(kVulkanHostSourceCodegenVersion));
  const std::string host_codegen_identity =
    hostSourceCodegenIdentity(host_program->identity);
  appendIdentityField(
    identity_input, "host_codegen_identity", host_codegen_identity);
  appendIdentityField(
    identity_input, "host_program_identity", host_program->identity);
  appendIdentityField(
    identity_input, "shader_count", std::to_string(ordered_shaders.size()));
  for (std::size_t index = 0; index < ordered_shaders.size(); ++index) {
    appendIdentityField(identity_input,
                        "shader_kernel_id",
                        std::to_string(ordered_shaders[index].kernel_id));
    appendIdentityField(identity_input,
                        "shader_binary_symbol",
                        ordered_shaders[index].binary_symbol);
    appendIdentityField(identity_input, "shader_sha256", shader_hashes[index]);
  }
  appendIdentityField(
    identity_input, "weight_count", std::to_string(ordered_weights.size()));
  for (const VulkanModelWeight& weight : ordered_weights) {
    appendIdentityField(
      identity_input, "weight_resource_id", std::to_string(weight.resource_id));
    appendIdentityField(identity_input,
                        "weight_sha256",
                        sha256Hex(std::string_view(
                          reinterpret_cast<const char*>(weight.bytes.data()),
                          weight.bytes.size())));
  }
  const std::string build_identity =
    std::string(kIdentityDomain) + "-" + sha256Hex(identity_input);
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
  const std::string header = makeGeneralHeader(model_name, *final_plan);
  const std::string wrapper_source =
    makeGeneralWrapperSource(model_name,
                             std::string(model_name) + "_vulkan.h",
                             *serialized,
                             ordered_shaders,
                             ordered_weights,
                             *final_plan);
  const std::string manifest =
    makeGeneralManifest(*final_plan,
                        model_name,
                        build_identity,
                        plan_sha256,
                        param_sha256,
                        bin_sha256,
                        ordered_shaders,
                        shader_hashes,
                        ordered_weights,
                        host_program->identity,
                        host_codegen_identity,
                        std::move(*host_program_value));
  const std::string first_binary_symbol = ordered_shaders.front().binary_symbol;
  const std::string first_shader_hash = shader_hashes.front();
  return VulkanModelPackage{.plan = std::move(*final_plan),
                            .host_program = std::move(*host_program),
                            .shader_words = ordered_shaders.front().words,
                            .shaders = std::move(ordered_shaders),
                            .weights = std::move(ordered_weights),
                            .shader_sha256s = shader_hashes,
                            .binary_symbol = first_binary_symbol,
                            .plan_json = std::move(*serialized),
                            .header = header,
                            .wrapper_source = wrapper_source,
                            .manifest_json = manifest,
                            .build_identity = build_identity,
                            .shader_sha256 = first_shader_hash,
                            .plan_sha256 = plan_sha256};
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
  std::vector<VulkanModelShader> packaged_shaders{
    VulkanModelShader{.kernel_id = parsed->kernels.front().id,
                      .binary_symbol = binary_symbol,
                      .words = *shader_words}};
  const std::array<VulkanModelWeight, 0> packaged_weights{};
  parsed->model = model_name;
  parsed->host_triple = host_triple;
  parsed->plan_hash.clear();
  auto host_program = ncnn_mlir::vulkan::makeVulkanHostProgram(
    *parsed, packaged_shaders, packaged_weights);
  if (!host_program) {
    return std::unexpected("cannot build Vulkan host program: " +
                           host_program.error());
  }
  if (auto verified = ncnn_mlir::vulkan::verifyVulkanHostProgram(
        *host_program, *parsed, packaged_shaders, packaged_weights);
      !verified) {
    return std::unexpected("invalid Vulkan host program: " + verified.error());
  }
  auto host_program_json =
    ncnn_mlir::vulkan::serializeVulkanHostProgram(*host_program);
  if (!host_program_json) {
    return std::unexpected("cannot serialize Vulkan host program: " +
                           host_program_json.error());
  }
  auto host_program_value = parseHostProgramJson(*host_program_json);
  if (!host_program_value) {
    return std::unexpected(host_program_value.error());
  }

  const std::string shader_sha256 = sha256Hex(spirv_bytes);
  const std::string param_sha256 = sha256Hex(param_bytes);
  const std::string bin_sha256 = sha256Hex(bin_bytes);
  const std::string plan_sha256_semantic = semanticPlanSHA256(*parsed);
  if (plan_sha256_semantic.empty()) {
    return std::unexpected("cannot calculate semantic Vulkan plan identity");
  }
  constexpr std::string_view kIdentityDomain = "ncnn-vulkan-product-v2";
  std::string identity_input;
  appendIdentityField(identity_input, "domain", kIdentityDomain);
  appendIdentityField(identity_input, "model", model_name);
  appendIdentityField(identity_input, "host_triple", host_triple);
  appendIdentityField(identity_input, "param_sha256", param_sha256);
  appendIdentityField(identity_input, "bin_sha256", bin_sha256);
  appendIdentityField(
    identity_input, "semantic_plan_sha256", plan_sha256_semantic);
  appendIdentityField(
    identity_input, "host_implementation", kHostImplementation);
  appendIdentityField(identity_input,
                      "host_codegen_version",
                      std::to_string(kHostCodegenVersion));
  appendIdentityField(identity_input,
                      "host_source_codegen_version",
                      std::string(kVulkanHostSourceCodegenVersion));
  const std::string host_codegen_identity =
    hostSourceCodegenIdentity(host_program->identity);
  appendIdentityField(
    identity_input, "host_codegen_identity", host_codegen_identity);
  appendIdentityField(
    identity_input, "host_program_identity", host_program->identity);
  appendIdentityField(
    identity_input, "shader_count", std::to_string(packaged_shaders.size()));
  appendIdentityField(identity_input,
                      "shader_kernel_id",
                      std::to_string(packaged_shaders.front().kernel_id));
  appendIdentityField(identity_input, "shader_binary_symbol", binary_symbol);
  appendIdentityField(identity_input, "shader_sha256", shader_sha256);
  appendIdentityField(
    identity_input, "weight_count", std::to_string(packaged_weights.size()));
  const std::string build_identity =
    std::string(kIdentityDomain) + "-" + sha256Hex(identity_input);
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
                                            bin_sha256,
                                            host_program->identity,
                                            host_codegen_identity,
                                            std::move(*host_program_value));
  return VulkanModelPackage{.plan = std::move(*final_plan),
                            .host_program = std::move(*host_program),
                            .shader_words = std::move(*shader_words),
                            .shaders = {},
                            .weights = {},
                            .shader_sha256s = {},
                            .binary_symbol = binary_symbol,
                            .plan_json = std::move(*serialized),
                            .header = header,
                            .wrapper_source = wrapper_source,
                            .manifest_json = manifest,
                            .build_identity = build_identity,
                            .shader_sha256 = shader_sha256,
                            .plan_sha256 = plan_sha256};
}

namespace {

std::expected<VulkanLifecycleSource, std::string> makeVulkanSource(
  const VulkanModelPackage& package, bool fixed_run) {
  const VulkanExecutionPlan& plan = package.plan;
  if (!isCIdentifier(plan.model) || plan.model.front() == '_' ||
      plan.model.back() == '_' || plan.model.find("__") != std::string::npos) {
    return std::unexpected(
      "lifecycle model name must be a non-reserved identifier prefix");
  }
  if (auto checked = ncnn_mlir::vulkan::verifyExecutionPlan(plan); !checked) {
    return std::unexpected("invalid lifecycle execution plan: " +
                           checked.error());
  }
  if (!plan.host_triple.starts_with("x86_64-") ||
      plan.host_triple.find("linux") == std::string::npos ||
      plan.device_target != "vulkan1.1-spirv1.0") {
    return std::unexpected(
      "lifecycle source requires x86-64 Linux and Vulkan 1.1");
  }
  for (const TensorResource& resource : plan.resources) {
    if (resource.byte_size == 0 || resource.byte_size % sizeof(float) != 0 ||
        !std::in_range<std::uint32_t>(resource.shape.size()) ||
        ((resource.kind == ResourceKind::Input ||
          resource.kind == ResourceKind::Output) &&
         resource.host_layout != resource.physical_layout)) {
      return std::unexpected(
        "lifecycle requires nonempty static FP32 public I/O");
    }
  }
  for (std::size_t count : {package.host_program.buffers.size(),
                            package.host_program.pipelines.size(),
                            plan.inputs.size(),
                            plan.outputs.size()}) {
    if (!std::in_range<std::uint32_t>(count)) {
      return std::unexpected("lifecycle object or I/O count exceeds uint32");
    }
  }
  std::vector<VulkanModelShader> single_shader;
  std::span<const VulkanModelShader> shaders = package.shaders;
  std::string header;
  if (shaders.empty()) {
    // The original single-ReLU packaging overload retains its payload in
    // shader_words rather than shaders. Normalize it only on the compiler side.
    if (plan.kernels.size() != 1 || plan.inputs.size() != 1 ||
        plan.outputs.size() != 1 || !package.weights.empty()) {
      return std::unexpected("lifecycle package has no shader payloads");
    }
    single_shader.push_back({.kernel_id = plan.kernels.front().id,
                             .binary_symbol = package.binary_symbol,
                             .words = package.shader_words});
    shaders = single_shader;
    header = makeHeader(plan.model,
                        *findResource(plan, plan.inputs.front()),
                        *findResource(plan, plan.outputs.front()));
  } else {
    header = makeGeneralHeader(plan.model, plan);
  }
  if (header != package.header) {
    return std::unexpected(
      "lifecycle header does not match verified public I/O");
  }
  if (auto verified = ncnn_mlir::vulkan::verifyVulkanHostProgram(
        package.host_program, plan, shaders, package.weights);
      !verified) {
    return std::unexpected("invalid lifecycle host program: " +
                           verified.error());
  }
  auto source = fixed_run ? generateVulkanHostImplementation(
                              package.host_program, shaders, package.weights)
                          : generateVulkanLifecycleImplementation(
                              package.host_program, shaders, package.weights);
  if (!source) {
    return std::unexpected(source.error());
  }
  return VulkanLifecycleSource{
    .header = std::move(header),
    .source = std::move(*source),
    .host_program_identity = package.host_program.identity};
}

}  // namespace

std::expected<VulkanLifecycleSource, std::string> makeVulkanLifecycleSource(
  const VulkanModelPackage& package) {
  return makeVulkanSource(package, false);
}

std::expected<VulkanHostSource, std::string> makeVulkanHostSource(
  const VulkanModelPackage& package) {
  auto generated = makeVulkanSource(package, true);
  if (!generated) {
    return std::unexpected(generated.error());
  }
  const std::string codegen_identity =
    hostSourceCodegenIdentity(generated->host_program_identity);
  generated->source.insert(
    0,
    std::format("// Fixed-run codegen: {}\n// Host program: {}\n"
                "// Codegen identity: {}\n",
                kVulkanHostSourceCodegenVersion,
                generated->host_program_identity,
                codegen_identity));
  return VulkanHostSource{
    .header = std::move(generated->header),
    .source = std::move(generated->source),
    .host_program_identity = std::move(generated->host_program_identity),
    .codegen_identity = codegen_identity,
    .codegen_version = std::string(kVulkanHostSourceCodegenVersion)};
}

std::expected<std::string, std::string> updateVulkanManifestDependencies(
  std::string_view manifest_json,
  std::vector<std::string> dependencies,
  std::vector<std::string> sanitizer_dependencies) {
  auto parsed = llvm::json::parse(
    llvm::StringRef(manifest_json.data(), manifest_json.size()));
  if (!parsed) {
    return std::unexpected("cannot parse generated Vulkan manifest JSON: " +
                           llvm::toString(parsed.takeError()));
  }
  llvm::json::Object* object = parsed->getAsObject();
  if (object == nullptr) {
    return std::unexpected("generated Vulkan manifest is not a JSON object");
  }
  const auto kind = object->getString("kind");
  if (!kind || *kind != mlir::ncnn::contract::kVulkanModelManifestKind) {
    return std::unexpected("generated Vulkan manifest has an unexpected kind");
  }
  const auto normalize = [](std::vector<std::string>& list) {
    std::ranges::sort(list);
    const auto unique = std::ranges::unique(list);
    list.erase(unique.begin(), unique.end());
  };
  normalize(dependencies);
  normalize(sanitizer_dependencies);
  // The deployment boundary is independent of the compiler's own LLVM build
  // mode: no model artifact may record an LLVM/Clang shared object, and the
  // sanitizer classification is kept strictly separate from the platform
  // dependency list.
  const auto isCompilerLibrary = [](std::string_view name) {
    // libclang_rt.* archives are compiler-provided runtime instrumentation
    // (asan/ubsan), not compiler code: they belong to the sanitizer
    // classification, never the platform dependency list.
    if (name.starts_with("libclang_rt.")) {
      return false;
    }
    return name.starts_with("libLLVM") || name.starts_with("libclang") ||
           name.starts_with("libMLIR") || name.starts_with("libmlir");
  };
  for (const std::string& dependency : dependencies) {
    if (isCompilerLibrary(dependency)) {
      return std::unexpected(
        std::format("Vulkan model library must not depend on the compiler "
                    "runtime: {}",
                    dependency));
    }
  }
  for (const std::string& dependency : sanitizer_dependencies) {
    if (isCompilerLibrary(dependency)) {
      return std::unexpected(std::format(
        "sanitizer dependency is not a platform library: {}", dependency));
    }
    if (std::ranges::contains(dependencies, dependency)) {
      return std::unexpected(std::format(
        "sanitizer dependency {} is duplicated in the platform dependency list",
        dependency));
    }
  }
  llvm::json::Array dependency_array;
  for (std::string& dependency : dependencies) {
    dependency_array.push_back(std::move(dependency));
  }
  llvm::json::Array sanitizer_array;
  for (std::string& dependency : sanitizer_dependencies) {
    sanitizer_array.push_back(std::move(dependency));
  }
  (*object)["dependencies"] = std::move(dependency_array);
  (*object)["sanitizer_dependencies"] = std::move(sanitizer_array);
  return llvm::formatv("{0:2}\n", std::move(*parsed)).str();
}

}  // namespace ncnn_compile
