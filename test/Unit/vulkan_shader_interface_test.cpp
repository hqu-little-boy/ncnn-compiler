#include "vulkan_shader_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ncnn-mlir/Support/VulkanShaderInterface.hpp"

namespace ncnn_mlir::vulkan::test {
namespace {

constexpr std::uint16_t kOpEntryPoint = 15;
constexpr std::uint16_t kOpExecutionMode = 16;
constexpr std::uint16_t kOpTypeFloat = 22;
constexpr std::uint16_t kOpTypeArray = 28;
constexpr std::uint16_t kOpConstant = 43;
constexpr std::uint16_t kOpDecorate = 71;
constexpr std::uint16_t kOpAccessChain = 65;
constexpr std::uint16_t kOpLoad = 61;
constexpr std::uint16_t kOpStore = 62;
constexpr std::uint16_t kOpCopyObject = 83;
constexpr std::uint32_t kDescriptorSetDecoration = 34;
constexpr std::uint32_t kBindingDecoration = 33;

class ShaderBuilder final {
 public:
  ShaderBuilder() : words_(), next_id_(1) {}

  std::uint32_t allocate() { return next_id_++; }

  void emit(std::uint16_t opcode,
            std::initializer_list<std::uint32_t> operands) {
    words_.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) |
                     opcode);
    words_.insert(words_.end(), operands.begin(), operands.end());
  }

  void emit(std::uint16_t opcode, std::span<const std::uint32_t> operands) {
    words_.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) |
                     opcode);
    words_.insert(words_.end(), operands.begin(), operands.end());
  }

  void emitString(std::uint16_t opcode,
                  std::span<const std::uint32_t> prefix,
                  std::string_view value,
                  std::span<const std::uint32_t> suffix = {}) {
    std::vector<std::uint32_t> operands(prefix.begin(), prefix.end());
    const std::size_t string_word_start = operands.size();
    operands.resize(string_word_start + ((value.size() + 1 + 3) / 4), 0);
    for (std::size_t index = 0; index < value.size(); ++index) {
      const std::size_t word_index = string_word_start + (index / 4);
      const auto byte_shift = static_cast<unsigned>((index % 4) * 8);
      operands[word_index] |=
        static_cast<std::uint32_t>(static_cast<unsigned char>(value[index]))
        << byte_shift;
    }
    operands.insert(operands.end(), suffix.begin(), suffix.end());
    emit(opcode, operands);
  }

  std::vector<std::uint32_t> finish() {
    std::vector<std::uint32_t> result{0x07230203, 0x00010000, 0, next_id_, 0};
    result.insert(result.end(), words_.begin(), words_.end());
    return result;
  }

 private:
  std::vector<std::uint32_t> words_;
  std::uint32_t next_id_;
};

struct DescriptorFixture final {
  const DescriptorBinding* binding;
  const TensorResource* resource;
  std::uint32_t length_constant;
  std::uint32_t array_type;
  std::uint32_t structure_type;
  std::uint32_t structure_pointer_type;
  std::uint32_t element_pointer_type;
  std::uint32_t variable;
};

std::expected<std::vector<std::uint32_t>, std::string> buildShaderWords(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  bool invert_access) {
  if (auto valid = verifyExecutionPlan(plan); !valid) {
    return std::unexpected("invalid fixture plan: " + valid.error());
  }
  const auto source_kernel =
    std::ranges::find(plan.kernels, kernel.id, &KernelExecution::id);
  if (kernel.local_size[1] != 1 || kernel.local_size[2] != 1) {
    return std::unexpected(
      "fixture supports only one-dimensional [x, 1, 1] workgroups");
  }
  if (source_kernel == plan.kernels.end() ||
      source_kernel->entry_point != kernel.entry_point ||
      source_kernel->local_size != kernel.local_size ||
      source_kernel->bindings.size() != kernel.bindings.size() ||
      source_kernel->push_constants.size() != kernel.push_constants.size()) {
    return std::unexpected("fixture kernel does not belong to the plan");
  }

  ShaderBuilder builder;
  const std::uint32_t void_type = builder.allocate();
  const std::uint32_t function_type = builder.allocate();
  const std::uint32_t float_type = builder.allocate();
  const std::uint32_t uint_type = builder.allocate();
  const std::uint32_t zero_index = builder.allocate();
  const std::uint32_t zero_float = builder.allocate();
  const std::uint32_t glsl_std_450_import = builder.allocate();
  const std::uint32_t bool_type = builder.allocate();
  const std::uint32_t false_constant = builder.allocate();
  const std::uint32_t true_constant = builder.allocate();
  const std::uint32_t function_id = builder.allocate();
  const std::uint32_t label_id = builder.allocate();
  const std::uint32_t element_pointer_type = builder.allocate();

  std::map<std::uint32_t, const TensorResource*> resources;
  for (const TensorResource& resource : plan.resources) {
    resources.emplace(resource.id, &resource);
  }
  std::vector<DescriptorFixture> descriptors;
  descriptors.reserve(kernel.bindings.size());
  for (const DescriptorBinding& binding : kernel.bindings) {
    const auto resource = resources.find(binding.resource_id);
    if (resource == resources.end() || resource->second->byte_size == 0 ||
        resource->second->byte_size % sizeof(float) != 0 ||
        resource->second->byte_size / sizeof(float) >
          std::numeric_limits<std::uint32_t>::max()) {
      return std::unexpected(
        "fixture descriptor resource has an unsupported byte extent");
    }
    descriptors.push_back(
      DescriptorFixture{.binding = &binding,
                        .resource = resource->second,
                        .length_constant = builder.allocate(),
                        .array_type = builder.allocate(),
                        .structure_type = builder.allocate(),
                        .structure_pointer_type = builder.allocate(),
                        .element_pointer_type = element_pointer_type,
                        .variable = builder.allocate()});
  }

  std::uint32_t push_structure = 0;
  std::uint32_t push_pointer = 0;
  std::uint32_t push_element_pointer = 0;
  std::uint32_t push_variable = 0;
  std::uint32_t push_read_member = 0;
  std::vector<std::uint32_t> push_member_constants;
  std::vector<std::uint32_t> push_members;
  if (!kernel.push_constants.empty()) {
    std::uint32_t max_end = 0;
    for (const PushConstantBlock& block : kernel.push_constants) {
      const std::uint32_t end =
        block.offset + static_cast<std::uint32_t>(block.bytes.size());
      max_end = std::max(max_end, end);
    }
    if (max_end % sizeof(std::uint32_t) != 0 || max_end > 128) {
      return std::unexpected("fixture push-constant layout is unsupported");
    }
    const std::size_t member_count = max_end / sizeof(std::uint32_t);
    push_read_member =
      kernel.push_constants.front().offset / sizeof(std::uint32_t);
    push_structure = builder.allocate();
    push_pointer = builder.allocate();
    push_element_pointer = builder.allocate();
    push_variable = builder.allocate();
    push_member_constants.reserve(member_count);
    push_members.assign(member_count, uint_type);
    for (std::size_t member = 0; member < member_count; ++member) {
      push_member_constants.push_back(member == 0 ? zero_index
                                                  : builder.allocate());
    }
  }

  builder.emit(17, {1});  // OpCapability Shader
  const std::array<std::uint32_t, 1> glsl_import_prefix{glsl_std_450_import};
  builder.emitString(11, glsl_import_prefix, "GLSL.std.450");
  builder.emit(14, {0, 1});  // OpMemoryModel Logical GLSL450
  const std::array<std::uint32_t, 2> entry_prefix{5, function_id};
  builder.emitString(kOpEntryPoint, entry_prefix, kernel.entry_point);
  builder.emit(
    16, {function_id, 17, kernel.local_size[0], 1, 1});  // OpExecutionMode
                                                         // LocalSize

  for (const DescriptorFixture& descriptor : descriptors) {
    builder.emit(71, {descriptor.structure_type, 3});         // BufferBlock
    builder.emit(72, {descriptor.structure_type, 0, 35, 0});  // Offset
    builder.emit(71, {descriptor.array_type, 6, 4});          // ArrayStride
    builder.emit(
      71,
      {descriptor.variable, kDescriptorSetDecoration, descriptor.binding->set});
    builder.emit(
      71,
      {descriptor.variable, kBindingDecoration, descriptor.binding->binding});
  }
  if (push_structure != 0) {
    builder.emit(71, {push_structure, 2});  // Block
    for (std::size_t member = 0; member < push_members.size(); ++member) {
      builder.emit(
        72,
        {push_structure,
         static_cast<std::uint32_t>(member),
         35,
         static_cast<std::uint32_t>(member * sizeof(std::uint32_t))});
    }
  }

  builder.emit(19, {void_type});                  // OpTypeVoid
  builder.emit(33, {function_type, void_type});   // OpTypeFunction
  builder.emit(22, {float_type, 32});             // OpTypeFloat
  builder.emit(21, {uint_type, 32, 0});           // OpTypeInt
  builder.emit(20, {bool_type});                  // OpTypeBool
  builder.emit(43, {uint_type, zero_index, 0});   // OpConstant 0
  builder.emit(43, {float_type, zero_float, 0});  // OpConstant 0.0
  builder.emit(42, {bool_type, false_constant});  // OpConstantFalse
  builder.emit(41, {bool_type, true_constant});   // OpConstantTrue
  for (const DescriptorFixture& descriptor : descriptors) {
    builder.emit(43,
                 {uint_type,
                  descriptor.length_constant,
                  static_cast<std::uint32_t>(descriptor.resource->byte_size /
                                             sizeof(float))});
  }
  for (std::size_t member = 1; member < push_member_constants.size();
       ++member) {
    builder.emit(43,
                 {uint_type,
                  push_member_constants[member],
                  static_cast<std::uint32_t>(member)});
  }
  for (const DescriptorFixture& descriptor : descriptors) {
    builder.emit(
      28, {descriptor.array_type, float_type, descriptor.length_constant});
    builder.emit(30, {descriptor.structure_type, descriptor.array_type});
    builder.emit(
      32, {descriptor.structure_pointer_type, 2, descriptor.structure_type});
  }
  builder.emit(32, {element_pointer_type, 2, float_type});
  if (push_structure != 0) {
    std::vector<std::uint32_t> structure_operands{push_structure};
    structure_operands.insert(
      structure_operands.end(), push_members.begin(), push_members.end());
    builder.emit(30, structure_operands);
    builder.emit(32, {push_pointer, 9, push_structure});
    builder.emit(32, {push_element_pointer, 9, uint_type});
  }
  for (const DescriptorFixture& descriptor : descriptors) {
    builder.emit(59,
                 {descriptor.structure_pointer_type,
                  descriptor.variable,
                  2});  // OpVariable Uniform
  }
  if (push_structure != 0) {
    builder.emit(59, {push_pointer, push_variable, 9});
  }

  builder.emit(54, {void_type, function_id, 0, function_type});
  builder.emit(248, {label_id});
  std::optional<std::uint32_t> read_value;
  std::vector<std::uint32_t> write_pointers;
  for (const DescriptorFixture& descriptor : descriptors) {
    const std::uint32_t pointer = builder.allocate();
    builder.emit(kOpAccessChain,
                 {descriptor.element_pointer_type,
                  pointer,
                  descriptor.variable,
                  zero_index,
                  zero_index});
    const bool plan_reads = descriptor.binding->access == ResourceAccess::Read;
    const bool generate_load = invert_access ? !plan_reads : plan_reads;
    if (generate_load) {
      const std::uint32_t value = builder.allocate();
      builder.emit(kOpLoad, {float_type, value, pointer});
      if (!invert_access && plan_reads && !read_value) {
        read_value = value;
      }
    } else if (!invert_access && !plan_reads) {
      write_pointers.push_back(pointer);
    } else {
      builder.emit(kOpStore, {pointer, zero_float});
    }
  }
  if (!invert_access) {
    if (!write_pointers.empty() && !read_value) {
      return std::unexpected(
        "fixture write descriptor has no FP32 input value");
    }
    for (const std::uint32_t pointer : write_pointers) {
      const std::uint32_t maximum = builder.allocate();
      builder.emit(12,
                   {float_type,
                    maximum,
                    glsl_std_450_import,
                    40,
                    *read_value,
                    zero_float});  // GLSL.std.450 FMax
      const std::uint32_t sum = builder.allocate();
      builder.emit(129, {float_type, sum, maximum, zero_float});  // OpFAdd
      const std::uint32_t selected = builder.allocate();
      builder.emit(
        169, {float_type, selected, true_constant, sum, maximum});  // OpSelect
      builder.emit(kOpStore, {pointer, selected});
    }
  }
  if (push_structure != 0) {
    const std::uint32_t member_pointer = builder.allocate();
    builder.emit(kOpAccessChain,
                 {push_element_pointer,
                  member_pointer,
                  push_variable,
                  push_member_constants[push_read_member]});
    const std::uint32_t loaded_constant = builder.allocate();
    builder.emit(kOpLoad, {uint_type, loaded_constant, member_pointer});
  }
  builder.emit(253, {});  // OpReturn
  builder.emit(56, {});   // OpFunctionEnd
  return builder.finish();
}

std::expected<void, std::string> mutateShader(
  std::vector<std::uint32_t>& words, TestVulkanShaderMutation mutation) {
  if (mutation == TestVulkanShaderMutation::None) {
    return {};
  }
  if (mutation == TestVulkanShaderMutation::TruncateFinalInstruction) {
    if (words.size() <= 5) {
      return std::unexpected("fixture has no instruction to truncate");
    }
    words.pop_back();
    return {};
  }
  if (mutation == TestVulkanShaderMutation::WrongVersion) {
    words[1] = 0x00010100;
    return {};
  }
  if (mutation == TestVulkanShaderMutation::AccessesOnlyInDeadFunction) {
    std::size_t entry_function_word = words.size();
    std::size_t execution_mode_word = words.size();
    std::uint32_t void_type = 0;
    std::uint32_t function_type = 0;
    std::size_t cursor = 5;
    while (cursor < words.size()) {
      const std::size_t word_count = words[cursor] >> 16;
      const auto opcode = static_cast<std::uint16_t>(words[cursor] & 0xffffU);
      if (word_count == 0 || word_count > words.size() - cursor) {
        return std::unexpected("fixture instruction stream is invalid");
      }
      if (opcode == kOpEntryPoint) {
        entry_function_word = cursor + 2;
      } else if (opcode == kOpExecutionMode && word_count == 6) {
        execution_mode_word = cursor + 1;
      } else if (opcode == 54 && word_count == 5) {
        void_type = words[cursor + 1];
        function_type = words[cursor + 4];
      }
      cursor += word_count;
    }
    if (entry_function_word == words.size() ||
        execution_mode_word == words.size() || void_type == 0 ||
        function_type == 0 ||
        words[3] > std::numeric_limits<std::uint32_t>::max() - 2) {
      return std::unexpected("fixture entry function was not found");
    }
    const std::uint32_t function_id = words[3];
    const std::uint32_t label_id = function_id + 1;
    words[3] = function_id + 2;
    words[entry_function_word] = function_id;
    words[execution_mode_word] = function_id;
    const std::array<std::uint32_t, 9> dead_function{(5U << 16) | 54U,
                                                     void_type,
                                                     function_id,
                                                     0,
                                                     function_type,
                                                     (2U << 16) | 248U,
                                                     label_id,
                                                     ((1U << 16) | 253U),
                                                     ((1U << 16) | 56U)};
    words.insert(words.end(), dead_function.begin(), dead_function.end());
    return {};
  }
  if (mutation == TestVulkanShaderMutation::AccessesOnlyInDeadBlock ||
      mutation == TestVulkanShaderMutation::AccessesOnlyInConstantFalseBranch) {
    std::size_t body_start = words.size();
    std::size_t return_word = words.size();
    std::size_t cursor = 5;
    bool in_function = false;
    std::uint32_t false_constant = 0;
    while (cursor < words.size()) {
      const std::size_t word_count = words[cursor] >> 16;
      const auto opcode = static_cast<std::uint16_t>(words[cursor] & 0xffffU);
      if (word_count == 0 || word_count > words.size() - cursor) {
        return std::unexpected("fixture instruction stream is invalid");
      }
      if (opcode == 42 && word_count == 3) {
        false_constant = words[cursor + 2];
      }
      if (opcode == 54) {
        in_function = true;
      } else if (opcode == 248 && in_function && body_start == words.size()) {
        body_start = cursor + word_count;
      } else if (opcode == 253 && in_function && body_start != words.size()) {
        return_word = cursor;
        break;
      }
      cursor += word_count;
    }
    if (body_start == words.size() || return_word == words.size() ||
        words[3] > std::numeric_limits<std::uint32_t>::max() - 2 ||
        (mutation ==
           TestVulkanShaderMutation::AccessesOnlyInConstantFalseBranch &&
         false_constant == 0)) {
      return std::unexpected("fixture function body was not found");
    }
    const std::uint32_t dead_label = words[3];
    const std::uint32_t exit_label = dead_label + 1;
    words[3] = exit_label + 1;
    std::vector<std::uint32_t> replacement;
    if (mutation ==
        TestVulkanShaderMutation::AccessesOnlyInConstantFalseBranch) {
      replacement.insert(replacement.end(),
                         {(3U << 16) | 247U,
                          exit_label,
                          0,
                          (4U << 16) | 250U,
                          false_constant,
                          dead_label,
                          exit_label});
    } else {
      replacement.insert(replacement.end(), {(2U << 16) | 249U, exit_label});
    }
    replacement.insert(replacement.end(), {(2U << 16) | 248U, dead_label});
    replacement.insert(
      replacement.end(),
      words.begin() + static_cast<std::ptrdiff_t>(body_start),
      words.begin() + static_cast<std::ptrdiff_t>(return_word));
    if (mutation ==
        TestVulkanShaderMutation::AccessesOnlyInConstantFalseBranch) {
      replacement.insert(replacement.end(), {(2U << 16) | 249U, exit_label});
    } else {
      replacement.push_back(((1U << 16) | 253U));
    }
    replacement.insert(replacement.end(),
                       {(2U << 16) | 248U, exit_label, ((1U << 16) | 253U)});
    words.erase(words.begin() + static_cast<std::ptrdiff_t>(body_start),
                words.begin() + static_cast<std::ptrdiff_t>(return_word + 1));
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(body_start),
                 replacement.begin(),
                 replacement.end());
    return {};
  }
  if (mutation == TestVulkanShaderMutation::MixedLocalStore) {
    std::optional<std::size_t> first_global_variable;
    std::optional<std::uint32_t> uint_type_id;
    std::optional<std::uint32_t> zero_uint_id;
    std::optional<std::uint32_t> false_constant;
    bool in_function = false;
    std::size_t cursor = 5;
    while (cursor < words.size()) {
      const std::size_t word_count = words[cursor] >> 16;
      const auto opcode = static_cast<std::uint16_t>(words[cursor] & 0xffffU);
      if (word_count == 0 || word_count > words.size() - cursor) {
        return std::unexpected("fixture instruction stream is invalid");
      }
      if (opcode == 21 && word_count == 4) {
        uint_type_id = words[cursor + 1];
      } else if (opcode == 42 && word_count == 3) {
        false_constant = words[cursor + 2];
      } else if (opcode == kOpConstant && word_count == 4 && uint_type_id &&
                 words[cursor + 1] == *uint_type_id && words[cursor + 3] == 0) {
        zero_uint_id = words[cursor + 2];
      }
      if (opcode == 54) {
        in_function = true;
      } else if (opcode == 59 && !in_function && !first_global_variable) {
        first_global_variable = cursor;
      } else if (opcode == 56) {
        in_function = false;
      }
      cursor += word_count;
    }
    if (!first_global_variable || !uint_type_id || !zero_uint_id ||
        !false_constant ||
        words[3] > std::numeric_limits<std::uint32_t>::max() - 4) {
      return std::unexpected(
        "fixture scalar-local insertion points are missing");
    }
    const std::uint32_t pointer_type_id = words[3];
    const std::uint32_t local_id = pointer_type_id + 1;
    const std::uint32_t local_block_id = pointer_type_id + 2;
    const std::uint32_t merge_block_id = pointer_type_id + 3;
    words[3] = pointer_type_id + 4;
    const std::array<std::uint32_t, 4> pointer_type_instruction{
      (4U << 16) | 32U, pointer_type_id, 7, *uint_type_id};
    words.insert(
      words.begin() + static_cast<std::ptrdiff_t>(*first_global_variable),
      pointer_type_instruction.begin(),
      pointer_type_instruction.end());

    std::optional<std::size_t> after_entry_label;
    std::optional<std::size_t> return_offset;
    cursor = 5;
    in_function = false;
    while (cursor < words.size()) {
      const std::size_t word_count = words[cursor] >> 16;
      const auto opcode = static_cast<std::uint16_t>(words[cursor] & 0xffffU);
      if (opcode == 54) {
        in_function = true;
      } else if (opcode == 248 && in_function && !after_entry_label) {
        after_entry_label = cursor + word_count;
      } else if (opcode == 253 && in_function && !return_offset) {
        return_offset = cursor;
      } else if (opcode == 56) {
        in_function = false;
      }
      cursor += word_count;
    }
    if (!after_entry_label || !return_offset) {
      return std::unexpected("fixture function block was not found");
    }
    const std::array<std::uint32_t, 4> local_variable_instruction{
      (4U << 16) | 59U, pointer_type_id, local_id, 7};
    words.insert(
      words.begin() + static_cast<std::ptrdiff_t>(*after_entry_label),
      local_variable_instruction.begin(),
      local_variable_instruction.end());
    const std::size_t adjusted_return =
      *return_offset + local_variable_instruction.size();
    const std::array<std::uint32_t, 15> control_flow_and_local_store{
      (3U << 16) | 247U,
      merge_block_id,
      0,
      (4U << 16) | 250U,
      *false_constant,
      local_block_id,
      merge_block_id,
      (2U << 16) | 248U,
      local_block_id,
      (3U << 16) | 62U,
      local_id,
      *zero_uint_id,
      (2U << 16) | 249U,
      merge_block_id,
      (2U << 16) | 248U};
    words.erase(words.begin() + static_cast<std::ptrdiff_t>(adjusted_return));
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(adjusted_return),
                 control_flow_and_local_store.begin(),
                 control_flow_and_local_store.end());
    words.insert(
      words.begin() + static_cast<std::ptrdiff_t>(
                        adjusted_return + control_flow_and_local_store.size()),
      {merge_block_id, ((1U << 16) | 253U)});
    return {};
  }
  if (mutation == TestVulkanShaderMutation::WrongActualAccess) {
    return {};
  }

  std::size_t cursor = 5;
  std::size_t first_access_chain = words.size();
  std::uint32_t access_pointer_id = 0;
  std::size_t copy_offset = 0;
  std::size_t pointer_use_offset = 0;
  std::uint32_t pointer_result_type = 0;
  std::optional<std::uint32_t> first_binding;
  std::optional<std::uint32_t> array_length_id;
  std::optional<std::uint32_t> uint_type_id;
  std::optional<std::uint32_t> integer_zero_id;
  while (cursor < words.size()) {
    const std::uint32_t first_word = words[cursor];
    const std::size_t word_count = first_word >> 16;
    const auto opcode = static_cast<std::uint16_t>(first_word & 0xffffU);
    if (word_count == 0 || word_count > words.size() - cursor) {
      return std::unexpected(
        "fixture instruction stream is unexpectedly invalid");
    }
    if (mutation == TestVulkanShaderMutation::WrongMemoryModel &&
        opcode == 14 && word_count == 3) {
      words[cursor + 1] = 1;
      return {};
    }
    if (mutation == TestVulkanShaderMutation::UnsupportedCapability &&
        opcode == 17 && word_count == 2) {
      const std::array<std::uint32_t, 2> extra_capability{(2U << 16) | 17U, 2};
      words.insert(words.begin() + static_cast<std::ptrdiff_t>(cursor + 2),
                   extra_capability.begin(),
                   extra_capability.end());
      return {};
    }
    if (mutation == TestVulkanShaderMutation::WrongExecutionModel &&
        opcode == kOpEntryPoint && word_count >= 3) {
      words[cursor + 1] = 0;
      return {};
    }
    if (mutation == TestVulkanShaderMutation::WrongEntryPoint &&
        opcode == kOpEntryPoint && word_count >= 5) {
      words[cursor + 3] ^= 1U;
      return {};
    }
    if (mutation == TestVulkanShaderMutation::WrongLocalSize &&
        opcode == kOpExecutionMode && word_count == 6) {
      ++words[cursor + 3];
      return {};
    }
    if (mutation == TestVulkanShaderMutation::WrongStoreValueType &&
        opcode == 21 && word_count == 4) {
      uint_type_id = words[cursor + 1];
    }
    if (mutation == TestVulkanShaderMutation::WrongStoreValueType &&
        opcode == kOpConstant && word_count == 4 && uint_type_id &&
        words[cursor + 1] == *uint_type_id && words[cursor + 3] == 0) {
      integer_zero_id = words[cursor + 2];
    }
    if (mutation == TestVulkanShaderMutation::WrongStoreValueType &&
        opcode == kOpStore && word_count >= 3 && integer_zero_id) {
      words[cursor + 2] = *integer_zero_id;
      return {};
    }
    if (mutation == TestVulkanShaderMutation::WrongFloatWidth &&
        opcode == kOpTypeFloat && word_count == 3) {
      words[cursor + 2] = 16;
      return {};
    }
    if (mutation == TestVulkanShaderMutation::DuplicateBinding &&
        opcode == kOpDecorate && word_count == 4 &&
        words[cursor + 2] == kBindingDecoration) {
      if (first_binding) {
        words[cursor + 3] = *first_binding;
        return {};
      }
      first_binding = words[cursor + 3];
    }
    if (mutation == TestVulkanShaderMutation::WrongBinding &&
        opcode == kOpDecorate && word_count == 4 &&
        words[cursor + 2] == kBindingDecoration) {
      ++words[cursor + 3];
      return {};
    }
    if (mutation == TestVulkanShaderMutation::OutOfBoundsArrayIndex &&
        opcode == kOpTypeArray && word_count == 4) {
      array_length_id = words[cursor + 3];
    }
    if (mutation == TestVulkanShaderMutation::OutOfBoundsArrayIndex &&
        opcode == kOpAccessChain && word_count == 6 && array_length_id) {
      words[cursor + 5] = *array_length_id;
      return {};
    }
    if (mutation == TestVulkanShaderMutation::UseRuntimeArray &&
        opcode == kOpTypeArray && word_count == 4) {
      words[cursor] = (3U << 16) | 29U;
      words.erase(words.begin() + static_cast<std::ptrdiff_t>(cursor + 3));
      return {};
    }
    if (mutation == TestVulkanShaderMutation::WrongArrayExtent &&
        opcode == kOpTypeArray && word_count == 4) {
      const std::uint32_t length_id = words[cursor + 3];
      std::size_t constant_cursor = 5;
      while (constant_cursor < words.size()) {
        const std::uint32_t constant_word = words[constant_cursor];
        const std::size_t constant_count = constant_word >> 16;
        const auto constant_opcode =
          static_cast<std::uint16_t>(constant_word & 0xffffU);
        if (constant_opcode == kOpConstant && constant_count == 4 &&
            words[constant_cursor + 2] == length_id) {
          ++words[constant_cursor + 3];
          return {};
        }
        constant_cursor += constant_count;
      }
      return std::unexpected("fixture array length constant was not found");
    }
    if (mutation == TestVulkanShaderMutation::UnsupportedPointerFlow &&
        opcode == kOpAccessChain && word_count >= 4 &&
        first_access_chain == words.size()) {
      first_access_chain = cursor;
      access_pointer_id = words[cursor + 2];
      pointer_result_type = words[cursor + 1];
      copy_offset = cursor + word_count;
      pointer_use_offset = cursor;
    } else if (mutation == TestVulkanShaderMutation::UnsupportedPointerFlow &&
               first_access_chain != words.size() &&
               ((opcode == kOpLoad && word_count >= 4 &&
                 words[cursor + 3] == access_pointer_id) ||
                (opcode == kOpStore && word_count >= 3 &&
                 words[cursor + 1] == access_pointer_id))) {
      pointer_use_offset = cursor;
      break;
    }
    cursor += word_count;
  }
  if (mutation == TestVulkanShaderMutation::UnsupportedPointerFlow &&
      first_access_chain != words.size() &&
      pointer_use_offset != first_access_chain) {
    const std::uint32_t copy_id = words[3];
    words[3] = copy_id + 1;
    const std::array<std::uint32_t, 4> copy_instruction{
      (4U << 16) | kOpCopyObject,
      pointer_result_type,
      copy_id,
      access_pointer_id};
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(copy_offset),
                 copy_instruction.begin(),
                 copy_instruction.end());
    const std::size_t adjusted_use =
      pointer_use_offset + copy_instruction.size();
    const auto use_opcode =
      static_cast<std::uint16_t>(words[adjusted_use] & 0xffffU);
    const std::size_t pointer_operand = use_opcode == kOpLoad ? 3 : 1;
    words[adjusted_use + pointer_operand] = copy_id;
    return {};
  }
  return std::unexpected("requested shader mutation target was not found");
}

VulkanExecutionPlan makePlan(bool with_push_constants = true) {
  return VulkanExecutionPlan{
    .schema_version = kExecutionPlanSchemaVersion,
    .model = "shader_interface_fixture",
    .build_identity = "shader-interface-fixture-build",
    .host_triple = "x86_64-pc-linux-gnu",
    .device_target = "vulkan1.1-spirv1.0",
    .resources = {TensorResource{.id = 10,
                                 .kind = ResourceKind::Input,
                                 .name = "input",
                                 .shape = {4},
                                 .logical_layout = "linear",
                                 .physical_layout = "linear",
                                 .host_layout = "linear",
                                 .layout_transform = std::nullopt,
                                 .weight_sha256 = "",
                                 .byte_size = 16,
                                 .first_kernel = 0,
                                 .last_kernel = 0},
                  TensorResource{.id = 20,
                                 .kind = ResourceKind::Output,
                                 .name = "output",
                                 .shape = {4},
                                 .logical_layout = "linear",
                                 .physical_layout = "linear",
                                 .host_layout = "linear",
                                 .layout_transform = std::nullopt,
                                 .weight_sha256 = "",
                                 .byte_size = 16,
                                 .first_kernel = 0,
                                 .last_kernel = 0}},
    .kernels = {KernelExecution{
      .id = 7,
      .operation = "copy_fixture",
      .entry_point = "compute_main",
      .local_size = {4, 1, 1},
      .reads = {10},
      .writes = {20},
      .bindings = {{.set = 2,
                    .binding = 5,
                    .resource_id = 10,
                    .access = ResourceAccess::Read},
                   {.set = 2,
                    .binding = 9,
                    .resource_id = 20,
                    .access = ResourceAccess::Write}},
      .push_constants =
        with_push_constants
          ? std::vector<PushConstantBlock>{{.offset = 0, .bytes = {4, 0, 0, 0}}}
          : std::vector<PushConstantBlock>{}}},
    .inputs = {10},
    .outputs = {20},
    .plan_hash = ""};
}

std::expected<VulkanModelShader, std::string> makeTestVulkanShaderImpl(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  TestVulkanShaderMutation mutation,
  std::string binary_symbol);

std::expected<VulkanModelShader, std::string> makeTestVulkanShaderImpl(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  std::string binary_symbol) {
  return makeTestVulkanShaderImpl(
    plan, kernel, TestVulkanShaderMutation::None, std::move(binary_symbol));
}

std::expected<VulkanModelShader, std::string> makeTestVulkanShaderImpl(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  TestVulkanShaderMutation mutation,
  std::string binary_symbol) {
  const bool invert_access =
    mutation == TestVulkanShaderMutation::WrongActualAccess;
  auto words = buildShaderWords(plan, kernel, invert_access);
  if (!words) {
    return std::unexpected(words.error());
  }
  if (auto mutated = mutateShader(*words, mutation); !mutated) {
    return std::unexpected(mutated.error());
  }
  if (binary_symbol.empty()) {
    binary_symbol = std::format("vulkan_kernel_{}_shader", kernel.id);
  }
  return VulkanModelShader{.kernel_id = kernel.id,
                           .binary_symbol = std::move(binary_symbol),
                           .words = std::move(*words)};
}

TEST(VulkanShaderInterfaceTest, ReflectsTypedBindingsEntryAndPushRanges) {
  const VulkanExecutionPlan plan = makePlan();
  const KernelExecution& kernel = plan.kernels.front();
  auto shader = makeTestVulkanShader(plan, kernel, "fixture_symbol");
  ASSERT_TRUE(shader) << shader.error();

  auto reflected = reflectVulkanShaderInterface(plan, kernel, *shader);
  ASSERT_TRUE(reflected) << reflected.error();
  EXPECT_EQ(reflected->entry_point, kernel.entry_point);
  EXPECT_EQ(reflected->local_size, (std::array<std::uint32_t, 3>{4, 1, 1}));
  ASSERT_EQ(reflected->bindings.size(), 2);
  EXPECT_EQ(reflected->bindings[0].set, 2);
  EXPECT_EQ(reflected->bindings[0].binding, 5);
  EXPECT_EQ(reflected->bindings[0].access, ResourceAccess::Read);
  EXPECT_EQ(reflected->bindings[0].byte_size, 16);
  EXPECT_EQ(reflected->bindings[1].set, 2);
  EXPECT_EQ(reflected->bindings[1].binding, 9);
  EXPECT_EQ(reflected->bindings[1].access, ResourceAccess::Write);
  EXPECT_EQ(reflected->bindings[1].byte_size, 16);
  ASSERT_EQ(reflected->push_constants.size(), 1);
  EXPECT_EQ(reflected->push_constants[0].offset, 0);
  EXPECT_EQ(reflected->push_constants[0].byte_size, 4);

  auto mixed_shader = makeTestVulkanShader(
    plan, kernel, TestVulkanShaderMutation::MixedLocalStore);
  ASSERT_TRUE(mixed_shader) << mixed_shader.error();
  auto mixed_reflection =
    reflectVulkanShaderInterface(plan, kernel, *mixed_shader);
  ASSERT_TRUE(mixed_reflection) << mixed_reflection.error();
  EXPECT_EQ(mixed_reflection->bindings.size(), 2);
}

TEST(VulkanShaderInterfaceTest, ReflectsShaderWithoutPushConstantBlock) {
  const VulkanExecutionPlan plan = makePlan(false);
  const KernelExecution& kernel = plan.kernels.front();
  auto shader = makeTestVulkanShader(plan, kernel);
  ASSERT_TRUE(shader) << shader.error();

  auto reflected = reflectVulkanShaderInterface(plan, kernel, *shader);
  ASSERT_TRUE(reflected) << reflected.error();
  EXPECT_TRUE(reflected->push_constants.empty());

  auto runtime_array_shader = makeTestVulkanShader(
    plan, kernel, TestVulkanShaderMutation::UseRuntimeArray);
  ASSERT_TRUE(runtime_array_shader) << runtime_array_shader.error();
  auto runtime_array_reflection =
    reflectVulkanShaderInterface(plan, kernel, *runtime_array_shader);
  ASSERT_TRUE(runtime_array_reflection) << runtime_array_reflection.error();
  EXPECT_EQ(runtime_array_reflection->bindings[0].byte_size, 16);
  EXPECT_EQ(runtime_array_reflection->bindings[1].byte_size, 16);
}

TEST(VulkanShaderInterfaceTest, RejectsTruncatedWordsAndWrongTargetVersion) {
  const VulkanExecutionPlan plan = makePlan();
  const KernelExecution& kernel = plan.kernels.front();
  for (const TestVulkanShaderMutation mutation :
       {TestVulkanShaderMutation::TruncateFinalInstruction,
        TestVulkanShaderMutation::WrongVersion,
        TestVulkanShaderMutation::WrongMemoryModel,
        TestVulkanShaderMutation::UnsupportedCapability}) {
    auto shader = makeTestVulkanShader(plan, kernel, mutation);
    ASSERT_TRUE(shader) << shader.error();
    auto result = reflectVulkanShaderInterface(plan, kernel, *shader);
    EXPECT_FALSE(result) << static_cast<unsigned>(mutation);
  }
}

TEST(VulkanShaderInterfaceTest, RejectsMismatchedEntryAndLocalSize) {
  const VulkanExecutionPlan plan = makePlan();
  const KernelExecution& kernel = plan.kernels.front();
  for (const TestVulkanShaderMutation mutation :
       {TestVulkanShaderMutation::WrongExecutionModel,
        TestVulkanShaderMutation::WrongEntryPoint,
        TestVulkanShaderMutation::WrongLocalSize}) {
    auto shader = makeTestVulkanShader(plan, kernel, mutation);
    ASSERT_TRUE(shader) << shader.error();
    auto result = reflectVulkanShaderInterface(plan, kernel, *shader);
    EXPECT_FALSE(result) << static_cast<unsigned>(mutation);
  }

  VulkanExecutionPlan non_1d_plan = plan;
  non_1d_plan.kernels.front().local_size[1] = 2;
  auto shader = makeTestVulkanShader(plan, kernel);
  ASSERT_TRUE(shader) << shader.error();
  EXPECT_FALSE(reflectVulkanShaderInterface(
    non_1d_plan, non_1d_plan.kernels.front(), *shader));
}

TEST(VulkanShaderInterfaceTest,
     RejectsBindingTypeExtentAndActualAccessMismatch) {
  const VulkanExecutionPlan plan = makePlan();
  const KernelExecution& kernel = plan.kernels.front();
  for (const TestVulkanShaderMutation mutation :
       {TestVulkanShaderMutation::WrongBinding,
        TestVulkanShaderMutation::DuplicateBinding,
        TestVulkanShaderMutation::WrongFloatWidth,
        TestVulkanShaderMutation::WrongStoreValueType,
        TestVulkanShaderMutation::WrongActualAccess,
        TestVulkanShaderMutation::WrongArrayExtent,
        TestVulkanShaderMutation::OutOfBoundsArrayIndex}) {
    auto shader = makeTestVulkanShader(plan, kernel, mutation);
    ASSERT_TRUE(shader) << shader.error();
    auto result = reflectVulkanShaderInterface(plan, kernel, *shader);
    EXPECT_FALSE(result) << static_cast<unsigned>(mutation);
  }
}

TEST(VulkanShaderInterfaceTest, RejectsUnsupportedPointerFlow) {
  const VulkanExecutionPlan plan = makePlan();
  const KernelExecution& kernel = plan.kernels.front();
  for (const TestVulkanShaderMutation mutation :
       {TestVulkanShaderMutation::UnsupportedPointerFlow,
        TestVulkanShaderMutation::AccessesOnlyInDeadFunction,
        TestVulkanShaderMutation::AccessesOnlyInDeadBlock,
        TestVulkanShaderMutation::AccessesOnlyInConstantFalseBranch}) {
    auto shader = makeTestVulkanShader(plan, kernel, mutation);
    ASSERT_TRUE(shader) << shader.error();
    EXPECT_FALSE(reflectVulkanShaderInterface(plan, kernel, *shader))
      << static_cast<unsigned>(mutation);
  }
}

TEST(VulkanShaderInterfaceTest, RequiresPlanCoverageForActualPushReads) {
  const VulkanExecutionPlan shader_plan = makePlan();
  auto shader = makeTestVulkanShader(shader_plan, shader_plan.kernels.front());
  ASSERT_TRUE(shader) << shader.error();

  VulkanExecutionPlan plan_without_payload = makePlan(false);
  auto reflected = reflectVulkanShaderInterface(
    plan_without_payload, plan_without_payload.kernels.front(), *shader);
  ASSERT_FALSE(reflected);
  EXPECT_NE(reflected.error().find("push-constant range"), std::string::npos);
}

TEST(VulkanShaderInterfaceTest, RequiresKernelAndShaderIdsToMatchPlan) {
  const VulkanExecutionPlan plan = makePlan();
  KernelExecution copied_kernel = plan.kernels.front();
  copied_kernel.entry_point = "different_entry";
  auto shader = makeTestVulkanShader(plan, plan.kernels.front());
  ASSERT_TRUE(shader) << shader.error();
  EXPECT_FALSE(reflectVulkanShaderInterface(plan, copied_kernel, *shader));

  VulkanModelShader copied_shader = *shader;
  ++copied_shader.kernel_id;
  EXPECT_FALSE(
    reflectVulkanShaderInterface(plan, plan.kernels.front(), copied_shader));
}

}  // namespace

std::expected<VulkanModelShader, std::string> makeTestVulkanShader(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  std::string binary_symbol) {
  return makeTestVulkanShaderImpl(plan, kernel, std::move(binary_symbol));
}

std::expected<VulkanModelShader, std::string> makeTestVulkanShader(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  TestVulkanShaderMutation mutation,
  std::string binary_symbol) {
  return makeTestVulkanShaderImpl(
    plan, kernel, mutation, std::move(binary_symbol));
}

}  // namespace ncnn_mlir::vulkan::test
