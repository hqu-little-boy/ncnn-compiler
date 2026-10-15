#include "ncnn-mlir/Support/VulkanShaderInterface.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace ncnn_mlir::vulkan {
namespace {

constexpr std::uint32_t kSpirvMagic = 0x07230203;
constexpr std::uint32_t kSpirvVersion10 = 0x00010000;
constexpr std::uint32_t kExecutionModelGLCompute = 5;
constexpr std::uint32_t kExecutionModeLocalSize = 17;
constexpr std::uint32_t kStorageClassUniform = 2;
constexpr std::uint32_t kStorageClassPushConstant = 9;
constexpr std::uint32_t kStorageClassStorageBuffer = 12;
constexpr std::uint32_t kDecorationBlock = 2;
constexpr std::uint32_t kDecorationBufferBlock = 3;
constexpr std::uint32_t kDecorationArrayStride = 6;
constexpr std::uint32_t kDecorationNonWritable = 24;
constexpr std::uint32_t kDecorationNonReadable = 25;
constexpr std::uint32_t kDecorationBinding = 33;
constexpr std::uint32_t kDecorationDescriptorSet = 34;
constexpr std::uint32_t kDecorationOffset = 35;
constexpr std::uint32_t kMaximumPushConstantBytes = 128;

struct Instruction final {
  std::uint16_t opcode;
  std::span<const std::uint32_t> operands;
};

enum class TypeKind : std::uint8_t {
  Unknown,
  Void,
  Boolean,
  Integer,
  Float,
  Vector,
  Matrix,
  Array,
  RuntimeArray,
  Structure,
  Pointer
};

struct Type final {
  TypeKind kind;
  std::uint32_t width;
  std::uint32_t element_type;
  std::uint32_t length_id;
  std::uint32_t storage_class;
  std::uint32_t pointee_type;
  std::vector<std::uint32_t> members;
};

struct Decorations final {
  std::map<std::uint32_t, std::uint32_t> values;
  std::set<std::uint32_t> flags;
};

struct MemberKey final {
  std::uint32_t type_id;
  std::uint32_t member;
  auto operator<=>(const MemberKey&) const = default;
};

struct Variable final {
  std::uint32_t result_type;
  std::uint32_t storage_class;
  bool in_function;
};

struct PointerOrigin final {
  std::uint32_t variable_id;
  std::uint32_t selected_member;
  bool selected_element;
};

struct ModuleData final {
  std::vector<Instruction> instructions;
  std::map<std::uint32_t, Type> types;
  std::set<std::string> extensions;
  std::set<std::uint32_t> capabilities;
  std::set<std::uint32_t> functions;
  std::set<std::uint32_t> reachable_blocks;
  std::map<std::uint32_t, std::uint64_t> constants;
  std::map<std::uint32_t, Variable> variables;
  std::map<std::uint32_t, Decorations> decorations;
  std::map<MemberKey, Decorations> member_decorations;
  std::map<std::uint32_t, std::uint32_t> value_types;
  bool storage_buffer_store_values_are_f32;
  std::uint32_t entry_function;
  std::string entry_point;
  std::array<std::uint32_t, 3> local_size;
  bool has_local_size;
};

std::expected<std::vector<Instruction>, std::string> parseInstructions(
  std::span<const std::uint32_t> words) {
  if (words.size() < 5) {
    return std::unexpected(
      "SPIR-V module is shorter than its five-word header");
  }
  if (words[0] != kSpirvMagic) {
    return std::unexpected("SPIR-V magic is invalid or not little-endian");
  }
  if (words[1] != kSpirvVersion10) {
    return std::unexpected("Vulkan shader must target SPIR-V 1.0");
  }
  if (words[4] != 0) {
    return std::unexpected("SPIR-V reserved schema word must be zero");
  }
  if (words[3] < 1) {
    return std::unexpected("SPIR-V ID bound must be nonzero");
  }

  std::vector<Instruction> instructions;
  std::size_t cursor = 5;
  while (cursor < words.size()) {
    const std::uint32_t first_word = words[cursor];
    const std::uint32_t word_count = first_word >> 16;
    const std::uint32_t opcode = first_word & 0xffffU;
    if (word_count == 0 || word_count > words.size() - cursor) {
      return std::unexpected(
        std::format("SPIR-V instruction at word {} is truncated", cursor));
    }
    if (opcode == 0) {
      return std::unexpected("SPIR-V instruction opcode zero is invalid");
    }
    instructions.push_back(
      Instruction{.opcode = static_cast<std::uint16_t>(opcode),
                  .operands = words.subspan(cursor + 1, word_count - 1)});
    cursor += word_count;
  }
  return instructions;
}

std::expected<std::string, std::string> decodeLiteralString(
  std::span<const std::uint32_t> words) {
  std::string result;
  for (std::uint32_t word : words) {
    for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
      const char value = static_cast<char>((word >> (byte_index * 8)) & 0xffU);
      if (value == '\0') {
        return result;
      }
      result.push_back(value);
    }
  }
  return std::unexpected("SPIR-V literal string is not null terminated");
}

std::expected<void, std::string> insertType(ModuleData& module,
                                            std::uint32_t id,
                                            Type type) {
  if (id == 0 || !module.types.emplace(id, std::move(type)).second) {
    return std::unexpected("SPIR-V type ID is zero or defined more than once");
  }
  return {};
}

std::expected<void, std::string> insertValueType(ModuleData& module,
                                                 std::uint32_t id,
                                                 std::uint32_t type_id) {
  if (id == 0 || !module.value_types.emplace(id, type_id).second) {
    return std::unexpected(
      "SPIR-V result ID is zero or defined more than once");
  }
  return {};
}

std::expected<ModuleData, std::string> parseModule(
  std::span<const std::uint32_t> words) {
  auto parsed_instructions = parseInstructions(words);
  if (!parsed_instructions) {
    return std::unexpected(parsed_instructions.error());
  }
  ModuleData module{.instructions = std::move(*parsed_instructions),
                    .types = {},
                    .extensions = {},
                    .capabilities = {},
                    .functions = {},
                    .reachable_blocks = {},
                    .constants = {},
                    .variables = {},
                    .decorations = {},
                    .member_decorations = {},
                    .value_types = {},
                    .storage_buffer_store_values_are_f32 = false,
                    .entry_function = 0,
                    .entry_point = {},
                    .local_size = {},
                    .has_local_size = false};
  std::size_t entry_points = 0;
  std::size_t memory_models = 0;
  bool in_function = false;

  for (const Instruction& instruction : module.instructions) {
    const std::span<const std::uint32_t> operands = instruction.operands;
    const auto require_count = [&](std::size_t count) {
      return operands.size() == count;
    };
    switch (instruction.opcode) {
      case 10: {  // OpExtension
        auto extension = decodeLiteralString(operands);
        if (!extension ||
            *extension != "SPV_KHR_storage_buffer_storage_class" ||
            operands.size() != (extension->size() + 1 + 3) / 4 ||
            !module.extensions.insert(std::move(*extension)).second) {
          return std::unexpected("OpExtension is malformed or duplicated");
        }
        break;
      }
      case 14:  // OpMemoryModel
        if (!require_count(2)) {
          return std::unexpected("OpMemoryModel has an invalid word count");
        }
        if (operands[0] != 0 || operands[1] != 1) {
          return std::unexpected(
            "Vulkan SPIR-V 1.0 requires Logical GLSL450 memory model");
        }
        ++memory_models;
        break;
      case 17:  // OpCapability
        if (!require_count(1) ||
            !module.capabilities.insert(operands[0]).second) {
          return std::unexpected("OpCapability is malformed or duplicated");
        }
        break;
      case 15: {  // OpEntryPoint
        if (operands.size() < 3) {
          return std::unexpected("OpEntryPoint is truncated");
        }
        auto name = decodeLiteralString(operands.subspan(2));
        if (!name) {
          return std::unexpected(name.error());
        }
        const std::size_t name_words = (name->size() + 1 + 3) / 4;
        if (operands.size() < 2 + name_words) {
          return std::unexpected("OpEntryPoint interface list is truncated");
        }
        if (++entry_points != 1 || operands[0] != kExecutionModelGLCompute) {
          return std::unexpected(
            "SPIR-V module must have exactly one GLCompute entry point");
        }
        module.entry_function = operands[1];
        module.entry_point = std::move(*name);
        break;
      }
      case 16:  // OpExecutionMode
        if (operands.size() >= 2 && operands[0] == module.entry_function &&
            operands[1] == kExecutionModeLocalSize) {
          if (!require_count(5) || module.has_local_size) {
            return std::unexpected(
              "entry point must have exactly one literal LocalSize mode");
          }
          module.local_size = {operands[2], operands[3], operands[4]};
          module.has_local_size = true;
        }
        break;
      case 19:  // OpTypeVoid
        if (!require_count(1)) {
          return std::unexpected("OpTypeVoid has an invalid word count");
        }
        if (auto result = insertType(module,
                                     operands[0],
                                     Type{.kind = TypeKind::Void,
                                          .width = 0,
                                          .element_type = 0,
                                          .length_id = 0,
                                          .storage_class = 0,
                                          .pointee_type = 0,
                                          .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 20:  // OpTypeBool
        if (!require_count(1)) {
          return std::unexpected("OpTypeBool has an invalid word count");
        }
        if (auto result = insertType(module,
                                     operands[0],
                                     Type{.kind = TypeKind::Boolean,
                                          .width = 1,
                                          .element_type = 0,
                                          .length_id = 0,
                                          .storage_class = 0,
                                          .pointee_type = 0,
                                          .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 21:    // OpTypeInt
      case 22: {  // OpTypeFloat
        if (!require_count(instruction.opcode == 21 ? 3 : 2)) {
          return std::unexpected("scalar type has an invalid word count");
        }
        const TypeKind kind =
          instruction.opcode == 21 ? TypeKind::Integer : TypeKind::Float;
        if (auto result = insertType(module,
                                     operands[0],
                                     Type{.kind = kind,
                                          .width = operands[1],
                                          .element_type = 0,
                                          .length_id = 0,
                                          .storage_class = 0,
                                          .pointee_type = 0,
                                          .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      }
      case 23:    // OpTypeVector
      case 24: {  // OpTypeMatrix
        if (!require_count(3)) {
          return std::unexpected(
            "vector/matrix type has an invalid word count");
        }
        if (auto result = insertType(
              module,
              operands[0],
              Type{.kind = instruction.opcode == 23 ? TypeKind::Vector
                                                    : TypeKind::Matrix,
                   .width = operands[2],
                   .element_type = operands[1],
                   .length_id = 0,
                   .storage_class = 0,
                   .pointee_type = 0,
                   .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      }
      case 28:  // OpTypeArray
        if (!require_count(3)) {
          return std::unexpected("OpTypeArray has an invalid word count");
        }
        if (auto result = insertType(module,
                                     operands[0],
                                     Type{.kind = TypeKind::Array,
                                          .width = 0,
                                          .element_type = operands[1],
                                          .length_id = operands[2],
                                          .storage_class = 0,
                                          .pointee_type = 0,
                                          .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 29:  // OpTypeRuntimeArray
        if (!require_count(2)) {
          return std::unexpected(
            "OpTypeRuntimeArray has an invalid word count");
        }
        if (auto result = insertType(module,
                                     operands[0],
                                     Type{.kind = TypeKind::RuntimeArray,
                                          .width = 0,
                                          .element_type = operands[1],
                                          .length_id = 0,
                                          .storage_class = 0,
                                          .pointee_type = 0,
                                          .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 30: {  // OpTypeStruct
        if (operands.empty()) {
          return std::unexpected("OpTypeStruct has an invalid word count");
        }
        Type type{.kind = TypeKind::Structure,
                  .width = 0,
                  .element_type = 0,
                  .length_id = 0,
                  .storage_class = 0,
                  .pointee_type = 0,
                  .members = {}};
        type.members.assign(operands.begin() + 1, operands.end());
        if (auto result = insertType(module, operands[0], std::move(type));
            !result) {
          return std::unexpected(result.error());
        }
        break;
      }
      case 32:  // OpTypePointer
        if (!require_count(3)) {
          return std::unexpected("OpTypePointer has an invalid word count");
        }
        if (auto result = insertType(module,
                                     operands[0],
                                     Type{.kind = TypeKind::Pointer,
                                          .width = 0,
                                          .element_type = 0,
                                          .length_id = 0,
                                          .storage_class = operands[1],
                                          .pointee_type = operands[2],
                                          .members = {}});
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 41:    // OpConstantTrue
      case 42: {  // OpConstantFalse
        if (!require_count(2)) {
          return std::unexpected(
            "OpConstantTrue/False has an invalid word count");
        }
        const auto type = module.types.find(operands[0]);
        if (type == module.types.end() ||
            type->second.kind != TypeKind::Boolean ||
            !module.constants
               .emplace(operands[1], instruction.opcode == 41 ? 1 : 0)
               .second) {
          return std::unexpected("boolean constant is invalid or duplicated");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      }
      case 43: {  // OpConstant
        if (operands.size() < 3) {
          return std::unexpected("OpConstant is truncated");
        }
        const auto type = module.types.find(operands[0]);
        if (type == module.types.end() ||
            (type->second.kind != TypeKind::Integer &&
             type->second.kind != TypeKind::Float)) {
          return std::unexpected("OpConstant has an unsupported scalar type");
        }
        if (type->second.width > 64 || type->second.width == 0 ||
            operands.size() != 2 + (type->second.width > 32 ? 2 : 1)) {
          return std::unexpected("OpConstant has an invalid scalar width");
        }
        std::uint64_t value = operands[2];
        if (type->second.width > 32) {
          value |= static_cast<std::uint64_t>(operands[3]) << 32;
        }
        if (!module.constants.emplace(operands[1], value).second) {
          return std::unexpected(
            "SPIR-V constant ID is defined more than once");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      }
      case 59: {  // OpVariable
        if (operands.size() < 3 || operands.size() > 4) {
          return std::unexpected("OpVariable has an invalid word count");
        }
        const bool local = in_function;
        if (!module.variables
               .emplace(operands[1],
                        Variable{.result_type = operands[0],
                                 .storage_class = operands[2],
                                 .in_function = local})
               .second) {
          return std::unexpected(
            "SPIR-V variable ID is defined more than once");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      }
      case 71: {  // OpDecorate
        if (operands.size() < 2) {
          return std::unexpected("OpDecorate is truncated");
        }
        Decorations& decorations = module.decorations[operands[0]];
        if (operands[1] == kDecorationBinding ||
            operands[1] == kDecorationDescriptorSet ||
            operands[1] == kDecorationArrayStride) {
          if (operands.size() != 3 ||
              !decorations.values.emplace(operands[1], operands[2]).second) {
            return std::unexpected(
              "SPIR-V decoration is missing or duplicated");
          }
        } else {
          if (operands.size() > 3) {
            return std::unexpected("unsupported SPIR-V decoration operands");
          }
          if (!decorations.flags.insert(operands[1]).second) {
            return std::unexpected("SPIR-V decoration is duplicated");
          }
        }
        break;
      }
      case 72: {  // OpMemberDecorate
        if (operands.size() < 3) {
          return std::unexpected("OpMemberDecorate is truncated");
        }
        Decorations& decorations = module.member_decorations[MemberKey{
          .type_id = operands[0], .member = operands[1]}];
        if (operands[2] == kDecorationOffset) {
          if (operands.size() != 4 ||
              !decorations.values.emplace(operands[2], operands[3]).second) {
            return std::unexpected(
              "SPIR-V member Offset is missing or duplicated");
          }
        } else {
          if (operands.size() > 4 ||
              !decorations.flags.insert(operands[2]).second) {
            return std::unexpected("SPIR-V member decoration is invalid");
          }
        }
        break;
      }
      case 54:  // OpFunction
        if (operands.size() != 4 || in_function) {
          return std::unexpected("OpFunction has an invalid word count");
        }
        in_function = true;
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        if (!module.functions.insert(operands[1]).second) {
          return std::unexpected(
            "SPIR-V function ID is defined more than once");
        }
        break;
      case 56:  // OpFunctionEnd
        if (!operands.empty() || !in_function) {
          return std::unexpected("OpFunctionEnd is invalid");
        }
        in_function = false;
        break;
      case 57:  // OpFunctionCall
        return std::unexpected(
          "function calls are unsupported for interface reflection");
      case 61:  // OpLoad
        if (operands.size() < 3) {
          return std::unexpected("OpLoad is truncated");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 65:  // OpAccessChain
      case 66:  // OpInBoundsAccessChain
        if (operands.size() < 4) {
          return std::unexpected("OpAccessChain is truncated");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 67:  // OpPtrAccessChain
      case 70:  // OpInBoundsPtrAccessChain
        if (operands.size() < 4) {
          return std::unexpected("OpPtrAccessChain is truncated");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 245:  // OpPhi
        if (operands.size() < 4 || operands.size() % 2 != 0) {
          return std::unexpected("OpPhi has an invalid word count");
        }
        if (auto result = insertValueType(module, operands[1], operands[0]);
            !result) {
          return std::unexpected(result.error());
        }
        break;
      case 247:  // OpSelectionMerge
        if (operands.size() != 2) {
          return std::unexpected("OpSelectionMerge has an invalid word count");
        }
        break;
      case 248:  // OpLabel
        if (!require_count(1)) {
          return std::unexpected("OpLabel has an invalid word count");
        }
        break;
      case 62:  // OpStore
        if (operands.size() < 2) {
          return std::unexpected("OpStore is truncated");
        }
        break;
      default:
        break;
    }
  }
  if (in_function) {
    return std::unexpected("SPIR-V function is missing OpFunctionEnd");
  }
  if (memory_models != 1 || entry_points != 1 || module.entry_function == 0 ||
      !module.has_local_size) {
    return std::unexpected(
      "SPIR-V module requires one memory model, one compute entry point, and "
      "LocalSize");
  }
  if (module.capabilities.size() != 1 || !module.capabilities.contains(1)) {
    return std::unexpected(
      "Vulkan SPIR-V 1.0 shader must use the Shader capability only");
  }
  if (module.functions.size() != 1 ||
      !module.functions.contains(module.entry_function)) {
    return std::unexpected(
      "only the selected entry-point function is supported");
  }
  if (module.entry_point.empty()) {
    return std::unexpected("SPIR-V compute entry-point name is empty");
  }
  if (module.instructions.empty()) {
    return std::unexpected("SPIR-V instruction stream is empty");
  }
  return module;
}

std::expected<void, std::string> computeReachableBlocks(ModuleData& module) {
  std::map<std::uint32_t, std::vector<std::uint32_t>> successors;
  bool in_function = false;
  bool block_terminated = true;
  std::uint32_t current_block = 0;
  std::uint32_t entry_block = 0;
  for (const Instruction& instruction : module.instructions) {
    const std::span<const std::uint32_t> operands = instruction.operands;
    if (instruction.opcode == 54) {
      in_function = true;
      continue;
    }
    if (instruction.opcode == 56) {
      if (current_block == 0 || !block_terminated) {
        return std::unexpected("SPIR-V function block has no terminator");
      }
      in_function = false;
      current_block = 0;
      continue;
    }
    if (!in_function) {
      continue;
    }
    if (instruction.opcode == 248) {  // OpLabel
      if (current_block != 0 && !block_terminated) {
        return std::unexpected("SPIR-V basic block is missing a terminator");
      }
      current_block = operands[0];
      if (current_block == 0 ||
          !successors.emplace(current_block, std::vector<std::uint32_t>{})
             .second) {
        return std::unexpected("SPIR-V block label is zero or duplicated");
      }
      if (entry_block == 0) {
        entry_block = current_block;
      }
      block_terminated = false;
      continue;
    }
    if (current_block == 0 || block_terminated) {
      return std::unexpected("SPIR-V instruction is outside a basic block");
    }
    if (instruction.opcode == 249) {  // OpBranch
      if (operands.size() != 1) {
        return std::unexpected("OpBranch has an invalid word count");
      }
      successors[current_block].push_back(operands[0]);
      block_terminated = true;
    } else if (instruction.opcode == 250) {  // OpBranchConditional
      if (operands.size() != 3 && operands.size() != 5) {
        return std::unexpected("OpBranchConditional has an invalid word count");
      }
      const auto condition_type_id = module.value_types.find(operands[0]);
      if (condition_type_id != module.value_types.end()) {
        const auto condition_type =
          module.types.find(condition_type_id->second);
        if (condition_type == module.types.end() ||
            condition_type->second.kind != TypeKind::Boolean) {
          return std::unexpected(
            "OpBranchConditional condition must have boolean type");
        }
      }
      const auto condition = module.constants.find(operands[0]);
      if (condition == module.constants.end()) {
        successors[current_block].push_back(operands[1]);
        successors[current_block].push_back(operands[2]);
      } else {
        successors[current_block].push_back(
          condition->second != 0 ? operands[1] : operands[2]);
      }
      block_terminated = true;
    } else if (instruction.opcode == 251) {  // OpSwitch
      return std::unexpected("OpSwitch control flow is unsupported");
    } else if (instruction.opcode == 252 || instruction.opcode == 253 ||
               instruction.opcode == 254 || instruction.opcode == 255) {
      block_terminated = true;
    }
  }
  if (entry_block == 0) {
    return std::unexpected("SPIR-V entry-point function has no basic block");
  }

  std::vector<std::uint32_t> pending{entry_block};
  while (!pending.empty()) {
    const std::uint32_t block = pending.back();
    pending.pop_back();
    if (!module.reachable_blocks.insert(block).second) {
      continue;
    }
    const auto block_successors = successors.find(block);
    if (block_successors == successors.end()) {
      return std::unexpected("SPIR-V branch targets an unknown block");
    }
    pending.insert(pending.end(),
                   block_successors->second.begin(),
                   block_successors->second.end());
  }
  return {};
}

const Decorations* getDecorations(const ModuleData& module, std::uint32_t id) {
  const auto found = module.decorations.find(id);
  return found == module.decorations.end() ? nullptr : &found->second;
}

const Decorations* getMemberDecorations(const ModuleData& module,
                                        std::uint32_t type_id,
                                        std::uint32_t member) {
  const auto found = module.member_decorations.find(
    MemberKey{.type_id = type_id, .member = member});
  return found == module.member_decorations.end() ? nullptr : &found->second;
}

std::expected<std::uint32_t, std::string> requiredDecoration(
  const Decorations* decorations,
  std::uint32_t decoration,
  std::string_view name) {
  if (decorations == nullptr) {
    return std::unexpected(std::format("{} decoration is missing", name));
  }
  const auto found = decorations->values.find(decoration);
  if (found == decorations->values.end()) {
    return std::unexpected(std::format("{} decoration is missing", name));
  }
  return found->second;
}

std::expected<std::uint64_t, std::string> typeAlignment(
  const ModuleData& module,
  std::uint32_t type_id,
  std::set<std::uint32_t>& active_types) {
  if (!active_types.insert(type_id).second) {
    return std::unexpected(
      "recursive push-constant type layout is unsupported");
  }
  const auto type_iter = module.types.find(type_id);
  if (type_iter == module.types.end()) {
    return std::unexpected("push constant references an unknown SPIR-V type");
  }
  const Type& type = type_iter->second;
  if (type.kind == TypeKind::Integer || type.kind == TypeKind::Float) {
    if (type.width == 0 || type.width % 8 != 0) {
      return std::unexpected("push-constant scalar width is not byte aligned");
    }
    active_types.erase(type_id);
    return type.width / 8;
  }
  if (type.kind == TypeKind::Vector) {
    if (type.width < 2 || type.width > 4) {
      return std::unexpected("push-constant vector width is unsupported");
    }
    auto element_alignment =
      typeAlignment(module, type.element_type, active_types);
    if (!element_alignment) {
      return std::unexpected(element_alignment.error());
    }
    const std::uint64_t alignment =
      *element_alignment * (type.width == 2 ? 2 : 4);
    active_types.erase(type_id);
    return alignment;
  }
  if (type.kind == TypeKind::Array) {
    auto element_alignment =
      typeAlignment(module, type.element_type, active_types);
    if (!element_alignment) {
      return std::unexpected(element_alignment.error());
    }
    const auto stride = requiredDecoration(getDecorations(module, type_id),
                                           kDecorationArrayStride,
                                           "push-constant ArrayStride");
    if (!stride || *stride % *element_alignment != 0) {
      return std::unexpected("push-constant array stride is misaligned");
    }
    active_types.erase(type_id);
    return *element_alignment;
  }
  if (type.kind == TypeKind::Structure) {
    std::uint64_t alignment = 1;
    for (std::size_t member = 0; member < type.members.size(); ++member) {
      const Decorations* decorations = getMemberDecorations(
        module, type_id, static_cast<std::uint32_t>(member));
      auto offset = requiredDecoration(
        decorations, kDecorationOffset, "push-constant member Offset");
      auto member_alignment =
        typeAlignment(module, type.members[member], active_types);
      if (!offset || !member_alignment || *offset % *member_alignment != 0) {
        return std::unexpected("push-constant member offset is misaligned");
      }
      alignment = std::max(alignment, *member_alignment);
    }
    active_types.erase(type_id);
    return alignment;
  }
  active_types.erase(type_id);
  return std::unexpected("push-constant type alignment is unsupported");
}

std::expected<std::uint64_t, std::string> typeAlignment(
  const ModuleData& module, std::uint32_t type_id) {
  std::set<std::uint32_t> active_types;
  return typeAlignment(module, type_id, active_types);
}

std::expected<std::uint64_t, std::string> typeByteSize(
  const ModuleData& module,
  std::uint32_t type_id,
  std::set<std::uint32_t>& active_types) {
  if (!active_types.insert(type_id).second) {
    return std::unexpected(
      "recursive push-constant type layout is unsupported");
  }
  const auto type_iter = module.types.find(type_id);
  if (type_iter == module.types.end()) {
    return std::unexpected("push constant references an unknown SPIR-V type");
  }
  const Type& type = type_iter->second;
  if (type.kind == TypeKind::Integer || type.kind == TypeKind::Float) {
    if (type.width == 0 || type.width % 8 != 0) {
      return std::unexpected("push-constant scalar width is not byte aligned");
    }
    active_types.erase(type_id);
    return type.width / 8;
  }
  if (type.kind == TypeKind::Vector) {
    if (type.width < 2 || type.width > 4) {
      return std::unexpected("push-constant vector width is unsupported");
    }
    auto element_size = typeByteSize(module, type.element_type, active_types);
    if (!element_size) {
      return std::unexpected(element_size.error());
    }
    if (*element_size >
        std::numeric_limits<std::uint64_t>::max() / type.width) {
      return std::unexpected("push-constant vector size overflows");
    }
    active_types.erase(type_id);
    return *element_size * type.width;
  }
  if (type.kind == TypeKind::Matrix) {
    active_types.erase(type_id);
    return std::unexpected(
      "push-constant matrix layout requires unsupported MatrixStride data");
  }
  if (type.kind == TypeKind::Array) {
    const auto length = module.constants.find(type.length_id);
    if (length == module.constants.end() || length->second == 0) {
      return std::unexpected(
        "push-constant array length is not a positive constant");
    }
    auto element_size = typeByteSize(module, type.element_type, active_types);
    if (!element_size) {
      return std::unexpected(element_size.error());
    }
    const Decorations* decorations = getDecorations(module, type_id);
    auto stride = requiredDecoration(
      decorations, kDecorationArrayStride, "push-constant ArrayStride");
    auto element_alignment = typeAlignment(module, type.element_type);
    if (!stride || !element_alignment || *stride < *element_size ||
        *stride % *element_alignment != 0 ||
        length->second > std::numeric_limits<std::uint64_t>::max() / *stride) {
      return std::unexpected("push-constant array layout is invalid");
    }
    active_types.erase(type_id);
    return length->second * *stride;
  }
  if (type.kind == TypeKind::Structure) {
    std::uint64_t extent = 0;
    for (std::size_t member = 0; member < type.members.size(); ++member) {
      const Decorations* decorations = getMemberDecorations(
        module, type_id, static_cast<std::uint32_t>(member));
      auto offset = requiredDecoration(
        decorations, kDecorationOffset, "push-constant member Offset");
      auto member_size =
        typeByteSize(module, type.members[member], active_types);
      auto member_alignment = typeAlignment(module, type.members[member]);
      if (!offset || !member_size || !member_alignment ||
          *offset % *member_alignment != 0 ||
          *offset > std::numeric_limits<std::uint64_t>::max() - *member_size) {
        return std::unexpected("push-constant structure layout is invalid");
      }
      extent =
        std::max(extent, static_cast<std::uint64_t>(*offset) + *member_size);
    }
    auto alignment = typeAlignment(module, type_id);
    if (!alignment ||
        extent > std::numeric_limits<std::uint64_t>::max() - (*alignment - 1)) {
      return std::unexpected("push-constant structure size overflows");
    }
    extent = ((extent + *alignment - 1) / *alignment) * *alignment;
    active_types.erase(type_id);
    return extent;
  }
  active_types.erase(type_id);
  return std::unexpected("push-constant type layout is unsupported");
}

std::expected<std::uint64_t, std::string> typeByteSize(const ModuleData& module,
                                                       std::uint32_t type_id) {
  std::set<std::uint32_t> active_types;
  return typeByteSize(module, type_id, active_types);
}

bool hasFlag(const Decorations* decorations, std::uint32_t flag) {
  return decorations != nullptr && decorations->flags.contains(flag);
}

std::expected<void, std::string> validatePlanKernel(
  const VulkanExecutionPlan& plan, const KernelExecution& kernel) {
  if (auto valid = verifyExecutionPlan(plan); !valid) {
    return std::unexpected("invalid Vulkan execution plan: " + valid.error());
  }
  const auto found =
    std::ranges::find(plan.kernels, kernel.id, &KernelExecution::id);
  if (found == plan.kernels.end()) {
    return std::unexpected("kernel ID is not present in the execution plan");
  }
  if (found->operation != kernel.operation ||
      found->entry_point != kernel.entry_point ||
      found->local_size != kernel.local_size || found->reads != kernel.reads ||
      found->writes != kernel.writes ||
      found->bindings.size() != kernel.bindings.size() ||
      found->push_constants.size() != kernel.push_constants.size()) {
    return std::unexpected(
      "kernel argument does not exactly match its execution-plan entry");
  }
  for (std::size_t index = 0; index < kernel.bindings.size(); ++index) {
    const DescriptorBinding& lhs = found->bindings[index];
    const DescriptorBinding& rhs = kernel.bindings[index];
    if (lhs.set != rhs.set || lhs.binding != rhs.binding ||
        lhs.resource_id != rhs.resource_id || lhs.access != rhs.access) {
      return std::unexpected(
        "kernel descriptor map does not exactly match its execution-plan "
        "entry");
    }
  }
  for (std::size_t index = 0; index < kernel.push_constants.size(); ++index) {
    const PushConstantBlock& lhs = found->push_constants[index];
    const PushConstantBlock& rhs = kernel.push_constants[index];
    if (lhs.offset != rhs.offset || lhs.bytes != rhs.bytes) {
      return std::unexpected(
        "kernel push constants do not exactly match its execution-plan entry");
    }
  }
  return {};
}

struct DescriptorState final {
  VulkanShaderBinding reflected;
  std::uint32_t variable_id;
  std::uint32_t structure_type;
  bool non_readable;
  bool non_writable;
  bool has_load;
  bool has_store;
};

std::expected<void, std::string> validateDescriptorType(
  const ModuleData& module,
  std::uint32_t variable_id,
  const Variable& variable,
  std::uint32_t set,
  std::uint32_t binding,
  const DescriptorBinding& planned_binding,
  const TensorResource& resource,
  DescriptorState& state) {
  const auto pointer_type = module.types.find(variable.result_type);
  if (pointer_type == module.types.end() ||
      pointer_type->second.kind != TypeKind::Pointer ||
      pointer_type->second.storage_class != variable.storage_class) {
    return std::unexpected("descriptor variable has a mismatched pointer type");
  }
  const std::uint32_t expected_storage =
    variable.storage_class == kStorageClassUniform ||
        variable.storage_class == kStorageClassStorageBuffer
      ? variable.storage_class
      : 0;
  if (expected_storage == 0) {
    return std::unexpected(
      "descriptor is not a SPIR-V storage-buffer variable");
  }
  if (variable.storage_class == kStorageClassStorageBuffer &&
      !module.extensions.contains("SPV_KHR_storage_buffer_storage_class")) {
    return std::unexpected(
      "SPIR-V 1.0 StorageBuffer class requires its KHR extension");
  }
  const auto structure = module.types.find(pointer_type->second.pointee_type);
  if (structure == module.types.end() ||
      structure->second.kind != TypeKind::Structure ||
      structure->second.members.size() != 1) {
    return std::unexpected(
      "storage-buffer descriptor must point to a one-member block structure");
  }
  const Decorations* structure_decorations =
    getDecorations(module, pointer_type->second.pointee_type);
  if (variable.storage_class == kStorageClassUniform) {
    if (!hasFlag(structure_decorations, kDecorationBufferBlock)) {
      return std::unexpected(
        "SPIR-V 1.0 Uniform storage buffer requires BufferBlock decoration");
    }
  } else if (!hasFlag(structure_decorations, kDecorationBlock)) {
    return std::unexpected("StorageBuffer block requires Block decoration");
  }
  const Decorations* member_decorations =
    getMemberDecorations(module, pointer_type->second.pointee_type, 0);
  auto member_offset = requiredDecoration(
    member_decorations, kDecorationOffset, "storage-buffer member Offset");
  if (!member_offset || *member_offset != 0) {
    return std::unexpected(
      "storage-buffer data member must start at byte zero");
  }

  const std::uint32_t array_type_id = structure->second.members[0];
  const auto array_type = module.types.find(array_type_id);
  if (array_type == module.types.end() ||
      (array_type->second.kind != TypeKind::Array &&
       array_type->second.kind != TypeKind::RuntimeArray)) {
    return std::unexpected("storage-buffer block member must be an FP32 array");
  }
  const auto float_type = module.types.find(array_type->second.element_type);
  if (float_type == module.types.end() ||
      float_type->second.kind != TypeKind::Float ||
      float_type->second.width != 32) {
    return std::unexpected("storage-buffer element type must be FP32");
  }
  auto stride = requiredDecoration(getDecorations(module, array_type_id),
                                   kDecorationArrayStride,
                                   "storage-buffer ArrayStride");
  if (!stride || *stride != sizeof(float)) {
    return std::unexpected(
      "storage-buffer FP32 ArrayStride must be four bytes");
  }
  std::uint64_t byte_size = resource.byte_size;
  if (array_type->second.kind == TypeKind::Array) {
    const auto length = module.constants.find(array_type->second.length_id);
    if (length == module.constants.end() || length->second == 0 ||
        length->second > std::numeric_limits<std::uint64_t>::max() / *stride) {
      return std::unexpected("storage-buffer array length is invalid");
    }
    const std::uint64_t declared_size = length->second * *stride;
    if (declared_size != resource.byte_size) {
      return std::unexpected(
        std::format("descriptor set {} binding {} FP32 array extent {} differs "
                    "from planned {}",
                    set,
                    binding,
                    declared_size,
                    resource.byte_size));
    }
  } else if (byte_size == 0 || byte_size % *stride != 0) {
    return std::unexpected(
      "runtime storage-buffer byte range is not FP32-array compatible");
  }
  if (planned_binding.set != set || planned_binding.binding != binding ||
      planned_binding.resource_id != resource.id) {
    return std::unexpected("shader descriptor does not match the planned map");
  }

  const Decorations* variable_decorations = getDecorations(module, variable_id);
  const Decorations* array_decorations = getDecorations(module, array_type_id);
  state = DescriptorState{
    .reflected = VulkanShaderBinding{.set = set,
                                     .binding = binding,
                                     .access = planned_binding.access,
                                     .byte_size = byte_size},
    .variable_id = variable_id,
    .structure_type = pointer_type->second.pointee_type,
    .non_readable = hasFlag(variable_decorations, kDecorationNonReadable) ||
                    hasFlag(member_decorations, kDecorationNonReadable) ||
                    hasFlag(structure_decorations, kDecorationNonReadable) ||
                    hasFlag(array_decorations, kDecorationNonReadable),
    .non_writable = hasFlag(variable_decorations, kDecorationNonWritable) ||
                    hasFlag(member_decorations, kDecorationNonWritable) ||
                    hasFlag(structure_decorations, kDecorationNonWritable) ||
                    hasFlag(array_decorations, kDecorationNonWritable),
    .has_load = false,
    .has_store = false};
  if (state.non_readable && state.non_writable) {
    return std::unexpected(
      "storage-buffer descriptor is both NonReadable and NonWritable");
  }
  return {};
}

bool rangeCovered(std::uint32_t begin,
                  std::uint32_t size,
                  std::span<const PushConstantBlock> planned) {
  if (size == 0 || begin > std::numeric_limits<std::uint32_t>::max() - size) {
    return false;
  }
  std::uint32_t cursor = begin;
  const std::uint32_t end = begin + size;
  while (cursor < end) {
    std::uint32_t furthest = cursor;
    for (const PushConstantBlock& block : planned) {
      if (block.bytes.size() >
          std::numeric_limits<std::uint32_t>::max() - block.offset) {
        continue;
      }
      const std::uint32_t block_end =
        block.offset + static_cast<std::uint32_t>(block.bytes.size());
      if (block.offset <= cursor && block_end > furthest) {
        furthest = block_end;
      }
    }
    if (furthest == cursor) {
      return false;
    }
    cursor = std::min(furthest, end);
  }
  return true;
}

std::expected<void, std::string> validatePushConstantRoot(
  const ModuleData& module,
  const Variable& variable,
  std::vector<VulkanShaderPushRange>& ranges,
  std::map<std::uint32_t, std::uint64_t>& member_sizes,
  std::uint32_t& structure_id) {
  const auto pointer_type = module.types.find(variable.result_type);
  if (pointer_type == module.types.end() ||
      pointer_type->second.kind != TypeKind::Pointer ||
      pointer_type->second.storage_class != kStorageClassPushConstant ||
      variable.storage_class != kStorageClassPushConstant) {
    return std::unexpected(
      "push-constant variable has an invalid pointer type");
  }
  structure_id = pointer_type->second.pointee_type;
  const auto structure = module.types.find(structure_id);
  if (structure == module.types.end() ||
      structure->second.kind != TypeKind::Structure ||
      structure->second.members.empty() ||
      !hasFlag(getDecorations(module, structure_id), kDecorationBlock)) {
    return std::unexpected(
      "push-constant variable must point to a nonempty Block structure");
  }
  for (std::size_t member = 0; member < structure->second.members.size();
       ++member) {
    const Decorations* decorations = getMemberDecorations(
      module, structure_id, static_cast<std::uint32_t>(member));
    const std::uint32_t member_type = structure->second.members[member];
    auto offset = requiredDecoration(
      decorations, kDecorationOffset, "push-constant member Offset");
    auto size = typeByteSize(module, member_type);
    auto alignment = typeAlignment(module, member_type);
    if (!offset || !size || !alignment || *size == 0 ||
        *size > kMaximumPushConstantBytes || *offset % *alignment != 0 ||
        *offset % 4 != 0 || *size % 4 != 0 ||
        *offset > kMaximumPushConstantBytes - *size) {
      return std::unexpected(
        "push-constant member layout must be aligned and fit 128 bytes");
    }
    const auto member_index = static_cast<std::uint32_t>(member);
    member_sizes.emplace(member_index, *size);
    ranges.push_back(VulkanShaderPushRange{
      .offset = *offset, .byte_size = static_cast<std::uint32_t>(*size)});
  }
  std::ranges::sort(ranges, {}, &VulkanShaderPushRange::offset);
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    const VulkanShaderPushRange& previous = ranges[index - 1];
    if (previous.offset + previous.byte_size > ranges[index].offset) {
      return std::unexpected("push-constant member byte ranges overlap");
    }
  }
  return {};
}

std::expected<void, std::string> reflectMemoryAccesses(
  ModuleData& module,
  std::map<std::uint32_t, DescriptorState>& descriptors,
  std::uint32_t push_variable,
  std::uint32_t push_structure,
  const std::map<std::uint32_t, std::uint64_t>& push_member_sizes,
  std::vector<std::pair<std::uint32_t, std::uint32_t>>& push_reads) {
  std::map<std::uint32_t, PointerOrigin> origins;
  for (const auto& descriptor : descriptors) {
    const std::uint32_t variable_id = descriptor.first;
    origins.emplace(variable_id,
                    PointerOrigin{.variable_id = variable_id,
                                  .selected_member =
                                    std::numeric_limits<std::uint32_t>::max(),
                                  .selected_element = false});
  }
  if (push_variable != 0) {
    origins.emplace(push_variable,
                    PointerOrigin{.variable_id = push_variable,
                                  .selected_member =
                                    std::numeric_limits<std::uint32_t>::max(),
                                  .selected_element = false});
  }

  bool in_function = false;
  std::uint32_t current_block = 0;
  for (const Instruction& instruction : module.instructions) {
    const std::span<const std::uint32_t> operands = instruction.operands;
    if (instruction.opcode == 54) {
      in_function = true;
      current_block = 0;
      continue;
    }
    if (instruction.opcode == 56) {
      in_function = false;
      current_block = 0;
      continue;
    }
    if (in_function && instruction.opcode == 248) {
      current_block = operands[0];
    }
    if (in_function && !module.reachable_blocks.contains(current_block)) {
      continue;
    }
    if (instruction.opcode == 65 || instruction.opcode == 66) {
      const std::uint32_t result_id = operands[1];
      const std::uint32_t base_id = operands[2];
      const auto base = origins.find(base_id);
      if (base == origins.end()) {
        continue;
      }
      const auto variable = module.variables.find(base->second.variable_id);
      if (variable == module.variables.end()) {
        return std::unexpected("descriptor pointer origin is not a variable");
      }
      const bool is_push =
        variable->second.storage_class == kStorageClassPushConstant;
      const std::size_t expected_indices = is_push ? 1 : 2;
      if (base->second.selected_element ||
          (is_push && base->second.selected_member !=
                        std::numeric_limits<std::uint32_t>::max()) ||
          operands.size() != 3 + expected_indices) {
        return std::unexpected(
          "unsupported descriptor or push-constant access-chain shape");
      }
      const auto member_constant = module.constants.find(operands[3]);
      const auto member_type_id = module.value_types.find(operands[3]);
      const auto member_type = member_type_id == module.value_types.end()
                                 ? module.types.end()
                                 : module.types.find(member_type_id->second);
      if (member_constant == module.constants.end() ||
          member_constant->second > std::numeric_limits<std::uint32_t>::max() ||
          member_type == module.types.end() ||
          member_type->second.kind != TypeKind::Integer ||
          member_type->second.width != 32) {
        return std::unexpected(
          "descriptor block member index must be a constant uint32");
      }
      const auto member = static_cast<std::uint32_t>(member_constant->second);
      if (!is_push && member != 0) {
        return std::unexpected(
          "storage-buffer block member index must be zero");
      }
      if (is_push && member >= push_member_sizes.size()) {
        return std::unexpected("push-constant member index is out of range");
      }
      if (!is_push) {
        const auto element_type = module.value_types.find(operands[4]);
        if (element_type != module.value_types.end()) {
          const auto type = module.types.find(element_type->second);
          if (type == module.types.end() ||
              type->second.kind != TypeKind::Integer ||
              type->second.width != 32) {
            return std::unexpected(
              "storage-buffer array index must be a 32-bit integer");
          }
        }
      }
      const auto result_pointer = module.types.find(operands[0]);
      if (result_pointer == module.types.end() ||
          result_pointer->second.kind != TypeKind::Pointer ||
          result_pointer->second.storage_class !=
            variable->second.storage_class) {
        return std::unexpected("access chain result pointer type is invalid");
      }
      std::uint32_t expected_pointee = 0;
      if (is_push) {
        const auto structure = module.types.find(push_structure);
        if (structure == module.types.end() ||
            member >= structure->second.members.size()) {
          return std::unexpected("push-constant access member is out of range");
        }
        expected_pointee = structure->second.members[member];
      } else {
        const auto descriptor = descriptors.find(base->second.variable_id);
        if (descriptor == descriptors.end()) {
          return std::unexpected("descriptor access has no plan mapping");
        }
        const auto structure =
          module.types.find(descriptor->second.structure_type);
        if (structure == module.types.end() ||
            structure->second.kind != TypeKind::Structure ||
            structure->second.members.size() != 1) {
          return std::unexpected("descriptor block type is invalid");
        }
        const auto array = module.types.find(structure->second.members[0]);
        if (array == module.types.end() ||
            (array->second.kind != TypeKind::Array &&
             array->second.kind != TypeKind::RuntimeArray)) {
          return std::unexpected("descriptor FP32 array type is invalid");
        }
        const auto constant_index = module.constants.find(operands[4]);
        if (constant_index != module.constants.end()) {
          std::uint64_t element_count = 0;
          if (array->second.kind == TypeKind::Array) {
            const auto length = module.constants.find(array->second.length_id);
            if (length == module.constants.end()) {
              return std::unexpected("storage-buffer array length is invalid");
            }
            element_count = length->second;
          } else {
            element_count =
              descriptor->second.reflected.byte_size / sizeof(float);
          }
          if (constant_index->second >= element_count) {
            return std::unexpected(
              "constant storage-buffer array index exceeds descriptor extent");
          }
        }
        expected_pointee = array->second.element_type;
      }
      if (result_pointer->second.pointee_type != expected_pointee) {
        return std::unexpected(
          "access-chain pointer element type differs from its interface "
          "layout");
      }
      origins.emplace(result_id,
                      PointerOrigin{.variable_id = base->second.variable_id,
                                    .selected_member = member,
                                    .selected_element = !is_push});
      continue;
    }
    if (instruction.opcode == 67 || instruction.opcode == 70) {
      const auto base = origins.find(operands[2]);
      if (base != origins.end()) {
        return std::unexpected(
          "unsupported PtrAccessChain flow from an interface variable");
      }
      continue;
    }
    if (instruction.opcode == 61) {  // OpLoad
      const auto pointer = origins.find(operands[2]);
      if (pointer == origins.end()) {
        continue;
      }
      const Variable& variable =
        module.variables.at(pointer->second.variable_id);
      if (variable.storage_class == kStorageClassPushConstant) {
        if (pointer->second.selected_element ||
            pointer->second.variable_id != push_variable ||
            pointer->second.selected_member >= push_member_sizes.size()) {
          return std::unexpected("unsupported push-constant pointer load flow");
        }
        const auto structure = module.types.find(push_structure);
        if (structure == module.types.end() ||
            pointer->second.selected_member >=
              structure->second.members.size() ||
            operands[0] !=
              structure->second.members[pointer->second.selected_member]) {
          return std::unexpected(
            "push-constant load type differs from its declared member");
        }
        const auto offset = requiredDecoration(
          getMemberDecorations(
            module, push_structure, pointer->second.selected_member),
          kDecorationOffset,
          "push-constant member Offset");
        if (!offset || push_member_sizes.at(pointer->second.selected_member) >
                         std::numeric_limits<std::uint32_t>::max()) {
          return std::unexpected("push-constant read range is invalid");
        }
        push_reads.emplace_back(*offset,
                                static_cast<std::uint32_t>(push_member_sizes.at(
                                  pointer->second.selected_member)));
        continue;
      }
      if (!pointer->second.selected_element) {
        return std::unexpected(
          "storage-buffer load bypasses the typed FP32 array element");
      }
      const auto state = descriptors.find(pointer->second.variable_id);
      if (state == descriptors.end()) {
        return std::unexpected("storage-buffer load has no descriptor binding");
      }
      const auto pointer_type = module.types.find(variable.result_type);
      const auto block =
        pointer_type == module.types.end()
          ? module.types.end()
          : module.types.find(pointer_type->second.pointee_type);
      const auto array =
        block == module.types.end() || block->second.kind != TypeKind::Structure
          ? module.types.end()
          : module.types.find(block->second.members.front());
      if (array == module.types.end() || array->second.element_type == 0 ||
          operands[0] != array->second.element_type) {
        return std::unexpected("storage-buffer load result type is not FP32");
      }
      state->second.has_load = true;
      continue;
    }
    if (instruction.opcode == 62) {  // OpStore
      if (origins.contains(operands[1])) {
        return std::unexpected(
          "unsupported interface pointer escape through OpStore value");
      }
      const auto pointer = origins.find(operands[0]);
      if (pointer == origins.end()) {
        continue;
      }
      const Variable& variable =
        module.variables.at(pointer->second.variable_id);
      if (variable.storage_class == kStorageClassPushConstant) {
        return std::unexpected("shader may not store to push constants");
      }
      if (!pointer->second.selected_element) {
        return std::unexpected(
          "storage-buffer store bypasses the typed FP32 array element");
      }
      const auto state = descriptors.find(pointer->second.variable_id);
      if (state == descriptors.end()) {
        return std::unexpected(
          "storage-buffer store has no descriptor binding");
      }
      if (!module.storage_buffer_store_values_are_f32) {
        return std::unexpected(
          "StorageBuffer StoreOp FP32 SSA type proof is unavailable");
      }
      state->second.has_store = true;
      continue;
    }
    if (instruction.opcode == 59 && operands.size() == 4 &&
        origins.contains(operands[3])) {
      return std::unexpected(
        "unsupported interface pointer escape through OpVariable initializer");
    }
    if (instruction.opcode == 254 && !operands.empty() &&
        origins.contains(operands[0])) {
      return std::unexpected(
        "unsupported interface pointer escape through OpReturnValue");
    }
    if (instruction.opcode == 68 && operands.size() >= 3 &&
        origins.contains(operands[2])) {
      return std::unexpected(
        "unsupported OpArrayLength flow from an interface variable");
    }
    if ((instruction.opcode == 117 ||
         (instruction.opcode >= 120 && instruction.opcode <= 124)) &&
        operands.size() >= 3 && origins.contains(operands[2])) {
      return std::unexpected(
        "unsupported pointer conversion flow from an interface variable");
    }
    if ((instruction.opcode == 227 ||
         (instruction.opcode >= 229 && instruction.opcode <= 242) ||
         instruction.opcode == 318) &&
        operands.size() >= 3 && origins.contains(operands[2])) {
      return std::unexpected(
        "unsupported atomic pointer flow from an interface variable");
    }
    if ((instruction.opcode == 228 || instruction.opcode == 319) &&
        !operands.empty() && origins.contains(operands[0])) {
      return std::unexpected(
        "unsupported atomic pointer flow from an interface variable");
    }
    if (instruction.opcode == 80) {  // OpCompositeConstruct
      for (std::size_t index = 2; index < operands.size(); ++index) {
        if (origins.contains(operands[index])) {
          return std::unexpected(
            "unsupported composite pointer flow from an interface variable");
        }
      }
      continue;
    }
    if (instruction.opcode == 82 && operands.size() >= 4 &&
        (origins.contains(operands[2]) || origins.contains(operands[3]))) {
      return std::unexpected(
        "unsupported composite pointer flow from an interface variable");
    }
    if (instruction.opcode == 12) {  // OpExtInst
      for (std::size_t index = 4; index < operands.size(); ++index) {
        if (origins.contains(operands[index])) {
          return std::unexpected(
            "unsupported extended-instruction pointer flow");
        }
      }
      continue;
    }
    if (instruction.opcode == 83) {  // OpCopyObject
      if (operands.size() >= 3 && origins.contains(operands[2])) {
        return std::unexpected(
          "unsupported OpCopyObject pointer flow from an interface variable");
      }
      continue;
    }
    if (instruction.opcode == 245) {  // OpPhi
      for (std::size_t index = 2; index < operands.size(); index += 2) {
        if (origins.contains(operands[index])) {
          return std::unexpected(
            "unsupported OpPhi pointer flow from an interface variable");
        }
      }
      continue;
    }
    if (instruction.opcode == 169) {  // OpSelect
      if (operands.size() >= 5 &&
          (origins.contains(operands[3]) || origins.contains(operands[4]))) {
        return std::unexpected(
          "unsupported OpSelect pointer flow from an interface variable");
      }
      continue;
    }
    if (instruction.opcode == 57) {  // OpFunctionCall
      for (std::size_t index = 3; index < operands.size(); ++index) {
        if (origins.contains(operands[index])) {
          return std::unexpected(
            "unsupported function-call pointer flow from an interface "
            "variable");
        }
      }
      continue;
    }
    if ((instruction.opcode == 63 || instruction.opcode == 64) &&
        operands.size() >= 2 &&
        (origins.contains(operands[0]) || origins.contains(operands[1]))) {
      return std::unexpected(
        "unsupported memory-copy pointer flow from an interface variable");
    }
  }

  return {};
}

std::expected<ModuleData, std::string> deserializeAndParse(
  std::span<const std::uint32_t> words) {
  auto module = parseModule(words);
  if (!module) {
    return std::unexpected(module.error());
  }
  if (auto reachable = computeReachableBlocks(*module); !reachable) {
    return std::unexpected(reachable.error());
  }

  mlir::MLIRContext context;
  context.loadDialect<mlir::spirv::SPIRVDialect>();
  mlir::ScopedDiagnosticHandler handler(
    &context, [](mlir::Diagnostic&) { return mlir::success(); });
  auto deserialized = mlir::spirv::deserialize(
    llvm::ArrayRef<std::uint32_t>(words.data(), words.size()), &context);
  if (!deserialized) {
    return std::unexpected("MLIR 21 rejected the SPIR-V module");
  }
  bool invalid_storage_buffer_store = false;
  std::size_t binary_store_count = 0;
  std::size_t typed_store_count = 0;
  for (const Instruction& instruction : module->instructions) {
    if (instruction.opcode == 62) {
      ++binary_store_count;
    }
  }
  deserialized->walk([&](mlir::spirv::StoreOp store) {
    ++typed_store_count;
    const auto pointer_type =
      mlir::dyn_cast<mlir::spirv::PointerType>(store.getPtr().getType());
    if (!pointer_type) {
      invalid_storage_buffer_store = true;
      return;
    }
    const mlir::spirv::StorageClass storage_class =
      pointer_type.getStorageClass();
    if (storage_class != mlir::spirv::StorageClass::StorageBuffer &&
        storage_class != mlir::spirv::StorageClass::Uniform) {
      return;
    }
    invalid_storage_buffer_store = invalid_storage_buffer_store ||
                                   !pointer_type.getPointeeType().isF32() ||
                                   !store.getValue().getType().isF32();
  });
  if (typed_store_count != binary_store_count) {
    return std::unexpected(
      "MLIR SPIR-V store count differs from the binary instruction stream");
  }
  if (invalid_storage_buffer_store) {
    return std::unexpected(
      "StorageBuffer StoreOp must store FP32 through a pointer to FP32");
  }
  module->storage_buffer_store_values_are_f32 = true;
  return module;
}

}  // namespace

std::expected<VulkanShaderInterface, std::string> reflectVulkanShaderInterface(
  const VulkanExecutionPlan& plan,
  const KernelExecution& kernel,
  const VulkanModelShader& shader) {
  if (auto valid = validatePlanKernel(plan, kernel); !valid) {
    return std::unexpected(valid.error());
  }
  if (shader.kernel_id != kernel.id) {
    return std::unexpected("shader kernel_id does not match the plan kernel");
  }
  if (shader.words.size() < 5) {
    return std::unexpected(
      "SPIR-V module is shorter than its five-word header");
  }
  if (shader.words[0] != kSpirvMagic || shader.words[1] != kSpirvVersion10 ||
      shader.words[4] != 0 || shader.words[3] == 0) {
    return std::unexpected("SPIR-V header is invalid or not Vulkan SPIR-V 1.0");
  }
  auto parsed = deserializeAndParse(shader.words);
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  ModuleData& module = *parsed;
  if (module.entry_point != kernel.entry_point) {
    return std::unexpected(
      std::format("SPIR-V entry point '{}' differs from planned '{}'",
                  module.entry_point,
                  kernel.entry_point));
  }
  if (kernel.local_size[1] != 1 || kernel.local_size[2] != 1) {
    return std::unexpected(
      "planned kernel local_size must be one-dimensional [x, 1, 1]");
  }
  if (module.local_size != kernel.local_size) {
    return std::unexpected("SPIR-V LocalSize differs from planned [x, 1, 1]");
  }

  std::map<std::pair<std::uint32_t, std::uint32_t>, DescriptorBinding>
    planned_bindings;
  std::map<std::uint32_t, const TensorResource*> resources;
  for (const TensorResource& resource : plan.resources) {
    resources.emplace(resource.id, &resource);
  }
  for (const DescriptorBinding& binding : kernel.bindings) {
    planned_bindings.emplace(std::pair{binding.set, binding.binding}, binding);
  }

  std::map<std::uint32_t, DescriptorState> descriptors;
  std::set<std::pair<std::uint32_t, std::uint32_t>> reflected_pairs;
  std::uint32_t push_variable = 0;
  std::uint32_t push_structure = 0;
  std::vector<VulkanShaderPushRange> push_ranges;
  std::map<std::uint32_t, std::uint64_t> push_member_sizes;
  for (const auto& [variable_id, variable] : module.variables) {
    if (variable.in_function) {
      continue;
    }
    const auto pointer_type = module.types.find(variable.result_type);
    if (pointer_type == module.types.end() ||
        pointer_type->second.kind != TypeKind::Pointer) {
      return std::unexpected("global variable result type is not a pointer");
    }
    if (variable.storage_class == kStorageClassPushConstant) {
      if (push_variable != 0) {
        return std::unexpected("shader has multiple push-constant blocks");
      }
      push_variable = variable_id;
      if (auto valid = validatePushConstantRoot(
            module, variable, push_ranges, push_member_sizes, push_structure);
          !valid) {
        return std::unexpected(valid.error());
      }
      continue;
    }

    const Decorations* decorations = getDecorations(module, variable_id);
    const bool has_set = decorations != nullptr &&
                         decorations->values.contains(kDecorationDescriptorSet);
    const bool has_binding = decorations != nullptr &&
                             decorations->values.contains(kDecorationBinding);
    if (!has_set && !has_binding) {
      if (variable.storage_class == kStorageClassUniform ||
          variable.storage_class == kStorageClassStorageBuffer) {
        return std::unexpected(
          "global storage-buffer variable lacks descriptor decorations");
      }
      continue;
    }
    if (!has_set || !has_binding) {
      return std::unexpected(
        "descriptor variable must declare both DescriptorSet and Binding");
    }
    const std::uint32_t set = decorations->values.at(kDecorationDescriptorSet);
    const std::uint32_t binding = decorations->values.at(kDecorationBinding);
    const std::pair pair{set, binding};
    if (!reflected_pairs.insert(pair).second) {
      return std::unexpected("SPIR-V has duplicate set/binding descriptors");
    }
    const auto planned = planned_bindings.find(pair);
    if (planned == planned_bindings.end()) {
      return std::unexpected(std::format(
        "SPIR-V descriptor set {} binding {} is absent from the plan",
        set,
        binding));
    }
    const auto resource = resources.find(planned->second.resource_id);
    if (resource == resources.end()) {
      return std::unexpected(
        "planned descriptor references an unknown resource");
    }
    DescriptorState state{};
    if (auto valid = validateDescriptorType(module,
                                            variable_id,
                                            variable,
                                            set,
                                            binding,
                                            planned->second,
                                            *resource->second,
                                            state);
        !valid) {
      return std::unexpected(valid.error());
    }
    descriptors.emplace(variable_id, std::move(state));
  }
  if (reflected_pairs.size() != planned_bindings.size()) {
    return std::unexpected(
      "SPIR-V descriptor set/binding map does not exactly match the plan");
  }

  std::vector<std::pair<std::uint32_t, std::uint32_t>> push_reads;
  if (auto valid = reflectMemoryAccesses(module,
                                         descriptors,
                                         push_variable,
                                         push_structure,
                                         push_member_sizes,
                                         push_reads);
      !valid) {
    return std::unexpected(valid.error());
  }

  std::vector<VulkanShaderBinding> reflected_bindings;
  reflected_bindings.reserve(descriptors.size());
  for (auto& [variable_id, descriptor] : descriptors) {
    const auto plan_binding = std::ranges::find_if(
      kernel.bindings, [&](const DescriptorBinding& value) {
        return value.set == descriptor.reflected.set &&
               value.binding == descriptor.reflected.binding;
      });
    if (plan_binding == kernel.bindings.end()) {
      return std::unexpected(
        "shader descriptor mapping disappeared during reflection");
    }
    if (descriptor.non_readable && descriptor.has_load) {
      return std::unexpected("NonReadable descriptor is loaded by the shader");
    }
    if (descriptor.non_writable && descriptor.has_store) {
      return std::unexpected("NonWritable descriptor is stored by the shader");
    }
    if (plan_binding->access == ResourceAccess::Read) {
      if (!descriptor.has_load || descriptor.has_store ||
          descriptor.non_readable) {
        return std::unexpected(
          std::format("planned read descriptor set {} binding {} does not have "
                      "read-only shader access",
                      plan_binding->set,
                      plan_binding->binding));
      }
    } else if (!descriptor.has_store || descriptor.has_load ||
               descriptor.non_writable) {
      return std::unexpected(
        std::format("planned write descriptor set {} binding {} does not have "
                    "write-only shader access",
                    plan_binding->set,
                    plan_binding->binding));
    }
    reflected_bindings.push_back(descriptor.reflected);
    (void)variable_id;
  }
  std::ranges::sort(
    reflected_bindings,
    [](const VulkanShaderBinding& lhs, const VulkanShaderBinding& rhs) {
      return std::pair{lhs.set, lhs.binding} < std::pair{rhs.set, rhs.binding};
    });

  for (const auto& [offset, byte_size] : push_reads) {
    if (!rangeCovered(offset, byte_size, kernel.push_constants)) {
      return std::unexpected(
        std::format("shader reads push-constant range [{}, {}) not covered by "
                    "planned payload",
                    offset,
                    static_cast<std::uint64_t>(offset) + byte_size));
    }
  }
  std::ranges::sort(
    push_ranges,
    [](const VulkanShaderPushRange& lhs, const VulkanShaderPushRange& rhs) {
      return lhs.offset < rhs.offset;
    });
  return VulkanShaderInterface{.entry_point = module.entry_point,
                               .local_size = module.local_size,
                               .bindings = std::move(reflected_bindings),
                               .push_constants = std::move(push_ranges)};
}

}  // namespace ncnn_mlir::vulkan
