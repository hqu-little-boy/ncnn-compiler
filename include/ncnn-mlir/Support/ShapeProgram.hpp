#pragma once

#include <cstdint>
#include <expected>
#include <limits>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::ncnn {

enum class ShapeOpcode : int64_t { Add = 0, Multiply = 1, Divide = 2 };

struct ShapeInstruction {
  ShapeOpcode opcode;
  int64_t operand;

  bool operator==(const ShapeInstruction&) const = default;
};

struct ShapeConstraint {
  unsigned inputIndex;
  unsigned inputDimension;
  int64_t minimum;
  int64_t multipleOf;
};

class DimensionExpr {
 public:
  DimensionExpr(unsigned inputIndex, unsigned inputDimension)
    : inputIndex_(inputIndex), inputDimension_(inputDimension) {}

  static std::expected<DimensionExpr, std::string> deserialize(
    unsigned inputIndex,
    unsigned inputDimension,
    llvm::ArrayRef<int64_t> program) {
    if (program.size() % 2 != 0) {
      return std::unexpected("shape program must contain opcode/operand pairs");
    }
    DimensionExpr result(inputIndex, inputDimension);
    for (std::size_t index = 0; index < program.size(); index += 2) {
      if (program[index] < static_cast<int64_t>(ShapeOpcode::Add) ||
          program[index] > static_cast<int64_t>(ShapeOpcode::Divide)) {
        return std::unexpected("shape program opcode is invalid");
      }
      auto opcode = static_cast<ShapeOpcode>(program[index]);
      if (opcode == ShapeOpcode::Divide && program[index + 1] <= 0) {
        return std::unexpected("shape program divisor must be positive");
      }
      result.append(opcode, program[index + 1]);
    }
    return result;
  }

  unsigned getInputIndex() const noexcept { return inputIndex_; }
  unsigned getInputDimension() const noexcept { return inputDimension_; }
  llvm::ArrayRef<ShapeInstruction> getInstructions() const noexcept {
    return instructions_;
  }

  void append(ShapeOpcode opcode, int64_t operand) {
    if ((opcode == ShapeOpcode::Add && operand == 0) ||
        ((opcode == ShapeOpcode::Multiply || opcode == ShapeOpcode::Divide) &&
         operand == 1)) {
      return;
    }
    instructions_.push_back({opcode, operand});
    normalize();
  }

  void normalize() {
    llvm::SmallVector<ShapeInstruction> normalized;
    for (ShapeInstruction instruction : instructions_) {
      if (!normalized.empty() &&
          normalized.back().opcode == instruction.opcode &&
          instruction.opcode != ShapeOpcode::Divide) {
        int64_t combined = 0;
        bool overflow =
          instruction.opcode == ShapeOpcode::Add
            ? llvm::AddOverflow(
                normalized.back().operand, instruction.operand, combined)
            : llvm::MulOverflow(
                normalized.back().operand, instruction.operand, combined);
        if (!overflow) {
          normalized.back().operand = combined;
          continue;
        }
      }
      normalized.push_back(instruction);
    }
    instructions_ = std::move(normalized);
  }

  bool isIdentity() const noexcept { return instructions_.empty(); }

  bool operator==(const DimensionExpr& other) const {
    return inputIndex_ == other.inputIndex_ &&
           inputDimension_ == other.inputDimension_ &&
           instructions_ == other.instructions_;
  }

  bool equivalentUnder(llvm::ArrayRef<ShapeConstraint> constraints,
                       const DimensionExpr& other) const {
    if (inputIndex_ != other.inputIndex_ ||
        inputDimension_ != other.inputDimension_) {
      return false;
    }
    return canonical(constraints) == other.canonical(constraints);
  }

  std::expected<int64_t, std::string> evaluateChecked(int64_t extent) const {
    for (ShapeInstruction instruction : instructions_) {
      int64_t result = 0;
      if (instruction.opcode == ShapeOpcode::Add) {
        if (llvm::AddOverflow(extent, instruction.operand, result)) {
          return std::unexpected("shape addition overflows");
        }
      } else if (instruction.opcode == ShapeOpcode::Multiply) {
        if (llvm::MulOverflow(extent, instruction.operand, result)) {
          return std::unexpected("shape multiplication overflows");
        }
      } else {
        if (instruction.operand <= 0 ||
            (extent == std::numeric_limits<int64_t>::min() &&
             instruction.operand == -1)) {
          return std::unexpected("shape division is invalid");
        }
        result = extent / instruction.operand;
      }
      extent = result;
    }
    return extent;
  }

  llvm::SmallVector<int64_t> serialize() const {
    llvm::SmallVector<int64_t> result;
    result.reserve(instructions_.size() * 2);
    for (ShapeInstruction instruction : instructions_) {
      result.push_back(static_cast<int64_t>(instruction.opcode));
      result.push_back(instruction.operand);
    }
    return result;
  }

 private:
  struct CanonicalForm {
    int64_t coefficient;
    int64_t offset;
    int64_t divisor;

    bool operator==(const CanonicalForm&) const = default;
  };

  CanonicalForm canonical(llvm::ArrayRef<ShapeConstraint> constraints) const {
    int64_t coefficient = 1;
    int64_t offset = 0;
    int64_t divisor = 1;
    int64_t modulus = 1;
    for (const ShapeConstraint& constraint : constraints) {
      if (constraint.inputIndex == inputIndex_ &&
          constraint.inputDimension == inputDimension_) {
        modulus = constraint.multipleOf;
        break;
      }
    }
    coefficient = modulus;
    for (ShapeInstruction instruction : instructions_) {
      if (instruction.opcode == ShapeOpcode::Add) {
        int64_t term = 0;
        if (!llvm::MulOverflow(instruction.operand, divisor, term) &&
            !llvm::AddOverflow(offset, term, offset)) {
          continue;
        }
      } else if (instruction.opcode == ShapeOpcode::Multiply) {
        if (!llvm::MulOverflow(coefficient, instruction.operand, coefficient) &&
            !llvm::MulOverflow(offset, instruction.operand, offset)) {
          continue;
        }
      } else if (instruction.opcode == ShapeOpcode::Divide) {
        const int64_t d = instruction.operand;
        int64_t combined = 0;
        if (!llvm::MulOverflow(divisor, d, combined)) {
          divisor = combined;
          if (coefficient % divisor == 0) {
            coefficient /= divisor;
            offset = llvm::divideFloorSigned(offset, divisor);
            divisor = 1;
          }
          continue;
        }
      }
      return {.coefficient = std::numeric_limits<int64_t>::min(),
              .offset = 0,
              .divisor = 0};
    }
    return {.coefficient = coefficient, .offset = offset, .divisor = divisor};
  }

  unsigned inputIndex_;
  unsigned inputDimension_;
  llvm::SmallVector<ShapeInstruction> instructions_;
};

}  // namespace mlir::ncnn
