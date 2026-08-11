#pragma once

#include <algorithm>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::ncnn {

enum class ShapeOpcode : int64_t {
  Add = 0,
  Multiply = 1,
  Divide = 2,
  SourceDimension = 3
};

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

class ShapeExpr;

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
    std::size_t start = 0;
    if (!program.empty() &&
        program.front() == static_cast<int64_t>(ShapeOpcode::SourceDimension)) {
      if (program[1] < 0) {
        return std::unexpected("shape source dimension must be non-negative");
      }
      inputDimension = static_cast<unsigned>(program[1]);
      start = 2;
    }
    DimensionExpr result(inputIndex, inputDimension);
    for (std::size_t index = start; index < program.size(); index += 2) {
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

  llvm::SmallVector<int64_t> serialize(unsigned outputDimension) const {
    llvm::SmallVector<int64_t> result;
    if (inputDimension_ != outputDimension) {
      result.push_back(static_cast<int64_t>(ShapeOpcode::SourceDimension));
      result.push_back(inputDimension_);
    }
    llvm::append_range(result, serialize());
    return result;
  }

  ShapeExpr toV2() const;

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

// V2 expressions are serialized as a prefix tree. Each node starts with an
// opcode; InputDim and Constant have two and one operands respectively, while
// binary operators recursively consume two nodes.
enum class ShapeExprOpcode : int64_t {
  Constant = 0,
  InputDimension = 1,
  Add = 2,
  Multiply = 3,
  FloorDivide = 4,
  CeilDivide = 5,
  Max = 6
};

class ShapeExpr {
 public:
  ShapeExpr() = default;

  static ShapeExpr constant(int64_t value) {
    return ShapeExpr(
      std::make_shared<Node>(Node{.opcode = ShapeExprOpcode::Constant,
                                  .value = value,
                                  .input = 0,
                                  .lhs = nullptr,
                                  .rhs = nullptr}));
  }

  static ShapeExpr inputDimension(unsigned input, unsigned dimension) {
    return ShapeExpr(
      std::make_shared<Node>(Node{.opcode = ShapeExprOpcode::InputDimension,
                                  .value = static_cast<int64_t>(dimension),
                                  .input = input,
                                  .lhs = nullptr,
                                  .rhs = nullptr}));
  }

  static ShapeExpr binary(ShapeExprOpcode opcode,
                          ShapeExpr lhs,
                          ShapeExpr rhs) {
    return ShapeExpr(std::make_shared<Node>(Node{.opcode = opcode,
                                                 .value = 0,
                                                 .input = 0,
                                                 .lhs = std::move(lhs.node_),
                                                 .rhs = std::move(rhs.node_)}));
  }

  ShapeExprOpcode getOpcode() const noexcept { return node_->opcode; }
  int64_t getValue() const noexcept { return node_->value; }
  unsigned getInput() const noexcept { return node_->input; }
  ShapeExpr getLhs() const noexcept {
    return ShapeExpr(node_ ? node_->lhs : nullptr);
  }
  ShapeExpr getRhs() const noexcept {
    return ShapeExpr(node_ ? node_->rhs : nullptr);
  }
  bool isValid() const noexcept { return static_cast<bool>(node_); }

  std::expected<int64_t, std::string> evaluateChecked(
    llvm::ArrayRef<llvm::ArrayRef<int64_t>> inputShapes) const {
    if (!node_) {
      return std::unexpected("shape expression is empty");
    }
    return evaluate(*node_, inputShapes);
  }

  std::expected<void, std::string> validateInputRanks(
    llvm::ArrayRef<unsigned> inputRanks) const {
    if (!node_) {
      return std::unexpected("shape expression is empty");
    }
    return validateInputRanks(*node_, inputRanks);
  }

  llvm::SmallVector<int64_t> serialize() const {
    llvm::SmallVector<int64_t> result;
    if (node_) {
      serializeNode(*node_, result);
    }
    return result;
  }

  std::expected<llvm::SmallVector<int64_t>, std::string> serializeChecked()
    const {
    if (!node_) {
      return std::unexpected("shape expression is empty");
    }
    llvm::SmallVector<int64_t> result;
    std::size_t nodes = 0;
    if (auto status = serializeNodeChecked(*node_, result, nodes, 0); !status) {
      return std::unexpected(status.error());
    }
    return result;
  }

  static std::expected<ShapeExpr, std::string> deserialize(
    llvm::ArrayRef<int64_t> program) {
    constexpr std::size_t kMaximumNodes = 4096;
    if (program.size() > kMaximumNodes * 3) {
      return std::unexpected("shape V2 expression is too large");
    }
    std::size_t index = 0;
    std::size_t nodes = 0;
    auto node = deserializeNode(program, index, nodes, 0);
    if (!node) {
      return std::unexpected(node.error());
    }
    if (index != program.size()) {
      return std::unexpected("shape V2 expression has trailing operands");
    }
    return ShapeExpr(std::move(*node));
  }

 private:
  struct Node {
    ShapeExprOpcode opcode;
    int64_t value;
    unsigned input;
    std::shared_ptr<Node> lhs;
    std::shared_ptr<Node> rhs;
  };

  explicit ShapeExpr(std::shared_ptr<Node> node) : node_(std::move(node)) {}

  static std::expected<std::shared_ptr<Node>, std::string> deserializeNode(
    llvm::ArrayRef<int64_t> program,
    std::size_t& index,
    std::size_t& nodes,
    std::size_t depth) {
    constexpr std::size_t kMaximumNodes = 4096;
    constexpr std::size_t kMaximumDepth = 256;
    if (++nodes > kMaximumNodes || depth > kMaximumDepth) {
      return std::unexpected("shape V2 expression is too complex");
    }
    if (index >= program.size()) {
      return std::unexpected("shape V2 expression is truncated");
    }
    const int64_t opcodeValue = program[index++];
    if (opcodeValue < 0 || opcodeValue > 6) {
      return std::unexpected("shape V2 expression opcode is invalid");
    }
    const auto opcode = static_cast<ShapeExprOpcode>(opcodeValue);
    if (opcode == ShapeExprOpcode::Constant) {
      if (index >= program.size()) {
        return std::unexpected("shape constant is truncated");
      }
      return std::make_shared<Node>(Node{.opcode = opcode,
                                         .value = program[index++],
                                         .input = 0,
                                         .lhs = nullptr,
                                         .rhs = nullptr});
    }
    if (opcode == ShapeExprOpcode::InputDimension) {
      if (index + 1 >= program.size() || program[index] < 0 ||
          program[index + 1] < 0 ||
          std::cmp_greater(program[index],
                           std::numeric_limits<unsigned>::max()) ||
          std::cmp_greater(program[index + 1],
                           std::numeric_limits<unsigned>::max())) {
        return std::unexpected("shape input dimension is invalid");
      }
      auto input = static_cast<unsigned>(program[index++]);
      auto dimension = static_cast<unsigned>(program[index++]);
      return std::make_shared<Node>(
        Node{.opcode = opcode,
             .value = static_cast<int64_t>(dimension),
             .input = input,
             .lhs = nullptr,
             .rhs = nullptr});
    }
    auto lhs = deserializeNode(program, index, nodes, depth + 1);
    if (!lhs) {
      return std::unexpected(lhs.error());
    }
    auto rhs = deserializeNode(program, index, nodes, depth + 1);
    if (!rhs) {
      return std::unexpected(rhs.error());
    }
    return std::make_shared<Node>(Node{.opcode = opcode,
                                       .value = 0,
                                       .input = 0,
                                       .lhs = std::move(*lhs),
                                       .rhs = std::move(*rhs)});
  }

  static std::expected<int64_t, std::string> evaluate(
    const Node& node, llvm::ArrayRef<llvm::ArrayRef<int64_t>> inputShapes) {
    if (node.opcode == ShapeExprOpcode::Constant) {
      return node.value;
    }
    if (node.opcode == ShapeExprOpcode::InputDimension) {
      if (node.input >= inputShapes.size() || node.value < 0 ||
          static_cast<std::size_t>(node.value) >=
            inputShapes[node.input].size()) {
        return std::unexpected("shape input dimension is out of range");
      }
      return inputShapes[node.input][node.value];
    }
    auto lhs = evaluate(*node.lhs, inputShapes);
    auto rhs = evaluate(*node.rhs, inputShapes);
    if (!lhs || !rhs) {
      return std::unexpected(!lhs ? lhs.error() : rhs.error());
    }
    int64_t result = 0;
    switch (node.opcode) {
      case ShapeExprOpcode::Add:
        if (llvm::AddOverflow(*lhs, *rhs, result)) {
          return std::unexpected("shape addition overflows");
        }
        return result;
      case ShapeExprOpcode::Multiply:
        if (llvm::MulOverflow(*lhs, *rhs, result)) {
          return std::unexpected("shape multiplication overflows");
        }
        return result;
      case ShapeExprOpcode::FloorDivide:
      case ShapeExprOpcode::CeilDivide: {
        if (*rhs == 0 ||
            (*lhs == std::numeric_limits<int64_t>::min() && *rhs == -1)) {
          return std::unexpected("shape division is invalid");
        }
        int64_t quotient = *lhs / *rhs;
        int64_t remainder = *lhs % *rhs;
        if (node.opcode == ShapeExprOpcode::FloorDivide && remainder != 0 &&
            ((*lhs < 0) != (*rhs < 0))) {
          --quotient;
        }
        if (node.opcode == ShapeExprOpcode::CeilDivide && remainder != 0 &&
            ((*lhs < 0) == (*rhs < 0))) {
          ++quotient;
        }
        return quotient;
      }
      case ShapeExprOpcode::Max:
        return std::max(*lhs, *rhs);
      default:
        return std::unexpected("shape expression opcode is invalid");
    }
  }

  static std::expected<void, std::string> validateInputRanks(
    const Node& node, llvm::ArrayRef<unsigned> inputRanks) {
    if (node.opcode == ShapeExprOpcode::Constant) {
      return {};
    }
    if (node.opcode == ShapeExprOpcode::InputDimension) {
      if (node.input >= inputRanks.size() || node.value < 0 ||
          std::cmp_greater_equal(node.value, inputRanks[node.input])) {
        return std::unexpected("shape input dimension is out of range");
      }
      return {};
    }
    if (auto lhs = validateInputRanks(*node.lhs, inputRanks); !lhs) {
      return lhs;
    }
    return validateInputRanks(*node.rhs, inputRanks);
  }

  static void serializeNode(const Node& node, llvm::SmallVector<int64_t>& out) {
    out.push_back(static_cast<int64_t>(node.opcode));
    if (node.opcode == ShapeExprOpcode::Constant) {
      out.push_back(node.value);
    } else if (node.opcode == ShapeExprOpcode::InputDimension) {
      out.push_back(node.input);
      out.push_back(node.value);
    } else {
      serializeNode(*node.lhs, out);
      serializeNode(*node.rhs, out);
    }
  }

  static std::expected<void, std::string> serializeNodeChecked(
    const Node& node,
    llvm::SmallVector<int64_t>& out,
    std::size_t& nodes,
    std::size_t depth) {
    constexpr std::size_t kMaximumNodes = 4096;
    constexpr std::size_t kMaximumDepth = 256;
    if (++nodes > kMaximumNodes || depth > kMaximumDepth) {
      return std::unexpected("shape V2 expression is too complex");
    }
    out.push_back(static_cast<int64_t>(node.opcode));
    if (node.opcode == ShapeExprOpcode::Constant) {
      out.push_back(node.value);
      return {};
    }
    if (node.opcode == ShapeExprOpcode::InputDimension) {
      out.push_back(node.input);
      out.push_back(node.value);
      return {};
    }
    if (auto lhs = serializeNodeChecked(*node.lhs, out, nodes, depth + 1);
        !lhs) {
      return lhs;
    }
    return serializeNodeChecked(*node.rhs, out, nodes, depth + 1);
  }

  std::shared_ptr<Node> node_;
};

inline ShapeExpr DimensionExpr::toV2() const {
  ShapeExpr result = ShapeExpr::inputDimension(inputIndex_, inputDimension_);
  for (ShapeInstruction instruction : instructions_) {
    ShapeExprOpcode opcode;
    switch (instruction.opcode) {
      case ShapeOpcode::Add:
        opcode = ShapeExprOpcode::Add;
        break;
      case ShapeOpcode::Multiply:
        opcode = ShapeExprOpcode::Multiply;
        break;
      case ShapeOpcode::Divide:
        opcode = ShapeExprOpcode::FloorDivide;
        break;
      default:
        llvm_unreachable("source dimension is not a linear instruction");
    }
    result = ShapeExpr::binary(
      opcode, std::move(result), ShapeExpr::constant(instruction.operand));
  }
  return result;
}

}  // namespace mlir::ncnn
