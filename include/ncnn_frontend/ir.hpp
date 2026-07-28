#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ncnn_frontend {

class OpId {
 public:
  explicit OpId(std::size_t index) noexcept;
  std::size_t get_index() const noexcept;
  auto operator<=>(const OpId&) const = default;

 private:
  std::size_t index_;
};

class ValueId {
 public:
  explicit ValueId(std::size_t index) noexcept;
  std::size_t get_index() const noexcept;
  auto operator<=>(const ValueId&) const = default;

 private:
  std::size_t index_;
};

enum class ElementType { Float32, Float16, Int8 };
enum class TensorLayout { Scalar, NcnnW, NcnnHW, NcnnCHW, NcnnCDHW, OIHW };

class TensorType {
 public:
  [[nodiscard]] static std::expected<TensorType, std::string> create(
    std::vector<std::int64_t> shape,
    ElementType element_type,
    TensorLayout layout);

  std::span<const std::int64_t> get_shape() const noexcept;
  ElementType get_element_type() const noexcept;
  TensorLayout get_layout() const noexcept;
  std::size_t get_element_count() const noexcept;
  std::size_t get_byte_size() const noexcept;
  bool operator==(const TensorType&) const = default;

 private:
  TensorType(std::vector<std::int64_t> shape,
             ElementType element_type,
             TensorLayout layout,
             std::size_t element_count,
             std::size_t byte_size);

  std::vector<std::int64_t> shape_;
  ElementType element_type_;
  TensorLayout layout_;
  std::size_t element_count_;
  std::size_t byte_size_;
};

class TensorLiteral {
 public:
  [[nodiscard]] static std::expected<TensorLiteral, std::string> create(
    TensorType type, std::vector<std::byte> data);

  const TensorType& get_type() const noexcept;
  std::span<const std::byte> get_data() const noexcept;
  bool operator==(const TensorLiteral&) const = default;

 private:
  TensorLiteral(TensorType type, std::vector<std::byte> data);

  TensorType type_;
  std::vector<std::byte> data_;
};

class GraphInputDef {
 public:
  explicit GraphInputDef(std::size_t input_index) noexcept;
  std::size_t get_input_index() const noexcept;
  bool operator==(const GraphInputDef&) const = default;

 private:
  std::size_t input_index_;
};

class OpResultDef {
 public:
  OpResultDef(OpId op, std::size_t result_index) noexcept;
  OpId get_op() const noexcept;
  std::size_t get_result_index() const noexcept;
  bool operator==(const OpResultDef&) const = default;

 private:
  OpId op_;
  std::size_t result_index_;
};

using ValueDef = std::variant<GraphInputDef, OpResultDef>;

class Use {
 public:
  Use(OpId user, std::size_t operand_index) noexcept;
  OpId get_user() const noexcept;
  std::size_t get_operand_index() const noexcept;
  bool operator==(const Use&) const = default;

 private:
  OpId user_;
  std::size_t operand_index_;
};

class Value {
 public:
  Value(std::string name,
        TensorType type,
        ValueDef definition,
        std::vector<Use> uses);

  std::string_view get_name() const noexcept;
  const TensorType& get_type() const noexcept;
  const ValueDef& get_definition() const noexcept;
  std::span<const Use> get_uses() const noexcept;

 private:
  std::string name_;
  TensorType type_;
  ValueDef definition_;
  std::vector<Use> uses_;
};

class ConstOp {
 public:
  explicit ConstOp(TensorLiteral literal);
  const TensorLiteral& get_literal() const noexcept;

 private:
  TensorLiteral literal_;
};

enum class ConvQuantizationMode { None, Dequantize, Requantize };

class Conv2DOp {
 public:
  Conv2DOp(std::int64_t kernel_height,
           std::int64_t kernel_width,
           std::int64_t stride_height,
           std::int64_t stride_width,
           std::int64_t dilation_height,
           std::int64_t dilation_width,
           std::int64_t pad_top,
           std::int64_t pad_bottom,
           std::int64_t pad_left,
           std::int64_t pad_right,
           bool has_bias,
           std::int64_t int8_scale_term = 0) noexcept;

  std::int64_t get_kernel_height() const noexcept;
  std::int64_t get_kernel_width() const noexcept;
  std::int64_t get_stride_height() const noexcept;
  std::int64_t get_stride_width() const noexcept;
  std::int64_t get_dilation_height() const noexcept;
  std::int64_t get_dilation_width() const noexcept;
  std::int64_t get_pad_top() const noexcept;
  std::int64_t get_pad_bottom() const noexcept;
  std::int64_t get_pad_left() const noexcept;
  std::int64_t get_pad_right() const noexcept;
  bool get_has_bias() const noexcept;
  std::int64_t get_int8_scale_term() const noexcept;
  ConvQuantizationMode get_quantization_mode() const noexcept;

 private:
  std::int64_t kernel_height_;
  std::int64_t kernel_width_;
  std::int64_t stride_height_;
  std::int64_t stride_width_;
  std::int64_t dilation_height_;
  std::int64_t dilation_width_;
  std::int64_t pad_top_;
  std::int64_t pad_bottom_;
  std::int64_t pad_left_;
  std::int64_t pad_right_;
  bool has_bias_;
  std::int64_t int8_scale_term_;
};

class ReluOp {
 public:
  explicit ReluOp(float negative_slope) noexcept;
  float get_negative_slope() const noexcept;

 private:
  float negative_slope_;
};

enum class PoolKind { Maximum, Average };
enum class PoolMode { Regular, Global, Adaptive };

class Pool2DOp {
 public:
  Pool2DOp(PoolKind kind,
           PoolMode mode,
           std::int64_t kernel_height,
           std::int64_t kernel_width,
           std::int64_t stride_height,
           std::int64_t stride_width,
           std::int64_t pad_top,
           std::int64_t pad_bottom,
           std::int64_t pad_left,
           std::int64_t pad_right,
           int pad_mode,
           bool include_pad) noexcept;

  PoolKind get_kind() const noexcept;
  PoolMode get_mode() const noexcept;
  std::int64_t get_kernel_height() const noexcept;
  std::int64_t get_kernel_width() const noexcept;
  std::int64_t get_stride_height() const noexcept;
  std::int64_t get_stride_width() const noexcept;
  std::int64_t get_pad_top() const noexcept;
  std::int64_t get_pad_bottom() const noexcept;
  std::int64_t get_pad_left() const noexcept;
  std::int64_t get_pad_right() const noexcept;
  int get_pad_mode() const noexcept;
  bool get_include_pad() const noexcept;

 private:
  PoolKind kind_;
  PoolMode mode_;
  std::int64_t kernel_height_;
  std::int64_t kernel_width_;
  std::int64_t stride_height_;
  std::int64_t stride_width_;
  std::int64_t pad_top_;
  std::int64_t pad_bottom_;
  std::int64_t pad_left_;
  std::int64_t pad_right_;
  int pad_mode_;
  bool include_pad_;
};

class SplitOp {};

class ConcatOp {
 public:
  explicit ConcatOp(std::int64_t axis) noexcept;
  std::int64_t get_axis() const noexcept;

 private:
  std::int64_t axis_;
};

class DropoutOp {
 public:
  explicit DropoutOp(float scale) noexcept;
  float get_scale() const noexcept;

 private:
  float scale_;
};

class SoftmaxOp {
 public:
  explicit SoftmaxOp(std::int64_t axis) noexcept;
  std::int64_t get_axis() const noexcept;

 private:
  std::int64_t axis_;
};

using OperationAttributes = std::variant<ConstOp,
                                         Conv2DOp,
                                         ReluOp,
                                         Pool2DOp,
                                         SplitOp,
                                         ConcatOp,
                                         DropoutOp,
                                         SoftmaxOp>;

enum class OperationKind {
  Constant,
  Convolution,
  Relu,
  Pooling,
  Split,
  Concat,
  Dropout,
  Softmax,
};

class Operation {
 public:
  Operation(std::string name,
            OperationAttributes attributes,
            std::vector<ValueId> operands,
            std::vector<ValueId> results,
            std::size_t source_layer_index);

  std::string_view get_name() const noexcept;
  OperationKind get_kind() const noexcept;
  const OperationAttributes& get_attributes() const noexcept;
  std::span<const ValueId> get_operands() const noexcept;
  std::span<const ValueId> get_results() const noexcept;
  std::size_t get_source_layer_index() const noexcept;

 private:
  std::string name_;
  OperationAttributes attributes_;
  std::vector<ValueId> operands_;
  std::vector<ValueId> results_;
  std::size_t source_layer_index_;
};

class Graph {
 public:
  Graph(std::vector<Operation> operations,
        std::vector<Value> values,
        std::vector<ValueId> inputs,
        std::vector<ValueId> outputs);

  std::span<const Operation> get_operations() const noexcept;
  std::span<const Value> get_values() const noexcept;
  std::span<const ValueId> get_inputs() const noexcept;
  std::span<const ValueId> get_outputs() const noexcept;
  const Operation& get_operation(OpId id) const;
  const Value& get_value(ValueId id) const;
  std::size_t operation_count_of(OperationKind kind) const noexcept;
  std::string dump() const;

 private:
  std::vector<Operation> operations_;
  std::vector<Value> values_;
  std::vector<ValueId> inputs_;
  std::vector<ValueId> outputs_;
};

}  // namespace ncnn_frontend
