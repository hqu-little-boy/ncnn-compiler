#include "ncnn_frontend/ir.hpp"

#include <array>
#include <format>
#include <limits>
#include <utility>

namespace ncnn_frontend {
namespace {

std::size_t element_size(ElementType type) noexcept {
  switch (type) {
    case ElementType::Float32:
      return 4;
    case ElementType::Float16:
      return 2;
    case ElementType::Int8:
      return 1;
  }
  return 0;
}

bool valid_element_type(ElementType type) noexcept {
  switch (type) {
    case ElementType::Float32:
    case ElementType::Float16:
    case ElementType::Int8:
      return true;
  }
  return false;
}

std::size_t expected_rank(TensorLayout layout) noexcept {
  switch (layout) {
    case TensorLayout::Scalar:
      return 0;
    case TensorLayout::NcnnW:
      return 1;
    case TensorLayout::NcnnHW:
      return 2;
    case TensorLayout::NcnnCHW:
      return 3;
    case TensorLayout::NcnnCDHW:
    case TensorLayout::OIHW:
      return 4;
  }
  return 0;
}

bool valid_layout(TensorLayout layout) noexcept {
  switch (layout) {
    case TensorLayout::Scalar:
    case TensorLayout::NcnnW:
    case TensorLayout::NcnnHW:
    case TensorLayout::NcnnCHW:
    case TensorLayout::NcnnCDHW:
    case TensorLayout::OIHW:
      return true;
  }
  return false;
}

}  // namespace

OpId::OpId(std::size_t index) noexcept : index_(index) {}

std::size_t OpId::get_index() const noexcept {
  return index_;
}

ValueId::ValueId(std::size_t index) noexcept : index_(index) {}

std::size_t ValueId::get_index() const noexcept {
  return index_;
}

TensorType::TensorType(std::vector<std::int64_t> shape,
                       ElementType element_type,
                       TensorLayout layout,
                       std::size_t element_count,
                       std::size_t byte_size)
  : shape_(std::move(shape)),
    element_type_(element_type),
    layout_(layout),
    element_count_(element_count),
    byte_size_(byte_size) {}

std::expected<TensorType, std::string> TensorType::create(
  std::vector<std::int64_t> shape,
  ElementType element_type,
  TensorLayout layout) {
  if (!valid_element_type(element_type)) {
    return std::unexpected("tensor element type enum is invalid");
  }
  if (!valid_layout(layout)) {
    return std::unexpected("tensor layout enum is invalid");
  }
  if (shape.size() != expected_rank(layout)) {
    return std::unexpected(
      std::format("layout rank {} does not match shape rank {}",
                  expected_rank(layout),
                  shape.size()));
  }

  std::size_t count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension < 0) {
      return std::unexpected("tensor dimensions must be non-negative");
    }
    const auto converted = static_cast<std::uint64_t>(dimension);
    if (converted > std::numeric_limits<std::size_t>::max()) {
      return std::unexpected("tensor dimension does not fit size_t");
    }
    const auto size = static_cast<std::size_t>(converted);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) {
      return std::unexpected("tensor element count overflows size_t");
    }
    count *= size;
  }

  const std::size_t width = element_size(element_type);
  if (count != 0 && width > std::numeric_limits<std::size_t>::max() / count) {
    return std::unexpected("tensor byte size overflows size_t");
  }
  return TensorType(
    std::move(shape), element_type, layout, count, count * width);
}

std::span<const std::int64_t> TensorType::get_shape() const noexcept {
  return shape_;
}

ElementType TensorType::get_element_type() const noexcept {
  return element_type_;
}

TensorLayout TensorType::get_layout() const noexcept {
  return layout_;
}

std::size_t TensorType::get_element_count() const noexcept {
  return element_count_;
}

std::size_t TensorType::get_byte_size() const noexcept {
  return byte_size_;
}

TensorLiteral::TensorLiteral(TensorType type, std::vector<std::byte> data)
  : type_(std::move(type)), data_(std::move(data)) {}

std::expected<TensorLiteral, std::string> TensorLiteral::create(
  TensorType type, std::vector<std::byte> data) {
  if (data.size() != type.get_byte_size()) {
    return std::unexpected(
      std::format("constant payload has {} bytes, expected {}",
                  data.size(),
                  type.get_byte_size()));
  }
  return TensorLiteral(std::move(type), std::move(data));
}

const TensorType& TensorLiteral::get_type() const noexcept {
  return type_;
}

std::span<const std::byte> TensorLiteral::get_data() const noexcept {
  return data_;
}

GraphInputDef::GraphInputDef(std::size_t input_index) noexcept
  : input_index_(input_index) {}

std::size_t GraphInputDef::get_input_index() const noexcept {
  return input_index_;
}

OpResultDef::OpResultDef(OpId op, std::size_t result_index) noexcept
  : op_(op), result_index_(result_index) {}

OpId OpResultDef::get_op() const noexcept {
  return op_;
}

std::size_t OpResultDef::get_result_index() const noexcept {
  return result_index_;
}

Use::Use(OpId user, std::size_t operand_index) noexcept
  : user_(user), operand_index_(operand_index) {}

OpId Use::get_user() const noexcept {
  return user_;
}

std::size_t Use::get_operand_index() const noexcept {
  return operand_index_;
}

Value::Value(std::string name,
             TensorType type,
             ValueDef definition,
             std::vector<Use> uses)
  : name_(std::move(name)),
    type_(std::move(type)),
    definition_(std::move(definition)),
    uses_(std::move(uses)) {}

std::string_view Value::get_name() const noexcept {
  return name_;
}

const TensorType& Value::get_type() const noexcept {
  return type_;
}

const ValueDef& Value::get_definition() const noexcept {
  return definition_;
}

std::span<const Use> Value::get_uses() const noexcept {
  return uses_;
}

ConstOp::ConstOp(TensorLiteral literal) : literal_(std::move(literal)) {}

const TensorLiteral& ConstOp::get_literal() const noexcept {
  return literal_;
}

Conv2DOp::Conv2DOp(std::int64_t kernel_height,
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
                   std::int64_t int8_scale_term) noexcept
  : kernel_height_(kernel_height),
    kernel_width_(kernel_width),
    stride_height_(stride_height),
    stride_width_(stride_width),
    dilation_height_(dilation_height),
    dilation_width_(dilation_width),
    pad_top_(pad_top),
    pad_bottom_(pad_bottom),
    pad_left_(pad_left),
    pad_right_(pad_right),
    has_bias_(has_bias),
    int8_scale_term_(int8_scale_term) {}

std::int64_t Conv2DOp::get_kernel_height() const noexcept {
  return kernel_height_;
}
std::int64_t Conv2DOp::get_kernel_width() const noexcept {
  return kernel_width_;
}
std::int64_t Conv2DOp::get_stride_height() const noexcept {
  return stride_height_;
}
std::int64_t Conv2DOp::get_stride_width() const noexcept {
  return stride_width_;
}
std::int64_t Conv2DOp::get_dilation_height() const noexcept {
  return dilation_height_;
}
std::int64_t Conv2DOp::get_dilation_width() const noexcept {
  return dilation_width_;
}
std::int64_t Conv2DOp::get_pad_top() const noexcept {
  return pad_top_;
}
std::int64_t Conv2DOp::get_pad_bottom() const noexcept {
  return pad_bottom_;
}
std::int64_t Conv2DOp::get_pad_left() const noexcept {
  return pad_left_;
}
std::int64_t Conv2DOp::get_pad_right() const noexcept {
  return pad_right_;
}
bool Conv2DOp::get_has_bias() const noexcept {
  return has_bias_;
}

std::int64_t Conv2DOp::get_int8_scale_term() const noexcept {
  return int8_scale_term_;
}

ConvQuantizationMode Conv2DOp::get_quantization_mode() const noexcept {
  if (int8_scale_term_ == 0) {
    return ConvQuantizationMode::None;
  }
  if (int8_scale_term_ > 100) {
    return ConvQuantizationMode::Requantize;
  }
  return ConvQuantizationMode::Dequantize;
}

ReluOp::ReluOp(float negative_slope) noexcept
  : negative_slope_(negative_slope) {}

float ReluOp::get_negative_slope() const noexcept {
  return negative_slope_;
}

Pool2DOp::Pool2DOp(PoolKind kind,
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
                   bool include_pad) noexcept
  : kind_(kind),
    mode_(mode),
    kernel_height_(kernel_height),
    kernel_width_(kernel_width),
    stride_height_(stride_height),
    stride_width_(stride_width),
    pad_top_(pad_top),
    pad_bottom_(pad_bottom),
    pad_left_(pad_left),
    pad_right_(pad_right),
    pad_mode_(pad_mode),
    include_pad_(include_pad) {}

PoolKind Pool2DOp::get_kind() const noexcept {
  return kind_;
}
PoolMode Pool2DOp::get_mode() const noexcept {
  return mode_;
}
std::int64_t Pool2DOp::get_kernel_height() const noexcept {
  return kernel_height_;
}
std::int64_t Pool2DOp::get_kernel_width() const noexcept {
  return kernel_width_;
}
std::int64_t Pool2DOp::get_stride_height() const noexcept {
  return stride_height_;
}
std::int64_t Pool2DOp::get_stride_width() const noexcept {
  return stride_width_;
}
std::int64_t Pool2DOp::get_pad_top() const noexcept {
  return pad_top_;
}
std::int64_t Pool2DOp::get_pad_bottom() const noexcept {
  return pad_bottom_;
}
std::int64_t Pool2DOp::get_pad_left() const noexcept {
  return pad_left_;
}
std::int64_t Pool2DOp::get_pad_right() const noexcept {
  return pad_right_;
}
int Pool2DOp::get_pad_mode() const noexcept {
  return pad_mode_;
}
bool Pool2DOp::get_include_pad() const noexcept {
  return include_pad_;
}

ConcatOp::ConcatOp(std::int64_t axis) noexcept : axis_(axis) {}

std::int64_t ConcatOp::get_axis() const noexcept {
  return axis_;
}

DropoutOp::DropoutOp(float scale) noexcept : scale_(scale) {}

float DropoutOp::get_scale() const noexcept {
  return scale_;
}

SoftmaxOp::SoftmaxOp(std::int64_t axis) noexcept : axis_(axis) {}

std::int64_t SoftmaxOp::get_axis() const noexcept {
  return axis_;
}

Operation::Operation(std::string name,
                     OperationAttributes attributes,
                     std::vector<ValueId> operands,
                     std::vector<ValueId> results,
                     std::size_t source_layer_index)
  : name_(std::move(name)),
    attributes_(std::move(attributes)),
    operands_(std::move(operands)),
    results_(std::move(results)),
    source_layer_index_(source_layer_index) {}

std::string_view Operation::get_name() const noexcept {
  return name_;
}

OperationKind Operation::get_kind() const noexcept {
  constexpr std::array kinds = {OperationKind::Constant,
                                OperationKind::Convolution,
                                OperationKind::Relu,
                                OperationKind::Pooling,
                                OperationKind::Split,
                                OperationKind::Concat,
                                OperationKind::Dropout,
                                OperationKind::Softmax};
  return kinds[attributes_.index()];
}

const OperationAttributes& Operation::get_attributes() const noexcept {
  return attributes_;
}

std::span<const ValueId> Operation::get_operands() const noexcept {
  return operands_;
}

std::span<const ValueId> Operation::get_results() const noexcept {
  return results_;
}

std::size_t Operation::get_source_layer_index() const noexcept {
  return source_layer_index_;
}

Graph::Graph(std::vector<Operation> operations,
             std::vector<Value> values,
             std::vector<ValueId> inputs,
             std::vector<ValueId> outputs)
  : operations_(std::move(operations)),
    values_(std::move(values)),
    inputs_(std::move(inputs)),
    outputs_(std::move(outputs)) {}

std::span<const Operation> Graph::get_operations() const noexcept {
  return operations_;
}

std::span<const Value> Graph::get_values() const noexcept {
  return values_;
}

std::span<const ValueId> Graph::get_inputs() const noexcept {
  return inputs_;
}

std::span<const ValueId> Graph::get_outputs() const noexcept {
  return outputs_;
}

const Operation& Graph::get_operation(OpId id) const {
  return operations_.at(id.get_index());
}

const Value& Graph::get_value(ValueId id) const {
  return values_.at(id.get_index());
}

std::size_t Graph::operation_count_of(OperationKind kind) const noexcept {
  std::size_t count = 0;
  for (const auto& operation : operations_) {
    if (operation.get_kind() == kind) {
      ++count;
    }
  }
  return count;
}

}  // namespace ncnn_frontend
