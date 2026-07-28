#include "ncnn_graph/graph.hpp"

#include "ncnn_graph/parser.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

namespace ncnn_graph {

// ───────────────────────── ParamValue ─────────────────────────
ParamValue ParamValue::make_int(std::int64_t value) {
  return ParamValue(Storage(value));
}

ParamValue ParamValue::make_float(float value) {
  return ParamValue(Storage(value));
}

ParamValue ParamValue::make_int_array(std::vector<std::int64_t> value) {
  return ParamValue(Storage(std::move(value)));
}

ParamValue ParamValue::make_float_array(std::vector<float> value) {
  return ParamValue(Storage(std::move(value)));
}

ParamValue ParamValue::make_string(std::string value) {
  return ParamValue(Storage(std::move(value)));
}

ParamValue::Kind ParamValue::get_kind() const noexcept {
  switch (storage_.index()) {
    case 0:
      return Kind::Int;
    case 1:
      return Kind::Float;
    case 2:
      return Kind::IntArray;
    case 3:
      return Kind::FloatArray;
    case 4:
      return Kind::String;
    default:
      return Kind::Int;
  }
}

std::int64_t ParamValue::get_int() const noexcept {
  const auto* value = std::get_if<std::int64_t>(&storage_);
  return value ? *value : 0;
}

float ParamValue::get_float() const noexcept {
  const auto* value = std::get_if<float>(&storage_);
  return value ? *value : 0.0f;
}

std::span<const std::int64_t> ParamValue::get_int_array() const noexcept {
  const auto* value = std::get_if<std::vector<std::int64_t>>(&storage_);
  return value ? std::span<const std::int64_t>(*value)
               : std::span<const std::int64_t>();
}

std::span<const float> ParamValue::get_float_array() const noexcept {
  const auto* value = std::get_if<std::vector<float>>(&storage_);
  return value ? std::span<const float>(*value) : std::span<const float>();
}

std::string_view ParamValue::get_string() const noexcept {
  const auto* value = std::get_if<std::string>(&storage_);
  return value ? std::string_view(*value) : std::string_view();
}

ParamValue::ParamValue(Storage storage) : storage_(std::move(storage)) {}

// ───────────────────────── ParamDict ─────────────────────────
ParamDict::ParamDict() : entries_() {}

std::optional<std::reference_wrapper<const ParamValue>> ParamDict::find_value(
  int id) const noexcept {
  auto iterator = std::ranges::find(
    entries_, id, [](const auto& entry) { return entry.first; });
  if (iterator == entries_.end()) {
    return std::nullopt;
  }
  return std::cref(iterator->second);
}

bool ParamDict::has(int id) const noexcept {
  return find_value(id).has_value();
}

std::span<const std::pair<int, ParamValue>> ParamDict::get_entries()
  const noexcept {
  return entries_;
}

void ParamDict::set_value(int id, ParamValue value) {
  auto iterator = std::ranges::find(
    entries_, id, [](const auto& entry) { return entry.first; });
  if (iterator != entries_.end()) {
    iterator->second = std::move(value);
    return;
  }
  entries_.emplace_back(id, std::move(value));
}

std::int64_t ParamDict::get_int(int id, std::int64_t default_value) const {
  auto value = find_value(id);
  if (!value || value->get().get_kind() != ParamValue::Kind::Int) {
    return default_value;
  }
  return value->get().get_int();
}

float ParamDict::get_float(int id, float default_value) const {
  auto value = find_value(id);
  if (!value || value->get().get_kind() != ParamValue::Kind::Float) {
    return default_value;
  }
  return value->get().get_float();
}

std::optional<std::reference_wrapper<const std::string>> ParamDict::get_string(
  int id) const noexcept {
  auto value = find_value(id);
  if (!value || value->get().get_kind() != ParamValue::Kind::String) {
    return std::nullopt;
  }
  const auto* string_value = std::get_if<std::string>(&value->get().storage_);
  if (!string_value) {
    return std::nullopt;
  }
  return std::cref(*string_value);
}

std::span<const std::int64_t> ParamDict::get_int_array(int id) const {
  auto value = find_value(id);
  return value ? value->get().get_int_array() : std::span<const std::int64_t>();
}

std::span<const float> ParamDict::get_float_array(int id) const {
  auto value = find_value(id);
  return value ? value->get().get_float_array() : std::span<const float>();
}

// ───────────────────────── Tensor ─────────────────────────
Tensor::Tensor() : shape_(), dtype_(DataType::Unknown), data_() {}

std::span<const std::int64_t> Tensor::get_shape() const noexcept {
  return shape_;
}

std::expected<void, std::string> Tensor::set_shape(
  std::vector<std::int64_t> shape) {
  std::size_t count = 1;
  for (std::int64_t dimension : shape) {
    if (dimension < 0) {
      return std::unexpected("tensor shape contains a negative dimension");
    }
    if (!std::in_range<std::size_t>(dimension)) {
      return std::unexpected("tensor shape dimension does not fit size_t");
    }
    auto unsigned_dimension = static_cast<std::size_t>(dimension);
    if (count != 0 &&
        unsigned_dimension > std::numeric_limits<std::size_t>::max() / count) {
      return std::unexpected("tensor element count overflows size_t");
    }
    count *= unsigned_dimension;
  }
  constexpr std::size_t kMaximumElementWidth = sizeof(float);
  if (count > std::numeric_limits<std::size_t>::max() / kMaximumElementWidth) {
    return std::unexpected("tensor byte size overflows size_t");
  }
  shape_ = std::move(shape);
  return {};
}

DataType Tensor::get_dtype() const noexcept {
  return dtype_;
}

void Tensor::set_dtype(DataType dtype) noexcept {
  dtype_ = dtype;
}

std::span<const std::byte> Tensor::get_data() const noexcept {
  return data_;
}

void Tensor::set_data(std::vector<std::byte> data) {
  data_ = std::move(data);
}

std::size_t Tensor::element_count() const noexcept {
  std::size_t count = 1;
  for (std::int64_t dimension : shape_) {
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

std::size_t Tensor::byte_size() const noexcept {
  switch (dtype_) {
    case DataType::Unknown:
      return 0;
    case DataType::Float32:
      return element_count() * sizeof(float);
    case DataType::Float16:
      return element_count() * 2;
    case DataType::Int8:
      return element_count();
  }
  return 0;
}

// ───────────────────────── Blob ─────────────────────────
Blob::Blob(std::string name)
  : name_(std::move(name)), producer_(-1), consumer_(-1) {}

std::string_view Blob::get_name() const noexcept {
  return name_;
}

void Blob::set_name(std::string name) {
  name_ = std::move(name);
}

int Blob::get_producer() const noexcept {
  return producer_;
}

void Blob::set_producer(int producer) noexcept {
  producer_ = producer;
}

int Blob::get_consumer() const noexcept {
  return consumer_;
}

void Blob::set_consumer(int consumer) noexcept {
  consumer_ = consumer;
}

// ───────────────────────── Layer ─────────────────────────
Layer::Layer()
  : type_(), name_(), inputs_(), outputs_(), params_(), weights_() {}

std::string_view Layer::get_type() const noexcept {
  return type_;
}

void Layer::set_type(std::string type) {
  type_ = std::move(type);
}

std::string_view Layer::get_name() const noexcept {
  return name_;
}

void Layer::set_name(std::string name) {
  name_ = std::move(name);
}

std::span<const std::string> Layer::get_inputs() const noexcept {
  return inputs_;
}

void Layer::set_inputs(std::vector<std::string> inputs) {
  inputs_ = std::move(inputs);
}

void Layer::add_input(std::string input) {
  inputs_.push_back(std::move(input));
}

std::span<const std::string> Layer::get_outputs() const noexcept {
  return outputs_;
}

void Layer::set_outputs(std::vector<std::string> outputs) {
  outputs_ = std::move(outputs);
}

void Layer::add_output(std::string output) {
  outputs_.push_back(std::move(output));
}

const ParamDict& Layer::get_params() const noexcept {
  return params_;
}

void Layer::set_params(ParamDict params) {
  params_ = std::move(params);
}

std::span<const Tensor> Layer::get_weights() const noexcept {
  return weights_;
}

void Layer::set_weights(std::vector<Tensor> weights) {
  weights_ = std::move(weights);
}

void Layer::add_weight(Tensor weight) {
  weights_.push_back(std::move(weight));
}

std::int64_t Layer::get_param_int(int id, std::int64_t default_value) const {
  return params_.get_int(id, default_value);
}

float Layer::get_param_float(int id, float default_value) const {
  return params_.get_float(id, default_value);
}

// ───────────────────────── Graph ─────────────────────────
Graph::Graph()
  : layers_(),
    blobs_(),
    input_blob_names_(),
    output_blob_names_(),
    weights_loaded_(false) {}

std::span<const Layer> Graph::get_layers() const noexcept {
  return layers_;
}

void Graph::set_layers(std::vector<Layer> layers) {
  layers_ = std::move(layers);
}

void Graph::add_layer(Layer layer) {
  layers_.push_back(std::move(layer));
}

std::span<const Blob> Graph::get_blobs() const noexcept {
  return blobs_;
}

void Graph::set_blobs(std::vector<Blob> blobs) {
  blobs_ = std::move(blobs);
}

void Graph::add_blob(Blob blob) {
  blobs_.push_back(std::move(blob));
}

std::span<const std::string> Graph::get_input_blob_names() const noexcept {
  return input_blob_names_;
}

void Graph::set_input_blob_names(std::vector<std::string> names) {
  input_blob_names_ = std::move(names);
}

void Graph::add_input_blob_name(std::string name) {
  input_blob_names_.push_back(std::move(name));
}

std::span<const std::string> Graph::get_output_blob_names() const noexcept {
  return output_blob_names_;
}

void Graph::set_output_blob_names(std::vector<std::string> names) {
  output_blob_names_ = std::move(names);
}

void Graph::add_output_blob_name(std::string name) {
  output_blob_names_.push_back(std::move(name));
}

bool Graph::get_weights_loaded() const noexcept {
  return weights_loaded_;
}

void Graph::set_weights_loaded(bool weights_loaded) noexcept {
  weights_loaded_ = weights_loaded;
}

namespace {

constexpr std::uint32_t kFloat16Flag = 0x01306B47;
constexpr std::uint32_t kInt8Flag = 0x000D4B38;
constexpr std::uint32_t kFloat32Flag = 0x0002C056;
constexpr std::size_t kWeightAlignment = 4;

std::expected<std::size_t, std::string> checked_multiply(
  std::size_t left, std::size_t right, std::string_view description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return std::unexpected(std::format("{} overflows size_t", description));
  }
  return left * right;
}

std::expected<std::size_t, std::string> positive_size(
  std::int64_t value, std::string_view description) {
  if (value <= 0) {
    return std::unexpected(std::format("{} must be positive", description));
  }
  if (!std::in_range<std::size_t>(value)) {
    return std::unexpected(std::format("{} does not fit size_t", description));
  }
  return static_cast<std::size_t>(value);
}

template <typename Value>
std::expected<Value, std::string> parse_integer(std::string_view token,
                                                std::string_view description) {
  if (token.empty()) {
    return std::unexpected(std::format("empty {}", description));
  }
  Value value{};
  auto [end, error] =
    std::from_chars(token.data(), token.data() + token.size(), value);
  if (error == std::errc::result_out_of_range) {
    return std::unexpected(
      std::format("{} out of range: {}", description, token));
  }
  if (error != std::errc{} || end != token.data() + token.size()) {
    return std::unexpected(std::format("bad {}: {}", description, token));
  }
  return value;
}

std::expected<int, std::string> parse_nonnegative_int(
  std::string_view token, std::string_view description) {
  auto parsed = parse_integer<std::int64_t>(token, description);
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  if (*parsed < 0) {
    return std::unexpected(std::format("{} must be non-negative", description));
  }
  if (!std::in_range<int>(*parsed)) {
    return std::unexpected(std::format("{} does not fit int", description));
  }
  return static_cast<int>(*parsed);
}

std::expected<std::vector<std::byte>, std::string> read_file_bytes(
  const std::filesystem::path& path) {
  std::error_code error;
  std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error) {
    return std::unexpected(std::format(
      "cannot get file size for {}: {}", path.string(), error.message()));
  }

  std::vector<std::byte> buffer;
  if (file_size > buffer.max_size() ||
      file_size > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::streamsize>::max())) {
    return std::unexpected(std::format("file too large: {}", path.string()));
  }
  buffer.resize(static_cast<std::size_t>(file_size));

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("cannot open file: {}", path.string()));
  }
  if (!buffer.empty()) {
    auto expected_size = static_cast<std::streamsize>(buffer.size());
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
    if (file.gcount() != expected_size) {
      return std::unexpected(
        std::format("short read from file: {}", path.string()));
    }
  }
  return buffer;
}

std::expected<std::vector<std::string>, std::string> read_lines(
  const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::unexpected(
      std::format("cannot open param file: {}", path.string()));
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    auto first = line.find_first_not_of(" \t\r\n");
    auto last = line.find_last_not_of(" \t\r\n");
    if (first != std::string::npos) {
      lines.push_back(line.substr(first, last - first + 1));
    }
  }
  if (file.bad()) {
    return std::unexpected(std::format("read error: {}", path.string()));
  }
  return lines;
}

class BinCursor {
 public:
  explicit BinCursor(std::span<const std::byte> data);

  std::expected<std::uint32_t, std::string> read_u32_le();
  std::expected<std::span<const std::byte>, std::string> read_bytes(
    std::size_t count);
  std::expected<void, std::string> align4();

  std::size_t get_size() const noexcept;
  std::size_t get_position() const noexcept;
  std::size_t get_remaining() const noexcept;

 private:
  std::span<const std::byte> data_;
  std::size_t position_;
};

BinCursor::BinCursor(std::span<const std::byte> data)
  : data_(data), position_(0) {}

std::expected<std::span<const std::byte>, std::string> BinCursor::read_bytes(
  std::size_t count) {
  if (count > get_remaining()) {
    return std::unexpected(
      std::format("unexpected EOF at offset {}: requested {} bytes, {} remain",
                  position_,
                  count,
                  get_remaining()));
  }
  auto result = data_.subspan(position_, count);
  position_ += count;
  return result;
}

std::expected<std::uint32_t, std::string> BinCursor::read_u32_le() {
  auto bytes = read_bytes(sizeof(std::uint32_t));
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  std::array<std::byte, sizeof(std::uint32_t)> value_bytes;
  std::ranges::copy(*bytes, value_bytes.begin());
  auto value = std::bit_cast<std::uint32_t>(value_bytes);
  if constexpr (std::endian::native == std::endian::big) {
    value = std::byteswap(value);
  }
  return value;
}

std::expected<void, std::string> BinCursor::align4() {
  std::size_t remainder = position_ % kWeightAlignment;
  std::size_t padding = remainder == 0 ? 0 : kWeightAlignment - remainder;
  auto skipped = read_bytes(padding);
  if (!skipped) {
    return std::unexpected(skipped.error());
  }
  return {};
}

std::size_t BinCursor::get_size() const noexcept {
  return data_.size();
}

std::size_t BinCursor::get_position() const noexcept {
  return position_;
}

std::size_t BinCursor::get_remaining() const noexcept {
  return data_.size() - position_;
}

std::expected<Tensor, std::string> load_weight(BinCursor& cursor,
                                               std::int64_t element_count,
                                               int type) {
  auto count = positive_size(element_count, "weight element count");
  if (!count) {
    return std::unexpected(count.error());
  }

  Tensor tensor;
  std::size_t element_width = 0;
  bool needs_alignment = false;
  if (type == 0) {
    auto flag = cursor.read_u32_le();
    if (!flag) {
      return std::unexpected("weight flag: " + flag.error());
    }
    if (*flag == kFloat16Flag) {
      tensor.set_dtype(DataType::Float16);
      element_width = 2;
      needs_alignment = true;
    } else if (*flag == kInt8Flag) {
      tensor.set_dtype(DataType::Int8);
      element_width = 1;
      needs_alignment = true;
    } else if (*flag == kFloat32Flag || *flag == 0) {
      tensor.set_dtype(DataType::Float32);
      element_width = sizeof(float);
    } else {
      return std::unexpected(std::format(
        "quantized lookup-table weight flag 0x{:08x} is unsupported", *flag));
    }
  } else if (type == 1) {
    tensor.set_dtype(DataType::Float32);
    element_width = sizeof(float);
  } else {
    return std::unexpected(
      std::format("unsupported weight load type: {}", type));
  }

  auto byte_count = checked_multiply(*count, element_width, "weight byte size");
  if (!byte_count) {
    return std::unexpected(byte_count.error());
  }
  auto bytes = cursor.read_bytes(*byte_count);
  if (!bytes) {
    return std::unexpected("weight data: " + bytes.error());
  }
  tensor.set_data(std::vector<std::byte>(bytes->begin(), bytes->end()));
  if (needs_alignment) {
    auto aligned = cursor.align4();
    if (!aligned) {
      return std::unexpected("weight alignment: " + aligned.error());
    }
  }
  return tensor;
}

std::expected<void, std::string> validate_tensor_data(
  const Tensor& tensor, std::string_view description) {
  if (tensor.byte_size() != tensor.get_data().size()) {
    return std::unexpected(
      std::format("{} byte size mismatch: shape requires {}, data has {}",
                  description,
                  tensor.byte_size(),
                  tensor.get_data().size()));
  }
  return {};
}

std::expected<void, std::string> load_layer_weights(Layer& layer,
                                                    BinCursor& cursor) {
  if (layer.get_type() != "Convolution") {
    return {};
  }

  std::int64_t dynamic_weight = layer.get_param_int(19);
  if (dynamic_weight != 0 && dynamic_weight != 1) {
    return std::unexpected("convolution dynamic_weight must be 0 or 1");
  }
  if (dynamic_weight == 1) {
    return {};
  }

  std::int64_t num_output_value = layer.get_param_int(0);
  std::int64_t weight_count_value = layer.get_param_int(6);
  std::int64_t kernel_w_value = layer.get_param_int(1);
  std::int64_t kernel_h_value = layer.get_param_int(11, kernel_w_value);
  std::int64_t bias_term = layer.get_param_int(5);
  std::int64_t int8_scale_term = layer.get_param_int(8);
  if (bias_term != 0 && bias_term != 1) {
    return std::unexpected("convolution bias_term must be 0 or 1");
  }

  auto num_output = positive_size(num_output_value, "convolution num_output");
  auto weight_count =
    positive_size(weight_count_value, "convolution weight count");
  auto kernel_w = positive_size(kernel_w_value, "convolution kernel width");
  auto kernel_h = positive_size(kernel_h_value, "convolution kernel height");
  if (!num_output) {
    return std::unexpected(num_output.error());
  }
  if (!weight_count) {
    return std::unexpected(weight_count.error());
  }
  if (!kernel_w) {
    return std::unexpected(kernel_w.error());
  }
  if (!kernel_h) {
    return std::unexpected(kernel_h.error());
  }

  auto output_kernel =
    checked_multiply(*num_output, *kernel_h, "convolution output/kernel size");
  if (!output_kernel) {
    return std::unexpected(output_kernel.error());
  }
  output_kernel = checked_multiply(
    *output_kernel, *kernel_w, "convolution output/kernel size");
  if (!output_kernel) {
    return std::unexpected(output_kernel.error());
  }
  if (*weight_count % *output_kernel != 0) {
    return std::unexpected(
      "convolution weight count is not divisible by output/kernel size");
  }
  std::size_t num_input = *weight_count / *output_kernel;
  if (num_input == 0 ||
      num_input >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::unexpected("convolution input channel count is invalid");
  }

  auto weight = load_weight(cursor, weight_count_value, 0);
  if (!weight) {
    return std::unexpected("conv weight: " + weight.error());
  }
  auto shape_result = weight->set_shape({num_output_value,
                                         static_cast<std::int64_t>(num_input),
                                         kernel_h_value,
                                         kernel_w_value});
  if (!shape_result) {
    return std::unexpected("conv weight shape: " + shape_result.error());
  }
  auto weight_data_result = validate_tensor_data(*weight, "conv weight");
  if (!weight_data_result) {
    return std::unexpected(weight_data_result.error());
  }
  layer.add_weight(std::move(*weight));

  if (bias_term == 1) {
    auto bias = load_weight(cursor, num_output_value, 1);
    if (!bias) {
      return std::unexpected("conv bias: " + bias.error());
    }
    auto bias_shape_result = bias->set_shape({num_output_value});
    if (!bias_shape_result) {
      return std::unexpected("conv bias shape: " + bias_shape_result.error());
    }
    auto bias_data_result = validate_tensor_data(*bias, "conv bias");
    if (!bias_data_result) {
      return std::unexpected(bias_data_result.error());
    }
    layer.add_weight(std::move(*bias));
  }

  if (int8_scale_term != 0) {
    auto weight_scales = load_weight(cursor, num_output_value, 1);
    if (!weight_scales) {
      return std::unexpected("conv weight int8 scales: " +
                             weight_scales.error());
    }
    auto weight_scales_shape = weight_scales->set_shape({num_output_value});
    if (!weight_scales_shape) {
      return std::unexpected("conv weight int8 scales shape: " +
                             weight_scales_shape.error());
    }
    layer.add_weight(std::move(*weight_scales));

    auto bottom_scale = load_weight(cursor, 1, 1);
    if (!bottom_scale) {
      return std::unexpected("conv bottom int8 scale: " + bottom_scale.error());
    }
    auto bottom_scale_shape = bottom_scale->set_shape({1});
    if (!bottom_scale_shape) {
      return std::unexpected("conv bottom int8 scale shape: " +
                             bottom_scale_shape.error());
    }
    layer.add_weight(std::move(*bottom_scale));
  }
  if (int8_scale_term > 100) {
    auto top_scale = load_weight(cursor, 1, 1);
    if (!top_scale) {
      return std::unexpected("conv top int8 scale: " + top_scale.error());
    }
    auto top_scale_shape = top_scale->set_shape({1});
    if (!top_scale_shape) {
      return std::unexpected("conv top int8 scale shape: " +
                             top_scale_shape.error());
    }
    layer.add_weight(std::move(*top_scale));
  }
  return {};
}

std::vector<std::string> split_ws(std::string_view text) {
  std::vector<std::string> tokens;
  std::size_t position = 0;
  while (position < text.size()) {
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position]))) {
      ++position;
    }
    if (position >= text.size()) {
      break;
    }
    std::size_t end = position;
    while (end < text.size() &&
           !std::isspace(static_cast<unsigned char>(text[end]))) {
      ++end;
    }
    tokens.emplace_back(text.substr(position, end - position));
    position = end;
  }
  return tokens;
}

std::string_view get_dtype_name(DataType dtype) {
  switch (dtype) {
    case DataType::Unknown:
      return "unknown";
    case DataType::Float32:
      return "f32";
    case DataType::Float16:
      return "f16";
    case DataType::Int8:
      return "i8";
  }
  return "unknown";
}

}  // namespace

std::expected<Graph, std::string> Graph::load(std::string_view param_path,
                                              std::string_view bin_path) {
  auto lines = read_lines(std::filesystem::path(param_path));
  if (!lines) {
    return std::unexpected(lines.error());
  }
  if (lines->empty()) {
    return std::unexpected("empty param file");
  }

  std::size_t line_index = 0;
  auto first_line = split_ws((*lines)[0]);
  if (first_line.size() == 1) {
    auto magic = parse_integer<std::int64_t>(first_line[0], "param magic");
    if (!magic) {
      return std::unexpected(magic.error());
    }
    if (*magic != 7767517) {
      return std::unexpected("param magic mismatch (expected 7767517)");
    }
    line_index = 1;
  } else if (first_line.size() != 2) {
    return std::unexpected("bad param magic/header line");
  }
  if (line_index >= lines->size()) {
    return std::unexpected("missing layer/blob count header");
  }

  auto header = split_ws((*lines)[line_index]);
  if (header.size() != 2) {
    return std::unexpected("bad layer/blob count header");
  }
  auto layer_count_result = parse_nonnegative_int(header[0], "layer count");
  auto blob_count_result = parse_nonnegative_int(header[1], "blob count");
  if (!layer_count_result) {
    return std::unexpected(layer_count_result.error());
  }
  if (!blob_count_result) {
    return std::unexpected(blob_count_result.error());
  }
  int layer_count = *layer_count_result;
  int blob_count = *blob_count_result;
  ++line_index;
  if (static_cast<std::size_t>(layer_count) > lines->size() - line_index) {
    return std::unexpected("param truncated before all layers read");
  }

  bool bin_supplied = !bin_path.empty();
  std::vector<std::byte> bin_data;
  if (bin_supplied) {
    auto data = read_file_bytes(std::filesystem::path(bin_path));
    if (!data) {
      return std::unexpected(data.error());
    }
    bin_data = std::move(*data);
  }
  BinCursor cursor(bin_data);

  std::vector<Layer> layers;
  layers.reserve(static_cast<std::size_t>(layer_count));
  std::vector<Blob> blobs;
  std::vector<std::pair<std::string, int>> blob_indices;
  auto find_or_allocate_blob =
    [&](const std::string& name) -> std::expected<int, std::string> {
    auto iterator = std::ranges::find(
      blob_indices, name, [](const auto& entry) { return entry.first; });
    if (iterator != blob_indices.end()) {
      return iterator->second;
    }
    if (blobs.size() >= static_cast<std::size_t>(blob_count)) {
      return std::unexpected("model creates more blobs than declared");
    }
    if (!std::in_range<int>(blobs.size())) {
      return std::unexpected("blob index does not fit int");
    }
    int index = static_cast<int>(blobs.size());
    blobs.emplace_back(name);
    blob_indices.emplace_back(name, index);
    return index;
  };

  for (int index = 0; index < layer_count; ++index) {
    auto tokens = split_ws((*lines)[line_index++]);
    if (tokens.size() < 4) {
      return std::unexpected("layer line too short");
    }

    Layer layer;
    layer.set_type(tokens[0]);
    layer.set_name(tokens[1]);
    auto bottom_count_result = parse_nonnegative_int(tokens[2], "bottom count");
    auto top_count_result = parse_nonnegative_int(tokens[3], "top count");
    if (!bottom_count_result) {
      return std::unexpected(bottom_count_result.error());
    }
    if (!top_count_result) {
      return std::unexpected(top_count_result.error());
    }
    auto bottom_count = static_cast<std::size_t>(*bottom_count_result);
    auto top_count = static_cast<std::size_t>(*top_count_result);
    std::size_t offset = 4;
    if (bottom_count > tokens.size() - offset) {
      return std::unexpected("layer line missing bottom blob names");
    }
    for (std::size_t bottom = 0; bottom < bottom_count; ++bottom) {
      layer.add_input(tokens[offset++]);
    }
    if (top_count > tokens.size() - offset) {
      return std::unexpected("layer line missing top blob names");
    }
    for (std::size_t top = 0; top < top_count; ++top) {
      layer.add_output(tokens[offset++]);
    }

    std::string param_tail;
    for (std::size_t token_index = offset; token_index < tokens.size();
         ++token_index) {
      if (token_index > offset) {
        param_tail += ' ';
      }
      param_tail += tokens[token_index];
    }
    if (!param_tail.empty()) {
      auto params = parse_layer_params(param_tail);
      if (!params) {
        return std::unexpected(std::format(
          "layer {} ({}): {}", index, layer.get_type(), params.error()));
      }
      layer.set_params(std::move(*params));
    }

    for (const auto& name : layer.get_inputs()) {
      auto iterator = std::ranges::find(
        blob_indices, name, [](const auto& entry) { return entry.first; });
      if (iterator == blob_indices.end() ||
          blobs[iterator->second].get_producer() == -1) {
        return std::unexpected(
          std::format("layer {} has unresolved bottom blob: {}", index, name));
      }
      if (blobs[iterator->second].get_consumer() == -1) {
        blobs[iterator->second].set_consumer(index);
      }
    }
    for (const auto& name : layer.get_outputs()) {
      auto blob = find_or_allocate_blob(name);
      if (!blob) {
        return std::unexpected(blob.error());
      }
      if (blobs[*blob].get_producer() != -1) {
        return std::unexpected(
          std::format("blob has multiple producers: {}", name));
      }
      blobs[*blob].set_producer(index);
    }
    layers.push_back(std::move(layer));
  }
  if (line_index != lines->size()) {
    return std::unexpected("param contains trailing layer lines");
  }
  if (blobs.size() != static_cast<std::size_t>(blob_count)) {
    return std::unexpected(
      std::format("blob count mismatch: declared {}, constructed {}",
                  blob_count,
                  blobs.size()));
  }

  bool weights_loaded = false;
  if (bin_supplied) {
    for (auto& layer : layers) {
      auto result = load_layer_weights(layer, cursor);
      if (!result) {
        return std::unexpected(result.error());
      }
    }
    if (cursor.get_remaining() != 0) {
      return std::unexpected(std::format(
        "bin size mismatch: consumed {} bytes of {} (difference {})",
        cursor.get_position(),
        cursor.get_size(),
        cursor.get_remaining()));
    }
    weights_loaded = true;
  }

  std::vector<std::string> input_blob_names;
  for (const auto& layer : layers) {
    if (layer.get_type() == "Input") {
      for (const auto& name : layer.get_outputs()) {
        input_blob_names.push_back(name);
      }
    }
  }
  std::vector<std::string> output_blob_names;
  for (const auto& blob : blobs) {
    if (blob.get_consumer() == -1) {
      output_blob_names.emplace_back(blob.get_name());
    }
  }

  Graph graph;
  graph.set_layers(std::move(layers));
  graph.set_blobs(std::move(blobs));
  graph.set_input_blob_names(std::move(input_blob_names));
  graph.set_output_blob_names(std::move(output_blob_names));
  graph.set_weights_loaded(weights_loaded);
  return graph;
}

std::string Graph::dump() const {
  std::ostringstream output;
  auto layers = get_layers();
  auto blobs = get_blobs();
  auto input_names = get_input_blob_names();
  auto output_names = get_output_blob_names();
  output << std::format(
    "ncnn_graph: {} layers, {} blobs\n", layers.size(), blobs.size());
  output << std::format("inputs: {}\n", input_names.size());
  for (const auto& name : input_names) {
    output << std::format("  - {}\n", name);
  }
  output << std::format("outputs: {}\n", output_names.size());
  for (const auto& name : output_names) {
    output << std::format("  - {}\n", name);
  }
  output << "layers:\n";

  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto& layer = layers[index];
    output << std::format("  [{:>3}] {:<14} {:<22} in=[",
                          index,
                          layer.get_type(),
                          layer.get_name());
    auto inputs = layer.get_inputs();
    for (std::size_t input = 0; input < inputs.size(); ++input) {
      if (input) {
        output << ",";
      }
      output << inputs[input];
    }
    output << "] out=[";
    auto outputs = layer.get_outputs();
    for (std::size_t result = 0; result < outputs.size(); ++result) {
      if (result) {
        output << ",";
      }
      output << outputs[result];
    }
    output << "]";

    auto entries = layer.get_params().get_entries();
    if (!entries.empty()) {
      output << " {";
      for (std::size_t entry = 0; entry < entries.size(); ++entry) {
        if (entry) {
          output << " ";
        }
        const auto& [id, value] = entries[entry];
        output << id << "=";
        switch (value.get_kind()) {
          case ParamValue::Kind::Int:
            output << value.get_int();
            break;
          case ParamValue::Kind::Float:
            output << value.get_float();
            break;
          case ParamValue::Kind::String:
            output << "\"" << value.get_string() << "\"";
            break;
          case ParamValue::Kind::IntArray: {
            output << "[";
            auto values = value.get_int_array();
            for (std::size_t item = 0; item < values.size(); ++item) {
              if (item) {
                output << ",";
              }
              output << values[item];
            }
            output << "]";
          } break;
          case ParamValue::Kind::FloatArray: {
            output << "[";
            auto values = value.get_float_array();
            for (std::size_t item = 0; item < values.size(); ++item) {
              if (item) {
                output << ",";
              }
              output << values[item];
            }
            output << "]";
          } break;
        }
      }
      output << "}";
    }

    auto weights = layer.get_weights();
    if (!weights.empty()) {
      output << " w=[";
      for (std::size_t weight_index = 0; weight_index < weights.size();
           ++weight_index) {
        if (weight_index) {
          output << ",";
        }
        const auto& weight = weights[weight_index];
        output << "[";
        auto shape = weight.get_shape();
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
          if (dimension) {
            output << ",";
          }
          output << shape[dimension];
        }
        output << ":" << get_dtype_name(weight.get_dtype()) << ":"
               << weight.get_data().size() << "B]";
      }
      output << "]";
    }
    output << "\n";
  }
  return output.str();
}

std::size_t Graph::layer_count_of(std::string_view type) const {
  return static_cast<std::size_t>(
    std::ranges::count(get_layers(), type, &Layer::get_type));
}

}  // namespace ncnn_graph
