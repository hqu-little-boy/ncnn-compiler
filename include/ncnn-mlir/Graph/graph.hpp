// ncnn_graph: ncnn 模型计算图数据模型
// 参考 netron/source/ncnn.js 的解析逻辑，用 C++23 实现。
// 对应 ncnn 权威序列化格式：ncnn/src/{paramdict,modelbin,net,layer/*}.cpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ncnn_graph {

class ParamValue {
 public:
  enum class Kind { Int, Float, IntArray, FloatArray, String };

  static ParamValue make_int(std::int64_t value);
  static ParamValue make_float(float value);
  static ParamValue make_int_array(std::vector<std::int64_t> value);
  static ParamValue make_float_array(std::vector<float> value);
  static ParamValue make_string(std::string value);

  Kind get_kind() const noexcept;
  std::optional<std::int64_t> get_int() const noexcept;
  std::optional<float> get_float() const noexcept;
  std::optional<std::span<const std::int64_t>> get_int_array() const noexcept;
  std::optional<std::span<const float>> get_float_array() const noexcept;
  std::optional<std::string_view> get_string() const noexcept;

 private:
  friend class ParamDict;

  using Storage = std::variant<std::int64_t,
                               float,
                               std::vector<std::int64_t>,
                               std::vector<float>,
                               std::string>;

  explicit ParamValue(Storage storage);

  Storage storage_;
};

class ParamDict {
 public:
  ParamDict();

  std::int64_t get_int(int id, std::int64_t default_value = 0) const;
  float get_float(int id, float default_value = 0.0f) const;
  std::optional<std::reference_wrapper<const std::string>> get_string(
    int id) const noexcept;
  std::span<const std::int64_t> get_int_array(int id) const;
  std::span<const float> get_float_array(int id) const;
  std::span<const std::pair<int, ParamValue>> get_entries() const noexcept;

  bool has(int id) const noexcept;
  void set_value(int id, ParamValue value);

 private:
  std::optional<std::reference_wrapper<const ParamValue>> find_value(
    int id) const noexcept;

  std::vector<std::pair<int, ParamValue>> entries_;
};

struct ConvolutionParams {
  std::int64_t output_channels = 0;
  std::int64_t kernel_w = 0;
  std::int64_t kernel_h = 0;
  std::int64_t dilation_w = 1;
  std::int64_t dilation_h = 1;
  std::int64_t stride_w = 1;
  std::int64_t stride_h = 1;
  std::int64_t pad_left = 0;
  std::int64_t pad_right = 0;
  std::int64_t pad_top = 0;
  std::int64_t pad_bottom = 0;
  bool has_bias = false;
  std::int64_t weight_count = 0;
  std::int64_t int8_scale_term = 0;
  std::int64_t activation_type = 0;
  bool has_activation_params = false;
  bool dynamic_weight = false;
  float pad_value = 0.0F;

  std::size_t expected_weight_tensors() const noexcept;
};

[[nodiscard]] std::expected<ConvolutionParams, std::string>
decode_convolution_params(const ParamDict& params);

bool has_weight_loader(std::string_view layer_type) noexcept;
std::size_t get_weight_loader_count() noexcept;

enum class DataType { Unknown, Float32, Float16, Int8 };

class Tensor {
 public:
  Tensor();

  std::span<const std::int64_t> get_shape() const noexcept;
  [[nodiscard]] std::expected<void, std::string> set_shape(
    std::vector<std::int64_t> shape);
  DataType get_dtype() const noexcept;
  std::span<const std::byte> get_data() const noexcept;
  [[nodiscard]] std::expected<void, std::string> set_contents(
    std::vector<std::int64_t> shape,
    DataType dtype,
    std::vector<std::byte> data);

  std::size_t element_count() const noexcept;
  std::size_t byte_size() const noexcept;

 private:
  std::vector<std::int64_t> shape_;
  DataType dtype_;
  std::vector<std::byte> data_;
};

class Blob {
 public:
  explicit Blob(std::string name);

  std::string_view get_name() const noexcept;
  void set_name(std::string name);
  int get_producer() const noexcept;
  void set_producer(int producer) noexcept;
  int get_consumer() const noexcept;
  void set_consumer(int consumer) noexcept;

 private:
  std::string name_;
  int producer_;
  int consumer_;
};

class Layer {
 public:
  Layer();

  std::string_view get_type() const noexcept;
  void set_type(std::string type);
  std::string_view get_name() const noexcept;
  void set_name(std::string name);

  std::span<const std::string> get_inputs() const noexcept;
  void set_inputs(std::vector<std::string> inputs);
  void add_input(std::string input);
  std::span<const std::string> get_outputs() const noexcept;
  void set_outputs(std::vector<std::string> outputs);
  void add_output(std::string output);

  const ParamDict& get_params() const noexcept;
  void set_params(ParamDict params);
  std::span<const Tensor> get_weights() const noexcept;
  void set_weights(std::vector<Tensor> weights);
  void add_weight(Tensor weight);

  std::int64_t get_param_int(int id, std::int64_t default_value = 0) const;
  float get_param_float(int id, float default_value = 0.0f) const;

 private:
  std::string type_;
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  ParamDict params_;
  std::vector<Tensor> weights_;
};

class Graph {
 public:
  Graph();

  std::span<const Layer> get_layers() const noexcept;
  void set_layers(std::vector<Layer> layers);
  void add_layer(Layer layer);
  std::span<const Blob> get_blobs() const noexcept;
  void set_blobs(std::vector<Blob> blobs);
  void add_blob(Blob blob);

  std::span<const std::string> get_input_blob_names() const noexcept;
  void set_input_blob_names(std::vector<std::string> names);
  void add_input_blob_name(std::string name);
  std::span<const std::string> get_output_blob_names() const noexcept;
  void set_output_blob_names(std::vector<std::string> names);
  void add_output_blob_name(std::string name);

  bool get_weights_loaded() const noexcept;
  void set_weights_loaded(bool weights_loaded) noexcept;

  [[nodiscard]] static std::expected<Graph, std::string> load(
    std::string_view param_path, std::string_view bin_path);

  std::string dump() const;
  std::size_t layer_count_of(std::string_view type) const;

 private:
  std::vector<Layer> layers_;
  std::vector<Blob> blobs_;
  std::vector<std::string> input_blob_names_;
  std::vector<std::string> output_blob_names_;
  bool weights_loaded_;
};

}  // namespace ncnn_graph
