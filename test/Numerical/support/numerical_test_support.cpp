#include "numerical_test_support.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <utility>

#include "net.h"

namespace ncnn_compiler::test {
namespace {

std::vector<std::size_t> top_indices(std::span<const float> values,
                                     std::size_t count) {
  std::vector<std::size_t> indices(values.size());
  std::ranges::iota(indices, 0);
  std::partial_sort(indices.begin(),
                    indices.begin() + static_cast<std::ptrdiff_t>(count),
                    indices.end(),
                    [values](std::size_t left, std::size_t right) {
                      if (values[left] == values[right]) {
                        return left < right;
                      }
                      return values[left] > values[right];
                    });
  indices.resize(count);
  return indices;
}

}  // namespace

TensorShape::TensorShape(int width)
  : width_(width), height_(1), channels_(1), rank_(1) {}

TensorShape::TensorShape(int width, int height)
  : width_(width), height_(height), channels_(1), rank_(2) {}

TensorShape::TensorShape(int width, int height, int channels)
  : width_(width), height_(height), channels_(channels), rank_(3) {}

int TensorShape::get_width() const noexcept {
  return width_;
}

int TensorShape::get_height() const noexcept {
  return height_;
}

int TensorShape::get_channels() const noexcept {
  return channels_;
}

int TensorShape::get_rank() const noexcept {
  return rank_;
}

std::expected<std::size_t, std::string> TensorShape::element_count() const {
  if (width_ < 0 || height_ < 0 || channels_ < 0) {
    return std::unexpected("tensor shape dimensions must be non-negative");
  }
  const auto checked_multiply =
    [](std::size_t left,
       std::size_t right) -> std::expected<std::size_t, std::string> {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
      return std::unexpected("tensor element count overflows size_t");
    }
    return left * right;
  };
  auto elements = checked_multiply(static_cast<std::size_t>(width_),
                                   static_cast<std::size_t>(height_));
  if (!elements) {
    return std::unexpected(elements.error());
  }
  return checked_multiply(*elements, static_cast<std::size_t>(channels_));
}

std::expected<std::size_t, std::string> TensorShape::byte_count(
  std::size_t element_size) const {
  auto elements = element_count();
  if (!elements) {
    return std::unexpected(elements.error());
  }
  if (element_size != 0 &&
      *elements > std::numeric_limits<std::size_t>::max() / element_size) {
    return std::unexpected("tensor byte count overflows size_t");
  }
  return *elements * element_size;
}

ReferenceModel::ReferenceModel(std::string param_path,
                               std::string bin_path,
                               std::string input_blob,
                               std::string output_blob,
                               TensorShape input_shape)
  : param_path_(std::move(param_path)),
    bin_path_(std::move(bin_path)),
    input_blob_(std::move(input_blob)),
    output_blob_(std::move(output_blob)),
    input_shape_(input_shape) {}

const std::string& ReferenceModel::get_param_path() const noexcept {
  return param_path_;
}

const std::string& ReferenceModel::get_bin_path() const noexcept {
  return bin_path_;
}

const std::string& ReferenceModel::get_input_blob() const noexcept {
  return input_blob_;
}

const std::string& ReferenceModel::get_output_blob() const noexcept {
  return output_blob_;
}

const TensorShape& ReferenceModel::get_input_shape() const noexcept {
  return input_shape_;
}

ReferenceInput::ReferenceInput(std::string_view blob_name,
                               TensorShape shape,
                               std::span<const float> values)
  : blob_name_(blob_name),
    shape_(shape),
    bytes_(std::as_bytes(values)),
    value_count_(values.size()) {}

ReferenceInput::ReferenceInput(std::string_view blob_name,
                               TensorShape shape,
                               std::span<const std::int32_t> values)
  : blob_name_(blob_name),
    shape_(shape),
    bytes_(std::as_bytes(values)),
    value_count_(values.size()) {}

std::string_view ReferenceInput::get_blob_name() const noexcept {
  return blob_name_;
}

const TensorShape& ReferenceInput::get_shape() const noexcept {
  return shape_;
}

std::span<const std::byte> ReferenceInput::get_bytes() const noexcept {
  return bytes_;
}

std::size_t ReferenceInput::get_value_count() const noexcept {
  return value_count_;
}

CompiledModel::CompiledModel(std::string_view library_path,
                             std::string_view symbol_name)
  : handle_(nullptr), symbol_(nullptr), error_() {
  handle_ = dlopen(std::string(library_path).c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ == nullptr) {
    error_ = dlerror();
    return;
  }
  dlerror();
  void* symbol = dlsym(handle_, std::string(symbol_name).c_str());
  if (const char* error = dlerror(); error != nullptr) {
    error_ = error;
    dlclose(handle_);
    handle_ = nullptr;
    return;
  }
  symbol_ = symbol;
}

CompiledModel::~CompiledModel() {
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
}

bool CompiledModel::valid() const noexcept {
  return symbol_ != nullptr;
}

std::string_view CompiledModel::error() const noexcept {
  return error_;
}

int CompiledModel::run(std::span<const float> input,
                       std::span<float> output) const {
  if (symbol_ == nullptr) {
    return -1;
  }
  using Function = int (*)(const float*, float*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(input.data(), output.data());
}

int CompiledModel::run_dynamic(std::span<const float> input,
                               std::span<const std::int64_t> input_shape,
                               std::span<float> output,
                               std::uint64_t output_capacity) const {
  if (symbol_ == nullptr || input_shape.size() != 3) {
    return -1;
  }
  using Function =
    int (*)(const float*, const std::int64_t*, float*, std::uint64_t);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(
    input.data(), input_shape.data(), output.data(), output_capacity);
}

int CompiledModel::run_dynamic_fixed_output(
  std::span<const float> input,
  std::span<const std::int64_t> input_shape,
  std::span<float> output) const {
  if (symbol_ == nullptr || input_shape.size() != 3) {
    return -1;
  }
  using Function = int (*)(const float*, const std::int64_t*, float*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(
    input.data(), input_shape.data(), output.data());
}

int CompiledModel::run_dynamic_two_outputs(
  std::span<const float> input,
  std::span<const std::int64_t> input_shape,
  std::span<float> first_output,
  std::span<float> second_output) const {
  if (symbol_ == nullptr || input_shape.size() != 3) {
    return -1;
  }
  using Function = int (*)(const float*,
                           const std::int64_t*,
                           float*,
                           float*,
                           std::uint64_t,
                           std::uint64_t);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(input.data(),
                                          input_shape.data(),
                                          first_output.data(),
                                          second_output.data(),
                                          first_output.size(),
                                          second_output.size());
}

int CompiledModel::infer_dynamic(std::span<const std::int64_t> input_shape,
                                 std::span<std::int64_t> output_shape) const {
  if (symbol_ == nullptr || input_shape.size() != 3 || output_shape.empty()) {
    return -1;
  }
  using Function = int (*)(const std::int64_t*, std::int64_t*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(input_shape.data(),
                                          output_shape.data());
}


int CompiledModel::infer_dynamic_two_outputs(
  std::span<const std::int64_t> input_shape,
  std::span<std::int64_t> first_output_shape,
  std::span<std::int64_t> second_output_shape) const {
  if (symbol_ == nullptr || input_shape.size() != 3 ||
      first_output_shape.empty() || second_output_shape.empty()) {
    return -1;
  }
  using Function = int (*)(const std::int64_t*, std::int64_t*, std::int64_t*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(
    input_shape.data(), first_output_shape.data(), second_output_shape.data());
}

int CompiledModel::run_two_outputs(std::span<const float> input,
                                   std::span<float> first_output,
                                   std::span<float> second_output) const {
  if (symbol_ == nullptr) {
    return -1;
  }
  using Function = int (*)(const float*, float*, float*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(
    input.data(), first_output.data(), second_output.data());
}

int CompiledModel::run_two_inputs(std::span<const float> first_input,
                                  std::span<const float> second_input,
                                  std::span<float> output) const {
  if (symbol_ == nullptr) {
    return -1;
  }
  using Function = int (*)(const float*, const float*, float*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(
    first_input.data(), second_input.data(), output.data());
}

int CompiledModel::run_two_integer_inputs(
  std::span<const std::int32_t> first_input,
  std::span<const std::int32_t> second_input,
  std::span<float> output) const {
  if (symbol_ == nullptr) {
    return -1;
  }
  using Function = int (*)(const std::int32_t*, const std::int32_t*, float*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(
    first_input.data(), second_input.data(), output.data());
}

int CompiledModel::run_three_inputs_two_outputs(
  std::span<const float> first_input,
  std::span<const float> second_input,
  std::span<const float> third_input,
  std::span<float> first_output,
  std::span<std::int64_t> second_output) const {
  if (symbol_ == nullptr) {
    return -1;
  }
  using Function = int (*)(const float*,
                           const float*,
                           const float*,
                           float*,
                           std::int64_t*,
                           std::uint32_t,
                           std::uint32_t*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  std::uint32_t rank = 0;
  auto function = std::bit_cast<Function>(symbol_);
  if (function(first_input.data(),
               second_input.data(),
               third_input.data(),
               first_output.data(),
               second_output.data(),
               second_output.size() - 1,
               &rank) == 0) {
    return -3;
  }
  int status = function(first_input.data(),
                        second_input.data(),
                        third_input.data(),
                        first_output.data(),
                        second_output.data(),
                        second_output.size(),
                        &rank);
  return status == 0 && rank != second_output.size() ? -2 : status;
}

int CompiledModel::run_three_inputs_three_outputs(
  std::span<const float> first_input,
  std::span<const float> second_input,
  std::span<const float> third_input,
  std::span<float> first_output,
  std::span<float> second_output,
  std::span<float> third_output) const {
  if (symbol_ == nullptr) {
    return -1;
  }
  using Function =
    int (*)(const float*, const float*, const float*, float*, float*, float*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(first_input.data(),
                                          second_input.data(),
                                          third_input.data(),
                                          first_output.data(),
                                          second_output.data(),
                                          third_output.data());
}

int CompiledModel::run_formula_decoder(
  std::span<const float> memory,
  std::span<const float> embedding,
  std::span<const float> mask,
  std::span<const std::int64_t> mask_shape,
  std::span<const float> cache_k0,
  std::span<const std::int64_t> cache_k0_shape,
  std::span<const float> cache_v0,
  std::span<const std::int64_t> cache_v0_shape,
  std::span<const float> cache_k1,
  std::span<const std::int64_t> cache_k1_shape,
  std::span<const float> cache_v1,
  std::span<const std::int64_t> cache_v1_shape,
  std::span<float> output_cache_k0,
  std::span<float> output_cache_v0,
  std::span<float> output_cache_k1,
  std::span<float> output_cache_v1,
  std::span<float> logits) const {
  const auto valid_cache = [](std::span<const float> cache,
                              std::span<const std::int64_t> shape) {
    return shape.size() == 3 && shape[0] == 16 && shape[1] > 0 &&
           shape[2] == 24 &&
           std::cmp_less_equal(
             shape[1], std::numeric_limits<std::size_t>::max() / (16U * 24U)) &&
           cache.size() == static_cast<std::size_t>(shape[1]) * 16U * 24U;
  };
  if (symbol_ == nullptr || memory.size() != 144U * 2048U ||
      embedding.size() != 384 || mask_shape.size() != 2 || mask_shape[0] != 1 ||
      mask_shape[1] <= 0 ||
      mask.size() != static_cast<std::size_t>(mask_shape[1]) ||
      !valid_cache(cache_k0, cache_k0_shape) ||
      !valid_cache(cache_v0, cache_v0_shape) ||
      !valid_cache(cache_k1, cache_k1_shape) ||
      !valid_cache(cache_v1, cache_v1_shape) || logits.size() < 50000 ||
      output_cache_k0.empty() || output_cache_v0.empty() ||
      output_cache_k1.empty() || output_cache_v1.empty()) {
    return -1;
  }
  using Function = int (*)(const float*,
                           const float*,
                           const float*,
                           const std::int64_t*,
                           const float*,
                           const std::int64_t*,
                           const float*,
                           const std::int64_t*,
                           const float*,
                           const std::int64_t*,
                           const float*,
                           const std::int64_t*,
                           float*,
                           float*,
                           float*,
                           float*,
                           float*,
                           std::uint64_t,
                           std::uint64_t,
                           std::uint64_t,
                           std::uint64_t);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(memory.data(),
                                          embedding.data(),
                                          mask.data(),
                                          mask_shape.data(),
                                          cache_k0.data(),
                                          cache_k0_shape.data(),
                                          cache_v0.data(),
                                          cache_v0_shape.data(),
                                          cache_k1.data(),
                                          cache_k1_shape.data(),
                                          cache_v1.data(),
                                          cache_v1_shape.data(),
                                          output_cache_k0.data(),
                                          output_cache_v0.data(),
                                          output_cache_k1.data(),
                                          output_cache_v1.data(),
                                          logits.data(),
                                          output_cache_k0.size(),
                                          output_cache_v0.size(),
                                          output_cache_k1.size(),
                                          output_cache_v1.size());
}

int CompiledModel::infer_formula_decoder(
  std::span<const std::int64_t> mask_shape,
  std::span<const std::int64_t> cache_k0_shape,
  std::span<const std::int64_t> cache_v0_shape,
  std::span<const std::int64_t> cache_k1_shape,
  std::span<const std::int64_t> cache_v1_shape,
  std::span<std::int64_t> output_cache_k0_shape,
  std::span<std::int64_t> output_cache_v0_shape,
  std::span<std::int64_t> output_cache_k1_shape,
  std::span<std::int64_t> output_cache_v1_shape) const {
  const auto valid_cache_shape = [](std::span<const std::int64_t> shape) {
    return shape.size() == 3 && shape[0] == 16 && shape[1] > 0 &&
           shape[2] == 24;
  };
  if (symbol_ == nullptr || mask_shape.size() != 2 || mask_shape[0] != 1 ||
      !valid_cache_shape(cache_k0_shape) ||
      !valid_cache_shape(cache_v0_shape) ||
      !valid_cache_shape(cache_k1_shape) ||
      !valid_cache_shape(cache_v1_shape) || output_cache_k0_shape.size() != 3 ||
      output_cache_v0_shape.size() != 3 || output_cache_k1_shape.size() != 3 ||
      output_cache_v1_shape.size() != 3) {
    return -1;
  }
  using Function = int (*)(const std::int64_t*,
                           const std::int64_t*,
                           const std::int64_t*,
                           const std::int64_t*,
                           const std::int64_t*,
                           std::int64_t*,
                           std::int64_t*,
                           std::int64_t*,
                           std::int64_t*);
  static_assert(sizeof(Function) == sizeof(symbol_));
  return std::bit_cast<Function>(symbol_)(mask_shape.data(),
                                          cache_k0_shape.data(),
                                          cache_v0_shape.data(),
                                          cache_k1_shape.data(),
                                          cache_v1_shape.data(),
                                          output_cache_k0_shape.data(),
                                          output_cache_v0_shape.data(),
                                          output_cache_k1_shape.data(),
                                          output_cache_v1_shape.data());
}

std::vector<float> make_random_input(std::size_t size,
                                     std::uint32_t seed,
                                     float minimum,
                                     float maximum) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(minimum, maximum);
  std::vector<float> input(size);
  std::ranges::generate(
    input, [&generator, &distribution]() { return distribution(generator); });
  return input;
}

std::expected<std::vector<float>, std::string> run_ncnn_reference(
  const ReferenceModel& model,
  std::span<const float> input,
  ReferenceInferenceMode mode) {
  const ReferenceInput reference_input(
    model.get_input_blob(), model.get_input_shape(), input);
  const std::string_view output_name = model.get_output_blob();
  auto outputs = run_ncnn_reference(model.get_param_path(),
                                    model.get_bin_path(),
                                    std::span(&reference_input, 1),
                                    std::span(&output_name, 1),
                                    mode);
  if (!outputs) {
    return std::unexpected(outputs.error());
  }
  return std::move(outputs->front());
}

std::expected<std::vector<std::vector<float>>, std::string> run_ncnn_reference(
  std::string_view param_path,
  std::string_view bin_path,
  std::span<const ReferenceInput> inputs,
  std::span<const std::string_view> output_blob_names,
  ReferenceInferenceMode mode) {
  if (inputs.empty() || output_blob_names.empty()) {
    return std::unexpected("reference model requires inputs and outputs");
  }
  ncnn::Net network;
  network.opt.lightmode = false;
  network.opt.use_vulkan_compute = false;
  network.opt.use_int8_inference = mode == ReferenceInferenceMode::Int8;
  network.opt.use_fp16_packed = false;
  network.opt.use_fp16_storage = false;
  network.opt.use_fp16_arithmetic = false;
  network.opt.use_bf16_packed = false;
  network.opt.use_bf16_storage = false;
  network.opt.flush_denormals = 0;

  if (network.load_param(std::string(param_path).c_str()) != 0) {
    return std::unexpected("ncnn failed to load param file");
  }
  if (network.load_model(std::string(bin_path).c_str()) != 0) {
    return std::unexpected("ncnn failed to load bin file");
  }

  ncnn::Extractor extractor = network.create_extractor();
  std::vector<ncnn::Mat> input_mats;
  input_mats.reserve(inputs.size());
  for (const ReferenceInput& input : inputs) {
    auto input_elements = input.get_shape().element_count();
    if (!input_elements) {
      return std::unexpected(std::format("reference input blob '{}': {}",
                                         input.get_blob_name(),
                                         input_elements.error()));
    }
    auto input_bytes = input.get_shape().byte_count(sizeof(float));
    if (!input_bytes) {
      return std::unexpected(std::format("reference input blob '{}': {}",
                                         input.get_blob_name(),
                                         input_bytes.error()));
    }
    if (input.get_value_count() != *input_elements ||
        input.get_bytes().size() != *input_elements * sizeof(float)) {
      return std::unexpected(std::format(
        "reference input blob '{}': value count does not match its shape",
        input.get_blob_name()));
    }
    switch (input.get_shape().get_rank()) {
      case 1:
        input_mats.emplace_back(input.get_shape().get_width());
        break;
      case 2:
        input_mats.emplace_back(input.get_shape().get_width(),
                                input.get_shape().get_height());
        break;
      case 3:
        input_mats.emplace_back(input.get_shape().get_width(),
                                input.get_shape().get_height(),
                                input.get_shape().get_channels());
        break;
      default:
        return std::unexpected("reference input rank must be 1 through 3");
    }
    if (input_mats.back().empty()) {
      return std::unexpected(std::format(
        "ncnn failed to allocate input blob '{}'", input.get_blob_name()));
    }
    const std::size_t channel_bytes =
      input.get_shape().get_channels() == 0
        ? 0
        : *input_bytes /
            static_cast<std::size_t>(input.get_shape().get_channels());
    if (input.get_shape().get_rank() < 3) {
      std::memcpy(
        input_mats.back().data, input.get_bytes().data(), *input_bytes);
    } else {
      for (int channel = 0; channel < input.get_shape().get_channels();
           ++channel) {
        std::memcpy(input_mats.back().channel(channel),
                    input.get_bytes().data() +
                      (static_cast<std::size_t>(channel) * channel_bytes),
                    channel_bytes);
      }
    }
    if (extractor.input(std::string(input.get_blob_name()).c_str(),
                        input_mats.back()) != 0) {
      return std::unexpected(std::format("ncnn failed to bind input blob '{}'",
                                         input.get_blob_name()));
    }
  }

  std::vector<std::vector<float>> results;
  results.reserve(output_blob_names.size());
  for (std::string_view output_blob_name : output_blob_names) {
    ncnn::Mat output;
    if (extractor.extract(std::string(output_blob_name).c_str(), output) != 0) {
      return std::unexpected(std::format(
        "ncnn failed to extract output blob '{}'", output_blob_name));
    }
    if (output.elempack != 1 || output.elemsize != sizeof(float)) {
      return std::unexpected(
        std::format("ncnn reference output blob '{}' is not unpacked float32",
                    output_blob_name));
    }
    if (output.w < 0 || output.h < 0 || output.d < 0 || output.c < 0) {
      return std::unexpected(
        std::format("ncnn reference output blob '{}' has a negative dimension",
                    output_blob_name));
    }
    const auto checked_multiply =
      [output_blob_name](
        std::size_t left,
        std::size_t right) -> std::expected<std::size_t, std::string> {
      if (right != 0 &&
          left > std::numeric_limits<std::size_t>::max() / right) {
        return std::unexpected(
          std::format("ncnn reference output blob '{}' size overflows size_t",
                      output_blob_name));
      }
      return left * right;
    };
    auto channel_elements = checked_multiply(
      static_cast<std::size_t>(output.w), static_cast<std::size_t>(output.h));
    if (channel_elements) {
      channel_elements =
        checked_multiply(*channel_elements, static_cast<std::size_t>(output.d));
    }
    if (!channel_elements) {
      return std::unexpected(channel_elements.error());
    }
    auto logical_elements =
      checked_multiply(*channel_elements, static_cast<std::size_t>(output.c));
    if (!logical_elements) {
      return std::unexpected(logical_elements.error());
    }
    if (*logical_elements >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      return std::unexpected(std::format(
        "ncnn reference output blob '{}' byte count overflows size_t",
        output_blob_name));
    }
    std::vector<float> flattened;
    flattened.reserve(*logical_elements);
    if (output.dims >= 3) {
      for (int channel = 0; channel < output.c; ++channel) {
        const float* begin = output.channel(channel);
        flattened.insert(flattened.end(), begin, begin + *channel_elements);
      }
    } else {
      const auto* begin = static_cast<const float*>(output.data);
      flattened.assign(begin, begin + *logical_elements);
    }
    results.push_back(std::move(flattened));
  }
  return results;
}

::testing::AssertionResult compare_values(std::span<const float> actual,
                                          std::span<const float> expected,
                                          float maximum_absolute_error) {
  if (actual.size() != expected.size()) {
    return ::testing::AssertionFailure()
           << "output size mismatch: actual=" << actual.size()
           << " expected=" << expected.size();
  }
  float observed_maximum = 0.0F;
  std::size_t maximum_index = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) {
      return ::testing::AssertionFailure()
             << "non-finite compiled output at index " << index;
    }
    const float error = std::abs(actual[index] - expected[index]);
    if (error > observed_maximum) {
      observed_maximum = error;
      maximum_index = index;
    }
  }
  if (observed_maximum > maximum_absolute_error) {
    return ::testing::AssertionFailure()
           << "maximum absolute error " << observed_maximum << " at index "
           << maximum_index << " exceeds " << maximum_absolute_error
           << "; actual=" << actual[maximum_index]
           << " expected=" << expected[maximum_index];
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult check_softmax(std::span<const float> actual,
                                         std::span<const float> expected,
                                         double sum_tolerance) {
  if (actual.empty() || actual.size() != expected.size()) {
    return ::testing::AssertionFailure() << "invalid softmax output size";
  }
  const double sum = std::accumulate(actual.begin(), actual.end(), 0.0);
  if (std::abs(sum - 1.0) > sum_tolerance) {
    return ::testing::AssertionFailure()
           << "softmax sum is " << sum << ", expected 1";
  }
  const std::size_t top_count = std::min<std::size_t>(5, actual.size());
  const auto actual_top = top_indices(actual, top_count);
  const auto expected_top = top_indices(expected, top_count);
  if (actual_top.front() != expected_top.front()) {
    return ::testing::AssertionFailure()
           << "top-1 mismatch: actual=" << actual_top.front()
           << " expected=" << expected_top.front();
  }
  if (std::set<std::size_t>(actual_top.begin(), actual_top.end()) !=
      std::set<std::size_t>(expected_top.begin(), expected_top.end())) {
    return ::testing::AssertionFailure() << "top-5 class set mismatch";
  }
  return ::testing::AssertionSuccess();
}

}  // namespace ncnn_compiler::test
