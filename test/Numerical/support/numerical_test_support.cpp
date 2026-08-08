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

#include "net.h"

namespace ncnn_compiler::test {
namespace {

#define DEFINE_NAIVE_CREATOR(name)          \
  ncnn::Layer* create_naive_##name(void*) { \
    return ncnn::create_layer_naive(#name); \
  }

DEFINE_NAIVE_CREATOR(Input)
DEFINE_NAIVE_CREATOR(Convolution)
DEFINE_NAIVE_CREATOR(ConvolutionDepthWise)
DEFINE_NAIVE_CREATOR(HardSigmoid)
DEFINE_NAIVE_CREATOR(HardSwish)
DEFINE_NAIVE_CREATOR(Reshape)
DEFINE_NAIVE_CREATOR(BinaryOp)
DEFINE_NAIVE_CREATOR(InnerProduct)
DEFINE_NAIVE_CREATOR(ShuffleChannel)
DEFINE_NAIVE_CREATOR(Slice)
DEFINE_NAIVE_CREATOR(Reduction)
DEFINE_NAIVE_CREATOR(ReLU)
DEFINE_NAIVE_CREATOR(Pooling)
DEFINE_NAIVE_CREATOR(Split)
DEFINE_NAIVE_CREATOR(Concat)
DEFINE_NAIVE_CREATOR(Dropout)
DEFINE_NAIVE_CREATOR(Softmax)
DEFINE_NAIVE_CREATOR(GELU)
DEFINE_NAIVE_CREATOR(Squeeze)
DEFINE_NAIVE_CREATOR(BatchNorm)
DEFINE_NAIVE_CREATOR(ExpandDims)
DEFINE_NAIVE_CREATOR(Permute)
DEFINE_NAIVE_CREATOR(Gemm)
DEFINE_NAIVE_CREATOR(Padding)
DEFINE_NAIVE_CREATOR(Interp)
DEFINE_NAIVE_CREATOR(Deconvolution)
DEFINE_NAIVE_CREATOR(Sigmoid)

#undef DEFINE_NAIVE_CREATOR

bool register_naive_layers(ncnn::Net& network) {
  return network.register_custom_layer("Input", create_naive_Input) == 0 &&
         network.register_custom_layer("Convolution",
                                       create_naive_Convolution) == 0 &&
         network.register_custom_layer(
           "ConvolutionDepthWise", create_naive_ConvolutionDepthWise) == 0 &&
         network.register_custom_layer("HardSigmoid",
                                       create_naive_HardSigmoid) == 0 &&
         network.register_custom_layer("HardSwish", create_naive_HardSwish) ==
           0 &&
         network.register_custom_layer("Reshape", create_naive_Reshape) == 0 &&
         network.register_custom_layer("BinaryOp", create_naive_BinaryOp) ==
           0 &&
         network.register_custom_layer("InnerProduct",
                                       create_naive_InnerProduct) == 0 &&
         network.register_custom_layer("ShuffleChannel",
                                       create_naive_ShuffleChannel) == 0 &&
         network.register_custom_layer("Slice", create_naive_Slice) == 0 &&
         network.register_custom_layer("Reduction", create_naive_Reduction) ==
           0 &&
         network.register_custom_layer("ReLU", create_naive_ReLU) == 0 &&
         network.register_custom_layer("Pooling", create_naive_Pooling) == 0 &&
         network.register_custom_layer("Split", create_naive_Split) == 0 &&
         network.register_custom_layer("Concat", create_naive_Concat) == 0 &&
         network.register_custom_layer("Dropout", create_naive_Dropout) == 0 &&
         network.register_custom_layer("Softmax", create_naive_Softmax) == 0 &&
         network.register_custom_layer("GELU", create_naive_GELU) == 0 &&
         network.register_custom_layer("Squeeze", create_naive_Squeeze) == 0 &&
         network.register_custom_layer("BatchNorm", create_naive_BatchNorm) ==
           0 &&
         network.register_custom_layer("ExpandDims", create_naive_ExpandDims) ==
           0 &&
         network.register_custom_layer("Permute", create_naive_Permute) == 0 &&
         network.register_custom_layer("Gemm", create_naive_Gemm) == 0 &&
         network.register_custom_layer("Padding", create_naive_Padding) == 0 &&
         network.register_custom_layer("Interp", create_naive_Interp) == 0 &&
         network.register_custom_layer("Deconvolution",
                                       create_naive_Deconvolution) == 0 &&
         network.register_custom_layer("Sigmoid", create_naive_Sigmoid) == 0;
}

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

TensorShape::TensorShape(int width, int height, int channels)
  : width_(width), height_(height), channels_(channels) {}

int TensorShape::get_width() const noexcept {
  return width_;
}

int TensorShape::get_height() const noexcept {
  return height_;
}

int TensorShape::get_channels() const noexcept {
  return channels_;
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
  : blob_name_(blob_name), shape_(shape), values_(values) {}

std::string_view ReferenceInput::get_blob_name() const noexcept {
  return blob_name_;
}

const TensorShape& ReferenceInput::get_shape() const noexcept {
  return shape_;
}

std::span<const float> ReferenceInput::get_values() const noexcept {
  return values_;
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
  Function function = std::bit_cast<Function>(symbol_);
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
  const ReferenceModel& model, std::span<const float> input) {
  const ReferenceInput reference_input(
    model.get_input_blob(), model.get_input_shape(), input);
  const std::string_view output_name = model.get_output_blob();
  auto outputs = run_ncnn_reference(model.get_param_path(),
                                    model.get_bin_path(),
                                    std::span(&reference_input, 1),
                                    std::span(&output_name, 1));
  if (!outputs) {
    return std::unexpected(outputs.error());
  }
  return std::move(outputs->front());
}

std::expected<std::vector<std::vector<float>>, std::string> run_ncnn_reference(
  std::string_view param_path,
  std::string_view bin_path,
  std::span<const ReferenceInput> inputs,
  std::span<const std::string_view> output_blob_names) {
  if (inputs.empty() || output_blob_names.empty()) {
    return std::unexpected("reference model requires inputs and outputs");
  }
  ncnn::Net network;
  network.opt.num_threads = 1;
  network.opt.lightmode = false;
  network.opt.use_vulkan_compute = false;
  network.opt.use_packing_layout = false;
  network.opt.use_winograd_convolution = false;
  network.opt.use_winograd23_convolution = false;
  network.opt.use_winograd43_convolution = false;
  network.opt.use_winograd63_convolution = false;
  network.opt.use_sgemm_convolution = false;
  network.opt.use_int8_inference = false;
  network.opt.use_fp16_packed = false;
  network.opt.use_fp16_storage = false;
  network.opt.use_fp16_arithmetic = false;
  network.opt.use_bf16_packed = false;
  network.opt.use_bf16_storage = false;
  network.opt.flush_denormals = 0;
  if (!register_naive_layers(network)) {
    return std::unexpected("ncnn failed to register naive layers");
  }

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
    if (input.get_values().size() != *input_elements) {
      return std::unexpected(std::format(
        "reference input blob '{}': value count does not match its shape",
        input.get_blob_name()));
    }
    input_mats.emplace_back(input.get_shape().get_width(),
                            input.get_shape().get_height(),
                            input.get_shape().get_channels());
    if (input_mats.back().empty()) {
      return std::unexpected(std::format(
        "ncnn failed to allocate input blob '{}'", input.get_blob_name()));
    }
    const std::size_t channel_elements =
      input.get_shape().get_channels() == 0
        ? 0
        : *input_elements /
            static_cast<std::size_t>(input.get_shape().get_channels());
    const std::size_t channel_bytes =
      input.get_shape().get_channels() == 0
        ? 0
        : *input_bytes /
            static_cast<std::size_t>(input.get_shape().get_channels());
    for (int channel = 0; channel < input.get_shape().get_channels();
         ++channel) {
      std::memcpy(input_mats.back().channel(channel),
                  input.get_values().data() +
                    (static_cast<std::size_t>(channel) * channel_elements),
                  channel_bytes);
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
