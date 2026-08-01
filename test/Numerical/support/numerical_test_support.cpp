#include "numerical_test_support.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
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
DEFINE_NAIVE_CREATOR(ReLU)
DEFINE_NAIVE_CREATOR(Pooling)
DEFINE_NAIVE_CREATOR(Split)
DEFINE_NAIVE_CREATOR(Concat)
DEFINE_NAIVE_CREATOR(Dropout)
DEFINE_NAIVE_CREATOR(Softmax)

#undef DEFINE_NAIVE_CREATOR

bool register_naive_layers(ncnn::Net& network) {
  return network.register_custom_layer("Input", create_naive_Input) == 0 &&
         network.register_custom_layer("Convolution",
                                       create_naive_Convolution) == 0 &&
         network.register_custom_layer("ReLU", create_naive_ReLU) == 0 &&
         network.register_custom_layer("Pooling", create_naive_Pooling) == 0 &&
         network.register_custom_layer("Split", create_naive_Split) == 0 &&
         network.register_custom_layer("Concat", create_naive_Concat) == 0 &&
         network.register_custom_layer("Dropout", create_naive_Dropout) == 0 &&
         network.register_custom_layer("Softmax", create_naive_Softmax) == 0;
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

std::size_t TensorShape::element_count() const {
  return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
         static_cast<std::size_t>(channels);
}

CompiledModel::CompiledModel(std::string_view library_path,
                             std::string_view symbol_name) {
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

bool CompiledModel::valid() const {
  return symbol_ != nullptr;
}

std::string_view CompiledModel::error() const {
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
  const ReferenceInput reference_input{
    .blob_name = model.input_blob, .shape = model.input_shape, .values = input};
  const std::string_view output_name = model.output_blob;
  auto outputs = run_ncnn_reference(model.param_path,
                                    model.bin_path,
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
    if (input.values.size() != input.shape.element_count()) {
      return std::unexpected("reference input size does not match its shape");
    }
    input_mats.emplace_back(
      input.shape.width, input.shape.height, input.shape.channels);
    if (input_mats.back().empty()) {
      return std::unexpected("ncnn failed to allocate input tensor");
    }
    const std::size_t channel_elements =
      static_cast<std::size_t>(input.shape.width) *
      static_cast<std::size_t>(input.shape.height);
    for (int channel = 0; channel < input.shape.channels; ++channel) {
      std::memcpy(input_mats.back().channel(channel),
                  input.values.data() +
                    (static_cast<std::size_t>(channel) * channel_elements),
                  channel_elements * sizeof(float));
    }
    if (extractor.input(std::string(input.blob_name).c_str(),
                        input_mats.back()) != 0) {
      return std::unexpected("ncnn failed to bind input blob");
    }
  }

  std::vector<std::vector<float>> results;
  results.reserve(output_blob_names.size());
  for (std::string_view output_blob_name : output_blob_names) {
    ncnn::Mat output;
    if (extractor.extract(std::string(output_blob_name).c_str(), output) != 0) {
      return std::unexpected("ncnn failed to extract output blob");
    }
    if (output.elempack != 1 || output.elemsize != sizeof(float)) {
      return std::unexpected("ncnn reference output is not unpacked float32");
    }
    const auto width = static_cast<std::size_t>(output.w);
    const auto height = static_cast<std::size_t>(output.h);
    const auto depth = static_cast<std::size_t>(output.d);
    const auto channels = static_cast<std::size_t>(output.c);
    const std::size_t channel_elements = width * height * depth;
    const std::size_t logical_elements = channel_elements * channels;
    std::vector<float> flattened;
    flattened.reserve(logical_elements);
    if (output.dims >= 3) {
      for (int channel = 0; channel < output.c; ++channel) {
        const float* begin = output.channel(channel);
        flattened.insert(flattened.end(), begin, begin + channel_elements);
      }
    } else {
      const auto* begin = static_cast<const float*>(output.data);
      flattened.assign(begin, begin + logical_elements);
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
                                         std::span<const float> expected) {
  if (actual.size() < 5 || actual.size() != expected.size()) {
    return ::testing::AssertionFailure() << "invalid softmax output size";
  }
  const float sum = std::accumulate(actual.begin(), actual.end(), 0.0F);
  if (std::abs(sum - 1.0F) > 1.0e-5F) {
    return ::testing::AssertionFailure()
           << "softmax sum is " << sum << ", expected 1";
  }
  const auto actual_top = top_indices(actual, 5);
  const auto expected_top = top_indices(expected, 5);
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
