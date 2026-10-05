#include "vulkan_model_runner.hpp"

#include <net.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::vector<float> readFloats(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open input tensor: " + path);
  }
  const std::streamsize bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
    throw std::runtime_error("tensor file has a non-f32 byte count: " + path);
  }
  std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
  stream.seekg(0);
  if (!values.empty() &&
      !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
    throw std::runtime_error("cannot read tensor file: " + path);
  }
  return values;
}

void writeFloats(const std::string& path,
                 const float* values,
                 std::size_t count) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      (count != 0 &&
       !stream.write(reinterpret_cast<const char*>(values),
                     static_cast<std::streamsize>(count * sizeof(float))))) {
    throw std::runtime_error("cannot write tensor file: " + path);
  }
}

std::size_t checkedElementCount(const VulkanModelTensorFile& tensor) {
  if (tensor.channels <= 0 || tensor.height <= 0 || tensor.width <= 0) {
    throw std::runtime_error("tensor dimensions must be positive: " +
                             tensor.name);
  }
  return static_cast<std::size_t>(tensor.channels) *
         static_cast<std::size_t>(tensor.height) *
         static_cast<std::size_t>(tensor.width);
}

}  // namespace

int runNcnnReference(const VulkanModelRunnerOptions& options) {
  ncnn::Net network;
  network.opt.use_vulkan_compute = false;
  network.opt.use_fp16_packed = false;
  network.opt.use_fp16_storage = false;
  network.opt.use_fp16_arithmetic = false;
  network.opt.use_bf16_storage = false;
  network.opt.use_bf16_packed = false;
  network.opt.use_packing_layout = false;
  network.opt.num_threads = 1;

  if (network.load_param(options.param_path.c_str()) != 0) {
    throw std::runtime_error("ncnn failed to load param: " +
                             options.param_path);
  }
  if (network.load_model(options.bin_path.c_str()) != 0) {
    throw std::runtime_error("ncnn failed to load bin: " + options.bin_path);
  }

  ncnn::Extractor extractor = network.create_extractor();
  for (const VulkanModelTensorFile& input : options.inputs) {
    std::vector<float> values = readFloats(input.path);
    const std::size_t expected = checkedElementCount(input);
    if (values.size() != expected) {
      throw std::runtime_error("input count differs from shape for " +
                               input.name);
    }
    ncnn::Mat tensor(
      input.width, input.height, input.channels, sizeof(float), 1);
    if (tensor.empty()) {
      throw std::runtime_error("ncnn could not allocate input tensor: " +
                               input.name);
    }
    std::size_t value_index = 0;
    for (int channel = 0; channel < input.channels; ++channel) {
      ncnn::Mat channel_view = tensor.channel(channel);
      for (int row = 0; row < input.height; ++row) {
        float* destination = channel_view.row(row);
        std::ranges::copy(
          std::span<const float>(values).subspan(value_index, input.width),
          destination);
        value_index += static_cast<std::size_t>(input.width);
      }
    }
    if (extractor.input(input.name.c_str(), tensor) != 0) {
      throw std::runtime_error("ncnn rejected input blob: " + input.name);
    }
  }

  for (const VulkanModelTensorFile& output : options.outputs) {
    ncnn::Mat tensor;
    if (extractor.extract(output.name.c_str(), tensor) != 0 || tensor.empty()) {
      throw std::runtime_error("ncnn failed to extract output blob: " +
                               output.name);
    }
    const std::size_t expected = checkedElementCount(output);
    const std::size_t logical_elements = static_cast<std::size_t>(tensor.w) *
                                         static_cast<std::size_t>(tensor.h) *
                                         static_cast<std::size_t>(tensor.c);
    if (tensor.elembits() != 32 || tensor.elempack != 1 ||
        tensor.c != output.channels || tensor.h != output.height ||
        tensor.w != output.width || logical_elements != expected) {
      throw std::runtime_error(
        "ncnn output shape/type differs for " + output.name + " (dims=" +
        std::to_string(tensor.dims) + ", c=" + std::to_string(tensor.c) +
        ", h=" + std::to_string(tensor.h) + ", w=" + std::to_string(tensor.w) +
        ", elembits=" + std::to_string(tensor.elembits()) +
        ", elempack=" + std::to_string(tensor.elempack) +
        ", total=" + std::to_string(tensor.total()) +
        ", expected=" + std::to_string(expected) + ")");
    }
    std::vector<float> values;
    values.reserve(expected);
    for (int channel = 0; channel < tensor.c; ++channel) {
      const ncnn::Mat channel_view = tensor.channel(channel);
      for (int row = 0; row < tensor.h; ++row) {
        const float* source = channel_view.row(row);
        values.insert(values.end(), source, source + tensor.w);
      }
    }
    writeFloats(output.path, values.data(), values.size());
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    VulkanModelRunnerOptions options;
    int index = 1;
    while (index < argc) {
      const std::string_view argument(argv[index++]);
      if (argument == "--param" && index < argc) {
        options.param_path = argv[index++];
      } else if (argument == "--bin" && index < argc) {
        options.bin_path = argv[index++];
      } else if ((argument == "--input" || argument == "--output") &&
                 index + 4 < argc) {
        VulkanModelTensorFile tensor;
        tensor.name = argv[index++];
        tensor.path = argv[index++];
        tensor.channels = std::stoi(argv[index++]);
        tensor.height = std::stoi(argv[index++]);
        tensor.width = std::stoi(argv[index++]);
        (argument == "--input" ? options.inputs : options.outputs)
          .push_back(std::move(tensor));
      } else {
        throw std::runtime_error("invalid or incomplete argument: " +
                                 std::string(argument));
      }
    }
    if (options.param_path.empty() || options.bin_path.empty() ||
        options.inputs.empty() || options.outputs.empty()) {
      throw std::runtime_error(
        "usage: vulkan_model_runner --param FILE --bin FILE "
        "--input BLOB RAW C H W [...] --output BLOB RAW C H W [...]");
    }
    return runNcnnReference(options);
  } catch (const std::exception& error) {
    std::cerr << "vulkan_model_runner: " << error.what() << '\n';
    return 1;
  }
}
