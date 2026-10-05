#include "vulkan_model_runner.hpp"

#include <net.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::vector<float> readFloats(const std::string& path, std::size_t expected) {
  if (expected > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      expected >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) /
          sizeof(float)) {
    throw std::runtime_error("tensor byte count overflows: " + path);
  }
  const std::size_t expectedBytes = expected * sizeof(float);
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open input tensor: " + path);
  }
  const std::streamoff bytes = stream.tellg();
  if (bytes < 0 || std::cmp_not_equal(bytes, expectedBytes)) {
    throw std::runtime_error("tensor file size differs from its shape: " +
                             path);
  }
  std::vector<float> values(expected);
  stream.seekg(0);
  if (!values.empty() &&
      !stream.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(expectedBytes))) {
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
  if (tensor.shape.empty() || tensor.shape.size() > 4) {
    throw std::runtime_error("tensor rank must be between one and four: " +
                             tensor.name);
  }
  std::size_t count = 1;
  for (int extent : tensor.shape) {
    if (extent <= 0 || count > std::numeric_limits<std::size_t>::max() /
                                 static_cast<std::size_t>(extent)) {
      throw std::runtime_error("tensor dimensions are invalid: " + tensor.name);
    }
    count *= static_cast<std::size_t>(extent);
  }
  return count;
}

ncnn::Mat makeTensor(const VulkanModelTensorFile& tensor) {
  switch (tensor.shape.size()) {
    case 1:
      return {tensor.shape[0], sizeof(float)};
    case 2:
      return {tensor.shape[1], tensor.shape[0], sizeof(float)};
    case 3:
      return {
        tensor.shape[2], tensor.shape[1], tensor.shape[0], sizeof(float), 1};
    case 4:
      return {tensor.shape[3],
              tensor.shape[2],
              tensor.shape[1],
              tensor.shape[0],
              sizeof(float),
              1};
  }
  throw std::runtime_error("unsupported tensor rank: " + tensor.name);
}

std::vector<int> tensorShape(const ncnn::Mat& tensor) {
  switch (tensor.dims) {
    case 1:
      return {tensor.w};
    case 2:
      return {tensor.h, tensor.w};
    case 3:
      return {tensor.c, tensor.h, tensor.w};
    case 4:
      return {tensor.c, tensor.d, tensor.h, tensor.w};
  }
  return {};
}

void copyToTensor(ncnn::Mat& tensor,
                  const std::vector<int>& shape,
                  const std::vector<float>& values) {
  std::size_t offset = 0;
  if (shape.size() == 1) {
    std::ranges::copy(values, static_cast<float*>(tensor.data));
    return;
  }
  if (shape.size() == 2) {
    for (int row = 0; row < shape[0]; ++row) {
      std::copy_n(values.begin() + offset, shape[1], tensor.row(row));
      offset += static_cast<std::size_t>(shape[1]);
    }
    return;
  }
  for (int channel = 0; channel < shape[0]; ++channel) {
    ncnn::Mat channelView = tensor.channel(channel);
    if (shape.size() == 3) {
      for (int row = 0; row < shape[1]; ++row) {
        std::copy_n(values.begin() + offset, shape[2], channelView.row(row));
        offset += static_cast<std::size_t>(shape[2]);
      }
      continue;
    }
    for (int depth = 0; depth < shape[1]; ++depth) {
      ncnn::Mat depthView = channelView.depth(depth);
      for (int row = 0; row < shape[2]; ++row) {
        std::copy_n(values.begin() + offset, shape[3], depthView.row(row));
        offset += static_cast<std::size_t>(shape[3]);
      }
    }
  }
}

std::vector<float> flattenTensor(const ncnn::Mat& tensor,
                                 const std::vector<int>& shape,
                                 std::size_t expected) {
  std::vector<float> values;
  values.reserve(expected);
  if (shape.size() == 1) {
    const float* data = tensor;
    values.insert(values.end(), data, data + shape[0]);
    return values;
  }
  if (shape.size() == 2) {
    for (int row = 0; row < shape[0]; ++row) {
      const float* data = tensor.row(row);
      values.insert(values.end(), data, data + shape[1]);
    }
    return values;
  }
  for (int channel = 0; channel < shape[0]; ++channel) {
    const ncnn::Mat channelView = tensor.channel(channel);
    if (shape.size() == 3) {
      for (int row = 0; row < shape[1]; ++row) {
        const float* data = channelView.row(row);
        values.insert(values.end(), data, data + shape[2]);
      }
      continue;
    }
    for (int depth = 0; depth < shape[1]; ++depth) {
      const ncnn::Mat depthView = channelView.depth(depth);
      for (int row = 0; row < shape[2]; ++row) {
        const float* data = depthView.row(row);
        values.insert(values.end(), data, data + shape[3]);
      }
    }
  }
  return values;
}

void verifyRankAwareTensorRoundTrips() {
  const std::vector<std::vector<int>> shapes{
    {2}, {2, 3}, {2, 3, 5}, {2, 2, 3, 5}};
  for (const auto& shape : shapes) {
    VulkanModelTensorFile tensor{
      .name = "rank_test", .path = "", .shape = shape};
    const std::size_t expected = checkedElementCount(tensor);
    ncnn::Mat mat = makeTensor(tensor);
    if (mat.empty() || std::cmp_not_equal(mat.dims, shape.size()) ||
        mat.elembits() != 32 || mat.elempack != 1 ||
        tensorShape(mat) != shape) {
      throw std::runtime_error("rank-aware ncnn Mat creation failed");
    }
    std::vector<float> source(expected);
    for (std::size_t index = 0; index < expected; ++index) {
      source[index] = static_cast<float>(index) - 3.0F;
    }
    copyToTensor(mat, shape, source);
    if (flattenTensor(mat, shape, expected) != source) {
      throw std::runtime_error("rank-aware ncnn Mat round-trip changed data");
    }
  }
  VulkanModelTensorFile overflow{
    .name = "overflow", .path = "", .shape = {2147483647, 2147483647, 3, 3}};
  try {
    static_cast<void>(checkedElementCount(overflow));
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("rank-aware ncnn Mat count overflow was accepted");
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
    const std::size_t expected = checkedElementCount(input);
    std::vector<float> values = readFloats(input.path, expected);
    if (values.size() != expected) {
      throw std::runtime_error("input count differs from shape for " +
                               input.name);
    }
    ncnn::Mat tensor = makeTensor(input);
    if (tensor.empty()) {
      throw std::runtime_error("ncnn could not allocate input tensor: " +
                               input.name);
    }
    if (std::cmp_not_equal(tensor.dims, input.shape.size()) ||
        tensorShape(tensor) != input.shape || tensor.elembits() != 32 ||
        tensor.elempack != 1) {
      throw std::runtime_error("ncnn input rank/shape differs for " +
                               input.name);
    }
    copyToTensor(tensor, input.shape, values);
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
    if (tensor.elembits() != 32 || tensor.elempack != 1 ||
        std::cmp_not_equal(tensor.dims, output.shape.size()) ||
        tensorShape(tensor) != output.shape) {
      throw std::runtime_error(
        "ncnn output rank/shape/type differs for " + output.name +
        " (dims=" + std::to_string(tensor.dims) +
        ", elembits=" + std::to_string(tensor.elembits()) +
        ", elempack=" + std::to_string(tensor.elempack) +
        ", total=" + std::to_string(tensor.total()) +
        ", expected=" + std::to_string(expected) + ")");
    }
    std::vector<float> values = flattenTensor(tensor, output.shape, expected);
    if (values.size() != expected) {
      throw std::runtime_error("ncnn output element count differs for " +
                               output.name);
    }
    writeFloats(output.path, values.data(), values.size());
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test-ranks") {
      verifyRankAwareTensorRoundTrips();
      return 0;
    }
    VulkanModelRunnerOptions options;
    int index = 1;
    while (index < argc) {
      const std::string_view argument(argv[index++]);
      if (argument == "--param" && index < argc) {
        options.param_path = argv[index++];
      } else if (argument == "--bin" && index < argc) {
        options.bin_path = argv[index++];
      } else if ((argument == "--input" || argument == "--output") &&
                 index + 2 < argc) {
        VulkanModelTensorFile tensor;
        tensor.name = argv[index++];
        tensor.path = argv[index++];
        const int rank = std::stoi(argv[index++]);
        if (rank < 1 || rank > 4 || argc - index < rank) {
          throw std::runtime_error("tensor rank must be one through four");
        }
        tensor.shape.reserve(static_cast<std::size_t>(rank));
        for (int dimension = 0; dimension < rank; ++dimension) {
          tensor.shape.push_back(std::stoi(argv[index++]));
        }
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
        "--input BLOB RAW RANK D0... --output BLOB RAW RANK D0...");
    }
    return runNcnnReference(options);
  } catch (const std::exception& error) {
    std::cerr << "vulkan_model_runner: " << error.what() << '\n';
    return 1;
  }
}
