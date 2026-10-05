#pragma once

#include <string>
#include <vector>

struct VulkanModelTensorFile {
  std::string name;
  std::string path;
  int channels = 1;
  int height = 1;
  int width = 1;
};

struct VulkanModelRunnerOptions {
  std::string param_path;
  std::string bin_path;
  std::vector<VulkanModelTensorFile> inputs;
  std::vector<VulkanModelTensorFile> outputs;
};

int runNcnnReference(const VulkanModelRunnerOptions& options);
