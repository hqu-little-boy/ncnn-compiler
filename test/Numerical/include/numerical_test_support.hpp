#ifndef NCNN_COMPILER_TEST_NUMERICAL_TEST_SUPPORT_HPP
#define NCNN_COMPILER_TEST_NUMERICAL_TEST_SUPPORT_HPP

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {

struct TensorShape {
  int width;
  int height;
  int channels;

  [[nodiscard]] std::size_t element_count() const;
};

struct ReferenceModel {
  std::string param_path;
  std::string bin_path;
  std::string input_blob;
  std::string output_blob;
  TensorShape input_shape;
};

struct ReferenceInput {
  std::string_view blob_name;
  TensorShape shape;
  std::span<const float> values;
};

class CompiledModel final {
 public:
  CompiledModel(std::string_view library_path, std::string_view symbol_name);
  ~CompiledModel();

  CompiledModel(const CompiledModel&) = delete;
  CompiledModel& operator=(const CompiledModel&) = delete;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::string_view error() const;
  int run(std::span<const float> input, std::span<float> output) const;
  int run_two_outputs(std::span<const float> input,
                      std::span<float> first_output,
                      std::span<float> second_output) const;
  int run_two_inputs(std::span<const float> first_input,
                     std::span<const float> second_input,
                     std::span<float> output) const;

 private:
  void* handle_ = nullptr;
  void* symbol_ = nullptr;
  std::string error_;
};

std::vector<float> make_random_input(std::size_t size,
                                     std::uint32_t seed,
                                     float minimum,
                                     float maximum);
std::expected<std::vector<float>, std::string> run_ncnn_reference(
  const ReferenceModel& model,
  std::span<const float> input);
std::expected<std::vector<std::vector<float>>, std::string> run_ncnn_reference(
  std::string_view param_path,
  std::string_view bin_path,
  std::span<const ReferenceInput> inputs,
  std::span<const std::string_view> output_blob_names);
::testing::AssertionResult compare_values(std::span<const float> actual,
                                          std::span<const float> expected,
                                          float maximum_absolute_error);
::testing::AssertionResult check_softmax(std::span<const float> actual,
                                         std::span<const float> expected);

}  // namespace ncnn_compiler::test

#endif  // NCNN_COMPILER_TEST_NUMERICAL_TEST_SUPPORT_HPP
