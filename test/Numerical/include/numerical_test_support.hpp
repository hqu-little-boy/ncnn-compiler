#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {

class TensorShape final {
 public:
  explicit TensorShape(int width, int height, int channels);

  [[nodiscard]] int get_width() const noexcept;
  [[nodiscard]] int get_height() const noexcept;
  [[nodiscard]] int get_channels() const noexcept;

  [[nodiscard]] std::expected<std::size_t, std::string> element_count() const;
  [[nodiscard]] std::expected<std::size_t, std::string> byte_count(
    std::size_t element_size) const;

 private:
  int width_;
  int height_;
  int channels_;
};

class ReferenceModel final {
 public:
  explicit ReferenceModel(std::string param_path,
                          std::string bin_path,
                          std::string input_blob,
                          std::string output_blob,
                          TensorShape input_shape);

  [[nodiscard]] const std::string& get_param_path() const noexcept;
  [[nodiscard]] const std::string& get_bin_path() const noexcept;
  [[nodiscard]] const std::string& get_input_blob() const noexcept;
  [[nodiscard]] const std::string& get_output_blob() const noexcept;
  [[nodiscard]] const TensorShape& get_input_shape() const noexcept;

 private:
  std::string param_path_;
  std::string bin_path_;
  std::string input_blob_;
  std::string output_blob_;
  TensorShape input_shape_;
};

class ReferenceInput final {
 public:
  explicit ReferenceInput(std::string_view blob_name,
                          TensorShape shape,
                          std::span<const float> values);

  [[nodiscard]] std::string_view get_blob_name() const noexcept;
  [[nodiscard]] const TensorShape& get_shape() const noexcept;
  [[nodiscard]] std::span<const float> get_values() const noexcept;

 private:
  std::string_view blob_name_;
  TensorShape shape_;
  std::span<const float> values_;
};

class CompiledModel final {
 public:
  CompiledModel(std::string_view library_path, std::string_view symbol_name);
  ~CompiledModel();

  CompiledModel(const CompiledModel&) = delete;
  CompiledModel& operator=(const CompiledModel&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::string_view error() const noexcept;
  int run(std::span<const float> input, std::span<float> output) const;
  int run_two_outputs(std::span<const float> input,
                      std::span<float> first_output,
                      std::span<float> second_output) const;
  int run_two_inputs(std::span<const float> first_input,
                     std::span<const float> second_input,
                     std::span<float> output) const;

 private:
  void* handle_;
  void* symbol_;
  std::string error_;
};

std::vector<float> make_random_input(std::size_t size,
                                     std::uint32_t seed,
                                     float minimum,
                                     float maximum);
[[nodiscard]] std::expected<std::vector<float>, std::string> run_ncnn_reference(
  const ReferenceModel& model, std::span<const float> input);
[[nodiscard]] std::expected<std::vector<std::vector<float>>, std::string>
run_ncnn_reference(std::string_view param_path,
                   std::string_view bin_path,
                   std::span<const ReferenceInput> inputs,
                   std::span<const std::string_view> output_blob_names);
::testing::AssertionResult compare_values(std::span<const float> actual,
                                          std::span<const float> expected,
                                          float maximum_absolute_error);
::testing::AssertionResult check_softmax(std::span<const float> actual,
                                         std::span<const float> expected,
                                         double sum_tolerance = 1.0e-5);

}  // namespace ncnn_compiler::test
