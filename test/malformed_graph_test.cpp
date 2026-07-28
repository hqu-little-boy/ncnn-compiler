#include "ncnn_graph/graph.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace {

bool contains(std::string_view text, std::string_view fragment) {
  return text.find(fragment) != std::string_view::npos;
}

bool write_text(const std::filesystem::path& path, std::string_view text) {
  std::ofstream file(path);
  file << text;
  return static_cast<bool>(file);
}

bool write_bytes(const std::filesystem::path& path,
                 std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  if (!bytes.empty()) {
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  return static_cast<bool>(file);
}

void append_u32_le(std::vector<std::byte>& bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::byte>(value & 0xff));
  bytes.push_back(static_cast<std::byte>((value >> 8) & 0xff));
  bytes.push_back(static_cast<std::byte>((value >> 16) & 0xff));
  bytes.push_back(static_cast<std::byte>((value >> 24) & 0xff));
}

class MalformedGraphTest : public ::testing::Test {
 protected:
  std::filesystem::path fixture_dir;

  void SetUp() override {
    fixture_dir =
      std::filesystem::path(NCNN_GRAPH_TEST_TEMP_DIR) / "malformed-graph";
    std::error_code directory_error;
    std::filesystem::create_directories(fixture_dir, directory_error);
    ASSERT_FALSE(directory_error) << "temporary fixture directory created";
  }

  auto load_text(std::string_view name, std::string_view text,
                 std::string_view bin_path = {}) {
    auto path = fixture_dir / std::format("{}.param", name);
    if (!write_text(path, text)) {
      ADD_FAILURE() << std::format("write {} fixture", name);
      return ncnn_graph::Graph::load("/fixture-write-failed", bin_path);
    }
    return ncnn_graph::Graph::load(path.string(), bin_path);
  }

  void expect_graph_failure(std::string_view name, std::string_view text,
                            std::string_view error_fragment,
                            std::string_view description,
                            std::string_view bin_path = {}) {
    auto graph = load_text(name, text, bin_path);
    EXPECT_TRUE(!graph && contains(graph.error(), error_fragment))
      << description;
  }
};

constexpr std::string_view kInputGraph =
  "7767517\n"
  "1 1\n"
  "Input input 0 1 data\n";

constexpr std::string_view kConvolutionGraph =
  "7767517\n"
  "2 2\n"
  "Input input 0 1 in\n"
  "Convolution conv 1 1 in out 0=1 1=1 11=1 5=0 6=1\n";

constexpr std::string_view kInt8ConvolutionGraph =
  "7767517\n"
  "2 2\n"
  "Input input 0 1 in\n"
  "Convolution conv 1 1 in out 0=1 1=1 5=0 6=1 8=1\n";

}  // namespace

TEST_F(MalformedGraphTest, TensorShapeValidation) {
  ncnn_graph::Tensor tensor;
  auto scalar_shape = tensor.set_shape({});
  EXPECT_TRUE(scalar_shape && tensor.element_count() == 1)
    << "empty shape has scalar element count";
  tensor.set_dtype(ncnn_graph::DataType::Float32);
  auto matrix_shape = tensor.set_shape({2, 3});
  EXPECT_TRUE(matrix_shape && tensor.element_count() == 6 &&
              tensor.byte_size() == 24)
    << "valid tensor shape computes element and byte counts";
  auto zero_shape = tensor.set_shape({2, 0, 4});
  EXPECT_TRUE(zero_shape && tensor.element_count() == 0 &&
              tensor.byte_size() == 0)
    << "zero tensor dimension produces zero elements";
  auto negative_shape = tensor.set_shape({2, -1});
  EXPECT_TRUE(!negative_shape && tensor.get_shape().size() == 3)
    << "negative shape is rejected transactionally";
  auto overflow_shape =
    tensor.set_shape({std::numeric_limits<std::int64_t>::max(), 3});
  EXPECT_TRUE(!overflow_shape && contains(overflow_shape.error(), "overflows"))
    << "tensor element-count overflow is rejected";
}

TEST_F(MalformedGraphTest, ValidHeaders) {
  auto input_graph = load_text("valid-input", kInputGraph);
  EXPECT_TRUE(input_graph && input_graph->get_layers().size() == 1 &&
              !input_graph->get_weights_loaded())
    << "standard param graph loads without a bin path";
  auto legacy_graph = load_text("legacy-header", "1 1\nInput input 0 1 data\n");
  EXPECT_TRUE(legacy_graph && legacy_graph->get_layers().size() == 1)
    << "legacy two-integer header remains supported";
}

TEST_F(MalformedGraphTest, RejectsMalformedParam) {
  expect_graph_failure(
    "empty", "", "empty param file", "empty param file rejected");
  expect_graph_failure("bad-magic",
                       "123\n1 1\nInput input 0 1 data\n",
                       "magic mismatch",
                       "incorrect magic rejected");
  expect_graph_failure(
    "missing-header", "7767517\n", "missing", "missing count header rejected");
  expect_graph_failure("bad-header",
                       "7767517\n1 extra\n",
                       "bad blob count",
                       "non-numeric blob count rejected");
  expect_graph_failure("negative-count",
                       "7767517\n-1 0\n",
                       "non-negative",
                       "negative layer count rejected");
  expect_graph_failure("truncated-layers",
                       "7767517\n1 0\n",
                       "truncated",
                       "truncated layer list rejected");
  expect_graph_failure("short-layer",
                       "7767517\n1 0\nInput input 0\n",
                       "too short",
                       "short layer line rejected");
  expect_graph_failure("bad-layer-count",
                       "7767517\n1 0\nInput input bad 0\n",
                       "bad bottom",
                       "non-numeric layer arity rejected");
  expect_graph_failure("missing-top",
                       "7767517\n1 1\nInput input 0 1\n",
                       "missing top",
                       "missing top blob name rejected");
  expect_graph_failure("blob-overflow",
                       "7767517\n1 0\nInput input 0 1 data\n",
                       "more blobs",
                       "constructing more blobs than declared is rejected");
  expect_graph_failure("blob-underflow",
                       "7767517\n0 1\n",
                       "blob count mismatch",
                       "constructing fewer blobs than declared is rejected");
  expect_graph_failure("bad-param",
                       "7767517\n1 1\nInput input 0 1 data 0=1junk\n",
                       "layer 0 (Input)",
                       "parameter errors include layer context");
  expect_graph_failure("trailing-layer",
                       "7767517\n0 0\nInput input 0 1 data\n",
                       "trailing",
                       "layer text after declared count is rejected");
  expect_graph_failure("unresolved-bottom",
                       "7767517\n2 2\nReLU relu 1 1 future out\n"
                       "Input input 0 1 future\n",
                       "unresolved bottom",
                       "bottom blob must be produced by an earlier layer");
  expect_graph_failure(
    "duplicate-producer",
    "7767517\n2 1\nInput first 0 1 data\nInput second 0 1 data\n",
    "multiple producers",
    "duplicate blob producers are rejected");
}

TEST_F(MalformedGraphTest, WeightLoading) {
  auto empty_bin_path = fixture_dir / "empty.bin";
  EXPECT_TRUE(write_bytes(empty_bin_path, {})) << "empty bin fixture written";
  auto empty_input_bin =
    load_text("empty-input-bin", kInputGraph, empty_bin_path.string());
  EXPECT_TRUE(empty_input_bin && empty_input_bin->get_weights_loaded())
    << "provided empty bin is validated for a weightless graph";
  auto missing_bin_path = fixture_dir / "does-not-exist.bin";
  auto missing_bin =
    load_text("missing-bin", kInputGraph, missing_bin_path.string());
  EXPECT_TRUE(!missing_bin && contains(missing_bin.error(), "file size"))
    << "provided missing bin path is rejected";

  std::vector<std::byte> float32_weight;
  append_u32_le(float32_weight, 0);
  append_u32_le(float32_weight, 0x3f800000);
  auto float32_bin_path = fixture_dir / "float32.bin";
  EXPECT_TRUE(write_bytes(float32_bin_path, float32_weight))
    << "float32 bin fixture written";
  auto convolution = load_text(
    "valid-convolution", kConvolutionGraph, float32_bin_path.string());
  EXPECT_TRUE(convolution && convolution->get_weights_loaded() &&
              convolution->get_layers()[1].get_weights().size() == 1 &&
              convolution->get_layers()[1].get_weights()[0].byte_size() == 4)
    << "valid little-endian float32 convolution weight loads";

  auto empty_convolution =
    load_text("empty-convolution", kConvolutionGraph, empty_bin_path.string());
  EXPECT_TRUE(!empty_convolution &&
              contains(empty_convolution.error(), "EOF"))
    << "provided empty bin does not skip convolution weight loading";

  std::vector<std::byte> truncated_weight;
  append_u32_le(truncated_weight, 0);
  auto truncated_bin_path = fixture_dir / "truncated.bin";
  EXPECT_TRUE(write_bytes(truncated_bin_path, truncated_weight))
    << "truncated bin fixture written";
  auto truncated_weight_graph = load_text(
    "truncated-weight", kConvolutionGraph, truncated_bin_path.string());
  EXPECT_TRUE(!truncated_weight_graph &&
              contains(truncated_weight_graph.error(), "EOF"))
    << "truncated weight payload rejected";

  std::vector<std::byte> unknown_flag;
  append_u32_le(unknown_flag, 0x12345678);
  auto unknown_flag_path = fixture_dir / "unknown-flag.bin";
  EXPECT_TRUE(write_bytes(unknown_flag_path, unknown_flag))
    << "unknown flag fixture written";
  auto unknown_weight_graph =
    load_text("unknown-flag", kConvolutionGraph, unknown_flag_path.string());
  EXPECT_TRUE(!unknown_weight_graph &&
              contains(unknown_weight_graph.error(), "unsupported"))
    << "unknown weight flag rejected";

  auto trailing_weight = float32_weight;
  trailing_weight.push_back(std::byte{0});
  auto trailing_bin_path = fixture_dir / "trailing.bin";
  EXPECT_TRUE(write_bytes(trailing_bin_path, trailing_weight))
    << "trailing bin fixture written";
  auto trailing_bin_graph =
    load_text("trailing-bin", kConvolutionGraph, trailing_bin_path.string());
  EXPECT_TRUE(!trailing_bin_graph &&
              contains(trailing_bin_graph.error(), "size mismatch"))
    << "trailing bin bytes rejected";

  std::vector<std::byte> float16_without_padding;
  append_u32_le(float16_without_padding, 0x01306B47);
  float16_without_padding.push_back(std::byte{0});
  float16_without_padding.push_back(std::byte{0});
  auto missing_padding_path = fixture_dir / "missing-padding.bin";
  EXPECT_TRUE(write_bytes(missing_padding_path, float16_without_padding))
    << "missing padding fixture written";
  auto missing_padding_graph = load_text(
    "missing-padding", kConvolutionGraph, missing_padding_path.string());
  EXPECT_TRUE(!missing_padding_graph &&
              contains(missing_padding_graph.error(), "alignment"))
    << "missing float16 alignment padding rejected";

  auto float16_weight = float16_without_padding;
  float16_weight.push_back(std::byte{0});
  float16_weight.push_back(std::byte{0});
  auto float16_path = fixture_dir / "float16.bin";
  EXPECT_TRUE(write_bytes(float16_path, float16_weight))
    << "float16 bin fixture written";
  auto float16_graph =
    load_text("float16", kConvolutionGraph, float16_path.string());
  EXPECT_TRUE(float16_graph &&
              float16_graph->get_layers()[1].get_weights()[0].get_dtype() ==
                ncnn_graph::DataType::Float16)
    << "float16 weight with complete alignment padding loads";

  expect_graph_failure("bad-bias",
                       "7767517\n2 2\nInput input 0 1 in\n"
                       "Convolution conv 1 1 in out 0=1 1=1 5=2 6=1\n",
                       "bias_term",
                       "invalid convolution bias flag rejected",
                       empty_bin_path.string());
  expect_graph_failure("zero-output",
                       "7767517\n2 2\nInput input 0 1 in\n"
                       "Convolution conv 1 1 in out 0=0 1=1 5=0 6=1\n",
                       "num_output",
                       "zero convolution output count rejected",
                       empty_bin_path.string());
  expect_graph_failure("bad-weight-shape",
                       "7767517\n2 2\nInput input 0 1 in\n"
                       "Convolution conv 1 1 in out 0=2 1=2 11=1 5=0 6=3\n",
                       "not divisible",
                       "inconsistent convolution weight count rejected",
                       empty_bin_path.string());

  std::vector<std::byte> int8_weight;
  append_u32_le(int8_weight, 0x000D4B38);
  int8_weight.push_back(std::byte{1});
  int8_weight.insert(int8_weight.end(), 3, std::byte{0});
  append_u32_le(int8_weight, 0x3f800000);
  append_u32_le(int8_weight, 0x3f800000);
  auto int8_path = fixture_dir / "int8.bin";
  EXPECT_TRUE(write_bytes(int8_path, int8_weight))
    << "int8 bin fixture written";
  auto int8_graph =
    load_text("int8", kInt8ConvolutionGraph, int8_path.string());
  EXPECT_TRUE(
    int8_graph && int8_graph->get_layers()[1].get_weights().size() == 3 &&
    int8_graph->get_layers()[1].get_weights()[0].get_dtype() ==
      ncnn_graph::DataType::Int8 &&
    int8_graph->get_layers()[1].get_weights()[1].get_shape().size() == 1 &&
    int8_graph->get_layers()[1].get_weights()[2].get_shape()[0] == 1)
    << "int8 convolution weight and scale tensors load completely";

  auto int8_with_top_scale = int8_weight;
  append_u32_le(int8_with_top_scale, 0x3f800000);
  auto int8_top_scale_path = fixture_dir / "int8-top-scale.bin";
  EXPECT_TRUE(write_bytes(int8_top_scale_path, int8_with_top_scale))
    << "int8 top scale fixture written";
  constexpr std::string_view kInt8TopScaleGraph =
    "7767517\n"
    "2 2\n"
    "Input input 0 1 in\n"
    "Convolution conv 1 1 in out 0=1 1=1 5=0 6=1 8=101\n";
  auto int8_top_scale_graph = load_text(
    "int8-top-scale", kInt8TopScaleGraph, int8_top_scale_path.string());
  EXPECT_TRUE(int8_top_scale_graph &&
              int8_top_scale_graph->get_layers()[1].get_weights().size() == 4)
    << "int8 convolution top scale loads when scale term exceeds 100";

  std::vector<std::byte> int8_without_scales(int8_weight.begin(),
                                             int8_weight.begin() + 8);
  auto int8_without_scales_path = fixture_dir / "int8-without-scales.bin";
  EXPECT_TRUE(write_bytes(int8_without_scales_path, int8_without_scales))
    << "truncated int8 scales fixture written";
  auto int8_without_scales_graph = load_text("int8-without-scales",
                                             kInt8ConvolutionGraph,
                                             int8_without_scales_path.string());
  EXPECT_TRUE(!int8_without_scales_graph &&
              contains(int8_without_scales_graph.error(), "int8 scales"))
    << "missing int8 scale tensors are rejected";

  constexpr std::string_view kDynamicConvolutionGraph =
    "7767517\n"
    "3 3\n"
    "Input data 0 1 in\n"
    "Input kernel 0 1 weight\n"
    "Convolution conv 2 1 in weight out 19=1\n";
  auto dynamic_graph = load_text(
    "dynamic-convolution", kDynamicConvolutionGraph, empty_bin_path.string());
  EXPECT_TRUE(dynamic_graph && dynamic_graph->get_weights_loaded() &&
              dynamic_graph->get_layers()[2].get_weights().empty())
    << "dynamic convolution consumes no model-bin weights";
}
