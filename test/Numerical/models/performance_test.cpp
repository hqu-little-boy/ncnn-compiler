// 性能基准（performance 标签）：编译产物 vs upstream ncnn 的墙钟对比。
// 复用数值金标的同一批 compile_<model> fixture，不重复编译任何模型；
// 数值正确性由 numerical_tests 保证，这里只做一次宽松 sanity 兜底
// （防止计时了陈旧/错误的 .so），然后双侧各计时并输出加速比。

#include "numerical_test_support.hpp"
#include "performance_test_support.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

namespace ncnn_compiler::test {
namespace {

constexpr bool kUnderSanitizer = PERFORMANCE_UNDER_SANITIZER != 0;

constexpr std::size_t kYolov5GridSum = (80 * 80) + (40 * 40) + (20 * 20);
constexpr std::size_t kYolov5Anchors = 3;
constexpr std::size_t kDetectionAttributes = 85;
constexpr std::size_t kDetectionElements =
  kYolov5GridSum * kYolov5Anchors * kDetectionAttributes;
constexpr std::size_t kSegProtoElements = 32U * 160U * 160U;
constexpr std::size_t kSegDetectionElements =
  kYolov5GridSum * kYolov5Anchors * 117U;

struct ModelSpec final {
  std::string_view name;
  std::string_view param_path;
  std::string_view bin_path;
  std::string_view library_path;
  std::string_view symbol;
  std::string_view input_blob;
  // ncnn 参考侧输出 blob 名与编译产物输出一一对应；第二项为空即单输出。
  std::array<std::string_view, 2> output_blobs{};
  TensorShape input_shape;
  std::array<std::size_t, 2> output_element_counts{};
  std::uint32_t seed;
};

// 重模型（输入 >= 640x640x3）用更少的迭代次数控制整包时长。
TimingPolicy default_timing_policy(const ModelSpec& spec) {
  const auto elements = spec.input_shape.element_count();
  if (elements && *elements >= static_cast<std::size_t>(640) *
                                 static_cast<std::size_t>(640) * 3U) {
    return TimingPolicy{.warmup_iterations = 5, .timed_iterations = 10};
  }
  return TimingPolicy{.warmup_iterations = 10, .timed_iterations = 20};
}

bool sanity_check_enabled() {
  const char* raw = std::getenv("NCNN_PERF_SKIP_SANITY");
  return raw == nullptr || *raw == '\0' || std::strcmp(raw, "0") == 0;
}

// 线程口径与 benchncnn 一致：物理大核数，两侧统一。首次调用时解析并生效，
// NCNN_PERF_THREADS 配置错误直接终止进程。
int resolved_thread_count() {
  static const int threads = [] {
    auto value = resolve_benchmark_thread_count();
    if (!value.has_value()) {
      std::cerr << value.error() << "\n";
      std::exit(1);
    }
    apply_benchncnn_threading(*value);
    return *value;
  }();
  return threads;
}

void run_model_benchmark(const ModelSpec& spec) {
  if (kUnderSanitizer) {
    GTEST_SKIP() << "performance measurements are meaningless under sanitizers";
  }
  const auto input_elements = spec.input_shape.element_count();
  ASSERT_TRUE(input_elements.has_value()) << input_elements.error();
  const std::vector<float> input =
    make_random_input(*input_elements, spec.seed);

  CompiledModel compiled(spec.library_path, spec.symbol);
  ASSERT_TRUE(compiled.valid()) << compiled.error();
  NcnnBenchRunner runner(
    spec.param_path, spec.bin_path, resolved_thread_count());
  ASSERT_TRUE(runner.valid()) << runner.error();

  const bool two_outputs = spec.output_element_counts[1] != 0;
  std::vector<float> first_output(spec.output_element_counts[0]);
  std::vector<float> second_output(spec.output_element_counts[1]);
  auto compiled_inference = [&]() -> int {
    if (!two_outputs) {
      return compiled.run(input, first_output);
    }
    return compiled.run_two_outputs(input, first_output, second_output);
  };

  const ReferenceInput reference_input(
    spec.input_blob, spec.input_shape, input);
  const auto reference_names =
    std::span(spec.output_blobs).first(two_outputs ? 2U : 1U);
  std::array<std::vector<float>, 2> reference_outputs{
    std::vector<float>(spec.output_element_counts[0]),
    std::vector<float>(spec.output_element_counts[1])};
  const auto reference_buffers =
    std::span(reference_outputs).first(two_outputs ? 2U : 1U);
  auto reference_inference = [&]() -> int {
    return runner.run(
      std::span(&reference_input, 1), reference_names, reference_buffers);
  };

  // 宽松数值兜底：只拦"计时了错误产物"这类粗错，精确容差归 golden 测试管。
  if (sanity_check_enabled()) {
    ASSERT_EQ(reference_inference(), 0) << runner.error();
    ASSERT_EQ(compiled_inference(), 0);
    EXPECT_TRUE(
      compare_values(first_output, reference_outputs[0], 5.0e-3F, 5.0e-3F))
      << spec.name << ": compiled output diverges from ncnn reference";
    if (two_outputs) {
      EXPECT_TRUE(
        compare_values(second_output, reference_outputs[1], 5.0e-3F, 5.0e-3F))
        << spec.name << ": compiled second output diverges";
    }
  }

  auto policy = resolve_timing_policy(default_timing_policy(spec));
  ASSERT_TRUE(policy.has_value()) << policy.error();

  PairBenchmarkResult result;
  const auto ncnn_stats = time_repeated_inference(reference_inference, *policy);
  ASSERT_TRUE(ncnn_stats.has_value()) << ncnn_stats.error();
  result.ncnn = *ncnn_stats;
  const auto compiled_stats =
    time_repeated_inference(compiled_inference, *policy);
  ASSERT_TRUE(compiled_stats.has_value()) << compiled_stats.error();
  result.compiled = *compiled_stats;
  result.ratio = result.ncnn.mean_ms > 0.0
                   ? result.compiled.mean_ms / result.ncnn.mean_ms
                   : 0.0;

  emit_performance_report(spec.name, resolved_thread_count(), *policy, result);
  append_performance_json_record(
    spec.name, resolved_thread_count(), *policy, result);
  EXPECT_TRUE(check_performance_gate(spec.name, result));
}

TEST(PerformanceModel, SqueezeNetV11) {
  run_model_benchmark(ModelSpec{
    .name = "squeezenet_v1_1",
    .param_path = SQUEEZENET_PARAM_PATH,
    .bin_path = SQUEEZENET_BIN_PATH,
    .library_path = SQUEEZENET_LIBRARY_PATH,
    .symbol = "squeezenet_v1_1",
    .input_blob = "data",
    .output_blobs = {"prob"},
    .input_shape = TensorShape(227, 227, 3),
    .output_element_counts = {1000},
    .seed = 0x53515545U,
  });
}

TEST(PerformanceModel, ResNet18) {
  run_model_benchmark(ModelSpec{
    .name = "resnet18",
    .param_path = RESNET18_PARAM_PATH,
    .bin_path = RESNET18_BIN_PATH,
    .library_path = RESNET18_LIBRARY_PATH,
    .symbol = "resnet18",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x52183518U,
  });
}

TEST(PerformanceModel, ResNet34) {
  run_model_benchmark(ModelSpec{
    .name = "resnet34",
    .param_path = RESNET34_PARAM_PATH,
    .bin_path = RESNET34_BIN_PATH,
    .library_path = RESNET34_LIBRARY_PATH,
    .symbol = "resnet34",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x52333434U,
  });
}

TEST(PerformanceModel, ResNet50) {
  run_model_benchmark(ModelSpec{
    .name = "resnet50",
    .param_path = RESNET50_PARAM_PATH,
    .bin_path = RESNET50_BIN_PATH,
    .library_path = RESNET50_LIBRARY_PATH,
    .symbol = "resnet50",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x52353030U,
  });
}

TEST(PerformanceModel, ResNet101) {
  run_model_benchmark(ModelSpec{
    .name = "resnet101",
    .param_path = RESNET101_PARAM_PATH,
    .bin_path = RESNET101_BIN_PATH,
    .library_path = RESNET101_LIBRARY_PATH,
    .symbol = "resnet101",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x52313031U,
  });
}

TEST(PerformanceModel, Yolov5nCls) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5n_cls",
    .param_path = YOLOV5N_CLS_PARAM_PATH,
    .bin_path = YOLOV5N_CLS_BIN_PATH,
    .library_path = YOLOV5N_CLS_LIBRARY_PATH,
    .symbol = "yolov5n_cls",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x59356e63U,
  });
}

TEST(PerformanceModel, Yolov5sCls) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5s_cls",
    .param_path = YOLOV5S_CLS_PARAM_PATH,
    .bin_path = YOLOV5S_CLS_BIN_PATH,
    .library_path = YOLOV5S_CLS_LIBRARY_PATH,
    .symbol = "yolov5s_cls",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x59357363U,
  });
}

TEST(PerformanceModel, Yolov5mCls) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5m_cls",
    .param_path = YOLOV5M_CLS_PARAM_PATH,
    .bin_path = YOLOV5M_CLS_BIN_PATH,
    .library_path = YOLOV5M_CLS_LIBRARY_PATH,
    .symbol = "yolov5m_cls",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x59356d63U,
  });
}

TEST(PerformanceModel, Yolov5lCls) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5l_cls",
    .param_path = YOLOV5L_CLS_PARAM_PATH,
    .bin_path = YOLOV5L_CLS_BIN_PATH,
    .library_path = YOLOV5L_CLS_LIBRARY_PATH,
    .symbol = "yolov5l_cls",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x59356c63U,
  });
}

TEST(PerformanceModel, Yolov5xCls) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5x_cls",
    .param_path = YOLOV5X_CLS_PARAM_PATH,
    .bin_path = YOLOV5X_CLS_BIN_PATH,
    .library_path = YOLOV5X_CLS_LIBRARY_PATH,
    .symbol = "yolov5x_cls",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x59357863U,
  });
}

TEST(PerformanceModel, Yolov5nDet) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5n",
    .param_path = YOLOV5N_PARAM_PATH,
    .bin_path = YOLOV5N_BIN_PATH,
    .library_path = YOLOV5N_LIBRARY_PATH,
    .symbol = "yolov5n",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetectionElements},
    .seed = 0x5930356EU,
  });
}

TEST(PerformanceModel, Yolov5sDet) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5s",
    .param_path = YOLOV5S_PARAM_PATH,
    .bin_path = YOLOV5S_BIN_PATH,
    .library_path = YOLOV5S_LIBRARY_PATH,
    .symbol = "yolov5s",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetectionElements},
    .seed = 0x59357373U,
  });
}

TEST(PerformanceModel, Yolov5mDet) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5m",
    .param_path = YOLOV5M_PARAM_PATH,
    .bin_path = YOLOV5M_BIN_PATH,
    .library_path = YOLOV5M_LIBRARY_PATH,
    .symbol = "yolov5m",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetectionElements},
    .seed = 0x59356D6DU,
  });
}

TEST(PerformanceModel, Yolov5lDet) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5l",
    .param_path = YOLOV5L_PARAM_PATH,
    .bin_path = YOLOV5L_BIN_PATH,
    .library_path = YOLOV5L_LIBRARY_PATH,
    .symbol = "yolov5l",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetectionElements},
    .seed = 0x59356C6CU,
  });
}

TEST(PerformanceModel, Yolov5xDet) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5x",
    .param_path = YOLOV5X_PARAM_PATH,
    .bin_path = YOLOV5X_BIN_PATH,
    .library_path = YOLOV5X_LIBRARY_PATH,
    .symbol = "yolov5x",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetectionElements},
    .seed = 0x59357878U,
  });
}

TEST(PerformanceModel, EfficientNetB0) {
  run_model_benchmark(ModelSpec{
    .name = "efficientnet_b0",
    .param_path = EFFICIENTNET_B0_PARAM_PATH,
    .bin_path = EFFICIENTNET_B0_BIN_PATH,
    .library_path = EFFICIENTNET_B0_LIBRARY_PATH,
    .symbol = "efficientnet_b0",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {1000},
    .seed = 0x454e4230U,
  });
}

TEST(PerformanceModel, EfficientNetB1) {
  run_model_benchmark(ModelSpec{
    .name = "efficientnet_b1",
    .param_path = EFFICIENTNET_B1_PARAM_PATH,
    .bin_path = EFFICIENTNET_B1_BIN_PATH,
    .library_path = EFFICIENTNET_B1_LIBRARY_PATH,
    .symbol = "efficientnet_b1",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(240, 240, 3),
    .output_element_counts = {1000},
    .seed = 0x454e4231U,
  });
}

TEST(PerformanceModel, EfficientNetB2) {
  run_model_benchmark(ModelSpec{
    .name = "efficientnet_b2",
    .param_path = EFFICIENTNET_B2_PARAM_PATH,
    .bin_path = EFFICIENTNET_B2_BIN_PATH,
    .library_path = EFFICIENTNET_B2_LIBRARY_PATH,
    .symbol = "efficientnet_b2",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(260, 260, 3),
    .output_element_counts = {1000},
    .seed = 0x454e4232U,
  });
}

TEST(PerformanceModel, EfficientNetB3) {
  run_model_benchmark(ModelSpec{
    .name = "efficientnet_b3",
    .param_path = EFFICIENTNET_B3_PARAM_PATH,
    .bin_path = EFFICIENTNET_B3_BIN_PATH,
    .library_path = EFFICIENTNET_B3_LIBRARY_PATH,
    .symbol = "efficientnet_b3",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(300, 300, 3),
    .output_element_counts = {1000},
    .seed = 0x454e4233U,
  });
}

TEST(PerformanceModel, PPLCNetDocOri) {
  run_model_benchmark(ModelSpec{
    .name = "pp_lcnet_x1_0_doc_ori",
    .param_path = PP_LCNET_DOC_ORI_PARAM_PATH,
    .bin_path = PP_LCNET_DOC_ORI_BIN_PATH,
    .library_path = PP_LCNET_DOC_ORI_LIBRARY_PATH,
    .symbol = "pp_lcnet_x1_0_doc_ori",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {4},
    .seed = 0x4C434E45U,
  });
}

TEST(PerformanceModel, PPLCNetTextlineOri) {
  run_model_benchmark(ModelSpec{
    .name = "pp_lcnet_x1_0_textline_ori",
    .param_path = PP_LCNET_TEXTLINE_ORI_PARAM_PATH,
    .bin_path = PP_LCNET_TEXTLINE_ORI_BIN_PATH,
    .library_path = PP_LCNET_TEXTLINE_ORI_LIBRARY_PATH,
    .symbol = "pp_lcnet_x1_0_textline_ori",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(160, 80, 3),
    .output_element_counts = {2},
    .seed = 0x54455854U,
  });
}

TEST(PerformanceModel, ChineseOCRLiteAngleNet) {
  run_model_benchmark(ModelSpec{
    .name = "chineseocr_lite_anglenet",
    .param_path = CHINESEOCR_LITE_ANGLENET_PARAM_PATH,
    .bin_path = CHINESEOCR_LITE_ANGLENET_BIN_PATH,
    .library_path = CHINESEOCR_LITE_ANGLENET_LIBRARY_PATH,
    .symbol = "chineseocr_lite_anglenet",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(192, 32, 3),
    .output_element_counts = {2},
    .seed = 0x414E474CU,
  });
}

// seg 双输出：blob 序 out1(proto)/out0(detections)，与
// yolov5n_seg_test.cpp 的数值测试保持一致。
TEST(PerformanceModel, Yolov5nSeg) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5n_seg",
    .param_path = YOLOV5N_SEG_PARAM_PATH,
    .bin_path = YOLOV5N_SEG_BIN_PATH,
    .library_path = YOLOV5N_SEG_LIBRARY_PATH,
    .symbol = "yolov5n_seg",
    .input_blob = "in0",
    .output_blobs = {"out1", "out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kSegProtoElements, kSegDetectionElements},
    .seed = 0x53454731U,
  });
}

TEST(PerformanceModel, Yolov5sSeg) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5s_seg",
    .param_path = YOLOV5S_SEG_PARAM_PATH,
    .bin_path = YOLOV5S_SEG_BIN_PATH,
    .library_path = YOLOV5S_SEG_LIBRARY_PATH,
    .symbol = "yolov5s_seg",
    .input_blob = "in0",
    .output_blobs = {"out1", "out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kSegProtoElements, kSegDetectionElements},
    .seed = 0x53454732U,
  });
}

TEST(PerformanceModel, Yolov5mSeg) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5m_seg",
    .param_path = YOLOV5M_SEG_PARAM_PATH,
    .bin_path = YOLOV5M_SEG_BIN_PATH,
    .library_path = YOLOV5M_SEG_LIBRARY_PATH,
    .symbol = "yolov5m_seg",
    .input_blob = "in0",
    .output_blobs = {"out1", "out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kSegProtoElements, kSegDetectionElements},
    .seed = 0x53454733U,
  });
}

TEST(PerformanceModel, Yolov5lSeg) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5l_seg",
    .param_path = YOLOV5L_SEG_PARAM_PATH,
    .bin_path = YOLOV5L_SEG_BIN_PATH,
    .library_path = YOLOV5L_SEG_LIBRARY_PATH,
    .symbol = "yolov5l_seg",
    .input_blob = "in0",
    .output_blobs = {"out1", "out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kSegProtoElements, kSegDetectionElements},
    .seed = 0x53454734U,
  });
}

TEST(PerformanceModel, Yolov5xSeg) {
  run_model_benchmark(ModelSpec{
    .name = "yolov5x_seg",
    .param_path = YOLOV5X_SEG_PARAM_PATH,
    .bin_path = YOLOV5X_SEG_BIN_PATH,
    .library_path = YOLOV5X_SEG_LIBRARY_PATH,
    .symbol = "yolov5x_seg",
    .input_blob = "in0",
    .output_blobs = {"out1", "out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kSegProtoElements, kSegDetectionElements},
    .seed = 0x53454735U,
  });
}

}  // namespace
}  // namespace ncnn_compiler::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
