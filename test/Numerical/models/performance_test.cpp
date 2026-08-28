// 性能基准（performance 标签）：编译产物 vs upstream ncnn 的墙钟对比。
// 复用数值金标的同一批 compile_<model> fixture，不重复编译任何模型；
// 数值正确性由 numerical_tests 保证，这里只做一次宽松 sanity 兜底
// （防止计时了陈旧/错误的 .so），然后双侧各计时并输出加速比。

#include "numerical_test_support.hpp"
#include "performance_test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

// OCR rec 输出 = 40 帧序列 × 类别数（序列长度 = (输入宽 + 3) / 8）。
constexpr std::size_t kRecSequenceLength = 40;
constexpr std::size_t kTinyRecClasses = 6906;
constexpr std::size_t kMobileRecClasses = 18385;
constexpr std::size_t kServerRecClasses = 18385;
constexpr std::size_t kMediumRecClasses = 18710;
constexpr std::size_t kSmallRecClasses = 18710;
constexpr std::size_t kDetMapElements = 640U * 640U;
constexpr std::size_t kDetInt8MapElements = 32U * 32U;
constexpr std::size_t kSlanetCnnElements = 256U * 96U;
constexpr std::size_t kFormulaEncoderElements = 144U * 2048U;

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
  // int8 产物行用 Int8 参考（双侧同模式计时才有意义）。
  ReferenceInferenceMode reference_mode = ReferenceInferenceMode::Float32;
  // int8 金标契约只做稳定性校验、无跨厂商交叉对比，这些行以有限域检查替代
  // compare_values 宽松对比。
  bool verify_against_reference = true;
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
  NcnnBenchRunner runner(spec.param_path,
                         spec.bin_path,
                         resolved_thread_count(),
                         spec.reference_mode);
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
    if (spec.verify_against_reference) {
      EXPECT_TRUE(
        compare_values(first_output, reference_outputs[0], 5.0e-3F, 5.0e-3F))
        << spec.name << ": compiled output diverges from ncnn reference";
      if (two_outputs) {
        EXPECT_TRUE(
          compare_values(second_output, reference_outputs[1], 5.0e-3F, 5.0e-3F))
          << spec.name << ": compiled second output diverges";
      }
    } else {
      const auto finite = [](float value) {
        return std::isfinite(value);
      };
      EXPECT_TRUE(std::ranges::all_of(first_output, finite))
        << spec.name << ": compiled output is not finite";
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

TEST(PerformanceModel, PPLCNetDocOriInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_lcnet_x1_0_doc_ori_int8",
    .param_path = PP_LCNET_DOC_ORI_INT8_PARAM_PATH,
    .bin_path = PP_LCNET_DOC_ORI_INT8_BIN_PATH,
    .library_path = PP_LCNET_DOC_ORI_INT8_LIBRARY_PATH,
    .symbol = "pp_lcnet_x1_0_doc_ori_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(224, 224, 3),
    .output_element_counts = {4},
    .seed = 0x4C434938U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPLCNetTextlineOriInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_lcnet_x1_0_textline_ori_int8",
    .param_path = PP_LCNET_TEXTLINE_ORI_INT8_PARAM_PATH,
    .bin_path = PP_LCNET_TEXTLINE_ORI_INT8_BIN_PATH,
    .library_path = PP_LCNET_TEXTLINE_ORI_INT8_LIBRARY_PATH,
    .symbol = "pp_lcnet_x1_0_textline_ori_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(160, 80, 3),
    .output_element_counts = {2},
    .seed = 0x544C4938U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPOcrv6TinyRec) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_tiny_rec",
    .param_path = PP_OCRV6_TINY_REC_PARAM_PATH,
    .bin_path = PP_OCRV6_TINY_REC_BIN_PATH,
    .library_path = PP_OCRV6_TINY_REC_LIBRARY_PATH,
    .symbol = "pp_ocrv6_tiny_rec",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kTinyRecClasses},
    .seed = 0x4F435236U,
  });
}

TEST(PerformanceModel, PPOcrv6TinyRecInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_tiny_rec_int8",
    .param_path = PP_OCRV6_TINY_REC_INT8_PARAM_PATH,
    .bin_path = PP_OCRV6_TINY_REC_INT8_BIN_PATH,
    .library_path = PP_OCRV6_TINY_REC_INT8_LIBRARY_PATH,
    .symbol = "pp_ocrv6_tiny_rec_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kTinyRecClasses},
    .seed = 0x36544938U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPOcrv5MobileRec) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv5_mobile_rec",
    .param_path = PP_OCRV5_MOBILE_REC_PARAM_PATH,
    .bin_path = PP_OCRV5_MOBILE_REC_BIN_PATH,
    .library_path = PP_OCRV5_MOBILE_REC_LIBRARY_PATH,
    .symbol = "pp_ocrv5_mobile_rec",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kMobileRecClasses},
    .seed = 0x354D5245U,
  });
}

TEST(PerformanceModel, PPOcrv5MobileRecInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv5_mobile_rec_int8",
    .param_path = PP_OCRV5_MOBILE_REC_INT8_PARAM_PATH,
    .bin_path = PP_OCRV5_MOBILE_REC_INT8_BIN_PATH,
    .library_path = PP_OCRV5_MOBILE_REC_INT8_LIBRARY_PATH,
    .symbol = "pp_ocrv5_mobile_rec_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kMobileRecClasses},
    .seed = 0x354D4938U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPOcrv5ServerRec) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv5_server_rec",
    .param_path = PP_OCRV5_SERVER_REC_PARAM_PATH,
    .bin_path = PP_OCRV5_SERVER_REC_BIN_PATH,
    .library_path = PP_OCRV5_SERVER_REC_LIBRARY_PATH,
    .symbol = "pp_ocrv5_server_rec",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kServerRecClasses},
    .seed = 0x35535245U,
  });
}

TEST(PerformanceModel, PPOcrv6MediumRec) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_medium_rec",
    .param_path = PP_OCRV6_MEDIUM_REC_PARAM_PATH,
    .bin_path = PP_OCRV6_MEDIUM_REC_BIN_PATH,
    .library_path = PP_OCRV6_MEDIUM_REC_LIBRARY_PATH,
    .symbol = "pp_ocrv6_medium_rec",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kMediumRecClasses},
    .seed = 0x364D5245U,
  });
}

TEST(PerformanceModel, PPOcrv6MediumRecInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_medium_rec_int8",
    .param_path = PP_OCRV6_MEDIUM_REC_INT8_PARAM_PATH,
    .bin_path = PP_OCRV6_MEDIUM_REC_INT8_BIN_PATH,
    .library_path = PP_OCRV6_MEDIUM_REC_INT8_LIBRARY_PATH,
    .symbol = "pp_ocrv6_medium_rec_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kMediumRecClasses},
    .seed = 0x364D4938U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPOcrv6SmallRec) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_small_rec",
    .param_path = PP_OCRV6_SMALL_REC_PARAM_PATH,
    .bin_path = PP_OCRV6_SMALL_REC_BIN_PATH,
    .library_path = PP_OCRV6_SMALL_REC_LIBRARY_PATH,
    .symbol = "pp_ocrv6_small_rec",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kSmallRecClasses},
    .seed = 0x36534D45U,
  });
}

TEST(PerformanceModel, PPOcrv6SmallRecInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_small_rec_int8",
    .param_path = PP_OCRV6_SMALL_REC_INT8_PARAM_PATH,
    .bin_path = PP_OCRV6_SMALL_REC_INT8_BIN_PATH,
    .library_path = PP_OCRV6_SMALL_REC_INT8_LIBRARY_PATH,
    .symbol = "pp_ocrv6_small_rec_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(320, 48, 3),
    .output_element_counts = {kRecSequenceLength * kMediumRecClasses},
    // 该 fixture 无专属金标测试，种子为新增固定值。
    .seed = 0x36534D49U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPOcrv6TinyDet) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_tiny_det",
    .param_path = PP_OCRV6_TINY_DET_PARAM_PATH,
    .bin_path = PP_OCRV6_TINY_DET_BIN_PATH,
    .library_path = PP_OCRV6_TINY_DET_LIBRARY_PATH,
    .symbol = "pp_ocrv6_tiny_det",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetMapElements},
    .seed = 0x44455436U,
  });
}

TEST(PerformanceModel, PPOcrv6SmallDet) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_small_det",
    .param_path = PP_OCRV6_SMALL_DET_PARAM_PATH,
    .bin_path = PP_OCRV6_SMALL_DET_BIN_PATH,
    .library_path = PP_OCRV6_SMALL_DET_LIBRARY_PATH,
    .symbol = "pp_ocrv6_small_det",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetMapElements},
    .seed = 0x53444554U,
  });
}

TEST(PerformanceModel, PPOcrv6MediumDet) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_medium_det",
    .param_path = PP_OCRV6_MEDIUM_DET_PARAM_PATH,
    .bin_path = PP_OCRV6_MEDIUM_DET_BIN_PATH,
    .library_path = PP_OCRV6_MEDIUM_DET_LIBRARY_PATH,
    .symbol = "pp_ocrv6_medium_det",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetMapElements},
    .seed = 0x4D444554U,
  });
}

// det_int8 系列的上游 ncnn 参考在独立最小复现下即崩溃（FP32 全崩，Int8 仅
// medium 存活，见 docs/ncnn-suspected-issues.md 第 4 节），tiny/small/mobile
// 三个 det_int8 无法提供任何参考侧计时，暂无对应行；medium_det_int8 保留
// Int8 双侧同模式对比。
TEST(PerformanceModel, PPOcrv6MediumDetInt8) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv6_medium_det_int8",
    .param_path = PP_OCRV6_MEDIUM_DET_INT8_PARAM_PATH,
    .bin_path = PP_OCRV6_MEDIUM_DET_INT8_BIN_PATH,
    .library_path = PP_OCRV6_MEDIUM_DET_INT8_LIBRARY_PATH,
    .symbol = "pp_ocrv6_medium_det_int8",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(32, 32, 3),
    .output_element_counts = {kDetInt8MapElements},
    .seed = 0x49384D44U,
    .reference_mode = ReferenceInferenceMode::Int8,
    .verify_against_reference = false,
  });
}

TEST(PerformanceModel, PPOcrv5MobileDetStatic) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv5_mobile_det_static",
    .param_path = PP_OCRV5_MOBILE_DET_PARAM_PATH,
    .bin_path = PP_OCRV5_MOBILE_DET_BIN_PATH,
    .library_path = PP_OCRV5_MOBILE_DET_LIBRARY_PATH,
    .symbol = "pp_ocrv5_mobile_det_static",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetMapElements},
    .seed = 0x354D4445U,
  });
}

TEST(PerformanceModel, PPOcrv5ServerDetStatic) {
  run_model_benchmark(ModelSpec{
    .name = "pp_ocrv5_server_det_static",
    .param_path = PP_OCRV5_SERVER_DET_PARAM_PATH,
    .bin_path = PP_OCRV5_SERVER_DET_BIN_PATH,
    .library_path = PP_OCRV5_SERVER_DET_STATIC_LIBRARY_PATH,
    .symbol = "pp_ocrv5_server_det_static",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(640, 640, 3),
    .output_element_counts = {kDetMapElements},
    .seed = 0x35534445U,
  });
}

TEST(PerformanceModel, PPStructrureV2SlanetPlusCnn) {
  run_model_benchmark(ModelSpec{
    .name = "pp_structrurev2_slanet_plus_cnn",
    .param_path = PP_STRUCTRUREV2_SLANET_PLUS_CNN_PARAM_PATH,
    .bin_path = PP_STRUCTRUREV2_SLANET_PLUS_CNN_BIN_PATH,
    .library_path = PP_STRUCTRUREV2_SLANET_PLUS_CNN_LIBRARY_PATH,
    .symbol = "pp_structrurev2_slanet_plus_cnn",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(488, 488, 3),
    .output_element_counts = {kSlanetCnnElements},
    .seed = 0x534C414EU,
  });
}

TEST(PerformanceModel, PPFormulaNetPlusSEncoder) {
  run_model_benchmark(ModelSpec{
    .name = "pp_formulanet_plus_s_encoder",
    .param_path = PP_FORMULANET_PLUS_S_ENCODER_PARAM_PATH,
    .bin_path = PP_FORMULANET_PLUS_S_ENCODER_BIN_PATH,
    .library_path = PP_FORMULANET_PLUS_S_ENCODER_LIBRARY_PATH,
    .symbol = "pp_formulanet_plus_s_encoder",
    .input_blob = "in0",
    .output_blobs = {"out0"},
    .input_shape = TensorShape(384, 384, 1),
    .output_element_counts = {kFormulaEncoderElements},
    .seed = 0x464F524DU,
  });
}

}  // namespace
}  // namespace ncnn_compiler::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
