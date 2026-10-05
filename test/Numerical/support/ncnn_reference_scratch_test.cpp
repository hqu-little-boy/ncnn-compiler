#include <layer.h>
#include <mat.h>
#include <modelbin.h>
#include <option.h>
#include <paramdict.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace {

enum class Storage { Float32, BFloat16 };

void runConvolution(
  int channels, int width, bool naive, Storage storage, ncnn::Mat& output) {
  std::unique_ptr<ncnn::Layer> layer(naive
                                       ? ncnn::create_layer_naive("Convolution")
                                       : ncnn::create_layer_cpu("Convolution"));
  ASSERT_NE(layer, nullptr);
  if (storage == Storage::BFloat16) {
    ASSERT_TRUE(layer->support_bf16_storage);
  }
  ncnn::ParamDict parameters;
  parameters.set(0, 3);
  parameters.set(1, 1);
  parameters.set(5, 1);
  parameters.set(6, channels * 3);
  ASSERT_EQ(layer->load_param(parameters), 0);

  ncnn::Mat weights(channels * 3);
  ncnn::Mat bias(3);
  ncnn::Mat input(width, 1, channels, sizeof(float), 1);
  ASSERT_FALSE(weights.empty());
  ASSERT_FALSE(bias.empty());
  ASSERT_FALSE(input.empty());
  for (int channel = 0; channel < channels; ++channel) {
    ncnn::Mat view = input.channel(channel);
    float* row = view.row(0);
    for (int x = 0; x < width; ++x) {
      row[x] = static_cast<float>(((channel % 7) - 3) * ((x % 5) + 1)) / 128.0F;
    }
    for (int outputChannel = 0; outputChannel < 3; ++outputChannel) {
      weights[(outputChannel * channels) + channel] =
        static_cast<float>(((channel + outputChannel) % 5) - 2) / 32.0F;
    }
  }
  for (int channel = 0; channel < 3; ++channel) {
    bias[channel] = static_cast<float>(channel - 1) / 8.0F;
  }
  const ncnn::Mat payloads[] = {weights, bias};
  const ncnn::ModelBinFromMatArray model(payloads);
  ASSERT_EQ(layer->load_model(model), 0);

  ncnn::Option options;
  options.num_threads = 1;
  options.use_packing_layout = false;
  options.use_fp16_packed = false;
  options.use_fp16_storage = false;
  options.use_fp16_arithmetic = false;
  options.use_bf16_storage = storage == Storage::BFloat16;
  options.use_bf16_packed = false;
  ncnn::Mat layerInput = input;
  if (storage == Storage::BFloat16) {
    ncnn::cast_float32_to_bfloat16(input, layerInput, options);
    ASSERT_FALSE(layerInput.empty());
    ASSERT_EQ(layerInput.elembits(), 16);
  }
  ASSERT_EQ(layer->create_pipeline(options), 0);
  const int forwardStatus = layer->forward(layerInput, output, options);
  const int destroyStatus = layer->destroy_pipeline(options);
  ASSERT_EQ(forwardStatus, 0);
  ASSERT_EQ(destroyStatus, 0);
}

void compareConvolution(int channels, int width, Storage storage) {
  SCOPED_TRACE(::testing::Message()
               << "K=" << channels << ", width=" << width
               << ", BF16=" << (storage == Storage::BFloat16));
  ncnn::Mat optimized;
  ncnn::Mat scalar;
  runConvolution(channels, width, false, storage, optimized);
  runConvolution(channels, width, true, Storage::Float32, scalar);
  ASSERT_FALSE(optimized.empty());
  ASSERT_FALSE(scalar.empty());
  ASSERT_EQ(optimized.w, width);
  ASSERT_EQ(optimized.h, 1);
  ASSERT_EQ(optimized.c, 3);
  ASSERT_EQ(optimized.elempack, 1);
  ASSERT_EQ(scalar.w, optimized.w);
  ASSERT_EQ(scalar.h, optimized.h);
  ASSERT_EQ(scalar.c, optimized.c);
  ASSERT_EQ(scalar.elembits(), 32);
  ncnn::Mat expected = scalar;
  if (storage == Storage::BFloat16) {
    // All input/weight values are exactly BF16-representable dyadics. Round
    // only the independent FP32 scalar result to the requested output storage.
    ncnn::Option options;
    options.num_threads = 1;
    options.use_packing_layout = false;
    ncnn::cast_float32_to_bfloat16(scalar, expected, options);
    ASSERT_FALSE(expected.empty());
    ASSERT_EQ(expected.elembits(), 16);
    ASSERT_EQ(optimized.elembits(), 16);
  } else {
    ASSERT_EQ(optimized.elembits(), 32);
  }
  for (int channel = 0; channel < 3; ++channel) {
    const ncnn::Mat optimizedView = optimized.channel(channel);
    const ncnn::Mat expectedView = expected.channel(channel);
    if (storage == Storage::BFloat16) {
      const auto* actual = optimizedView.row<std::uint16_t>(0);
      const auto* reference = expectedView.row<std::uint16_t>(0);
      for (int x = 0; x < width; ++x) {
        EXPECT_EQ(actual[x], reference[x])
          << "channel=" << channel << ", x=" << x;
      }
    } else {
      const float* actual = optimizedView.row(0);
      const float* reference = expectedView.row(0);
      for (int x = 0; x < width; ++x) {
        EXPECT_EQ(actual[x], reference[x])
          << "channel=" << channel << ", x=" << x;
      }
    }
  }
}

TEST(NcnnReferenceScratchTest, SmallKDoesNotAdvanceAbsentScratch) {
  for (int width : {1, 2, 4, 7, 17}) {
    compareConvolution(1, width, Storage::Float32);
  }
}

TEST(NcnnReferenceScratchTest, SplitKPreservesPartialSumsAndTails) {
  // Force multiple cache-derived K tiles on the gate host, plus a width tail.
  // Exact dyadic inputs avoid making this dependency test an ISA tolerance
  // test.
  compareConvolution(500000, 17, Storage::Float32);
}

TEST(NcnnReferenceScratchTest, BFloat16SmallKDoesNotAdvanceAbsentScratch) {
  for (int width : {1, 2, 4, 7, 17}) {
    compareConvolution(1, width, Storage::BFloat16);
  }
}

TEST(NcnnReferenceScratchTest, BFloat16SplitKPreservesPartialSumsAndTails) {
  compareConvolution(500000, 17, Storage::BFloat16);
}

enum class GemmCBroadcast { None, Scalar, MVector, NVector, MColumn, Full };

struct GemmCase {
  GemmCase(int mValue,
           int nValue,
           int kValue,
           bool transAValue,
           bool transBValue,
           bool outputTransposeValue,
           GemmCBroadcast cBroadcastValue,
           const char* nameValue)
    : m(mValue),
      n(nValue),
      k(kValue),
      transA(transAValue),
      transB(transBValue),
      outputTranspose(outputTransposeValue),
      cBroadcast(cBroadcastValue),
      name(nameValue) {}

  int m;
  int n;
  int k;
  bool transA;
  bool transB;
  bool outputTranspose;
  GemmCBroadcast cBroadcast;
  const char* name;
};

float gemmAValue(int row, int column) {
  return static_cast<float>(((row + 2 * column) % 5) - 2) / 64.0F;
}

float gemmBValue(int row, int column) {
  return static_cast<float>(((row + 3 * column) % 5) - 2) / 64.0F;
}

float gemmCValue(int row, int column) {
  return static_cast<float>(((3 * row + column) % 5) - 2) / 16.0F;
}

ncnn::Mat makeGemmA(const GemmCase& test) {
  ncnn::Mat a(test.transA ? test.m : test.k, test.transA ? test.k : test.m);
  for (int row = 0; row < test.m; ++row) {
    for (int column = 0; column < test.k; ++column) {
      if (test.transA) {
        a.row(column)[row] = gemmAValue(row, column);
      } else {
        a.row(row)[column] = gemmAValue(row, column);
      }
    }
  }
  return a;
}

ncnn::Mat makeGemmB(const GemmCase& test) {
  ncnn::Mat b(test.transB ? test.k : test.n, test.transB ? test.n : test.k);
  for (int row = 0; row < test.k; ++row) {
    for (int column = 0; column < test.n; ++column) {
      if (test.transB) {
        b.row(column)[row] = gemmBValue(row, column);
      } else {
        b.row(row)[column] = gemmBValue(row, column);
      }
    }
  }
  return b;
}

ncnn::Mat makeGemmC(const GemmCase& test) {
  switch (test.cBroadcast) {
    case GemmCBroadcast::None:
      return {};
    case GemmCBroadcast::Scalar: {
      ncnn::Mat c(1);
      c[0] = 3.0F / 16.0F;
      return c;
    }
    case GemmCBroadcast::MVector: {
      ncnn::Mat c(test.m);
      for (int row = 0; row < test.m; ++row) {
        c[row] = gemmCValue(row, 0);
      }
      return c;
    }
    case GemmCBroadcast::NVector: {
      ncnn::Mat c(test.n);
      for (int column = 0; column < test.n; ++column) {
        c[column] = gemmCValue(0, column);
      }
      return c;
    }
    case GemmCBroadcast::MColumn: {
      ncnn::Mat c(1, test.m);
      for (int row = 0; row < test.m; ++row) {
        c.row(row)[0] = gemmCValue(row, 0);
      }
      return c;
    }
    case GemmCBroadcast::Full: {
      ncnn::Mat c(test.n, test.m);
      for (int row = 0; row < test.m; ++row) {
        for (int column = 0; column < test.n; ++column) {
          c.row(row)[column] = gemmCValue(row, column);
        }
      }
      return c;
    }
  }
  return {};
}

void runGemm(const GemmCase& test, bool naive, ncnn::Mat& output) {
  std::unique_ptr<ncnn::Layer> layer(naive ? ncnn::create_layer_naive("Gemm")
                                           : ncnn::create_layer_cpu("Gemm"));
  ASSERT_NE(layer, nullptr);

  ncnn::ParamDict parameters;
  parameters.set(0, 0.75F);
  parameters.set(1, 0.5F);
  parameters.set(2, test.transA ? 1 : 0);
  parameters.set(3, test.transB ? 1 : 0);
  parameters.set(14, test.outputTranspose ? 1 : 0);
  ASSERT_EQ(layer->load_param(parameters), 0);

  std::vector<ncnn::Mat> inputs{makeGemmA(test), makeGemmB(test)};
  ncnn::Mat c = makeGemmC(test);
  if (!c.empty()) {
    inputs.push_back(c);
  }
  ASSERT_FALSE(inputs[0].empty());
  ASSERT_FALSE(inputs[1].empty());

  ncnn::Option options;
  options.num_threads = 1;
  options.use_packing_layout = false;
  options.use_fp16_packed = false;
  options.use_fp16_storage = false;
  options.use_fp16_arithmetic = false;
  options.use_bf16_storage = false;
  options.use_bf16_packed = false;
  ASSERT_EQ(layer->create_pipeline(options), 0);
  std::vector<ncnn::Mat> outputs(1);
  const int forwardStatus = layer->forward(inputs, outputs, options);
  const int destroyStatus = layer->destroy_pipeline(options);
  ASSERT_EQ(forwardStatus, 0);
  ASSERT_EQ(destroyStatus, 0);
  ASSERT_FALSE(outputs[0].empty());
  output = outputs[0];
}

void compareGemm(const GemmCase& test) {
  SCOPED_TRACE(test.name);
  ncnn::Mat optimized;
  ncnn::Mat scalar;
  runGemm(test, false, optimized);
  runGemm(test, true, scalar);
  ASSERT_EQ(optimized.dims, scalar.dims);
  ASSERT_EQ(optimized.w, scalar.w);
  ASSERT_EQ(optimized.h, scalar.h);
  ASSERT_EQ(optimized.c, scalar.c);
  ASSERT_EQ(optimized.elembits(), 32);
  ASSERT_EQ(scalar.elembits(), 32);
  ASSERT_EQ(optimized.elempack, 1);
  ASSERT_EQ(scalar.elempack, 1);
  ASSERT_EQ(optimized.total(), scalar.total());

  for (int row = 0; row < optimized.h; ++row) {
    const float* actual = optimized.row(row);
    const float* expected = scalar.row(row);
    for (int column = 0; column < optimized.w; ++column) {
      EXPECT_TRUE(std::isfinite(actual[column]));
      EXPECT_TRUE(std::isfinite(expected[column]));
      EXPECT_EQ(actual[column], expected[column])
        << "row=" << row << ", column=" << column;
    }
  }
}

TEST(NcnnReferenceScratchTest, GemmNullScratchLayoutsAndSplitK) {
  const GemmCase cases[] = {
    {9, 17, 1, false, false, false, GemmCBroadcast::None, "small-null-tail"},
    {9, 17, 1, false, false, false, GemmCBroadcast::Scalar, "small-scalar-C"},
    {9,
     17,
     1,
     false,
     false,
     false,
     GemmCBroadcast::MVector,
     "small-M-vector-C"},
    {9,
     17,
     1,
     false,
     false,
     false,
     GemmCBroadcast::NVector,
     "small-N-vector-C"},
    {9, 17, 1, false, false, false, GemmCBroadcast::MColumn, "small-Mx1-C"},
    {9, 17, 1, true, false, false, GemmCBroadcast::None, "small-transA"},
    {9, 17, 1, false, true, false, GemmCBroadcast::None, "small-transB"},
    {9,
     17,
     1,
     false,
     false,
     true,
     GemmCBroadcast::None,
     "small-output-transpose"},
    {9, 17, 1, false, false, false, GemmCBroadcast::Full, "small-full-C"},
    {8,
     2,
     1,
     false,
     false,
     false,
     GemmCBroadcast::None,
     "small-avx-two-column"},
    {8,
     4,
     1,
     false,
     false,
     false,
     GemmCBroadcast::None,
     "small-avx-four-column"},
    {4,
     2,
     1,
     false,
     false,
     false,
     GemmCBroadcast::None,
     "small-sse-four-row-two-column"},
    {4,
     4,
     1,
     false,
     false,
     false,
     GemmCBroadcast::None,
     "small-sse-four-row-four-column"},
    {2,
     4,
     1,
     false,
     false,
     false,
     GemmCBroadcast::None,
     "small-pair-two-row-four-column"},
    {9, 17, 500000, false, false, false, GemmCBroadcast::None, "splitK-null-C"},
    {9, 17, 500000, false, false, false, GemmCBroadcast::Full, "splitK-full-C"},
    {9, 17, 500000, true, true, false, GemmCBroadcast::None, "splitK-transAB"},
  };
  for (const GemmCase& test : cases) {
    compareGemm(test);
  }
}

}  // namespace
