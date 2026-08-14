#include "ncnn-mlir/Support/Precision.hpp"

#include <gtest/gtest.h>

TEST(PrecisionTest, ParsesSupportedModes) {
  using ncnn_mlir::PrecisionMode;
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("auto"), PrecisionMode::Auto);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("f32"), PrecisionMode::Float32);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("fp16"), PrecisionMode::Float16);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("bf16"), PrecisionMode::BFloat16);
  EXPECT_EQ(ncnn_mlir::parse_precision_mode("int8"), PrecisionMode::Int8);
  EXPECT_FALSE(ncnn_mlir::parse_precision_mode("half"));
  EXPECT_EQ(ncnn_mlir::parse_fp16_accumulator_mode("f16"),
            ncnn_mlir::FP16AccumulatorMode::Float16);
  EXPECT_EQ(ncnn_mlir::parse_fp16_accumulator_mode("f32"),
            ncnn_mlir::FP16AccumulatorMode::Float32);
  EXPECT_FALSE(ncnn_mlir::parse_fp16_accumulator_mode("auto"));
}

TEST(PrecisionTest, ChecksTargetCapabilities) {
  ncnn_mlir::TargetSpec target{.triple = "x86_64-unknown-linux-gnu",
                               .march = "x86-64-v4",
                               .mcpu = "",
                               .features = {"+avx512fp16", "+avx512vnni"}};
  const auto capabilities = ncnn_mlir::infer_target_capabilities(target);
  EXPECT_TRUE(capabilities.fp16_storage);
  EXPECT_TRUE(capabilities.fp16_arithmetic);
  EXPECT_FALSE(capabilities.bf16);
  EXPECT_TRUE(capabilities.int8);
  EXPECT_TRUE(ncnn_mlir::validate_precision_target(
    ncnn_mlir::PrecisionMode::Float16, target));
  EXPECT_FALSE(ncnn_mlir::validate_precision_target(
    ncnn_mlir::PrecisionMode::BFloat16, target));

  target.features.emplace_back("-avx512fp16");
  EXPECT_FALSE(ncnn_mlir::infer_target_capabilities(target).fp16_arithmetic);

  target.features = {"+f16c"};
  EXPECT_TRUE(ncnn_mlir::infer_target_capabilities(target).fp16_storage);
  EXPECT_FALSE(ncnn_mlir::infer_target_capabilities(target).fp16_arithmetic);
  EXPECT_FALSE(
    ncnn_mlir::resolve_precision_policy(ncnn_mlir::PrecisionMode::Float16,
                                        ncnn_mlir::FP16AccumulatorMode::Float16,
                                        false,
                                        target));
  auto fallback =
    ncnn_mlir::resolve_precision_policy(ncnn_mlir::PrecisionMode::Float16,
                                        ncnn_mlir::FP16AccumulatorMode::Float16,
                                        true,
                                        target);
  ASSERT_TRUE(fallback);
  EXPECT_EQ(fallback->fp16_accumulator,
            ncnn_mlir::FP16AccumulatorMode::Float32);
  EXPECT_TRUE(fallback->used_fallback);
  EXPECT_EQ(ncnn_mlir::precision_execution_profile(*fallback, target),
            "x86-64-fp16-storage-fp32");
}

TEST(PrecisionTest, RecognizesArchitectureSpecificProfiles) {
  using ncnn_mlir::FP16AccumulatorMode;
  using ncnn_mlir::PrecisionMode;

  ncnn_mlir::TargetSpec arm{.triple = "aarch64-unknown-linux-gnu",
                            .march = "armv8.6-a+fp16+bf16+i8mm",
                            .mcpu = "",
                            .features = {}};
  auto capabilities = ncnn_mlir::infer_target_capabilities(arm);
  EXPECT_TRUE(capabilities.fp16_storage);
  EXPECT_TRUE(capabilities.fp16_arithmetic);
  EXPECT_TRUE(capabilities.bf16);
  EXPECT_TRUE(capabilities.int8);

  ncnn_mlir::TargetSpec riscv{.triple = "riscv64-unknown-linux-gnu",
                              .march = "rv64gcv_zfh_zvfh",
                              .mcpu = "",
                              .features = {}};
  capabilities = ncnn_mlir::infer_target_capabilities(riscv);
  EXPECT_TRUE(capabilities.fp16_storage);
  EXPECT_TRUE(capabilities.fp16_arithmetic);
  EXPECT_TRUE(capabilities.int8);
  auto policy = ncnn_mlir::resolve_precision_policy(
    PrecisionMode::Float16, FP16AccumulatorMode::Float16, false, riscv);
  ASSERT_TRUE(policy);
  EXPECT_EQ(ncnn_mlir::precision_execution_profile(*policy, riscv),
            "riscv-rvv-fp16");

  ncnn_mlir::TargetSpec x86{.triple = "x86_64-unknown-linux-gnu",
                            .march = "",
                            .mcpu = "sapphirerapids",
                            .features = {}};
  capabilities = ncnn_mlir::infer_target_capabilities(x86);
  EXPECT_TRUE(capabilities.fp16_arithmetic);
  EXPECT_TRUE(capabilities.bf16);
  EXPECT_TRUE(capabilities.int8);
}

TEST(PrecisionTest, RegistersOperatorCapabilities) {
  using ncnn_mlir::OperatorPrecisionCapability;
  EXPECT_EQ(ncnn_mlir::operator_precision_capability("ncnn.convolution"),
            OperatorPrecisionCapability::FP16Arithmetic);
  EXPECT_EQ(ncnn_mlir::operator_precision_capability("ncnn.softmax"),
            OperatorPrecisionCapability::LowPrecisionBoundary);
  for (const char* operation : {"ncnn.sigmoid",
                                "ncnn.hard_sigmoid",
                                "ncnn.hard_swish",
                                "ncnn.gelu",
                                "ncnn.batch_norm",
                                "ncnn.detection_output"}) {
    EXPECT_EQ(ncnn_mlir::operator_precision_capability(operation),
              OperatorPrecisionCapability::LowPrecisionBoundary);
  }
  for (const char* operation : {"ncnn.deconvolution",
                                "ncnn.gemm",
                                "ncnn.binary",
                                "ncnn.pooling",
                                "ncnn.concat"}) {
    EXPECT_EQ(ncnn_mlir::operator_precision_capability(operation),
              OperatorPrecisionCapability::LowPrecisionBoundary);
  }
  EXPECT_EQ(ncnn_mlir::operator_precision_capability("ncnn.relu"),
            OperatorPrecisionCapability::Float32Only);
}
