#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::ncnn {

struct NCNNTosaToLinalgPipelineOptions
  : public PassPipelineOptions<NCNNTosaToLinalgPipelineOptions> {
  Option<bool> int8CastChain{
    *this,
    "int8-cast-chain",
    llvm::cl::desc("Fuse proven static cast map consumers"),
    llvm::cl::init(false)};
  Option<int64_t> epilogueTileWidth{
    *this,
    "epilogue-tile-width",
    llvm::cl::desc("Output column tiling width for fused Linalg epilogues"),
    llvm::cl::init(16)};
  Option<std::string> convStrategy{
    *this,
    "conv-strategy",
    llvm::cl::desc(
      "Convolution operator-shape strategy: auto, gemm, conv, winograd"),
    llvm::cl::init("auto")};
  Option<int64_t> convGemmL2Bytes{
    *this,
    "conv-gemm-l2-bytes",
    llvm::cl::desc("L2 cache byte budget for the convolution prefer-GEMM "
                   "heuristic"),
    llvm::cl::init(524288)};
  Option<bool> selectiveFusion{
    *this,
    "selective-fusion",
    llvm::cl::desc("Enable selective static producer-to-epilogue fusion"),
    llvm::cl::init(true)};
  Option<bool> selectiveFusionResidual{
    *this,
    "selective-fusion-residual",
    llvm::cl::desc("Allow same-shaped identity-mapped residual fusion inputs"),
    llvm::cl::init(true)};
};

struct NCNNLinalgToMemRefPipelineOptions
  : public PassPipelineOptions<NCNNLinalgToMemRefPipelineOptions> {
  Option<unsigned> vectorLanes{
    *this,
    "vector-lanes",
    llvm::cl::desc(
      "Vectorize Linalg ops to this lane count before bufferization; "
      "0 disables"),
    llvm::cl::init(0)};
  Option<bool> vectorScalable{
    *this,
    "vector-scalable",
    llvm::cl::desc("Emit scalable vectors (SVE/RVV) with lanes as minimum VL"),
    llvm::cl::init(false)};
  Option<bool> vectorTail{
    *this,
    "vector-tail",
    llvm::cl::desc(
      "Downstream memref-to-llvm will run the vector lowering tail; "
      "gates passes that emit vector dialect ops"),
    llvm::cl::init(false)};
  Option<std::string> int8KernelPolicy{
    *this,
    "int8-kernel",
    llvm::cl::desc("INT8 row-dot kernel policy: portable, auto, vnni"),
    llvm::cl::init("portable")};
  Option<std::string> int8TargetCapability{
    *this,
    "int8-target",
    llvm::cl::desc(
      "Resolved backend INT8 capability: portable, avx-vnni, avx512-vnni"),
    llvm::cl::init("portable")};
  Option<bool> int8Depthwise{
    *this,
    "int8-depthwise",
    llvm::cl::desc("Enable the static int8 depthwise SIMD path"),
    llvm::cl::init(false)};
  Option<std::string> tuningProfile{
    *this,
    "tuning-profile",
    llvm::cl::desc("Bounded compile-time tuning profile"),
    llvm::cl::init("stable")};
  Option<std::string> tuningStatus{*this,
                                   "tuning-status",
                                   llvm::cl::desc("Resolved tuning status"),
                                   llvm::cl::init("stable")};
  Option<std::string> tuningFallbackReason{
    *this,
    "tuning-fallback-reason",
    llvm::cl::desc("Auditable reason for tuning fallback or override"),
    llvm::cl::init("")};
  Option<int64_t> matmulMRows{
    *this,
    "matmul-m-rows",
    llvm::cl::desc("M-direction matmul register tile rows"),
    llvm::cl::init(4)};
  Option<int64_t> matmulAccColumns{*this,
                                   "matmul-acc-columns",
                                   llvm::cl::desc("Matmul accumulator columns"),
                                   llvm::cl::init(16)};
  Option<unsigned> rowChunkLanes{
    *this,
    "row-chunk-lanes",
    llvm::cl::desc("Row-generic vector chunk lane budget"),
    llvm::cl::init(8)};
  Option<int64_t> matmulI8Rows{*this,
                               "matmul-i8-rows",
                               llvm::cl::desc("INT8 matmul register tile rows"),
                               llvm::cl::init(2)};
  Option<int64_t> matmulI8AccColumns{
    *this,
    "matmul-i8-acc-columns",
    llvm::cl::desc("INT8 matmul accumulator columns"),
    llvm::cl::init(4)};
  Option<std::string> executionPlanPath{
    *this,
    "execution-plan-path",
    llvm::cl::desc("Optional deterministic execution-plan JSON output path"),
    llvm::cl::init("")};
  Option<std::string> executionPlanModel{
    *this,
    "execution-plan-model",
    llvm::cl::desc("Model name recorded in the execution plan"),
    llvm::cl::init("")};
  Option<std::string> executionPlanTargetTriple{
    *this,
    "execution-plan-target-triple",
    llvm::cl::desc("Target triple recorded in the execution plan"),
    llvm::cl::init("")};
  Option<unsigned> executionPlanThreads{
    *this,
    "execution-plan-threads",
    llvm::cl::desc("Effective threads recorded in the execution plan"),
    llvm::cl::init(1)};
  Option<bool> profileInstrumentation{
    *this,
    "profile-instrumentation",
    llvm::cl::desc("Insert private diagnostic execution-profile callbacks"),
    llvm::cl::init(false)};
  Option<std::string> executionPlanCodegenIdentity{
    *this,
    "execution-plan-codegen-identity",
    llvm::cl::desc("Canonical code-generation identity included in the plan"),
    llvm::cl::init("")};
};

struct NCNNMemRefToLLVMPipelineOptions
  : public PassPipelineOptions<NCNNMemRefToLLVMPipelineOptions> {
  Option<unsigned> threads{
    *this,
    "threads",
    llvm::cl::desc(
      "OpenMP worker threads; 0 uses the runtime default, 1 is serial"),
    llvm::cl::init(1)};
  Option<unsigned> vectorSize{
    *this,
    "vector-size",
    llvm::cl::desc("Explicit SIMD lane count for serial lowering"),
    llvm::cl::init(0)};
  Option<bool> vectorLowering{
    *this,
    "vector-lowering",
    llvm::cl::desc(
      "IR already contains vector ops from earlier ncnn pipelines; append "
      "the Vector-to-LLVM lowering tail"),
    llvm::cl::init(false)};
  Option<std::string> vectorMath{
    *this,
    "vector-math",
    llvm::cl::desc("Resolved vector math backend: none, libmvec, sleef; "
                   "none keeps scalar MathToLibm lowering"),
    llvm::cl::init("none")};
  Option<unsigned> vectorMathLanes{
    *this,
    "vector-math-lanes",
    llvm::cl::desc("SIMD lane count used by the vector math backend for "
                   "chunking decisions"),
    llvm::cl::init(0)};
  Option<std::string> vectorMathAbi{
    *this,
    "vector-math-abi",
    llvm::cl::desc("Backend ABI fragment (libmvec ISA prefix such as "
                   "_ZGVdN8, or SLEEF accuracy/dispatch tag)"),
    llvm::cl::init("")};
};

void buildNCNNToTosaPipeline(OpPassManager& passManager);
void buildNCNNTosaToLinalgPipeline(OpPassManager& passManager);
void buildNCNNTosaToLinalgPipeline(
  OpPassManager& passManager, const NCNNTosaToLinalgPipelineOptions& options);
void buildNCNNLinalgToMemRefPipeline(OpPassManager& passManager);
void buildNCNNLinalgToMemRefPipeline(
  OpPassManager& passManager, const NCNNLinalgToMemRefPipelineOptions& options);
void buildNCNNMemRefToLLVMPipeline(OpPassManager& passManager);
void buildNCNNMemRefToLLVMPipeline(
  OpPassManager& passManager, const NCNNMemRefToLLVMPipelineOptions& options);
void registerNCNNPipelines();

}  // namespace mlir::ncnn
