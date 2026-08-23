#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::ncnn {

struct NCNNTosaToLinalgPipelineOptions
  : public PassPipelineOptions<NCNNTosaToLinalgPipelineOptions> {
  Option<int64_t> epilogueTileWidth{
    *this,
    "epilogue-tile-width",
    llvm::cl::desc("Output column tiling width for fused Linalg epilogues"),
    llvm::cl::init(16)};
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
