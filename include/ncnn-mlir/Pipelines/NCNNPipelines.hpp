#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::ncnn {

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
};

void buildNCNNToTosaPipeline(OpPassManager& passManager);
void buildNCNNTosaToLinalgPipeline(OpPassManager& passManager);
void buildNCNNLinalgToMemRefPipeline(OpPassManager& passManager);
void buildNCNNMemRefToLLVMPipeline(OpPassManager& passManager);
void buildNCNNMemRefToLLVMPipeline(
  OpPassManager& passManager, const NCNNMemRefToLLVMPipelineOptions& options);
void registerNCNNPipelines();

}  // namespace mlir::ncnn
