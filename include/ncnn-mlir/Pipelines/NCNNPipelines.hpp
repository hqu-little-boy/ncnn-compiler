#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::ncnn {

void buildNCNNToTosaPipeline(OpPassManager& passManager);
void buildNCNNTosaToLinalgPipeline(OpPassManager& passManager);
void buildNCNNLinalgToMemRefPipeline(OpPassManager& passManager);
void registerNCNNPipelines();

}  // namespace mlir::ncnn
