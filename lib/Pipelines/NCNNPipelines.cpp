#include "ncnn-mlir/Pipelines/NCNNPipelines.hpp"

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"
#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"
#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"

namespace mlir::ncnn {

void buildNCNNToTosaPipeline(OpPassManager& passManager) {
  passManager.addPass(createConvertNCNNModelToFuncPass());
  passManager.addPass(createNormalizeNCNNPass());
  passManager.addPass(createConvertNCNNToTosaPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createVerifyNoNCNNOpsPass());
}

void registerNCNNPipelines() {
  static PassPipelineRegistration<> registration(
    "ncnn-to-tosa-pipeline",
    "Strict ncnn model-to-TOSA pipeline",
    buildNCNNToTosaPipeline);
}

}  // namespace mlir::ncnn
