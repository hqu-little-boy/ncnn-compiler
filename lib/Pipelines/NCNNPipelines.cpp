#include "ncnn-mlir/Pipelines/NCNNPipelines.hpp"

#include <optional>

#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"
#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"
#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"
#include "ncnn-mlir/Transforms/VerifyNoTosaOps/VerifyNoTosaOps.hpp"

namespace mlir::ncnn {

void buildNCNNToTosaPipeline(OpPassManager& passManager) {
  passManager.addPass(createConvertNCNNModelToFuncPass());
  passManager.addPass(createNormalizeNCNNPass());
  passManager.addPass(createConvertNCNNToTosaPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createVerifyNoNCNNOpsPass());
}

void buildNCNNTosaToLinalgPipeline(OpPassManager& passManager) {
  TosaToLinalgNamedOptions namedOptions;
  namedOptions.preferConv2DKernelLayoutHWCF = true;
  tosa::addTosaToLinalgPasses(
    passManager, TosaToLinalgOptions(), namedOptions, std::nullopt);
  passManager.addNestedPass<func::FuncOp>(createTosaToTensorPass());
  passManager.addNestedPass<func::FuncOp>(createTosaToArithPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createVerifyNoTosaOpsPass());
}

void registerNCNNPipelines() {
  static PassPipelineRegistration<> ncnnToTosaRegistration(
    "ncnn-to-tosa-pipeline",
    "Strict ncnn model-to-TOSA pipeline",
    buildNCNNToTosaPipeline);
  static PassPipelineRegistration<> tosaToLinalgRegistration(
    "ncnn-tosa-to-linalg-pipeline",
    "Strict TOSA-to-Linalg pipeline for ncnn models",
    buildNCNNTosaToLinalgPipeline);
}

}  // namespace mlir::ncnn
