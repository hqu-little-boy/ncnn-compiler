#include "ncnn-mlir/Pipelines/NCNNPipelines.hpp"

#include <optional>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"
#include "ncnn-mlir/Transforms/BufferizeNCNN/BufferizeNCNN.hpp"
#include "ncnn-mlir/Transforms/FoldLinalgConstantTranspose/FoldLinalgConstantTranspose.hpp"
#include "ncnn-mlir/Transforms/FoldNCNNBatchNorm/FoldNCNNBatchNorm.hpp"
#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"
#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"
#include "ncnn-mlir/Transforms/VerifyBufferizedModel/VerifyBufferizedModel.hpp"
#include "ncnn-mlir/Transforms/VerifyModelShapeContracts/VerifyModelShapeContracts.hpp"
#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"
#include "ncnn-mlir/Transforms/VerifyNoTosaOps/VerifyNoTosaOps.hpp"

namespace mlir::ncnn {

void buildNCNNToTosaPipeline(OpPassManager& passManager) {
  passManager.addPass(createConvertNCNNModelToFuncPass());
  passManager.addPass(createFoldNCNNBatchNormPass());
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
  passManager.addPass(createFoldLinalgConstantTransposePass());
  passManager.addNestedPass<func::FuncOp>(createTosaToTensorPass());
  passManager.addNestedPass<func::FuncOp>(createTosaToArithPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createVerifyNoTosaOpsPass());
}

void buildNCNNLinalgToMemRefPipeline(OpPassManager& passManager) {
  passManager.addPass(createBufferizeNCNNPass());

  bufferization::BufferResultsToOutParamsPassOptions outParamOptions;
  outParamOptions.addResultAttribute = true;
  outParamOptions.hoistStaticAllocs = true;
  passManager.addPass(
    bufferization::createBufferResultsToOutParamsPass(outParamOptions));

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  deallocationOptions.privateFunctionDynamicOwnership = false;
  bufferization::buildBufferDeallocationPipeline(passManager,
                                                 deallocationOptions);
  passManager.addPass(createVerifyBufferizedModelPass());
  passManager.addPass(createVerifyModelShapeContractsPass());
}

void buildNCNNMemRefToLLVMPipeline(OpPassManager& passManager) {
  buildNCNNMemRefToLLVMPipeline(passManager, NCNNMemRefToLLVMPipelineOptions());
}

void buildNCNNMemRefToLLVMPipeline(
  OpPassManager& passManager, const NCNNMemRefToLLVMPipelineOptions& options) {
  if (options.threads != 1) {
    passManager.addPass(createConvertLinalgToParallelLoopsPass());
    passManager.addPass(createParallelLoopFusionPass());
    ConvertSCFToOpenMPPassOptions openmpOptions;
    if (options.threads > 1) {
      openmpOptions.numThreads = options.threads;
    }
    passManager.addPass(createConvertSCFToOpenMPPass(openmpOptions));
  } else if (options.vectorSize > 0) {
    passManager.addPass(createConvertLinalgToAffineLoopsPass());
    affine::AffineVectorizeOptions vectorOptions;
    vectorOptions.vectorSizes.push_back(options.vectorSize);
    vectorOptions.vectorizeReductions = true;
    passManager.addNestedPass<func::FuncOp>(
      affine::createAffineVectorize(vectorOptions));
  } else {
    passManager.addPass(createConvertLinalgToLoopsPass());
  }
  passManager.addPass(createLoopInvariantCodeMotionPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createSCFToControlFlowPass());
  passManager.addPass(createConvertMathToLibmPass());
  passManager.addPass(memref::createExpandStridedMetadataPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createArithToLLVMConversionPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(createConvertFuncToLLVMPass());
  passManager.addPass(createFinalizeCAPIPass());
  passManager.addPass(createConvertControlFlowToLLVMPass());
  if (options.vectorSize > 0 && options.threads == 1) {
    passManager.addPass(createConvertVectorToLLVMPass());
    passManager.addPass(createArithToLLVMConversionPass());
    passManager.addPass(createUBToLLVMConversionPass());
  }
  if (options.threads != 1) {
    passManager.addPass(createConvertOpenMPToLLVMPass());
  }
  passManager.addPass(createReconcileUnrealizedCastsPass());
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
  static PassPipelineRegistration<> linalgToMemRefRegistration(
    "ncnn-linalg-to-memref-pipeline",
    "Bufferize ncnn Linalg models with caller-owned output parameters",
    buildNCNNLinalgToMemRefPipeline);
  static PassPipelineRegistration<NCNNMemRefToLLVMPipelineOptions>
    memRefToLLVMRegistration(
      "ncnn-memref-to-llvm-pipeline",
      "Lower bufferized ncnn models to the LLVM dialect",
      [](OpPassManager& passManager,
         const NCNNMemRefToLLVMPipelineOptions& options) {
        buildNCNNMemRefToLLVMPipeline(passManager, options);
      });
}

}  // namespace mlir::ncnn
