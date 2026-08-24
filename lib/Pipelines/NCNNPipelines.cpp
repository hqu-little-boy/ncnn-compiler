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
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"
#include "ncnn-mlir/Transforms/BufferizeNCNN/BufferizeNCNN.hpp"
#include "ncnn-mlir/Transforms/FoldLinalgConstantTranspose/FoldLinalgConstantTranspose.hpp"
#include "ncnn-mlir/Transforms/FoldNCNNBatchNorm/FoldNCNNBatchNorm.hpp"
#include "ncnn-mlir/Transforms/FuseLinalgEpilogue/FuseLinalgEpilogue.hpp"
#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"
#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"
#include "ncnn-mlir/Transforms/RewriteLinalgCopies/RewriteLinalgCopies.hpp"
#include "ncnn-mlir/Transforms/StrategyNCNN/StrategyNCNN.hpp"
#include "ncnn-mlir/Transforms/VectorizeNCNN/LowerVectorTransfersNCNN.hpp"
#include "ncnn-mlir/Transforms/VectorizeNCNN/VectorizeNCNN.hpp"
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
  buildNCNNTosaToLinalgPipeline(passManager, NCNNTosaToLinalgPipelineOptions());
}

void buildNCNNTosaToLinalgPipeline(
  OpPassManager& passManager, const NCNNTosaToLinalgPipelineOptions& options) {
  TosaToLinalgNamedOptions namedOptions;
  namedOptions.preferConv2DKernelLayoutHWCF = true;
  tosa::addTosaToLinalgPasses(
    passManager, TosaToLinalgOptions(), namedOptions, std::nullopt);
  passManager.addPass(createFoldLinalgConstantTransposePass());
  passManager.addNestedPass<func::FuncOp>(createTosaToTensorPass());
  passManager.addNestedPass<func::FuncOp>(createTosaToArithPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  // 算子形态策略层（A1）：在 epilogue 融合与向量化之前把可改写卷积变为
  // matmul 形态，使计算大头进入投影映射的收缩主干；权重 collapse 由紧随
  // 其后的 canonicalizer 折叠为 .rodata 常量。
  StrategyNCNNPassOptions strategyOptions;
  strategyOptions.strategy = options.convStrategy.getValue();
  strategyOptions.gemmL2Bytes = options.convGemmL2Bytes.getValue();
  passManager.addPass(createStrategyNCNNPass(strategyOptions));
  passManager.addPass(createLinalgInlineScalarOperandsPass());
  passManager.addPass(createLinalgFoldIntoElementwisePass());
  passManager.addPass(createCanonicalizerPass());
  FuseLinalgEpiloguePassOptions epilogueOptions;
  epilogueOptions.tileWidth = options.epilogueTileWidth;
  passManager.addPass(createFuseLinalgEpiloguePass(epilogueOptions));
  passManager.addPass(createVerifyNoTosaOpsPass());
}

void buildNCNNLinalgToMemRefPipeline(OpPassManager& passManager) {
  buildNCNNLinalgToMemRefPipeline(passManager,
                                  NCNNLinalgToMemRefPipelineOptions());
}

void buildNCNNLinalgToMemRefPipeline(
  OpPassManager& passManager,
  const NCNNLinalgToMemRefPipelineOptions& options) {
  if (options.vectorLanes > 0) {
    // 向量化阶段：tensor 层先于 bufferize（Bufferization.md 指南），vector op
    // 由已注册的 BufferizableOpInterface 外部模型消费。
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
    VectorizeNCNNPassOptions vectorizeOptions;
    vectorizeOptions.lanes = options.vectorLanes;
    vectorizeOptions.scalable = options.vectorScalable;
    passManager.addPass(createVectorizeNCNNPass(vectorizeOptions));
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
  }

  passManager.addPass(createBufferizeNCNNPass());
  // bufferize 的拷贝以恒等 linalg.generic 形式存在（memCpyFn 产物）；
  // 改写为 memref.copy，避免多线程路径把它们当作可并行 linalg op 在
  // forall 区域内再并行化（嵌套 omp），并让尾段走更廉价的整块复制。
  passManager.addPass(createRewriteLinalgCopiesPass());

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
    // 过渡期双轨：convert-linalg-to-parallel-loops 仅服务残余 linalg op
    // （conv/matmul/pooling，A1 接管前保持多核标量）；向量化产生的
    // scf.forall 经上游 scf-forall-to-parallel 归一为 scf.parallel 后，
    // 与旧路径共用同一条 OpenMP 转换。forall 转换对 shared_outs +
    // parallel_insert_slice 的 bufferized 形态产出写不相交的
    // scf.parallel（外层线程级并行，内层 SIMD 不受影响）。
    passManager.addPass(createConvertLinalgToParallelLoopsPass());
    passManager.addPass(createParallelLoopFusionPass());
    passManager.addPass(createForallToParallelLoopPass());
    ConvertSCFToOpenMPPassOptions openmpOptions;
    if (options.threads > 1) {
      openmpOptions.numThreads = options.threads;
    }
    passManager.addPass(createConvertSCFToOpenMPPass(openmpOptions));
  } else {
    // 串行回退：threads=1 时 forall 无并行语义承载，先归一为 scf.for，
    // 再进入既有的 affine 向量化或标量循环下降。
    passManager.addPass(createForallToForLoopPass());
    if (options.vectorSize > 0) {
      passManager.addPass(createConvertLinalgToAffineLoopsPass());
      affine::AffineVectorizeOptions vectorOptions;
      vectorOptions.vectorSizes.push_back(options.vectorSize);
      vectorOptions.vectorizeReductions = true;
      passManager.addNestedPass<func::FuncOp>(
        affine::createAffineVectorize(vectorOptions));
    } else {
      passManager.addPass(createConvertLinalgToLoopsPass());
    }
  }
  // 门禁断言：OpenMP/串行转换对不认识的 SCF 形态会静默跳过；任何残留的
  // scf.forall/scf.parallel 都意味着并行性悄悄丢失，必须在此失败。
  passManager.addPass(createVerifyNoSCFForallPass());
  passManager.addPass(createLoopInvariantCodeMotionPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  const bool hasVectorIR = options.vectorSize > 0 || options.vectorLowering;
  if (hasVectorIR) {
    // multi_reduction/mask 不在 VectorToLLVM 覆盖范围内，必须在 ArithToLLVM
    // 之前降级，避免新生成的 arith op 漏掉整型/浮点转换。
    passManager.addNestedPass<func::FuncOp>(
      vector::createLowerVectorMultiReductionPass());
    passManager.addNestedPass<func::FuncOp>(
      vector::createLowerVectorMaskPass());
  }
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createSCFToControlFlowPass());
  passManager.addPass(createConvertMathToLibmPass());
  if (hasVectorIR) {
    // N-D transfer 规范化必须在 ExpandStridedMetadata/MemRefToLLVM 之前：
    // 此时周边仍是 memref/scf 语义，且其新建的 memref op 会被后续展开与
    // LLVM 化覆盖。
    passManager.addPass(createLowerVectorTransfersNCNNPass());
  }
  passManager.addPass(memref::createExpandStridedMetadataPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createArithToLLVMConversionPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(createConvertFuncToLLVMPass());
  passManager.addPass(createFinalizeCAPIPass());
  passManager.addPass(createConvertControlFlowToLLVMPass());
  if (hasVectorIR) {
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
  static PassPipelineRegistration<NCNNTosaToLinalgPipelineOptions>
    tosaToLinalgRegistration(
      "ncnn-tosa-to-linalg-pipeline",
      "Strict TOSA-to-Linalg pipeline for ncnn models",
      [](OpPassManager& passManager,
         const NCNNTosaToLinalgPipelineOptions& options) {
        buildNCNNTosaToLinalgPipeline(passManager, options);
      });
  static PassPipelineRegistration<NCNNLinalgToMemRefPipelineOptions>
    linalgToMemRefRegistration(
      "ncnn-linalg-to-memref-pipeline",
      "Bufferize ncnn Linalg models with caller-owned output parameters",
      [](OpPassManager& passManager,
         const NCNNLinalgToMemRefPipelineOptions& options) {
        buildNCNNLinalgToMemRefPipeline(passManager, options);
      });
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
