#include "ncnn-mlir/Pipelines/NCNNPipelines.hpp"

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"
#include "ncnn-mlir/Transforms/BufferizeNCNN/BufferizeNCNN.hpp"
#include "ncnn-mlir/Transforms/EmitModelPlan/EmitModelPlan.hpp"
#include "ncnn-mlir/Transforms/FoldLinalgConstantTranspose/FoldLinalgConstantTranspose.hpp"
#include "ncnn-mlir/Transforms/FoldNCNNBatchNorm/FoldNCNNBatchNorm.hpp"
#include "ncnn-mlir/Transforms/ForallizeDisjointTileLoops/ForallizeDisjointTileLoops.hpp"
#include "ncnn-mlir/Transforms/FuseLinalgEpilogue/FuseLinalgEpilogue.hpp"
#include "ncnn-mlir/Transforms/FuseQuantChainNCNN/FuseQuantChainNCNN.hpp"
#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNProfile.hpp"
#include "ncnn-mlir/Transforms/LowerVectorMathNCNN/LowerVectorMathNCNN.hpp"
#include "ncnn-mlir/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.hpp"
#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"
#include "ncnn-mlir/Transforms/ReuseWorkspaceSlots/ReuseWorkspaceSlots.hpp"
#include "ncnn-mlir/Transforms/RewriteLinalgCopies/RewriteLinalgCopies.hpp"
#include "ncnn-mlir/Transforms/StrategyNCNN/StrategyNCNN.hpp"
#include "ncnn-mlir/Transforms/TileMatmulForall/TileMatmulForall.hpp"
#include "ncnn-mlir/Transforms/VectorizeNCNN/LowerVectorTransfersNCNN.hpp"
#include "ncnn-mlir/Transforms/VectorizeNCNN/VectorizeNCNN.hpp"
#include "ncnn-mlir/Transforms/VerifyBufferizedModel/VerifyBufferizedModel.hpp"
#include "ncnn-mlir/Transforms/VerifyModelShapeContracts/VerifyModelShapeContracts.hpp"
#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"
#include "ncnn-mlir/Transforms/VerifyNoNestedOpenMP/VerifyNoNestedOpenMP.hpp"
#include "ncnn-mlir/Transforms/VerifyNoTosaOps/VerifyNoTosaOps.hpp"

namespace mlir::ncnn {

namespace {

// Linalg-to-parallel-loops does not distinguish an operation already enclosed
// by a tile forall from a top-level residual operation.  Lower the former to
// serial scf.for loops first so the subsequent SCF-to-OpenMP conversion emits
// one team for the outer tile boundary instead of a nested team per tile.
class ConvertNestedLinalgToLoopsPass final
  : public PassWrapper<ConvertNestedLinalgToLoopsPass,
                       OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNestedLinalgToLoopsPass)

  StringRef getArgument() const final {
    return "convert-nested-linalg-to-loops";
  }
  StringRef getDescription() const final {
    return "Lower nested Linalg and reductions to serial loops";
  }

  void runOnOperation() final {
    SmallVector<linalg::LinalgOp> nestedOps;
    getOperation().walk([&](linalg::LinalgOp op) {
      // Linalg reductions lower to a parallel loop for the reduction's
      // elementwise body in addition to their outer loop.  Keep the whole
      // reduction serial when the threaded pipeline will provide the single
      // outer team; this also avoids a hidden nested team for top-level
      // reductions.
      if (isa<linalg::ReduceOp>(op) || op->getParentOfType<scf::ForallOp>() ||
          op->getParentOfType<scf::ParallelOp>() ||
          op->getParentOfType<omp::ParallelOp>()) {
        nestedOps.push_back(op);
      }
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : nestedOps) {
      if (!llvm::all_of(op->getOperands(), [](Value value) {
            return isa<MemRefType>(value.getType());
          })) {
        op.emitError()
          << "nested Linalg operation must be bufferized before OpenMP "
             "lowering";
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(op);
      if (failed(linalg::linalgOpToLoops(rewriter, op))) {
        op.emitError() << "failed to lower nested Linalg operation to loops";
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(op);
    }
  }
};

class SetLowPrecisionOptionsPass final
  : public PassWrapper<SetLowPrecisionOptionsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetLowPrecisionOptionsPass)
  SetLowPrecisionOptionsPass(bool depthwise,
                             bool casts,
                             std::string policy,
                             std::string target,
                             bool preserveCasts = false)
    : depthwise(depthwise),
      casts(casts),
      policy(std::move(policy)),
      target(std::move(target)),
      preserveCasts(preserveCasts) {}
  void runOnOperation() final {
    Builder builder(&getContext());
    getOperation()->setAttr("ncnn.int8_depthwise",
                            builder.getBoolAttr(depthwise));
    if (!preserveCasts) {
      getOperation()->setAttr("ncnn.int8_cast_chain",
                              builder.getBoolAttr(casts));
    }
    getOperation()->setAttr("ncnn.int8_kernel", builder.getStringAttr(policy));
    getOperation()->setAttr("ncnn.int8_target", builder.getStringAttr(target));
  }

 private:
  bool depthwise;
  bool casts;
  std::string policy;
  std::string target;
  bool preserveCasts;
};

}  // namespace

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
  // int8 requant/dequant 尾部收拢：sitofp→scale mul→bias→[act]→quantize
  // 链融合为单一 generic（P4）。放在 strategy 之前，使量化卷积的
  // requant 尾部与激活一样成为卷积结果之后的单个可折叠逐元素 consumer；
  // 融合保持 op 顺序与常量不变，数值与链式形态逐位一致。
  passManager.addPass(std::make_unique<SetLowPrecisionOptionsPass>(
    false, options.int8CastChain, "portable", "portable"));
  passManager.addPass(createFuseQuantChainNCNNPass());
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
  epilogueOptions.enable = options.selectiveFusion;
  epilogueOptions.allowResidual = options.selectiveFusionResidual;
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
  passManager.addPass(
    std::make_unique<SetLowPrecisionOptionsPass>(options.int8Depthwise,
                                                 false,
                                                 options.int8KernelPolicy,
                                                 options.int8TargetCapability,
                                                 true));
  // 并行化单轨化（T4b 配套）：顶层 matmul 沿 M/N 切进 forall 网格，
  // epilogue 分块循环改写为 shared_outs forall——两者都在向量化之前
  // 完成，体内 generic 随后照常被行级向量化。K 维全程不被切分。
  // P20 在此之前把静态 f32 RHS 常量重排为物理 panel-NK 存储；关闭
  // policy 保留完全等价的 unpacked 基线路径，供 paired A/B 使用。
  PackStaticMatmulNCNNPassOptions packingOptions;
  // The initial panel schema is fixed at Nr=16. Keep the opt-in transform
  // disabled for register tiles that cannot be partitioned into that panel;
  // otherwise a direct fallback would interpret physically reordered storage.
  const bool packedPanelCompatible =
    options.matmulAccColumns > 0 && 16 % options.matmulAccColumns == 0;
  packingOptions.enabled =
    options.matmulPackingPolicy == "auto" && packedPanelCompatible;
  passManager.addPass(createPackStaticMatmulNCNNPass(packingOptions));
  passManager.addPass(createTileMatmulForallPass());
  passManager.addPass(createForallizeDisjointTileLoopsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  if (options.vectorLanes > 0) {
    // 向量化阶段：tensor 层先于 bufferize（Bufferization.md 指南），vector op
    // 由已注册的 BufferizableOpInterface 外部模型消费。
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
  if (options.vectorTail) {
    // A1b SIMD matmul 内核与 forall 路径标量热点清理。发射 vector/
    // ub.poison op，必须确保下游有向量下降尾（串行遗留路径没有），
    // 由调用方经 vector-tail 显式声明。INT8 kernel policy 与目标能力
    // 显式分离：portable 保持历史标量 MAC；vnni 只在目标能力匹配时
    // 生效（不匹配由 pass 直接报错，不做静默降级）。
    MatmulKernelNCNNPassOptions matmulKernelOptions;
    matmulKernelOptions.matmulMRows = options.matmulMRows;
    matmulKernelOptions.matmulAccColumns = options.matmulAccColumns;
    matmulKernelOptions.rowChunkLanes = options.rowChunkLanes;
    matmulKernelOptions.int8Kernel = options.int8KernelPolicy;
    matmulKernelOptions.int8Target = options.int8TargetCapability;
    matmulKernelOptions.matmulI8Rows = options.matmulI8Rows;
    matmulKernelOptions.matmulI8AccColumns = options.matmulI8AccColumns;
    passManager.addPass(createMatmulKernelNCNNPass(matmulKernelOptions));
  }

  bufferization::BufferResultsToOutParamsPassOptions outParamOptions;
  outParamOptions.addResultAttribute = true;
  outParamOptions.hoistStaticAllocs = true;
  passManager.addPass(
    bufferization::createBufferResultsToOutParamsPass(outParamOptions));

  bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  deallocationOptions.privateFunctionDynamicOwnership = false;
  bufferization::buildBufferDeallocationPipeline(passManager,
                                                 deallocationOptions);
  // P11：只在可证明的直线静态生命周期中复用局部 workspace slot；未知
  // alias、动态 shape 和控制流保留原 allocation，并由 verifier 继续兜底。
  passManager.addPass(createReuseWorkspaceSlotsPass());
  passManager.addPass(createVerifyBufferizedModelPass());
  passManager.addPass(createVerifyModelShapeContractsPass());
  if (!options.executionPlanPath.empty()) {
    EmitModelPlanPassOptions planOptions;
    planOptions.path = options.executionPlanPath;
    planOptions.model = options.executionPlanModel;
    planOptions.targetTriple = options.executionPlanTargetTriple;
    planOptions.threads = options.executionPlanThreads;
    planOptions.vectorLanes = options.vectorLanes;
    planOptions.vectorScalable = options.vectorScalable;
    planOptions.vectorTail = options.vectorTail;
    planOptions.codegenIdentity = options.executionPlanCodegenIdentity;
    planOptions.tuningProfile = options.tuningProfile;
    planOptions.tuningStatus = options.tuningStatus;
    planOptions.tuningFallbackReason = options.tuningFallbackReason;
    planOptions.matmulMRows = options.matmulMRows;
    planOptions.matmulAccColumns = options.matmulAccColumns;
    planOptions.rowChunkLanes = options.rowChunkLanes;
    planOptions.matmulI8Rows = options.matmulI8Rows;
    planOptions.matmulI8AccColumns = options.matmulI8AccColumns;
    passManager.addPass(createEmitModelPlanPass(planOptions));
  }
  if (options.profileInstrumentation) {
    passManager.addPass(createInstrumentNCNNProfilePass());
  }
}

void buildNCNNMemRefToLLVMPipeline(OpPassManager& passManager) {
  buildNCNNMemRefToLLVMPipeline(passManager, NCNNMemRefToLLVMPipelineOptions());
}

void buildNCNNMemRefToLLVMPipeline(
  OpPassManager& passManager, const NCNNMemRefToLLVMPipelineOptions& options) {
  if (options.threads != 1) {
    // 并行化发射：张量级 tile-matmul-forall / forallize-disjoint-tile-
    // loops 已产出分块 forall；残余顶层 linalg 仍可用上游全域并行化，
    // 而 tile 区域与 reduction 先改成串行 scf.for，保证整个算子只
    // 进入一个 OpenMP team。
    passManager.addPass(std::make_unique<ConvertNestedLinalgToLoopsPass>());
    passManager.addPass(createConvertLinalgToParallelLoopsPass());
    passManager.addPass(createForallToParallelLoopPass());
    ConvertSCFToOpenMPPassOptions openmpOptions;
    if (options.threads > 1) {
      openmpOptions.numThreads = options.threads;
    }
    passManager.addPass(createConvertSCFToOpenMPPass(openmpOptions));
    passManager.addPass(createVerifyNoNestedOpenMPPass());
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
  if (hasVectorIR && options.vectorMath != "none") {
    // 向量数学库调用下降：抢在标量 MathToLibm 之前把 ABI 等宽的 f32 向量
    // math op 整体替换为 libmvec/SLEEF 调用。位置刻意保持在 SCFToControl-
    // Flow 之前，为后续宽向量循环分块保留 scf 发射合法性。
    LowerVectorMathNCNNPassOptions mathOptions;
    mathOptions.backend = options.vectorMath.getValue();
    mathOptions.lanes = options.vectorMathLanes.getValue();
    mathOptions.abiFragment = options.vectorMathAbi.getValue();
    passManager.addPass(createLowerVectorMathNCNNPass(mathOptions));
  }
  passManager.addPass(createLowerAffinePass());
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
  // OpenMP lowering introduces single-block memref.alloca_scope regions.
  // Lower their stack save/restore boundary before SCF expands nested control
  // flow into multiple blocks; otherwise the intermediate IR is invalid.
  passManager.addPass(createSCFToControlFlowPass());
  // SCF lowering creates arithmetic even when no vector tail is enabled.
  passManager.addPass(createArithToLLVMConversionPass());
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
