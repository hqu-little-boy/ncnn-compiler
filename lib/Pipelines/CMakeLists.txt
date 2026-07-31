add_library(NCNNPipelines STATIC
    NCNNPipelines.cpp
)
target_compile_options(NCNNPipelines PRIVATE -fno-rtti)
target_link_libraries(NCNNPipelines PUBLIC
    BufferizeNCNN
    NCNNToFunc
    NCNNToTosa
    NormalizeNCNN
    VerifyBufferizedModel
    VerifyNoNCNNOps
    VerifyNoTosaOps
    MLIRFuncDialect
    MLIRBufferizationPipelines
    MLIRBufferizationTransforms
    MLIRAffineToStandard
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRLinalgTransforms
    MLIRMathToLibm
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToTensor
    MLIRTosaTransforms
    MLIRTransforms)
