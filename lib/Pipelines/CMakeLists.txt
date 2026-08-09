add_library(NCNNPipelines STATIC
    NCNNPipelines.cpp
)
add_dependencies(NCNNPipelines NCNNPassesIncGen)
target_include_directories(NCNNPipelines PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_compile_options(NCNNPipelines PRIVATE -fno-rtti)
target_link_libraries(NCNNPipelines PUBLIC
    BufferizeNCNN
    GenerateCAPI
    NCNNToFunc
    NCNNToTosa
    NormalizeNCNN
    VerifyBufferizedModel
    VerifyModelShapeContracts
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
