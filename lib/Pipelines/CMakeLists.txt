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
    FoldLinalgConstantTranspose
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
    MLIRAffineTransforms
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRLinalgTransforms
    MLIRMathToLibm
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
    MLIROpenMPToLLVM
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFToOpenMP
    MLIRSCFTransforms
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToTensor
    MLIRTosaTransforms
    MLIRUBToLLVM
    MLIRVectorToLLVM
    MLIRTransforms)
