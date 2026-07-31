add_library(NCNNPipelines STATIC
    NCNNPipelines.cpp
)
target_compile_options(NCNNPipelines PRIVATE -fno-rtti)
target_link_libraries(NCNNPipelines PUBLIC
    NCNNToFunc
    NCNNToTosa
    NormalizeNCNN
    VerifyBufferizedModel
    VerifyNoNCNNOps
    VerifyNoTosaOps
    MLIRFuncDialect
    MLIRBufferizationPipelines
    MLIRBufferizationTransforms
    MLIRPass
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToTensor
    MLIRTosaTransforms
    MLIRTransforms)
