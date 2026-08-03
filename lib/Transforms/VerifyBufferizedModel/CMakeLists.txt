add_library(VerifyBufferizedModel STATIC
    VerifyBufferizedModel.cpp
)
add_dependencies(VerifyBufferizedModel NCNNPassesIncGen)
target_compile_options(VerifyBufferizedModel PRIVATE -fno-rtti)
target_include_directories(VerifyBufferizedModel PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyBufferizedModel PUBLIC
    MLIRAnalysis
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRPass)
