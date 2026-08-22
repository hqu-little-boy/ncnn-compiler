add_library(FoldNCNNBatchNorm STATIC
    FoldNCNNBatchNorm.cpp
)
add_dependencies(FoldNCNNBatchNorm NCNNPassesIncGen)
target_include_directories(FoldNCNNBatchNorm PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(FoldNCNNBatchNorm SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(FoldNCNNBatchNorm PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(FoldNCNNBatchNorm PRIVATE -fno-rtti)
target_link_libraries(FoldNCNNBatchNorm PUBLIC
    NCNNDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass)
