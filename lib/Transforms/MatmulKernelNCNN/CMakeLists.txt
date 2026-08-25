add_library(MatmulKernelNCNN
    MatmulKernelNCNN.cpp
)
add_dependencies(MatmulKernelNCNN NCNNPassesIncGen)
target_include_directories(MatmulKernelNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(MatmulKernelNCNN SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(MatmulKernelNCNN PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(MatmulKernelNCNN PRIVATE -fno-rtti)
target_link_libraries(MatmulKernelNCNN PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRVectorDialect)
