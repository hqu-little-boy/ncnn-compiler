add_library(FuseQuantChainNCNN
    FuseQuantChainNCNN.cpp
)
add_dependencies(FuseQuantChainNCNN NCNNPassesIncGen)
target_include_directories(FuseQuantChainNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(FuseQuantChainNCNN SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(FuseQuantChainNCNN PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(FuseQuantChainNCNN PRIVATE -fno-rtti)
target_link_libraries(FuseQuantChainNCNN PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRTransforms)
