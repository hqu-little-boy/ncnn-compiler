add_library(StrategyNCNN
    StrategyNCNN.cpp
)
add_dependencies(StrategyNCNN NCNNPassesIncGen)
target_include_directories(StrategyNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(StrategyNCNN SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(StrategyNCNN PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(StrategyNCNN PRIVATE -fno-rtti)
target_link_libraries(StrategyNCNN PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRTensorDialect)
