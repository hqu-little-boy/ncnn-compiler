add_library(LowerVectorMathNCNN
    LowerVectorMathNCNN.cpp
)
add_dependencies(LowerVectorMathNCNN NCNNPassesIncGen)
target_include_directories(LowerVectorMathNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(LowerVectorMathNCNN SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(LowerVectorMathNCNN PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(LowerVectorMathNCNN PRIVATE -fno-rtti)
target_link_libraries(LowerVectorMathNCNN PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMathDialect
    MLIRPass
    MLIRSCFDialect
    MLIRVectorDialect)
