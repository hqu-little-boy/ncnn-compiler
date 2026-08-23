add_library(VectorizeNCNN
    VectorizeNCNN.cpp
    LowerVectorTransfersNCNN.cpp
)
add_dependencies(VectorizeNCNN NCNNPassesIncGen)
target_include_directories(VectorizeNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(VectorizeNCNN SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(VectorizeNCNN PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(VectorizeNCNN PRIVATE -fno-rtti)
target_link_libraries(VectorizeNCNN PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMathDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRUBDialect
    MLIRVectorDialect
    MLIRVectorTransforms)
