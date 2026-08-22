add_library(FoldLinalgConstantTranspose STATIC
    FoldLinalgConstantTranspose.cpp
)
add_dependencies(FoldLinalgConstantTranspose NCNNPassesIncGen)
target_include_directories(FoldLinalgConstantTranspose PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(FoldLinalgConstantTranspose SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(FoldLinalgConstantTranspose PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(FoldLinalgConstantTranspose PRIVATE -fno-rtti)
target_link_libraries(FoldLinalgConstantTranspose PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRPass)
