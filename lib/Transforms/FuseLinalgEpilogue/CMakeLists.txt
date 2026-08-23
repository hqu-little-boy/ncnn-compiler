add_library(FuseLinalgEpilogue STATIC
    FuseLinalgEpilogue.cpp
)
add_dependencies(FuseLinalgEpilogue NCNNPassesIncGen)
target_include_directories(FuseLinalgEpilogue PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(FuseLinalgEpilogue SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(FuseLinalgEpilogue PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(FuseLinalgEpilogue PRIVATE -fno-rtti)
target_link_libraries(FuseLinalgEpilogue PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect)
