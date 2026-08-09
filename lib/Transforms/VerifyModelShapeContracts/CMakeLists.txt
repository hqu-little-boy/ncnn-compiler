add_library(VerifyModelShapeContracts STATIC
    VerifyModelShapeContracts.cpp
)
add_dependencies(VerifyModelShapeContracts NCNNPassesIncGen)
target_compile_options(VerifyModelShapeContracts PRIVATE -fno-rtti)
target_include_directories(VerifyModelShapeContracts PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyModelShapeContracts PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass)
