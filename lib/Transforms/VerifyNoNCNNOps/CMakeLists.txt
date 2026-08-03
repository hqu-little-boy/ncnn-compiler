add_library(VerifyNoNCNNOps STATIC
    VerifyNoNCNNOps.cpp
)
add_dependencies(VerifyNoNCNNOps NCNNPassesIncGen)
target_compile_options(VerifyNoNCNNOps PRIVATE -fno-rtti)
target_include_directories(VerifyNoNCNNOps PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyNoNCNNOps PUBLIC
    MLIRIR
    MLIRPass)
