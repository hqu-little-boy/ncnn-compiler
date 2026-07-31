add_library(VerifyNoNCNNOps STATIC
    VerifyNoNCNNOps.cpp
)
target_compile_options(VerifyNoNCNNOps PRIVATE -fno-rtti)
target_include_directories(VerifyNoNCNNOps PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyNoNCNNOps PUBLIC
    MLIRIR
    MLIRPass)
