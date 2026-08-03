add_library(VerifyNoTosaOps STATIC
    VerifyNoTosaOps.cpp
)
add_dependencies(VerifyNoTosaOps NCNNPassesIncGen)
target_compile_options(VerifyNoTosaOps PRIVATE -fno-rtti)
target_include_directories(VerifyNoTosaOps PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyNoTosaOps PUBLIC
    MLIRIR
    MLIRPass)
