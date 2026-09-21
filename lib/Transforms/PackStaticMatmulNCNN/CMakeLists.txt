add_library(PackStaticMatmulNCNN
    PackStaticMatmulNCNN.cpp
)
add_dependencies(PackStaticMatmulNCNN NCNNPassesIncGen)
target_compile_options(PackStaticMatmulNCNN PRIVATE -fno-rtti)
target_include_directories(PackStaticMatmulNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(PackStaticMatmulNCNN PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRTensorDialect)
