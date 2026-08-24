add_library(TileMatmulForall
    TileMatmulForall.cpp
)
add_dependencies(TileMatmulForall NCNNPassesIncGen)
target_compile_options(TileMatmulForall PRIVATE -fno-rtti)
target_include_directories(TileMatmulForall PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(TileMatmulForall PUBLIC
    MLIRIR
    MLIRPass
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect)
