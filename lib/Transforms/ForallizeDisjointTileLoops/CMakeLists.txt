add_library(ForallizeDisjointTileLoops
    ForallizeDisjointTileLoops.cpp
)
add_dependencies(ForallizeDisjointTileLoops NCNNPassesIncGen)
target_compile_options(ForallizeDisjointTileLoops PRIVATE -fno-rtti)
target_include_directories(ForallizeDisjointTileLoops PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(ForallizeDisjointTileLoops PUBLIC
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect)
