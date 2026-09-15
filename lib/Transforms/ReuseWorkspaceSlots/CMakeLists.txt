add_library(ReuseWorkspaceSlots
    ReuseWorkspaceSlots.cpp
)
add_dependencies(ReuseWorkspaceSlots NCNNPassesIncGen)
target_compile_options(ReuseWorkspaceSlots PRIVATE -fno-rtti)
target_include_directories(ReuseWorkspaceSlots PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})
target_link_libraries(ReuseWorkspaceSlots PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass)
