# ncnn 方言库。
add_library(NCNNDialect STATIC
    NCNNDialect.cpp
    NCNNOps.cpp
)
add_dependencies(NCNNDialect NCNNOpsIncGen)
target_include_directories(NCNNDialect PUBLIC
    ${CMAKE_SOURCE_DIR}/include)
target_include_directories(NCNNDialect SYSTEM PUBLIC
    ${CMAKE_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNDialect PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNDialect PRIVATE -fno-rtti)
target_link_libraries(NCNNDialect PUBLIC
    MLIRIR
    MLIRSupport
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces)
