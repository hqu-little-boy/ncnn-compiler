# ncnn-mlir-driver：.param/.bin -> ncnn 方言 MLIR 模块的端到端驱动。
llvm_map_components_to_libnames(NCNN_DRIVER_LLVM_LIBS support)
add_executable(ncnn-mlir-driver ncnn-mlir-driver.cpp)
target_compile_options(ncnn-mlir-driver PRIVATE -fno-rtti)
target_link_libraries(ncnn-mlir-driver PRIVATE
    NCNNImporter
    NCNNDialect
    NCNNGraph
    MLIRFuncDialect
    MLIRArithDialect
    ${NCNN_DRIVER_LLVM_LIBS})
