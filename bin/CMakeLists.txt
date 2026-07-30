# ncnn-mlir-opt：含 ncnn 方言的 mlir-opt 克隆（round-trip / lit 测试用）。
llvm_map_components_to_libnames(NCNN_OPT_LLVM_LIBS support)
add_executable(ncnn-mlir-opt ncnn-mlir-opt.cpp)
# 本目录在 include 路径上，便于以 "RegisterNCNNDialects.h" 直接引用。
target_include_directories(ncnn-mlir-opt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ncnn-mlir-opt PRIVATE -fno-rtti)
target_link_libraries(ncnn-mlir-opt PRIVATE
    NCNNDialect
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSupport
    ${NCNN_OPT_LLVM_LIBS})
