# ncnn-mlir-opt：基于 MlirOptMain 的 mlir-opt 克隆。
# 注册全部上游方言/扩展/pass（故支持所有标准 mlir-opt 选项与 pass）+ ncnn 方言。
# 链接全部 dialect/conversion/extension 库，registerAll* 引用的符号才齐全。
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_executable(ncnn-mlir-opt ncnn-mlir-opt.cpp)
# 本目录在 include 路径上，便于以 "RegisterNCNNDialects.h" 直接引用。
target_include_directories(ncnn-mlir-opt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(ncnn-mlir-opt PRIVATE -fno-rtti)
target_link_libraries(ncnn-mlir-opt PRIVATE
    NCNNDialect
    NCNNToFunc
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRMlirOptMain
    MLIRPass)
