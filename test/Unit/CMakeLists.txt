# gtest 单元/集成测试：解析器 + MLIR importer + SqueezeNet 端到端。
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(unit_tests
    load_squeezenet.cpp
    parser_test.cpp
    shape_program_test.cpp
    malformed_graph_test.cpp
    layer_registry_test.cpp
    ncnn_importer_test.cpp
    load_squeezenet_typed.cpp
)
target_link_libraries(unit_tests PRIVATE
    NCNNImporter
    NCNNDialect
    NCNNGraph
    MLIRFuncDialect
    MLIRArithDialect
    GTest::gtest_main)
target_compile_options(unit_tests PRIVATE -fno-rtti)
# LLVM 的 -isystem include 目录里有自带 cxxabi.h，会遮蔽系统 <cxxabi.h> 并与
# libstdc++ 冲突；gtest 仅在 demangle 时用 <cxxabi.h>，关掉即可避免。
target_compile_definitions(unit_tests PRIVATE
    GTEST_HAS_CXXABI_H_=0
    NCNN_GRAPH_SOURCE_ROOT="${CMAKE_SOURCE_DIR}/test/third_party"
    NCNN_GRAPH_TEST_TEMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/test-fixtures")

gtest_discover_tests(unit_tests)
