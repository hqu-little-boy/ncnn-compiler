# lit（.mlir 方言/转换测试）+ gtest（C++ 单元/集成测试）。
add_subdirectory(Unit)

# ── lit (.mlir) 测试 ─────────────────────────────────────
find_program(FILECHECK NAMES FileCheck-21 FileCheck)
# LLVM 的 lit 运行器（Debian 安装在 <prefix>/build/utils/lit）。
set(NCNN_LIT_PY "${LLVM_DIR}/../../../build/utils/lit/lit.py")
get_filename_component(NCNN_LIT_PY "${NCNN_LIT_PY}" ABSOLUTE)

if(FILECHECK AND EXISTS "${NCNN_LIT_PY}")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        @ONLY)
    add_test(NAME ncnn-mlir-lit
        COMMAND python3 ${NCNN_LIT_PY} -v ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
    message(STATUS "lit 或 FileCheck 未找到，跳过 .mlir lit 测试")
endif()
