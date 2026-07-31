# lit（.mlir 方言/转换测试）+ gtest（C++ 单元/集成测试）。
add_subdirectory(Unit)

# ── lit (.mlir) 测试 ─────────────────────────────────────
find_program(FILECHECK NAMES FileCheck-21 FileCheck)
find_program(MLIR_OPT NAMES mlir-opt-21 mlir-opt REQUIRED)
find_program(MLIR_RUNNER NAMES mlir-runner-21 mlir-runner REQUIRED)
find_library(MLIR_RUNNER_UTILS NAMES mlir_runner_utils PATHS ${LLVM_LIBRARY_DIR} REQUIRED)
find_library(MLIR_C_RUNNER_UTILS NAMES mlir_c_runner_utils PATHS ${LLVM_LIBRARY_DIR} REQUIRED)
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

set(REPEATED_CALL_ARGS
    --mlir-opt ${MLIR_OPT}
    --mlir-runner ${MLIR_RUNNER}
    --input ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/repeated-calls.mlir
    --lowered ${CMAKE_CURRENT_BINARY_DIR}/Runtime/repeated-calls-llvm.mlir
    --runner-utils ${MLIR_RUNNER_UTILS}
    --c-runner-utils ${MLIR_C_RUNNER_UTILS})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Runtime)
if(CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE ASAN_RUNTIME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND REPEATED_CALL_ARGS --asan-runtime ${ASAN_RUNTIME})
endif()
add_test(NAME repeated-call-memory
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/check_repeated_calls.py
            ${REPEATED_CALL_ARGS})
set_tests_properties(repeated-call-memory PROPERTIES
    LABELS "memory;runtime"
    RUN_SERIAL TRUE
    TIMEOUT 120)
