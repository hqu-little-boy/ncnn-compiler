# lit（.mlir 方言/转换测试）+ gtest（C++ 单元/集成测试）。
add_subdirectory(Unit)

# ── lit (.mlir) 测试 ─────────────────────────────────────
find_program(FILECHECK NAMES FileCheck-21 REQUIRED)
find_program(MLIR_OPT NAMES mlir-opt-21 REQUIRED)
find_program(MLIR_RUNNER NAMES mlir-runner-21 REQUIRED)
find_program(MLIR_TRANSLATE NAMES mlir-translate-21 REQUIRED)
find_program(CLANG NAMES clang-21 REQUIRED)
find_program(LLVM_NM NAMES llvm-nm-21 REQUIRED)
find_program(LLVM_READELF NAMES llvm-readelf-21 REQUIRED)
find_library(MLIR_RUNNER_UTILS NAMES mlir_runner_utils PATHS ${LLVM_LIBRARY_DIR} REQUIRED)
find_library(MLIR_C_RUNNER_UTILS NAMES mlir_c_runner_utils PATHS ${LLVM_LIBRARY_DIR} REQUIRED)
add_subdirectory(Numerical)
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

add_test(NAME squeezenet-shared-library
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/examples/squeezenet_v1.1.param
        --bin ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/examples/squeezenet_v1.1.bin
        --model-name squeezenet_v1_1
        --expected-undefined expf,free,malloc,memset
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/squeezenet
        --emit-manifest
        --verify-execution)
set_tests_properties(squeezenet-shared-library PROPERTIES
    LABELS "native;integration"
    RUN_SERIAL TRUE
    TIMEOUT 180)

add_test(NAME ncnn-compile-cli
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_ncnn_compile.py
        --compiler ${CMAKE_BINARY_DIR}/tools/ncnn-compile
        --debug-compiler ${CMAKE_SOURCE_DIR}/tools/compile_ncnn_model.py
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --build-dir ${CMAKE_BINARY_DIR}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/Numerical/fixtures/relu.param
        --bin ${CMAKE_CURRENT_BINARY_DIR}/Numerical/fixtures/empty.bin
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/ncnn-compile-cli)
set_tests_properties(ncnn-compile-cli PROPERTIES
    LABELS "native;integration;cli"
    RUN_SERIAL TRUE
    TIMEOUT 180)
