# lit（.mlir 方言/转换测试）+ gtest（C++ 单元/集成测试）。
add_subdirectory(Unit)

# ── lit (.mlir) 测试 ─────────────────────────────────────
find_program(FILECHECK NAMES FileCheck-21 REQUIRED)
find_program(MLIR_OPT NAMES mlir-opt-21 REQUIRED)
# find_program(MLIR_RUNNER NAMES mlir-runner-21 REQUIRED)
find_program(MLIR_TRANSLATE NAMES mlir-translate-21 REQUIRED)
find_program(CLANG NAMES clang-21 REQUIRED)
find_program(LLVM_NM NAMES llvm-nm-21 REQUIRED)
find_program(LLVM_READELF NAMES llvm-readelf-21 REQUIRED)
find_program(LLVM_AS NAMES llvm-as-21 REQUIRED)
find_program(LLVM_OBJDUMP NAMES llvm-objdump-21 REQUIRED)
find_program(LLVM_SIZE NAMES llvm-size-21 REQUIRED)
find_program(LLVM_MCA NAMES llvm-mca-21 REQUIRED)
# find_library(MLIR_RUNNER_UTILS NAMES mlir_runner_utils PATHS ${LLVM_LIBRARY_DIR} REQUIRED)
# find_library(MLIR_C_RUNNER_UTILS NAMES mlir_c_runner_utils PATHS ${LLVM_LIBRARY_DIR} REQUIRED)
add_subdirectory(Numerical)
if(NCNN_ENABLE_VULKAN_V0_TESTS)
    add_subdirectory(Vulkan)
endif()
# LLVM 的 lit 运行器（Debian 安装在 <prefix>/build/utils/lit）。可在
# 独立 LLVM/MLIR build 或 sanitizer toolchain 中显式指定其源码路径。
set(NCNN_LIT_PY "" CACHE FILEPATH "Path to the LLVM lit.py test runner")
if(NOT NCNN_LIT_PY)
    set(NCNN_LIT_PY "${LLVM_DIR}/../../../build/utils/lit/lit.py")
endif()
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
    message(FATAL_ERROR
        "lit runner is required when BUILD_TESTING=ON; set NCNN_LIT_PY to a matching LLVM lit.py")
endif()

add_executable(vulkan_abi_contract_test Native/vulkan_abi_contract.c)
target_compile_features(vulkan_abi_contract_test PRIVATE c_std_11)
target_include_directories(vulkan_abi_contract_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/Numerical/generated/relu_scalar)
add_dependencies(vulkan_abi_contract_test compile_relu_scalar)
add_test(NAME vulkan-abi-c-contract
    COMMAND $<TARGET_FILE:vulkan_abi_contract_test>)
set_tests_properties(vulkan-abi-c-contract PROPERTIES
    LABELS "abi;tooling"
    TIMEOUT 30)

add_test(NAME vulkan-v1-cpu-baseline
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_vulkan_v1_cpu_baseline.py
        --compiler $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/Numerical/fixtures/relu.param
        --bin ${CMAKE_CURRENT_BINARY_DIR}/Numerical/fixtures/empty.bin
        --baseline ${CMAKE_CURRENT_SOURCE_DIR}/Native/vulkan_v1_cpu_baseline.json
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/vulkan-v1-cpu-baseline
        --result ${CMAKE_CURRENT_BINARY_DIR}/Native/vulkan-v1-cpu-baseline/evidence.json)
set_tests_properties(vulkan-v1-cpu-baseline PROPERTIES
    LABELS "abi;integration;cpu-baseline"
    RUN_SERIAL TRUE
    TIMEOUT 180)

add_test(NAME perf-json-summary
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_perf_json_summary.py)
set_tests_properties(perf-json-summary PROPERTIES
    LABELS "performance;tooling"
    RUN_SERIAL TRUE
    TIMEOUT 30)
# T-C1：plan-hash 字段白名单守卫（源码级）+ 行为级用例见
# test/Transforms/EmitModelPlan/plan-hash-whitelist.mlir。
add_test(NAME plan-hash-whitelist
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_plan_hash_whitelist.py)
set_tests_properties(plan-hash-whitelist PROPERTIES
    LABELS "tooling"
    TIMEOUT 30)
add_test(NAME perf-attribution
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_perf_attribution.py)
set_tests_properties(perf-attribution PROPERTIES
    LABELS "performance;tooling"
    RUN_SERIAL TRUE
    TIMEOUT 30)
# T-C2：tools/attr/ 拆分后的纯 Python 单元测试（无编译产物依赖）。
add_test(NAME attr-partition
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/test_attr_partition.py)
set_tests_properties(attr-partition PROPERTIES
    LABELS "performance;tooling"
    RUN_SERIAL TRUE
    TIMEOUT 30)
add_test(NAME attr-report
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/test_attr_report.py)
set_tests_properties(attr-report PROPERTIES
    LABELS "performance;tooling"
    RUN_SERIAL TRUE
    TIMEOUT 30)
add_test(NAME profile-runtime
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_profile_runtime.py
            --cc ${CLANG}
            --runtime ${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_runtime.c
            --runtime ${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_json_writer.c)
set_tests_properties(profile-runtime PROPERTIES
    LABELS "performance;runtime;tooling"
    RUN_SERIAL TRUE
    TIMEOUT 30)

# ncnn.* 属性键单一事实源：KernelContract.hpp 之外不得再出现裸键字面量。
add_test(NAME attribute-whitelist
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_attribute_whitelist.py)
set_tests_properties(attribute-whitelist PROPERTIES
    LABELS "tooling"
    TIMEOUT 30)

# repeated-call-memory 测试暂时屏蔽，当前阶段不做多次调用稳定性测试。
# set(REPEATED_CALL_ARGS
#     --mlir-opt ${MLIR_OPT}
#     --mlir-runner ${MLIR_RUNNER}
#     --input ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/repeated-calls.mlir
#     --lowered ${CMAKE_CURRENT_BINARY_DIR}/Runtime/repeated-calls-llvm.mlir
#     --runner-utils ${MLIR_RUNNER_UTILS}
#     --c-runner-utils ${MLIR_C_RUNNER_UTILS})
# file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Runtime)
# if(CMAKE_CXX_FLAGS MATCHES "fsanitize=address")
#     execute_process(
#         COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
#         OUTPUT_VARIABLE ASAN_RUNTIME
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#         COMMAND_ERROR_IS_FATAL ANY)
#     list(APPEND REPEATED_CALL_ARGS --asan-runtime ${ASAN_RUNTIME})
# endif()
# add_test(NAME repeated-call-memory
#     COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Runtime/check_repeated_calls.py
#             ${REPEATED_CALL_ARGS})
# set_tests_properties(repeated-call-memory PROPERTIES
#     LABELS "memory;runtime"
#     RUN_SERIAL TRUE
#     TIMEOUT 120)

add_test(NAME squeezenet-shared-library
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/examples/squeezenet_v1.1.param
        --bin ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/examples/squeezenet_v1.1.bin
        --model-name squeezenet_v1_1
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/squeezenet
        --emit-manifest
        --verify-execution)
set_tests_properties(squeezenet-shared-library PROPERTIES
    LABELS "native;integration;static-baseline"
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
        --static-baseline ${CMAKE_CURRENT_SOURCE_DIR}/Native/static_artifact_baseline.json
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/ncnn-compile-cli
        --llvm-lib-dir ${LLVM_LIBRARY_DIR})
set_tests_properties(ncnn-compile-cli PROPERTIES
    LABELS "native;integration;cli;static-baseline"
    RUN_SERIAL TRUE
    TIMEOUT 180)

add_test(NAME fp16-arithmetic-codegen
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_fp16_arithmetic.py
        --compiler $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --objdump ${LLVM_OBJDUMP}
        --size ${LLVM_SIZE}
        --mca ${LLVM_MCA}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/Numerical/fixtures/convolution.param
        --bin ${CMAKE_CURRENT_BINARY_DIR}/Numerical/fixtures/convolution.bin
        --codegen-dir ${CMAKE_CURRENT_BINARY_DIR}/Numerical/generated/convolution_codegen_fp16
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/fp16-arithmetic)
set_tests_properties(fp16-arithmetic-codegen PROPERTIES
    LABELS "native;integration;precision;performance"
    RUN_SERIAL TRUE
    TIMEOUT 180)

# P16: independent modulo-2^32 operator oracles and real ISA object checks.
add_test(NAME int8-target-cli
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_int8_target.py
        --compiler $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/Numerical/fixtures/relu.param
        --bin ${CMAKE_CURRENT_BINARY_DIR}/Numerical/fixtures/empty.bin
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/int8-target)
set_tests_properties(int8-target-cli PROPERTIES
    LABELS "native;integration;cli;int8"
    RUN_SERIAL TRUE
    TIMEOUT 180)

if(NCNN_ENABLE_VULKAN_PRODUCT AND NCNN_ENABLE_VULKAN_V0_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(_ncnn_vulkan_product_test_args
        --compiler $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --mlir-opt ${NCNN_MLIR_OPT_EXECUTABLE}
        --spirv-val ${NCNN_SPIRV_VAL_EXECUTABLE}
        --spirv-dis ${NCNN_SPIRV_DIS_EXECUTABLE}
        --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --host-cxx ${CMAKE_CXX_COMPILER}
        --loader-source ${CMAKE_CURRENT_SOURCE_DIR}/Native/vulkan_product_loader.cpp
        --vulkan-include-dir ${PROJECT_SOURCE_DIR}/include
        --llvm-major ${LLVM_VERSION_MAJOR}
        --llvm-lib-dir ${LLVM_LIBRARY_DIR}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/Vulkan/relu.param
        --bin ${CMAKE_CURRENT_BINARY_DIR}/Numerical/fixtures/empty.bin
        --unsupported-param ${CMAKE_CURRENT_SOURCE_DIR}/Numerical/fixtures/convolution.param
        --unsupported-bin ${CMAKE_CURRENT_BINARY_DIR}/Numerical/fixtures/convolution.bin
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/vulkan-product-cli)
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND _ncnn_vulkan_product_test_args --llvm-shared)
    endif()
    foreach(_dependency IN LISTS
            NCNN_VULKAN_STATIC_LLVM_NEEDED_LIBRARIES
            NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        list(APPEND _ncnn_vulkan_product_test_args
            --allowed-dependency=${_dependency})
    endforeach()
    foreach(_dependency IN LISTS NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        list(APPEND _ncnn_vulkan_product_test_args
            --sanitizer-dependency=${_dependency})
    endforeach()
    set(_ncnn_vulkan_library_dirs)
    if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
        foreach(_cxx_arg IN LISTS NCNN_VULKAN_HOST_SANITIZER_ARGS)
            list(APPEND _ncnn_vulkan_product_test_args --cxx-arg=${_cxx_arg})
        endforeach()
        list(APPEND _ncnn_vulkan_product_test_args
            --sanitizer-runtime-dir ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
        list(APPEND _ncnn_vulkan_library_dirs
            ${NCNN_VULKAN_SANITIZER_RUNTIME_DIR})
    endif()
    list(APPEND _ncnn_vulkan_library_dirs ${LLVM_LIBRARY_DIR})
    list(JOIN _ncnn_vulkan_library_dirs ":" _ncnn_vulkan_library_path)
    set(_ncnn_vulkan_product_environment
        "LD_LIBRARY_PATH=${_ncnn_vulkan_library_path}:$ENV{LD_LIBRARY_PATH}")
    if(COMPILER_SANITIZERS MATCHES "(^|;)(address|undefined)(;|$)")
        find_file(_ncnn_sanitizer_vulkan_icd
            NAMES lvp_icd.json
            PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d
            REQUIRED)
        list(APPEND _ncnn_vulkan_product_environment
            "VK_ICD_FILENAMES=${_ncnn_sanitizer_vulkan_icd}")
    endif()
    add_test(NAME ncnn-compile-vulkan-product
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_ncnn_compile_vulkan.py
            ${_ncnn_vulkan_product_test_args})
    set_tests_properties(ncnn-compile-vulkan-product PROPERTIES
        LABELS "native;integration;cli;vulkan;product"
        RUN_SERIAL TRUE
        TIMEOUT 900
        ENVIRONMENT "${_ncnn_vulkan_product_environment}")
endif()

# Unsupported hosts still cross-compile; the script explicitly skips execution.
foreach(INT8_KIND IN ITEMS matmul depthwise)
    add_test(NAME int8-${INT8_KIND}-native
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/Native/check_int8_kernels.py
            --kind ${INT8_KIND}
            --opt $<TARGET_FILE:ncnn-mlir-opt>
            --translate ${MLIR_TRANSLATE}
            --clang ${CLANG}
            --objdump ${LLVM_OBJDUMP}
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/Native/int8-kernels)
    set_tests_properties(int8-${INT8_KIND}-native PROPERTIES
        LABELS "native;integration;int8;codegen"
        RUN_SERIAL TRUE
        TIMEOUT 180)
endforeach()
