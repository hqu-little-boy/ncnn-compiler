find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
find_program(SPIRV_VAL NAMES spirv-val REQUIRED)
find_program(SPIRV_DIS NAMES spirv-dis REQUIRED)
find_program(VULKANINFO NAMES vulkaninfo REQUIRED)
add_test(NAME vulkan-v5-comparison-validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_run_v5.py)
set_tests_properties(vulkan-v5-comparison-validation PROPERTIES
    LABELS "vulkan;unit;v5"
    TIMEOUT 30)
find_file(V0_LLVMPIPE_ICD
    NAMES lvp_icd.json
    PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d
    REQUIRED)
find_file(V0_NVIDIA_SYSTEM_ICD
    NAMES nvidia_icd.json
    PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d
    REQUIRED)
find_library(V0_NVIDIA_EGL_LIBRARY
    NAMES EGL_nvidia libEGL_nvidia.so.0
    REQUIRED)
# NVIDIA supports the EGL frontend as a headless Vulkan ICD. The GLX frontend
# in driver 615.71.09 leaks a libdrm device on unload; EGL cleans up normally.
# Use the same device and normal library unloading in both test configurations.
file(READ "${V0_NVIDIA_SYSTEM_ICD}" V0_NVIDIA_ICD_DOCUMENT)
string(JSON V0_NVIDIA_ICD_DOCUMENT SET "${V0_NVIDIA_ICD_DOCUMENT}"
    ICD library_path "\"${V0_NVIDIA_EGL_LIBRARY}\"")
set(V0_NVIDIA_ICD "${CMAKE_CURRENT_BINARY_DIR}/nvidia-egl-icd.json")
file(WRITE "${V0_NVIDIA_ICD}" "${V0_NVIDIA_ICD_DOCUMENT}\n")

add_executable(vulkan_add_runner vulkan_add_runner.cpp)
target_link_libraries(vulkan_add_runner PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
target_compile_options(vulkan_add_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_runtime_runner
    vulkan_runtime_runner.cpp
    vulkan_runtime_abi_fixture.cpp)
target_link_libraries(vulkan_runtime_runner PRIVATE
    NCNNVulkanRuntime ${CMAKE_DL_LIBS} Threads::Threads)
target_compile_options(vulkan_runtime_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_runtime_weight_runner vulkan_runtime_weight_runner.cpp)
target_link_libraries(vulkan_runtime_weight_runner PRIVATE
    NCNNVulkanRuntime ${CMAKE_DL_LIBS})
target_compile_options(vulkan_runtime_weight_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_v0_spirv_extract extract_vulkan_spirv.cpp)
target_include_directories(vulkan_v0_spirv_extract PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})
target_compile_options(vulkan_v0_spirv_extract PRIVATE -fno-rtti)
target_link_libraries(vulkan_v0_spirv_extract PRIVATE
    MLIRGPUDialect MLIRSPIRVDialect MLIRParser MLIRIR MLIRSupport LLVMSupport)
set_target_properties(vulkan_v0_spirv_extract PROPERTIES
    BUILD_RPATH "${LLVM_LIBRARY_DIR}")

set(V0_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-add)
set(V0_CPU_LIBRARY ${V0_CPU_DIR}/libv0_add.so)
set(V0_CPU_HEADER ${V0_CPU_DIR}/v0_add.h)
set(V0_CPU_MANIFEST ${V0_CPU_DIR}/v0_add.json)
set(V0_EMPTY_BIN ${CMAKE_CURRENT_BINARY_DIR}/empty.bin)
file(WRITE ${V0_EMPTY_BIN} "")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/generate_bins.py
    ${CMAKE_CURRENT_SOURCE_DIR}/generate_conv1x1_bin.py)
add_custom_command(
    OUTPUT ${V0_CPU_LIBRARY} ${V0_CPU_HEADER} ${V0_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V0_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/add.param
        --bin ${V0_EMPTY_BIN}
        --model-name v0_add
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V0_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/add.param
    VERBATIM)
add_custom_target(v0_cpu_add_fixture
    DEPENDS ${V0_CPU_LIBRARY} ${V0_CPU_HEADER} ${V0_CPU_MANIFEST})
add_custom_target(vulkan-v0-build
    DEPENDS vulkan_add_runner vulkan_v0_spirv_extract v0_cpu_add_fixture)

set(V2_RELU_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-relu)
set(V2_RELU_CPU_LIBRARY ${V2_RELU_CPU_DIR}/libv2_relu.so)
set(V2_RELU_CPU_HEADER ${V2_RELU_CPU_DIR}/v2_relu.h)
set(V2_RELU_CPU_MANIFEST ${V2_RELU_CPU_DIR}/v2_relu.json)
add_custom_command(
    OUTPUT ${V2_RELU_CPU_LIBRARY} ${V2_RELU_CPU_HEADER} ${V2_RELU_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_RELU_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_relu
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_RELU_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/relu.param
    VERBATIM)
add_custom_target(v2_cpu_relu_fixture
    DEPENDS ${V2_RELU_CPU_LIBRARY} ${V2_RELU_CPU_HEADER} ${V2_RELU_CPU_MANIFEST})

set(V2_IDENTITY_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-identity)
set(V2_IDENTITY_CPU_LIBRARY ${V2_IDENTITY_CPU_DIR}/libv2_identity.so)
set(V2_IDENTITY_CPU_HEADER ${V2_IDENTITY_CPU_DIR}/v2_identity.h)
set(V2_IDENTITY_CPU_MANIFEST ${V2_IDENTITY_CPU_DIR}/v2_identity.json)
add_custom_command(
    OUTPUT ${V2_IDENTITY_CPU_LIBRARY} ${V2_IDENTITY_CPU_HEADER} ${V2_IDENTITY_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_IDENTITY_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/identity.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_identity
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_IDENTITY_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/identity.param
    VERBATIM)
add_custom_target(v2_cpu_identity_fixture
    DEPENDS ${V2_IDENTITY_CPU_LIBRARY} ${V2_IDENTITY_CPU_HEADER} ${V2_IDENTITY_CPU_MANIFEST})

set(V2_TRANSPOSE_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-transpose)
set(V2_TRANSPOSE_CPU_LIBRARY ${V2_TRANSPOSE_CPU_DIR}/libv2_transpose.so)
set(V2_TRANSPOSE_CPU_HEADER ${V2_TRANSPOSE_CPU_DIR}/v2_transpose.h)
set(V2_TRANSPOSE_CPU_MANIFEST ${V2_TRANSPOSE_CPU_DIR}/v2_transpose.json)
add_custom_command(
    OUTPUT ${V2_TRANSPOSE_CPU_LIBRARY} ${V2_TRANSPOSE_CPU_HEADER} ${V2_TRANSPOSE_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_TRANSPOSE_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/transpose.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_transpose
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_TRANSPOSE_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/transpose.param
    VERBATIM)
add_custom_target(v2_cpu_transpose_fixture
    DEPENDS ${V2_TRANSPOSE_CPU_LIBRARY} ${V2_TRANSPOSE_CPU_HEADER} ${V2_TRANSPOSE_CPU_MANIFEST})

set(V2_RESHAPE_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-reshape)
set(V2_RESHAPE_CPU_LIBRARY ${V2_RESHAPE_CPU_DIR}/libv2_reshape.so)
set(V2_RESHAPE_CPU_HEADER ${V2_RESHAPE_CPU_DIR}/v2_reshape.h)
set(V2_RESHAPE_CPU_MANIFEST ${V2_RESHAPE_CPU_DIR}/v2_reshape.json)
add_custom_command(
    OUTPUT ${V2_RESHAPE_CPU_LIBRARY} ${V2_RESHAPE_CPU_HEADER} ${V2_RESHAPE_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_RESHAPE_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/reshape.param
        --bin ${V0_EMPTY_BIN}
        --model-name v2_reshape
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_RESHAPE_CPU_DIR}
    DEPENDS
        ncnn-compile
        ncnn-mlir-driver
        ncnn-mlir-opt
        ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/reshape.param
    VERBATIM)
add_custom_target(v2_cpu_reshape_fixture
    DEPENDS ${V2_RESHAPE_CPU_LIBRARY} ${V2_RESHAPE_CPU_HEADER} ${V2_RESHAPE_CPU_MANIFEST})

set(V2_CONV3X3_BIN ${CMAKE_CURRENT_BINARY_DIR}/convolution-3x3.bin)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/generate_bins.py
        ${V2_CONV3X3_BIN} convolution
    COMMAND_ERROR_IS_FATAL ANY)
set(V2_CONV3X3_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/convolution.param)
set(V2_CONV3X3_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-3x3)
set(V2_CONV3X3_CPU_LIBRARY ${V2_CONV3X3_CPU_DIR}/libv2_conv3x3.so)
set(V2_CONV3X3_CPU_HEADER ${V2_CONV3X3_CPU_DIR}/v2_conv3x3.h)
set(V2_CONV3X3_CPU_MANIFEST ${V2_CONV3X3_CPU_DIR}/v2_conv3x3.json)
add_custom_command(
    OUTPUT ${V2_CONV3X3_CPU_LIBRARY} ${V2_CONV3X3_CPU_HEADER} ${V2_CONV3X3_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV3X3_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV3X3_PARAM}
        --bin ${V2_CONV3X3_BIN}
        --model-name v2_conv3x3
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV3X3_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV3X3_PARAM} ${V2_CONV3X3_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv3x3_fixture
    DEPENDS ${V2_CONV3X3_CPU_LIBRARY} ${V2_CONV3X3_CPU_HEADER}
            ${V2_CONV3X3_CPU_MANIFEST})

set(V2_CONV3X3_NO_BIAS_BIN ${CMAKE_CURRENT_BINARY_DIR}/convolution-3x3-no-bias.bin)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/generate_bins.py
        ${V2_CONV3X3_NO_BIAS_BIN} convolution_no_bias
    COMMAND_ERROR_IS_FATAL ANY)
set(V2_CONV3X3_NO_BIAS_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/../Numerical/fixtures/convolution_no_bias.param)
set(V2_CONV3X3_NO_BIAS_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-3x3-no-bias)
set(V2_CONV3X3_NO_BIAS_CPU_LIBRARY ${V2_CONV3X3_NO_BIAS_CPU_DIR}/libv2_conv3x3_no_bias.so)
set(V2_CONV3X3_NO_BIAS_CPU_HEADER ${V2_CONV3X3_NO_BIAS_CPU_DIR}/v2_conv3x3_no_bias.h)
set(V2_CONV3X3_NO_BIAS_CPU_MANIFEST ${V2_CONV3X3_NO_BIAS_CPU_DIR}/v2_conv3x3_no_bias.json)
add_custom_command(
    OUTPUT ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY} ${V2_CONV3X3_NO_BIAS_CPU_HEADER} ${V2_CONV3X3_NO_BIAS_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV3X3_NO_BIAS_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV3X3_NO_BIAS_PARAM}
        --bin ${V2_CONV3X3_NO_BIAS_BIN}
        --model-name v2_conv3x3_no_bias
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV3X3_NO_BIAS_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV3X3_NO_BIAS_PARAM} ${V2_CONV3X3_NO_BIAS_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv3x3_no_bias_fixture
    DEPENDS ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY} ${V2_CONV3X3_NO_BIAS_CPU_HEADER}
            ${V2_CONV3X3_NO_BIAS_CPU_MANIFEST})

set(V2_CONV3X3_STRIDE2_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/convolution_3x3_stride2.param)
set(V2_CONV3X3_STRIDE2_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-3x3-stride2)
set(V2_CONV3X3_STRIDE2_CPU_LIBRARY ${V2_CONV3X3_STRIDE2_CPU_DIR}/libv2_conv3x3_stride2.so)
set(V2_CONV3X3_STRIDE2_CPU_HEADER ${V2_CONV3X3_STRIDE2_CPU_DIR}/v2_conv3x3_stride2.h)
set(V2_CONV3X3_STRIDE2_CPU_MANIFEST ${V2_CONV3X3_STRIDE2_CPU_DIR}/v2_conv3x3_stride2.json)
add_custom_command(
    OUTPUT ${V2_CONV3X3_STRIDE2_CPU_LIBRARY} ${V2_CONV3X3_STRIDE2_CPU_HEADER} ${V2_CONV3X3_STRIDE2_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV3X3_STRIDE2_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV3X3_STRIDE2_PARAM}
        --bin ${V2_CONV3X3_BIN}
        --model-name v2_conv3x3_stride2
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV3X3_STRIDE2_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV3X3_STRIDE2_PARAM} ${V2_CONV3X3_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv3x3_stride2_fixture
    DEPENDS ${V2_CONV3X3_STRIDE2_CPU_LIBRARY} ${V2_CONV3X3_STRIDE2_CPU_HEADER}
            ${V2_CONV3X3_STRIDE2_CPU_MANIFEST})

set(V2_CONV1X1_BIN ${CMAKE_CURRENT_BINARY_DIR}/convolution-1x1.bin)
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_conv1x1_bin.py
            ${V2_CONV1X1_BIN}
    COMMAND_ERROR_IS_FATAL ANY)
set(V2_CONV1X1_PARAM ${CMAKE_CURRENT_SOURCE_DIR}/convolution_1x1.param)
set(V2_CONV1X1_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-convolution-1x1)
set(V2_CONV1X1_CPU_LIBRARY ${V2_CONV1X1_CPU_DIR}/libv2_conv1x1.so)
set(V2_CONV1X1_CPU_HEADER ${V2_CONV1X1_CPU_DIR}/v2_conv1x1.h)
set(V2_CONV1X1_CPU_MANIFEST ${V2_CONV1X1_CPU_DIR}/v2_conv1x1.json)
add_custom_command(
    OUTPUT ${V2_CONV1X1_CPU_LIBRARY} ${V2_CONV1X1_CPU_HEADER} ${V2_CONV1X1_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V2_CONV1X1_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${V2_CONV1X1_PARAM}
        --bin ${V2_CONV1X1_BIN}
        --model-name v2_conv1x1
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V2_CONV1X1_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${V2_CONV1X1_PARAM} ${V2_CONV1X1_BIN}
    VERBATIM)
add_custom_target(v2_cpu_conv1x1_fixture
    DEPENDS ${V2_CONV1X1_CPU_LIBRARY} ${V2_CONV1X1_CPU_HEADER}
            ${V2_CONV1X1_CPU_MANIFEST})

add_custom_target(vulkan-v2-build ALL
    DEPENDS vulkan-v0-build v2_cpu_relu_fixture v2_cpu_identity_fixture
            v2_cpu_transpose_fixture v2_cpu_reshape_fixture
            v2_cpu_conv1x1_fixture v2_cpu_conv3x3_fixture
            v2_cpu_conv3x3_no_bias_fixture v2_cpu_conv3x3_stride2_fixture)

set(V3_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-add-relu)
set(V3_CPU_LIBRARY ${V3_CPU_DIR}/libv3_add_relu.so)
set(V3_CPU_HEADER ${V3_CPU_DIR}/v3_add_relu.h)
set(V3_CPU_MANIFEST ${V3_CPU_DIR}/v3_add_relu.json)
add_custom_command(
    OUTPUT ${V3_CPU_LIBRARY} ${V3_CPU_HEADER} ${V3_CPU_MANIFEST}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${V3_CPU_DIR}
    COMMAND $<TARGET_FILE:ncnn-compile>
        --driver $<TARGET_FILE:ncnn-mlir-driver>
        --opt $<TARGET_FILE:ncnn-mlir-opt>
        --translate ${MLIR_TRANSLATE}
        --clang ${CLANG}
        --nm ${LLVM_NM}
        --readelf ${LLVM_READELF}
        --llvm-as ${LLVM_AS}
        --param ${CMAKE_CURRENT_SOURCE_DIR}/add_relu.param
        --bin ${V0_EMPTY_BIN}
        --model-name v3_add_relu
        --precision=f32
        --threads=1
        --emit-manifest
        --verify-execution
        --output-dir ${V3_CPU_DIR}
    DEPENDS ncnn-compile ncnn-mlir-driver ncnn-mlir-opt
            ${CMAKE_CURRENT_SOURCE_DIR}/add_relu.param ${V0_EMPTY_BIN}
    VERBATIM)
add_custom_target(v3_cpu_add_relu_fixture
    DEPENDS ${V3_CPU_LIBRARY} ${V3_CPU_HEADER} ${V3_CPU_MANIFEST})
add_custom_target(vulkan-v3-build ALL
    DEPENDS vulkan-v2-build vulkan_runtime_runner
            vulkan_runtime_weight_runner v3_cpu_add_relu_fixture)

function(ncnn_add_vulkan_v0_test test_name device_match output_name)
    set(test_args
        --mlir-opt ${MLIR_OPT}
        --spirv-val ${SPIRV_VAL}
        --spirv-dis ${SPIRV_DIS}
        --vulkaninfo ${VULKANINFO}
        --kernel ${CMAKE_CURRENT_SOURCE_DIR}/gpu_func_add_probe.mlir
        --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
        --runner $<TARGET_FILE:vulkan_add_runner>
        --cpu-library ${V0_CPU_LIBRARY}
        --cpu-symbol v0_add
        --device-match ${device_match}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/gpu-func/${output_name}
        --validation)
    if(ARGC GREATER 3)
        list(APPEND test_args ${ARGN})
    endif()
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_v0.py
            ${test_args})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "vulkan;device;integration"
        RUN_SERIAL TRUE
        TIMEOUT 180)
endfunction()

ncnn_add_vulkan_v0_test(vulkan-v0-add-nvidia NVIDIA nvidia
    --icd ${V0_NVIDIA_ICD} --run-negative)
ncnn_add_vulkan_v0_test(vulkan-v0-add-llvmpipe llvmpipe llvmpipe --icd ${V0_LLVMPIPE_ICD})

function(ncnn_add_vulkan_v2_test test_name device_match output_name operation
         kernel_file binary_symbol cpu_library cpu_symbol model_param model_bin)
    set(test_args
        --ncnn-driver $<TARGET_FILE:ncnn-mlir-driver>
        --ncnn-mlir-opt $<TARGET_FILE:ncnn-mlir-opt>
        --mlir-opt ${MLIR_OPT}
        --spirv-val ${SPIRV_VAL}
        --spirv-dis ${SPIRV_DIS}
        --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
        --runner $<TARGET_FILE:vulkan_add_runner>
        --operation ${operation}
        --elements 257
        --binary-symbol ${binary_symbol}
        --cpu-library ${cpu_library}
        --cpu-symbol ${cpu_symbol}
        --model-param ${CMAKE_CURRENT_SOURCE_DIR}/${model_param}
        --model-bin ${model_bin}
        --device-match ${device_match}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/v2/${output_name}
        --validation)
    if(device_match STREQUAL "NVIDIA")
        list(APPEND test_args --icd ${V0_NVIDIA_ICD})
    endif()
    if(NOT kernel_file STREQUAL "from-model")
        list(APPEND test_args
            --kernel ${CMAKE_CURRENT_SOURCE_DIR}/../Conversion/NCNNToVulkan/${kernel_file})
    endif()
    if(ARGC GREATER 10)
        list(APPEND test_args ${ARGN})
    endif()
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_v2.py
            ${test_args})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "vulkan;device;integration;v2"
        RUN_SERIAL TRUE
        TIMEOUT 180)
endfunction()

function(ncnn_add_vulkan_v3_runtime_test test_name device_match output_name)
    set(test_args
        --ncnn-driver $<TARGET_FILE:ncnn-mlir-driver>
        --ncnn-mlir-opt $<TARGET_FILE:ncnn-mlir-opt>
        --mlir-opt ${MLIR_OPT}
        --spirv-val ${SPIRV_VAL}
        --spirv-dis ${SPIRV_DIS}
        --spirv-extract $<TARGET_FILE:vulkan_v0_spirv_extract>
        --v2-runner $<TARGET_FILE:vulkan_add_runner>
        --runtime-runner $<TARGET_FILE:vulkan_runtime_runner>
        --weight-runner $<TARGET_FILE:vulkan_runtime_weight_runner>
        --conv-cpu-library ${V2_CONV3X3_CPU_LIBRARY}
        --conv-model-param ${V2_CONV3X3_PARAM}
        --conv-model-bin ${V2_CONV3X3_BIN}
        --cpu-library ${V3_CPU_LIBRARY}
        --cpu-symbol v3_add_relu
        --add-cpu-library ${V0_CPU_LIBRARY}
        --add-cpu-symbol v0_add
        --relu-cpu-library ${V2_RELU_CPU_LIBRARY}
        --relu-cpu-symbol v2_relu
        --empty-bin ${V0_EMPTY_BIN}
        --device-match ${device_match}
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/v3/${output_name}
        --validation)
    if(ARGC GREATER 3)
        list(APPEND test_args ${ARGN})
    endif()
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_v3.py
            ${test_args})
    set_tests_properties(${test_name} PROPERTIES
        LABELS "vulkan;device;integration;v3"
        RUN_SERIAL TRUE
        TIMEOUT 300)
endfunction()

ncnn_add_vulkan_v3_runtime_test(
    vulkan-v3-runtime-nvidia NVIDIA nvidia --icd ${V0_NVIDIA_ICD})
ncnn_add_vulkan_v3_runtime_test(
    vulkan-v3-runtime-llvmpipe llvmpipe llvmpipe --icd ${V0_LLVMPIPE_ICD})

ncnn_add_vulkan_v2_test(
    vulkan-v2-add-nvidia NVIDIA add-nvidia add from-model model_vulkan
    ${V0_CPU_LIBRARY} v0_add add.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-add-llvmpipe llvmpipe add-llvmpipe add from-model model_vulkan
    ${V0_CPU_LIBRARY} v0_add add.param ${V0_EMPTY_BIN} --icd ${V0_LLVMPIPE_ICD})
ncnn_add_vulkan_v2_test(
    vulkan-v2-relu-nvidia NVIDIA relu-nvidia relu from-model model_vulkan
    ${V2_RELU_CPU_LIBRARY} v2_relu relu.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-relu-llvmpipe llvmpipe relu-llvmpipe relu from-model model_vulkan
    ${V2_RELU_CPU_LIBRARY} v2_relu relu.param ${V0_EMPTY_BIN}
    --icd ${V0_LLVMPIPE_ICD})
ncnn_add_vulkan_v2_test(
    vulkan-v2-copy-nvidia NVIDIA copy-nvidia copy lower-copy.mlir copy_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-copy-llvmpipe llvmpipe copy-llvmpipe copy lower-copy.mlir copy_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN}
    --icd ${V0_LLVMPIPE_ICD})
ncnn_add_vulkan_v2_test(
    vulkan-v2-identity-nvidia NVIDIA identity-nvidia identity from-model model_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN})
ncnn_add_vulkan_v2_test(
    vulkan-v2-identity-llvmpipe llvmpipe identity-llvmpipe identity from-model model_vulkan
    ${V2_IDENTITY_CPU_LIBRARY} v2_identity identity.param ${V0_EMPTY_BIN}
    --icd ${V0_LLVMPIPE_ICD})
ncnn_add_vulkan_v2_test(
    vulkan-v2-transpose-nvidia NVIDIA transpose-nvidia transpose from-model model_vulkan
    ${V2_TRANSPOSE_CPU_LIBRARY} v2_transpose transpose.param ${V0_EMPTY_BIN}
    --elements 255 --input-shape 51x5x1 --permutation 1,0,2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-transpose-llvmpipe llvmpipe transpose-llvmpipe transpose from-model model_vulkan
    ${V2_TRANSPOSE_CPU_LIBRARY} v2_transpose transpose.param ${V0_EMPTY_BIN}
    --icd ${V0_LLVMPIPE_ICD} --elements 255
    --input-shape 51x5x1 --permutation 1,0,2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-reshape-nvidia NVIDIA reshape-nvidia reshape from-model model_vulkan
    ${V2_RESHAPE_CPU_LIBRARY} v2_reshape ../Numerical/fixtures/reshape.param
    ${V0_EMPTY_BIN} --elements 8)
ncnn_add_vulkan_v2_test(
    vulkan-v2-reshape-llvmpipe llvmpipe reshape-llvmpipe reshape from-model model_vulkan
    ${V2_RESHAPE_CPU_LIBRARY} v2_reshape ../Numerical/fixtures/reshape.param
    ${V0_EMPTY_BIN} --icd ${V0_LLVMPIPE_ICD} --elements 8)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-nvidia NVIDIA conv3x3-nvidia conv2d from-model model_vulkan
    ${V2_CONV3X3_CPU_LIBRARY} v2_conv3x3 ../Numerical/fixtures/convolution.param
    ${V2_CONV3X3_BIN} --elements 32 --input-shape 1x4x4
    --kernel-size 3x3 --stride 1x1 --padding 1,1,1,1
    --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-llvmpipe llvmpipe conv3x3-llvmpipe conv2d from-model model_vulkan
    ${V2_CONV3X3_CPU_LIBRARY} v2_conv3x3 ../Numerical/fixtures/convolution.param
    ${V2_CONV3X3_BIN} --icd ${V0_LLVMPIPE_ICD} --elements 32
    --input-shape 1x4x4 --kernel-size 3x3 --stride 1x1
    --padding 1,1,1,1 --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-no-bias-nvidia NVIDIA conv3x3-no-bias-nvidia conv2d
    from-model model_vulkan ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY} v2_conv3x3_no_bias
    ../Numerical/fixtures/convolution_no_bias.param ${V2_CONV3X3_NO_BIAS_BIN}
    --elements 32 --input-shape 1x4x4 --kernel-size 3x3 --stride 1x1
    --padding 1,1,1,1 --output-channels 2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-no-bias-llvmpipe llvmpipe conv3x3-no-bias-llvmpipe
    conv2d from-model model_vulkan ${V2_CONV3X3_NO_BIAS_CPU_LIBRARY}
    v2_conv3x3_no_bias ../Numerical/fixtures/convolution_no_bias.param
    ${V2_CONV3X3_NO_BIAS_BIN} --icd ${V0_LLVMPIPE_ICD} --elements 32
    --input-shape 1x4x4 --kernel-size 3x3 --stride 1x1
    --padding 1,1,1,1 --output-channels 2)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-stride2-nvidia NVIDIA conv3x3-stride2-nvidia conv2d
    from-model model_vulkan ${V2_CONV3X3_STRIDE2_CPU_LIBRARY} v2_conv3x3_stride2
    convolution_3x3_stride2.param ${V2_CONV3X3_BIN} --elements 8
    --input-shape 1x4x4 --kernel-size 3x3 --stride 2x2
    --padding 1,1,1,1 --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv3x3-stride2-llvmpipe llvmpipe conv3x3-stride2-llvmpipe
    conv2d from-model model_vulkan ${V2_CONV3X3_STRIDE2_CPU_LIBRARY}
    v2_conv3x3_stride2 convolution_3x3_stride2.param ${V2_CONV3X3_BIN}
    --icd ${V0_LLVMPIPE_ICD} --elements 8 --input-shape 1x4x4
    --kernel-size 3x3 --stride 2x2 --padding 1,1,1,1
    --output-channels 2 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv1x1-nvidia NVIDIA conv1x1-nvidia conv2d from-model model_vulkan
    ${V2_CONV1X1_CPU_LIBRARY} v2_conv1x1 convolution_1x1.param
    ${V2_CONV1X1_BIN} --elements 18 --input-shape 2x3x5
    --kernel-size 1x1 --stride 2x2 --padding 0,0,0,0
    --output-channels 3 --conv-bias)
ncnn_add_vulkan_v2_test(
    vulkan-v2-conv1x1-llvmpipe llvmpipe conv1x1-llvmpipe conv2d from-model model_vulkan
    ${V2_CONV1X1_CPU_LIBRARY} v2_conv1x1 convolution_1x1.param
    ${V2_CONV1X1_BIN} --icd ${V0_LLVMPIPE_ICD} --elements 18
    --input-shape 2x3x5 --kernel-size 1x1 --stride 2x2
    --padding 0,0,0,0 --output-channels 3 --conv-bias)

if(NCNN_ENABLE_VULKAN_PRODUCT)
    # V5.1 product multi-I/O fixtures use the vendored CPU-only ncnn reference
    # target directly; they deliberately do not link numerical_test_support.
    add_executable(vulkan_model_runner vulkan_model_runner.cpp)
    target_link_libraries(vulkan_model_runner PRIVATE ncnn)
    target_compile_options(vulkan_model_runner PRIVATE
        -Wall -Wextra -Wpedantic)
    set_source_files_properties(vulkan_model_runner.cpp PROPERTIES
        COMPILE_OPTIONS "-fexceptions")

    set(V5_CXX_SANITIZER_ARGS)
    set(V5_SANITIZER_RUNTIME_DIR)
    if(COMPILER_SANITIZERS MATCHES "(^|;)(address|undefined)(;|$)")
        list(APPEND V5_CXX_SANITIZER_ARGS
            --cxx-arg=-fsanitize=address,undefined
            --cxx-arg=-fno-sanitize-recover=all
            --cxx-arg=-fno-omit-frame-pointer)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND V5_CXX_SANITIZER_ARGS --cxx-arg=-shared-libasan)
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} --print-resource-dir
                OUTPUT_VARIABLE V5_CLANG_RESOURCE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
            set(V5_SANITIZER_RUNTIME_DIR
                "${V5_CLANG_RESOURCE_DIR}/lib/linux")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE V5_ASAN_LIBRARY
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
            get_filename_component(V5_SANITIZER_RUNTIME_DIR
                "${V5_ASAN_LIBRARY}" DIRECTORY)
        endif()
        if(V5_SANITIZER_RUNTIME_DIR AND
           NOT IS_DIRECTORY "${V5_SANITIZER_RUNTIME_DIR}")
            message(FATAL_ERROR
                "V5 sanitizer runtime directory is missing: ${V5_SANITIZER_RUNTIME_DIR}")
        endif()
    endif()
    set(V5_CONFIGURED_DEPENDENCY_ARGS)
    if(NCNN_VULKAN_USES_SHARED_LLVM)
        list(APPEND V5_CONFIGURED_DEPENDENCY_ARGS --shared-llvm)
    endif()
    foreach(dependency IN LISTS NCNN_VULKAN_STATIC_LLVM_NEEDED_LIBRARIES)
        list(APPEND V5_CONFIGURED_DEPENDENCY_ARGS
            --allowed-dependency=${dependency})
    endforeach()
    foreach(dependency IN LISTS NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        list(APPEND V5_CONFIGURED_DEPENDENCY_ARGS
            --allowed-dependency=${dependency}
            --sanitizer-dependency=${dependency})
    endforeach()

    set(V5_TEST_LIBRARY_PATH "${LLVM_LIBRARY_DIR}")
    if(V5_SANITIZER_RUNTIME_DIR)
        string(APPEND V5_TEST_LIBRARY_PATH ":${V5_SANITIZER_RUNTIME_DIR}")
    endif()
    string(APPEND V5_TEST_LIBRARY_PATH ":$ENV{LD_LIBRARY_PATH}")

    function(ncnn_add_vulkan_v5_test test_name device_match output_name icd)
        set(sanitizer_runtime_args)
        if(V5_SANITIZER_RUNTIME_DIR)
            list(APPEND sanitizer_runtime_args
                --sanitizer-runtime-dir=${V5_SANITIZER_RUNTIME_DIR})
        endif()
        add_test(NAME ${test_name}
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/run_v5.py
                --compiler $<TARGET_FILE:ncnn-compile>
                --driver $<TARGET_FILE:ncnn-mlir-driver>
                --opt $<TARGET_FILE:ncnn-mlir-opt>
                --mlir-opt ${MLIR_OPT}
                --translate ${MLIR_TRANSLATE}
                --clang ${CLANG}
                --nm ${LLVM_NM}
                --readelf ${LLVM_READELF}
                --llvm-as ${LLVM_AS}
                --spirv-val ${SPIRV_VAL}
                --spirv-dis ${SPIRV_DIS}
                --spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
                --host-cxx ${CMAKE_CXX_COMPILER}
                --llvm-major ${LLVM_VERSION_MAJOR}
                --llvm-lib-dir ${LLVM_LIBRARY_DIR}
                --runner $<TARGET_FILE:vulkan_model_runner>
                --vulkaninfo ${VULKANINFO}
                --project-include ${PROJECT_SOURCE_DIR}/include
                --device-match ${device_match}
                --icd ${icd}
                ${V5_CXX_SANITIZER_ARGS}
                ${sanitizer_runtime_args}
                ${V5_CONFIGURED_DEPENDENCY_ARGS}
                --output-dir ${CMAKE_CURRENT_BINARY_DIR}/evidence/v5/${output_name})
        set_tests_properties(${test_name} PROPERTIES
            LABELS "vulkan;device;integration;v5;product"
            RUN_SERIAL TRUE
            TIMEOUT 900
            ENVIRONMENT "LD_LIBRARY_PATH=${V5_TEST_LIBRARY_PATH}")
    endfunction()

ncnn_add_vulkan_v5_test(
    vulkan-v5-multiop-nvidia NVIDIA nvidia ${V0_NVIDIA_ICD})
ncnn_add_vulkan_v5_test(
    vulkan-v5-multiop-llvmpipe llvmpipe llvmpipe ${V0_LLVMPIPE_ICD})
endif()

