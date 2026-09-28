find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(SPIRV_VAL NAMES spirv-val REQUIRED)
find_program(SPIRV_DIS NAMES spirv-dis REQUIRED)
find_program(VULKANINFO NAMES vulkaninfo REQUIRED)
find_file(V0_LLVMPIPE_ICD
    NAMES lvp_icd.json
    PATHS /usr/share/vulkan/icd.d /etc/vulkan/icd.d
    REQUIRED)

add_executable(vulkan_add_runner vulkan_add_runner.cpp)
target_link_libraries(vulkan_add_runner PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS})
target_compile_options(vulkan_add_runner PRIVATE
    -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)

add_executable(vulkan_v0_spirv_extract extract_vulkan_spirv.cpp)
target_include_directories(vulkan_v0_spirv_extract PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})
target_compile_options(vulkan_v0_spirv_extract PRIVATE -fno-rtti)
target_link_libraries(vulkan_v0_spirv_extract PRIVATE MLIR LLVM)
set_target_properties(vulkan_v0_spirv_extract PROPERTIES
    BUILD_RPATH "${LLVM_LIBRARY_DIR}")

set(V0_CPU_DIR ${CMAKE_CURRENT_BINARY_DIR}/cpu-add)
set(V0_CPU_LIBRARY ${V0_CPU_DIR}/libv0_add.so)
set(V0_CPU_HEADER ${V0_CPU_DIR}/v0_add.h)
set(V0_CPU_MANIFEST ${V0_CPU_DIR}/v0_add.json)
set(V0_EMPTY_BIN ${CMAKE_CURRENT_BINARY_DIR}/empty.bin)
file(WRITE ${V0_EMPTY_BIN} "")
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

ncnn_add_vulkan_v0_test(vulkan-v0-add-nvidia NVIDIA nvidia --run-negative)
ncnn_add_vulkan_v0_test(vulkan-v0-add-llvmpipe llvmpipe llvmpipe --icd ${V0_LLVMPIPE_ICD})

