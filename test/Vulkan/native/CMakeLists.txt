cmake_minimum_required(VERSION 3.20)
project(ncnn_vulkan_device_helper_native LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTING "Register standalone helper unit test" OFF)
option(NCNN_VULKAN_HELPER_BUILD_STANDALONE
    "Build a private copy of the Vulkan device helper in this standalone tree" OFF)
set(NCNN_VULKAN_HELPER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../lib/VulkanDeviceHelper"
    CACHE PATH "Vulkan device helper source directory")
set(NCNN_VULKAN_HELPER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../include"
    CACHE PATH "ncnn-mlir public include directory")

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

if(NCNN_VULKAN_HELPER_BUILD_STANDALONE)
    add_library(NCNNVulkanDeviceHelper STATIC
        "${NCNN_VULKAN_HELPER_SOURCE_DIR}/VulkanDeviceHelper.cpp")
    target_include_directories(NCNNVulkanDeviceHelper PUBLIC
        "${NCNN_VULKAN_HELPER_INCLUDE_DIR}")
    target_link_libraries(NCNNVulkanDeviceHelper PUBLIC Vulkan::Vulkan Threads::Threads)
    target_compile_features(NCNNVulkanDeviceHelper PUBLIC cxx_std_23)
    target_compile_options(NCNNVulkanDeviceHelper PRIVATE
        -Wall -Wextra -Wpedantic)
    set_target_properties(NCNNVulkanDeviceHelper PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
elseif(NOT TARGET NCNNVulkanDeviceHelper)
    message(FATAL_ERROR
        "The parent build must define NCNNVulkanDeviceHelper, or configure with "
        "NCNN_VULKAN_HELPER_BUILD_STANDALONE=ON")
endif()

add_executable(vulkan_device_helper_runner device_helper_runner.cpp)
target_link_libraries(vulkan_device_helper_runner PRIVATE
    NCNNVulkanDeviceHelper Vulkan::Vulkan Threads::Threads ${CMAKE_DL_LIBS})
target_compile_features(vulkan_device_helper_runner PRIVATE cxx_std_23)
target_compile_options(vulkan_device_helper_runner PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(vulkan_device_helper_unit_test
    device_helper_unit_test.cpp
    fakeVulkan/fake_vulkan.cpp)
target_link_libraries(vulkan_device_helper_unit_test PRIVATE
    NCNNVulkanDeviceHelper Vulkan::Vulkan Threads::Threads)
target_compile_features(vulkan_device_helper_unit_test PRIVATE cxx_std_23)
target_compile_options(vulkan_device_helper_unit_test PRIVATE
    -Wall -Wextra -Wpedantic)
include("${CMAKE_CURRENT_LIST_DIR}/fakeVulkan/WrapVulkan.cmake")
ncnn_wrap_vulkan_for_test(vulkan_device_helper_unit_test)

if(TARGET NCNNVulkanProductSupport)
    find_package(GTest REQUIRED)
    if(NOT NCNN_SPIRV_VAL_EXECUTABLE)
        find_program(NCNN_SPIRV_VAL_EXECUTABLE NAMES spirv-val REQUIRED)
    endif()
    get_filename_component(_ncnn_compiler_source_dir
        "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

    set(_lifecycle_root "${CMAKE_CURRENT_BINARY_DIR}/lifecycle")
    set(_lifecycle_emitter_sources
        "${CMAKE_CURRENT_LIST_DIR}/lifecycle_fixture_emitter.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/../../Unit/vulkan_shader_interface_test.cpp")
    add_executable(vulkan_lifecycle_fixture_emitter
        ${_lifecycle_emitter_sources})
    target_include_directories(vulkan_lifecycle_fixture_emitter PRIVATE
        "${_ncnn_compiler_source_dir}/test/Unit"
        "${_ncnn_compiler_source_dir}/tools")
    target_link_libraries(vulkan_lifecycle_fixture_emitter PRIVATE
        NCNNVulkanProductSupport GTest::gtest)
    target_compile_definitions(vulkan_lifecycle_fixture_emitter PRIVATE
        GTEST_HAS_CXXABI_H_=0)
    target_compile_options(vulkan_lifecycle_fixture_emitter PRIVATE
        -Wall -Wextra -Wpedantic -fno-rtti)

    function(ncnn_add_vulkan_lifecycle_fixture kind)
        set(_fixture_dir "${_lifecycle_root}/${kind}")
        set(_fixture_source "${_fixture_dir}/candidate_source.cpp")
        set(_fixture_header "${_fixture_dir}/lifecycle_fixture_vulkan.h")
        set(_fixture_identity "${_fixture_dir}/host-program-identity.txt")
        set(_fixture_shader "${_fixture_dir}/shader_0.spv")
        set(_fixture_plan "${_fixture_dir}/plan.json")
        set(_fixture_host_program "${_fixture_dir}/host_program.json")
        set(_fixture_manifest "${_fixture_dir}/package_manifest.json")
        set(_emitter_kind "${kind}")
        if(kind STREQUAL "unweighted")
            set(_emitter_kind relu)
        endif()
        add_custom_command(
            OUTPUT "${_fixture_source}" "${_fixture_header}"
                "${_fixture_identity}" "${_fixture_shader}" "${_fixture_plan}"
                "${_fixture_host_program}" "${_fixture_manifest}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_fixture_dir}"
            COMMAND $<TARGET_FILE:vulkan_lifecycle_fixture_emitter>
                --kind "${_emitter_kind}"
                --output-dir "${_fixture_dir}"
            COMMAND "${NCNN_SPIRV_VAL_EXECUTABLE}"
                --target-env vulkan1.1 "${_fixture_shader}"
            DEPENDS vulkan_lifecycle_fixture_emitter
                ${_lifecycle_emitter_sources}
                "${CMAKE_CURRENT_LIST_DIR}/lifecycle_test_hooks.hpp"
            VERBATIM)
        set_source_files_properties("${_fixture_source}" PROPERTIES
            GENERATED TRUE)
        add_custom_target(vulkan_lifecycle_fixture_${kind}
            DEPENDS "${_fixture_source}" "${_fixture_header}"
                "${_fixture_identity}" "${_fixture_shader}" "${_fixture_plan}"
                "${_fixture_host_program}" "${_fixture_manifest}")

        foreach(_variant IN ITEMS release validation)
            set(_target "vulkan_lifecycle_${kind}_${_variant}")
            set(_output_name "vulkan_lifecycle_${kind}")
            if(_variant STREQUAL "validation")
                string(APPEND _output_name "_validation")
            endif()
            set(_map "${_fixture_dir}/${_output_name}.map")
            add_library(${_target} SHARED "${_fixture_source}")
            add_dependencies(${_target} vulkan_lifecycle_fixture_${kind})
            target_include_directories(${_target} PRIVATE
                "${_fixture_dir}"
                "${_ncnn_compiler_source_dir}/include")
            target_link_libraries(${_target} PRIVATE
                NCNNVulkanDeviceHelper Vulkan::Vulkan Threads::Threads)
            target_compile_options(${_target} PRIVATE
                -Wall -Wextra -Wpedantic)
            target_link_options(${_target} PRIVATE
                "-Wl,-z,defs"
                "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/lifecycle.exports.map"
                "-Wl,-Map=${_map}"
                "-Wl,--cref")
            set_target_properties(${_target} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${_fixture_dir}"
                OUTPUT_NAME "${_output_name}"
                LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/lifecycle.exports.map")
            if(_variant STREQUAL "validation")
                target_compile_definitions(${_target} PRIVATE
                    NCNN_VULKAN_LIFECYCLE_VALIDATION=1)
            endif()
        endforeach()
    endfunction()

    ncnn_add_vulkan_lifecycle_fixture(unweighted)
    ncnn_add_vulkan_lifecycle_fixture(weighted)

    add_executable(vulkan_lifecycle_c_harness lifecycle_c_harness.c)
    add_dependencies(vulkan_lifecycle_c_harness
        vulkan_lifecycle_fixture_unweighted)
    target_include_directories(vulkan_lifecycle_c_harness PRIVATE
        "${_lifecycle_root}/unweighted"
        "${_ncnn_compiler_source_dir}/include")
    target_link_libraries(vulkan_lifecycle_c_harness PRIVATE ${CMAKE_DL_LIBS})
    target_compile_features(vulkan_lifecycle_c_harness PRIVATE c_std_11)
    target_compile_options(vulkan_lifecycle_c_harness PRIVATE
        -Wall -Wextra -Wpedantic)

    add_executable(vulkan_lifecycle_fake_unit_test
        lifecycle_fake_unit_test.cpp
        fakeVulkan/fake_vulkan.cpp)
    add_dependencies(vulkan_lifecycle_fake_unit_test
        vulkan_lifecycle_fixture_weighted)
    target_include_directories(vulkan_lifecycle_fake_unit_test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${_lifecycle_root}/weighted"
        "${_ncnn_compiler_source_dir}/include")
    target_link_libraries(vulkan_lifecycle_fake_unit_test PRIVATE
        NCNNVulkanDeviceHelper Vulkan::Vulkan Threads::Threads)
    target_compile_definitions(vulkan_lifecycle_fake_unit_test PRIVATE
        NCNN_VULKAN_LIFECYCLE_TEST=1)
    target_compile_options(vulkan_lifecycle_fake_unit_test PRIVATE
        -Wall -Wextra -Wpedantic)
    ncnn_wrap_vulkan_for_test(vulkan_lifecycle_fake_unit_test)
    target_link_options(vulkan_lifecycle_fake_unit_test PRIVATE
        -Wl,--wrap=vkCmdCopyBuffer
        -Wl,--wrap=vkCmdPipelineBarrier)

    if(BUILD_TESTING)
        set(_lifecycle_fake_result
            "${CMAKE_BINARY_DIR}/test/Vulkan/evidence/va3/fake/fake-result.json")
        add_test(NAME vulkan-lifecycle-fake-unit
            COMMAND $<TARGET_FILE:vulkan_lifecycle_fake_unit_test>
                --result "${_lifecycle_fake_result}")
        set_tests_properties(vulkan-lifecycle-fake-unit PROPERTIES
            LABELS "vulkan;unit;offline;lifecycle"
            RUN_SERIAL TRUE
            TIMEOUT 120)
    endif()
endif()

if(TARGET NCNNVulkanProductSupport)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(NCNN_FIXEDRUN_SPIRV_VAL NAMES spirv-val REQUIRED)
    get_filename_component(_ncnn_compiler_source_dir
        "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    set(_fixedrun_root "${CMAKE_BINARY_DIR}/test/Vulkan/fixedrun-candidates")
    set(_fixedrun_empty_bin "${_fixedrun_root}/empty.bin")
    file(MAKE_DIRECTORY "${_fixedrun_root}")
    file(WRITE "${_fixedrun_empty_bin}" "")

    set(_fixedrun_fixtures relu add_relu conv_relu add_split_relu
        split_concat reshape transpose)
    set(_fixedrun_params
        relu "${_ncnn_compiler_source_dir}/test/Vulkan/relu.param"
        add_relu "${_ncnn_compiler_source_dir}/test/Vulkan/add_relu.param"
        conv_relu "${_ncnn_compiler_source_dir}/test/Vulkan/fixedrun_conv_relu.param"
        add_split_relu "${_ncnn_compiler_source_dir}/test/Vulkan/add_split_relu.param"
        split_concat "${_ncnn_compiler_source_dir}/test/Numerical/fixtures/split_three.param"
        reshape "${_ncnn_compiler_source_dir}/test/Vulkan/fixedrun_reshape_conv.param"
        transpose "${_ncnn_compiler_source_dir}/test/Vulkan/fixedrun_transpose.param")
    set(_fixedrun_exports
        "${_ncnn_compiler_source_dir}/test/Vulkan/native/fixedrun.exports.map")
    function(ncnn_add_vulkan_fixedrun_fixture fixture param_path)
        set(_fixture_dir "${_fixedrun_root}/${fixture}")
        set(_source "${_fixture_dir}/candidate_source.cpp")
        set(_header "${_fixture_dir}/${fixture}_vulkan.h")
        set(_metadata "${_fixture_dir}/metadata.json")
        set(_plan "${_fixture_dir}/plan.json")
        set(_host_program "${_fixture_dir}/host_program.json")
        set(_bin "${_fixedrun_empty_bin}")
        if(fixture STREQUAL "conv_relu" OR fixture STREQUAL "reshape")
            set(_bin "${_fixedrun_root}/${fixture}.bin")
            add_custom_command(
                OUTPUT "${_bin}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${_fixedrun_root}"
                COMMAND "${Python3_EXECUTABLE}"
                    "${_ncnn_compiler_source_dir}/test/Vulkan/generate_fixedrun_conv_bin.py"
                    "${_bin}"
                DEPENDS
                    "${_ncnn_compiler_source_dir}/test/Vulkan/generate_fixedrun_conv_bin.py"
                VERBATIM)
        endif()
        add_custom_command(
            OUTPUT "${_source}" "${_header}" "${_metadata}" "${_plan}"
                "${_host_program}" "${_fixture_dir}/candidate_tosa.mlir"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_fixture_dir}"
            COMMAND $<TARGET_FILE:ncnn-vulkan-fixedrun-capture>
                "${param_path}"
                --bin "${_bin}"
                --model-name "${fixture}"
                --output-dir "${_fixture_dir}"
                --backend=vulkan
                --precision=f32
                --driver $<TARGET_FILE:ncnn-mlir-driver>
                --opt $<TARGET_FILE:ncnn-mlir-opt>
                --translate ${MLIR_TRANSLATE}
                --clang ${CLANG}
                --nm ${LLVM_NM}
                --readelf ${LLVM_READELF}
                --llvm-as ${LLVM_AS}
                --vulkan-mlir-opt ${NCNN_MLIR_OPT_EXECUTABLE}
                --vulkan-spirv-val ${NCNN_SPIRV_VAL_EXECUTABLE}
                --vulkan-spirv-dis ${NCNN_SPIRV_DIS_EXECUTABLE}
                --vulkan-spirv-extract $<TARGET_FILE:ncnn-vulkan-spirv-extract>
                --vulkan-host-cxx ${CMAKE_CXX_COMPILER}
                --capture-vulkan-host-source
            COMMAND "${NCNN_FIXEDRUN_SPIRV_VAL}"
                --target-env vulkan1.1 "${_fixture_dir}/shader_0.spv"
            DEPENDS ncnn-vulkan-fixedrun-capture
                "${param_path}" "${_bin}"
            VERBATIM)
        set_source_files_properties("${_source}" PROPERTIES GENERATED TRUE)
        add_custom_target(vulkan_fixedrun_fixture_${fixture}
            DEPENDS "${_source}" "${_header}" "${_metadata}" "${_plan}"
                "${_host_program}" "${_fixture_dir}/candidate_tosa.mlir")
        set(_target "vulkan_fixedrun_${fixture}_validation")
        set(_library_dir "${_fixedrun_root}/libraries/${fixture}")
        set(_audit_dir "${_fixedrun_root}/audit/${fixture}")
        file(MAKE_DIRECTORY "${_library_dir}" "${_audit_dir}")
        set(_map "${_audit_dir}/${_target}.map")
        add_library(${_target} SHARED "${_source}")
        add_dependencies(${_target} vulkan_fixedrun_fixture_${fixture})
        target_include_directories(${_target} PRIVATE
            "${_fixture_dir}"
            "${_ncnn_compiler_source_dir}/include")
        target_link_libraries(${_target} PRIVATE
            NCNNVulkanDeviceHelper Vulkan::Vulkan Threads::Threads)
        target_compile_options(${_target} PRIVATE
            ${NCNN_VULKAN_HOST_SANITIZER_ARGS})
        target_link_options(${_target} PRIVATE
            ${NCNN_VULKAN_HOST_SANITIZER_ARGS}
            "-Wl,-z,defs"
            "-Wl,--version-script=${_fixedrun_exports}"
            "-Wl,-Map=${_map}"
            "-Wl,--cref")
        if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
            target_link_options(${_target} PRIVATE
                "-Wl,-rpath,${NCNN_VULKAN_SANITIZER_RUNTIME_DIR}")
        endif()
        set_target_properties(${_target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${_library_dir}"
            OUTPUT_NAME "${fixture}_fixedrun_validation"
            LINK_DEPENDS "${_fixedrun_exports}")
        target_compile_definitions(${_target} PRIVATE
            NCNN_VULKAN_LIFECYCLE_VALIDATION=1)

        string(TOUPPER "${fixture}" _fixture_upper)
        set(_loader_target "vulkan_fixedrun_loader_${fixture}")
        add_executable(${_loader_target} fixedrun_native_loader.cpp)
        add_dependencies(${_loader_target} ${_target})
        target_include_directories(${_loader_target} PRIVATE
            "${_fixture_dir}"
            "${_ncnn_compiler_source_dir}/include")
        target_compile_definitions(${_loader_target} PRIVATE
            NCNN_FIXEDRUN_MODEL=${fixture}
            NCNN_FIXEDRUN_MODEL_UPPER=${_fixture_upper}
            NCNN_FIXEDRUN_MODEL_HEADER="${fixture}_vulkan.h"
            NCNN_FIXEDRUN_BACKEND_VULKAN=1)
        target_compile_options(${_loader_target} PRIVATE
            -Wall -Wextra -Wpedantic
            ${NCNN_VULKAN_HOST_SANITIZER_ARGS})
        target_link_libraries(${_loader_target} PRIVATE
            ${CMAKE_DL_LIBS} Threads::Threads)
        target_link_options(${_loader_target} PRIVATE
            ${NCNN_VULKAN_HOST_SANITIZER_ARGS})
        if(NCNN_VULKAN_SANITIZER_RUNTIME_DIR)
            target_link_options(${_loader_target} PRIVATE
                "-Wl,-rpath,${NCNN_VULKAN_SANITIZER_RUNTIME_DIR}")
        endif()
        set_property(GLOBAL APPEND PROPERTY NCNN_VULKAN_FIXEDRUN_TARGETS
            "${_target}" "${_loader_target}")
    endfunction()
    foreach(_fixture IN LISTS _fixedrun_fixtures)
        list(FIND _fixedrun_params "${_fixture}" _fixture_index)
        math(EXPR _param_index "${_fixture_index} + 1")
        list(GET _fixedrun_params ${_param_index} _fixture_param)
        ncnn_add_vulkan_fixedrun_fixture("${_fixture}" "${_fixture_param}")
    endforeach()
    set(NCNN_VULKAN_FIXEDRUN_ROOT "${_fixedrun_root}" PARENT_SCOPE)
    set(NCNN_VULKAN_FIXEDRUN_CAPTURE "$<TARGET_FILE:ncnn-vulkan-fixedrun-capture>"
        PARENT_SCOPE)

    set(_fixedrun_fake_dir "${CMAKE_BINARY_DIR}/test/Vulkan/fixedrun-fake")
    set(_fixedrun_fake_source "${_fixedrun_fake_dir}/candidate_source.cpp")
    set(_fixedrun_fake_header "${_fixedrun_fake_dir}/va4_fixedrun_vulkan.h")
    set(_fixedrun_fake_plan "${_fixedrun_fake_dir}/plan.json")
    set(_fixedrun_fake_program "${_fixedrun_fake_dir}/host_program.json")
    set(_fixedrun_fake_expectations
        "${_fixedrun_fake_dir}/fixture_expectations.hpp")
    set(_fixedrun_fake_manifest "${_fixedrun_fake_dir}/package_manifest.json")
    set(_fixedrun_fake_metadata "${_fixedrun_fake_dir}/metadata.json")
    set(_fixedrun_fake_host_identity
        "${_fixedrun_fake_dir}/host-program-identity.txt")
    set(_fixedrun_fake_codegen_identity
        "${_fixedrun_fake_dir}/codegen-identity.txt")
    set(_fixedrun_fake_codegen_version
        "${_fixedrun_fake_dir}/codegen-version.txt")
    set(_fixedrun_fake_shader0 "${_fixedrun_fake_dir}/shader_0.spv")
    set(_fixedrun_fake_shader1 "${_fixedrun_fake_dir}/shader_1.spv")
    add_executable(vulkan_fixedrun_fake_fixture_emitter
        fixedrun_fixture_emitter.cpp
        "${_ncnn_compiler_source_dir}/test/Unit/vulkan_shader_interface_test.cpp")
    target_include_directories(vulkan_fixedrun_fake_fixture_emitter PRIVATE
        "${_ncnn_compiler_source_dir}/test/Unit"
        "${_ncnn_compiler_source_dir}/tools")
    target_compile_options(vulkan_fixedrun_fake_fixture_emitter PRIVATE
        -Wall -Wextra -Wpedantic -fno-rtti)
    target_compile_definitions(vulkan_fixedrun_fake_fixture_emitter PRIVATE
        GTEST_HAS_CXXABI_H_=0)
    target_link_libraries(vulkan_fixedrun_fake_fixture_emitter PRIVATE
        NCNNVulkanProductSupport GTest::gtest)
    add_custom_command(
        OUTPUT "${_fixedrun_fake_source}" "${_fixedrun_fake_header}"
            "${_fixedrun_fake_plan}" "${_fixedrun_fake_program}"
            "${_fixedrun_fake_manifest}" "${_fixedrun_fake_metadata}"
            "${_fixedrun_fake_expectations}" "${_fixedrun_fake_host_identity}"
            "${_fixedrun_fake_codegen_identity}"
            "${_fixedrun_fake_codegen_version}" "${_fixedrun_fake_shader0}"
            "${_fixedrun_fake_shader1}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_fixedrun_fake_dir}"
        COMMAND $<TARGET_FILE:vulkan_fixedrun_fake_fixture_emitter>
            --output-dir "${_fixedrun_fake_dir}"
        COMMAND "${NCNN_FIXEDRUN_SPIRV_VAL}"
            --target-env vulkan1.1 "${_fixedrun_fake_shader0}"
        COMMAND "${NCNN_FIXEDRUN_SPIRV_VAL}"
            --target-env vulkan1.1 "${_fixedrun_fake_shader1}"
        DEPENDS vulkan_fixedrun_fake_fixture_emitter
            fixedrun_fixture_emitter.cpp
            "${_ncnn_compiler_source_dir}/test/Unit/vulkan_shader_interface_test.cpp"
        VERBATIM)
    set_source_files_properties("${_fixedrun_fake_source}" PROPERTIES
        GENERATED TRUE)
    add_custom_target(vulkan_fixedrun_fake_fixture
        DEPENDS "${_fixedrun_fake_source}" "${_fixedrun_fake_header}"
            "${_fixedrun_fake_plan}" "${_fixedrun_fake_program}"
            "${_fixedrun_fake_manifest}" "${_fixedrun_fake_metadata}"
            "${_fixedrun_fake_expectations}" "${_fixedrun_fake_host_identity}"
            "${_fixedrun_fake_codegen_identity}"
            "${_fixedrun_fake_codegen_version}" "${_fixedrun_fake_shader0}"
            "${_fixedrun_fake_shader1}")
    add_executable(vulkan_fixedrun_fake_unit_test
        fixedrun_fake_unit_test.cpp
        fakeVulkan/fake_vulkan.cpp)
    add_dependencies(vulkan_fixedrun_fake_unit_test
        vulkan_fixedrun_fake_fixture)
    target_include_directories(vulkan_fixedrun_fake_unit_test PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
        "${_fixedrun_fake_dir}"
        "${_ncnn_compiler_source_dir}/include")
    target_link_libraries(vulkan_fixedrun_fake_unit_test PRIVATE
        NCNNVulkanDeviceHelper Vulkan::Vulkan Threads::Threads)
    target_compile_definitions(vulkan_fixedrun_fake_unit_test PRIVATE
        NCNN_VULKAN_LIFECYCLE_TEST=1)
    target_compile_options(vulkan_fixedrun_fake_unit_test PRIVATE
        -Wall -Wextra -Wpedantic)
    ncnn_wrap_vulkan_for_test(vulkan_fixedrun_fake_unit_test)
    target_link_options(vulkan_fixedrun_fake_unit_test PRIVATE
        -Wl,--wrap=vkCmdDispatch
        -Wl,--wrap=vkCmdBindPipeline
        -Wl,--wrap=vkCmdBindDescriptorSets
        -Wl,--wrap=vkCmdPushConstants
        -Wl,--wrap=vkCmdCopyBuffer
        -Wl,--wrap=vkCmdPipelineBarrier
        -Wl,--wrap=vkGetInstanceProcAddr)
    if(BUILD_TESTING)
        add_test(NAME vulkan-fixedrun-fake-unit
            COMMAND $<TARGET_FILE:vulkan_fixedrun_fake_unit_test>)
        set_tests_properties(vulkan-fixedrun-fake-unit PROPERTIES
            LABELS "vulkan;unit;offline;fixedrun"
            RUN_SERIAL TRUE
            TIMEOUT 120)
    endif()
endif()

if(COMPILER_SANITIZERS)
    set(_ncnn_helper_sanitizers ${COMPILER_SANITIZERS})
    list(REMOVE_DUPLICATES _ncnn_helper_sanitizers)
    list(LENGTH _ncnn_helper_sanitizers _ncnn_helper_sanitizer_count)
    if(NOT _ncnn_helper_sanitizer_count EQUAL 2 OR
       NOT "address" IN_LIST _ncnn_helper_sanitizers OR
       NOT "undefined" IN_LIST _ncnn_helper_sanitizers)
        message(FATAL_ERROR
            "Vulkan helper SAN tests require exactly address and undefined sanitizers")
    endif()
    set(_ncnn_helper_sanitizer_flags
        -fsanitize=address,undefined
        -fno-sanitize-recover=all
        -fno-omit-frame-pointer)
    foreach(_target IN ITEMS
            NCNNVulkanDeviceHelper
            vulkan_device_helper_runner
            vulkan_device_helper_unit_test)
        target_compile_options(${_target} PRIVATE ${_ncnn_helper_sanitizer_flags})
    endforeach()
    target_link_options(vulkan_device_helper_runner PRIVATE
        -fsanitize=address,undefined)
    target_link_options(vulkan_device_helper_unit_test PRIVATE
        -fsanitize=address,undefined)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR
            "Vulkan helper address-sanitizer tests require the configured Clang toolchain")
    endif()
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
        RESULT_VARIABLE _ncnn_helper_resource_status
        OUTPUT_VARIABLE _ncnn_helper_resource_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    if(NOT _ncnn_helper_resource_status EQUAL 0 OR
       NOT IS_DIRECTORY "${_ncnn_helper_resource_dir}/lib/linux")
        message(FATAL_ERROR
            "Cannot locate the shared Clang ASan runtime for the helper tests")
    endif()
    file(GLOB _ncnn_helper_asan_runtimes
        "${_ncnn_helper_resource_dir}/lib/linux/libclang_rt.asan-*.so")
    if(NOT _ncnn_helper_asan_runtimes)
        message(FATAL_ERROR
            "Shared Clang ASan runtime is missing from ${_ncnn_helper_resource_dir}/lib/linux")
    endif()
    set(NCNN_VULKAN_HELPER_SANITIZER_RUNTIME_DIR
        "${_ncnn_helper_resource_dir}/lib/linux")
    foreach(_target IN ITEMS vulkan_device_helper_runner
            vulkan_device_helper_unit_test)
        target_link_options(${_target} PRIVATE
            -shared-libasan
            "-Wl,-rpath,${NCNN_VULKAN_HELPER_SANITIZER_RUNTIME_DIR}")
    endforeach()
endif()

target_link_options(vulkan_device_helper_runner PRIVATE
    "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/vulkan_device_helper_runner.map,--cref")
target_link_options(vulkan_device_helper_unit_test PRIVATE
    "-Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/vulkan_device_helper_unit_test.map,--cref")

function(ncnn_vulkan_helper_json_list output_var input_var)
    set(_items ${${input_var}})
    set(_json "[")
    set(_separator "")
    foreach(_item IN LISTS _items)
        string(REPLACE "\\" "\\\\" _escaped "${_item}")
        string(REPLACE "\"" "\\\"" _escaped "${_escaped}")
        string(APPEND _json "${_separator}\"${_escaped}\"")
        set(_separator ",")
    endforeach()
    string(APPEND _json "]")
    set(${output_var} "${_json}" PARENT_SCOPE)
endfunction()

function(ncnn_vulkan_helper_closure_entry target output_var)
    get_target_property(_type "${target}" TYPE)
    get_target_property(_sources "${target}" SOURCES)
    get_target_property(_source_dir "${target}" SOURCE_DIR)
    set(_absolute_sources)
    foreach(_source IN LISTS _sources)
        if(NOT IS_ABSOLUTE "${_source}")
            get_filename_component(_source "${_source}" ABSOLUTE
                BASE_DIR "${_source_dir}")
        endif()
        list(APPEND _absolute_sources "${_source}")
    endforeach()
    set(_sources "${_absolute_sources}")
    get_target_property(_includes "${target}" INCLUDE_DIRECTORIES)
    get_target_property(_links "${target}" LINK_LIBRARIES)
    get_target_property(_interface_links "${target}" INTERFACE_LINK_LIBRARIES)
    get_target_property(_dependencies "${target}" MANUALLY_ADDED_DEPENDENCIES)
    foreach(_property IN ITEMS _sources _includes _links _interface_links _dependencies)
        if(NOT ${_property} OR "${${_property}}" MATCHES "-NOTFOUND$")
            set(${_property})
        endif()
    endforeach()
    ncnn_vulkan_helper_json_list(_sources_json _sources)
    ncnn_vulkan_helper_json_list(_includes_json _includes)
    ncnn_vulkan_helper_json_list(_links_json _links)
    ncnn_vulkan_helper_json_list(_interface_links_json _interface_links)
    ncnn_vulkan_helper_json_list(_dependencies_json _dependencies)
    set(${output_var}
        "{\"name\":\"${target}\",\"type\":\"${_type}\",\"sources\":${_sources_json},\"include_directories\":${_includes_json},\"link_libraries\":${_links_json},\"interface_link_libraries\":${_interface_links_json},\"dependencies\":${_dependencies_json},\"artifact\":\"$<TARGET_FILE:${target}>\"}"
        PARENT_SCOPE)
endfunction()

# Queries created as files during configure are only observed by a later CMake
# invocation. Request the reply for this invocation when the project API exists.
if(COMMAND cmake_file_api)
    cmake_file_api(QUERY API_VERSION 1 CODEMODEL 2 CACHE 2 CMAKEFILES 1)
endif()

set(_helper_closure_entries)
ncnn_vulkan_helper_closure_entry(NCNNVulkanDeviceHelper _helper_closure_entry)
list(APPEND _helper_closure_entries "${_helper_closure_entry}")
if(NCNN_VULKAN_HELPER_BUILD_STANDALONE)
    ncnn_vulkan_helper_closure_entry(vulkan_device_helper_runner _runner_closure_entry)
    list(APPEND _helper_closure_entries "${_runner_closure_entry}")
    ncnn_vulkan_helper_closure_entry(vulkan_device_helper_unit_test _unit_closure_entry)
    list(APPEND _helper_closure_entries "${_unit_closure_entry}")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/.cmake/api/v1/query")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/codemodel-v2" "")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/cache-v2" "")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/cmakeFiles-v1" "")
    set(_closure_path "${CMAKE_BINARY_DIR}/device-helper-target-link-closure.json")
else()
    set(_closure_path "${CMAKE_BINARY_DIR}/test/Vulkan/device-helper-target-link-closure.json")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/.cmake/api/v1/query")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/codemodel-v2" "")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/cache-v2" "")
    file(WRITE "${CMAKE_BINARY_DIR}/.cmake/api/v1/query/cmakeFiles-v1" "")
endif()
string(JOIN "," _closure_json_targets ${_helper_closure_entries})
file(GENERATE OUTPUT "${_closure_path}"
    CONTENT "{\"schema_version\":1,\"kind\":\"ncnn.vulkan_device_helper_target_link_closure\",\"targets\":[${_closure_json_targets}]}\n")

if(NCNN_VULKAN_HELPER_BUILD_STANDALONE AND BUILD_TESTING)
    enable_testing()
    add_test(NAME vulkan-device-helper-unit
        COMMAND $<TARGET_FILE:vulkan_device_helper_unit_test>)
    set_tests_properties(vulkan-device-helper-unit PROPERTIES
        LABELS "vulkan;unit;offline;device-helper"
        TIMEOUT 60)
endif()
