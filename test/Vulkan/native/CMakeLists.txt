cmake_minimum_required(VERSION 3.20)
project(ncnn_vulkan_device_helper_native LANGUAGES CXX)
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
