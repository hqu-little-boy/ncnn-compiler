cmake_minimum_required(VERSION 3.20)

project(ncnn_compiler LANGUAGES C CXX)
include(GNUInstallDirs)

# C++23, 严格点
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 导出 compile_commands.json 供 clang-tidy / clangd 读取编译参数
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── LLVM / MLIR 21（OOT 方言）────────────────────────────────
if(NOT LLVM_DIR)
    set(LLVM_DIR "/usr/lib/llvm-21/lib/cmake/llvm")
endif()
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (cmake dir: ${LLVM_DIR})")

if(NOT MLIR_DIR)
    set(MLIR_DIR "/usr/lib/llvm-21/lib/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION} (cmake dir: ${MLIR_DIR})")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

enable_testing()

# Debian 的 LLVM/MLIR 以 -fno-rtti 构建；链接 MLIR 的 target 统一关闭 RTTI。
# 在需要的 target 上单独 target_compile_options(<t> PRIVATE -fno-rtti)。

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(tools)
add_subdirectory(test)

# Apply project warnings only to first-party compiled targets. Directory-wide
# flags would otherwise leak into the vendored ncnn targets under test/.
function(ncnn_enable_first_party_warnings directory)
    get_property(NCNN_DIRECTORY_TARGETS DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(NCNN_TARGET IN LISTS NCNN_DIRECTORY_TARGETS)
        get_target_property(NCNN_TARGET_TYPE "${NCNN_TARGET}" TYPE)
        get_target_property(NCNN_TARGET_SOURCE_DIR "${NCNN_TARGET}" SOURCE_DIR)
        if(NCNN_TARGET_TYPE MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$"
           AND NCNN_TARGET_SOURCE_DIR MATCHES "^${CMAKE_SOURCE_DIR}(/|$)"
           AND NOT NCNN_TARGET_SOURCE_DIR MATCHES "^${CMAKE_SOURCE_DIR}/test/third_party(/|$)")
            target_compile_options("${NCNN_TARGET}" PRIVATE
                -Wall -Wextra -Wpedantic)
        endif()
    endforeach()

    get_property(NCNN_SUBDIRECTORIES DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(NCNN_SUBDIRECTORY IN LISTS NCNN_SUBDIRECTORIES)
        ncnn_enable_first_party_warnings("${NCNN_SUBDIRECTORY}")
    endforeach()
endfunction()
ncnn_enable_first_party_warnings("${CMAKE_SOURCE_DIR}")

# ── 辅助目标：clang-format / clang-tidy ───────────────────
# Format first-party sources and headers, including test support headers.
file(GLOB_RECURSE NCNN_FORMAT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/lib/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/bin/*.cpp"
    "${CMAKE_SOURCE_DIR}/bin/*.h"
    "${CMAKE_SOURCE_DIR}/bin/*.hpp"
    "${CMAKE_SOURCE_DIR}/tools/*.cpp"
    "${CMAKE_SOURCE_DIR}/tools/*.h"
    "${CMAKE_SOURCE_DIR}/tools/*.hpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.h"
    "${CMAKE_SOURCE_DIR}/test/*.hpp")
list(FILTER NCNN_FORMAT_SOURCES EXCLUDE REGEX
    "/(build[^/]*|third_party|generated)/")

# Headers are checked through their owning translation units, never as main
# files, so clang-tidy receives only first-party .cpp compile database entries.
file(GLOB_RECURSE NCNN_TIDY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/lib/*.cpp"
    "${CMAKE_SOURCE_DIR}/bin/*.cpp"
    "${CMAKE_SOURCE_DIR}/tools/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp")
list(FILTER NCNN_TIDY_SOURCES EXCLUDE REGEX
    "/(build[^/]*|third_party|generated)/")

find_program(CLANG_FORMAT NAMES clang-format-21 REQUIRED)
find_program(CLANG_TIDY NAMES clang-tidy-21 REQUIRED)
set(NCNN_TIDY_HEADER_FILTER
    "^${CMAKE_SOURCE_DIR}/(include|lib|bin|tools|test)/")
set(NCNN_TIDY_EXCLUDE_HEADER_FILTER
    "^${CMAKE_SOURCE_DIR}/(build[^/]*|test/third_party|third_party|(.*/)?generated)(/|$)")

# clang-format：检查格式是否合规（不改文件，发现违例即失败）
add_custom_target(format_check
    COMMAND ${CLANG_FORMAT} --dry-run --Werror ${NCNN_FORMAT_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking code style with clang-format (Google)"
    VERBATIM)
# clang-format：原地修正格式
add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${NCNN_FORMAT_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Reformatting code with clang-format (Google)"
    VERBATIM)

# clang-tidy：静态检查（依赖 compile_commands.json）
add_custom_target(tidy
    COMMAND ${CLANG_TIDY} -p=${CMAKE_BINARY_DIR}
            --header-filter=${NCNN_TIDY_HEADER_FILTER}
            --exclude-header-filter=${NCNN_TIDY_EXCLUDE_HEADER_FILTER}
            ${NCNN_TIDY_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-tidy (Google)"
    VERBATIM)
add_custom_target(tidy_fix
    COMMAND ${CLANG_TIDY} -p=${CMAKE_BINARY_DIR} --fix --fix-errors
            --header-filter=${NCNN_TIDY_HEADER_FILTER}
            --exclude-header-filter=${NCNN_TIDY_EXCLUDE_HEADER_FILTER}
            ${NCNN_TIDY_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Auto-fixing with clang-tidy (Google)"
    VERBATIM)
