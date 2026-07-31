cmake_minimum_required(VERSION 3.20)

project(ncnn_compiler LANGUAGES C CXX)

# C++23, 严格点
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 导出 compile_commands.json 供 clang-tidy / clangd 读取编译参数
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

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

# ── 辅助目标：clang-format / clang-tidy ───────────────────
# 收集所有源/头文件（排除 build 目录与生成代码）。
file(GLOB_RECURSE NCNN_ALL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/lib/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.h"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/bin/*.cpp"
    "${CMAKE_SOURCE_DIR}/bin/*.h"
    "${CMAKE_SOURCE_DIR}/tools/*.cpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp")
list(FILTER NCNN_ALL_SOURCES EXCLUDE REGEX "/test/third_party/")

# clang-format：检查格式是否合规（不改文件，发现违例即失败）
add_custom_target(format_check
    COMMAND /usr/bin/clang-format-21 --dry-run --Werror ${NCNN_ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking code style with clang-format (Google)")
# clang-format：原地修正格式
add_custom_target(format
    COMMAND /usr/bin/clang-format-21 -i ${NCNN_ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Reformatting code with clang-format (Google)")

# clang-tidy：静态检查（依赖 compile_commands.json）
find_program(CLANG_TIDY clang-tidy-21 /usr/bin/clang-tidy-21)
if(CLANG_TIDY)
    add_custom_target(tidy
        COMMAND ${CLANG_TIDY} -p=${CMAKE_BINARY_DIR}
                --extra-arg=-std=c++23 ${NCNN_ALL_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy (Google)")
    add_custom_target(tidy_fix
        COMMAND ${CLANG_TIDY} -p=${CMAKE_BINARY_DIR} --fix --fix-errors
                --extra-arg=-std=c++23 ${NCNN_ALL_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Auto-fixing with clang-tidy (Google)")
endif()
