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

# ── 可配置选项 ───────────────────────────────────────────────
option(BUILD_TESTING "Build tests (pulls in GTest, lit and vendored ncnn)" ON)
option(BUILD_SHARED_LIBS
    "Build first-party libraries as shared instead of static (needs PIC-capable LLVM archives)" OFF)
option(COMPILER_ENABLE_WARNINGS
    "Enable -Wall -Wextra -Wpedantic on first-party targets" ON)
option(COMPILER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors (-Werror)" OFF)
set(COMPILER_SANITIZERS "" CACHE STRING
    "Sanitizers for all targets, semicolon-separated subset of: address, undefined")
option(COMPILER_ENABLE_FORMAT_TARGETS
    "Add clang-format/clang-tidy helper targets (requires the tools at configure time)" ON)
set(COMPILER_TIDY_JOBS "" CACHE STRING
    "Parallel jobs for the tidy target via run-clang-tidy.py (empty = all cores)")
option(COMPILER_INSTALL_RPATH
    "Embed LLVM_LIBRARY_DIR as INSTALL_RPATH (makes installed binaries non-relocatable)" OFF)
option(COMPILER_ENABLE_IPO "Enable interprocedural optimization / LTO" OFF)
option(COMPILER_USE_CCACHE "Use ccache as the compiler launcher" OFF)

# sanitizers：注入全局 flags。test/Numerical 通过探测 "fsanitize" 字样把参数
# 透传给生成的测试代码，因此必须走 CMAKE_*_FLAGS 而非 per-target 属性。
if(COMPILER_SANITIZERS)
    list(JOIN COMPILER_SANITIZERS "," _compiler_sanitize_list)
    set(_compiler_sanitize_flags "-fsanitize=${_compiler_sanitize_list} -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_compiler_sanitize_flags}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_compiler_sanitize_flags}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=${_compiler_sanitize_list}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=${_compiler_sanitize_list}")
endif()

if(COMPILER_USE_CCACHE)
    find_program(CCACHE_PROGRAM NAMES ccache REQUIRED)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# 共享库要求 PIC；IPO 需要工具链支持，不支持时降级并告警。
if(BUILD_SHARED_LIBS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if(COMPILER_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _compiler_ipo_supported OUTPUT _compiler_ipo_error)
    if(_compiler_ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(WARNING "IPO not supported by the toolchain, ignoring: ${_compiler_ipo_error}")
    endif()
endif()

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

if(BUILD_TESTING)
    enable_testing()
endif()

# Debian 的 LLVM/MLIR 以 -fno-rtti 构建；链接 MLIR 的 target 统一关闭 RTTI。
# 在需要的 target 上单独 target_compile_options(<t> PRIVATE -fno-rtti)。

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(tools)

# vendored SLEEF（向量数学后端，静态档案）。测试/DFT/GNUABI 全关，只产出
# <build>/lib/libsleef.a 供 ncnn-compile 定位注入产物链接。
set(SLEEF_BUILD_SHARED_LIBS OFF)
set(SLEEF_BUILD_TESTS OFF)
set(SLEEF_BUILD_GNUABI_LIBS OFF)
set(SLEEF_BUILD_DFT OFF)
set(SLEEF_BUILD_QUAD OFF)
set(SLEEF_BUILD_INLINE_HEADERS OFF)
set(SLEEF_SHOW_CONFIG OFF)
add_subdirectory(third_party/sleef)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

# Apply project warnings only to first-party compiled targets. Directory-wide
# flags would otherwise leak into the vendored ncnn targets under test/.
function(ncnn_enable_first_party_warnings directory)
    if(NOT COMPILER_ENABLE_WARNINGS)
        return()
    endif()
    set(NCNN_WARNING_FLAGS -Wall -Wextra -Wpedantic)
    if(COMPILER_WARNINGS_AS_ERRORS)
        list(APPEND NCNN_WARNING_FLAGS -Werror)
    endif()
    get_property(NCNN_DIRECTORY_TARGETS DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(NCNN_TARGET IN LISTS NCNN_DIRECTORY_TARGETS)
        get_target_property(NCNN_TARGET_TYPE "${NCNN_TARGET}" TYPE)
        get_target_property(NCNN_TARGET_SOURCE_DIR "${NCNN_TARGET}" SOURCE_DIR)
        if(NCNN_TARGET_TYPE MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$"
           AND NCNN_TARGET_SOURCE_DIR MATCHES "^${CMAKE_SOURCE_DIR}(/|$)"
           AND NOT NCNN_TARGET_SOURCE_DIR MATCHES "^${CMAKE_SOURCE_DIR}/test/third_party(/|$)")
            target_compile_options("${NCNN_TARGET}" PRIVATE ${NCNN_WARNING_FLAGS})
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
    "/([^/]*build[^/]*|third_party|generated)/")

# clang-tidy 的检查范围由 tidy 目标里的 -source-filter 正则从 compile_commands.json
# 中筛选（头文件经由所属翻译单元检查，不作为 main file 传入）。

if(COMPILER_ENABLE_FORMAT_TARGETS)
    find_program(CLANG_FORMAT NAMES clang-format-21 REQUIRED)
    find_program(CLANG_TIDY NAMES clang-tidy-21 REQUIRED)
    # tidy_fix 经 run-clang-tidy.py 合并修复，需要 clang-apply-replacements。
    find_program(CLANG_APPLY_REPLACEMENTS NAMES clang-apply-replacements-21 REQUIRED)
    set(NCNN_TIDY_HEADER_FILTER
        "^${CMAKE_SOURCE_DIR}/(include|lib|bin|tools|test)/")
    set(NCNN_TIDY_EXCLUDE_HEADER_FILTER
        "^${CMAKE_SOURCE_DIR}/([^/]*build[^/]*|test/third_party|third_party|(.*/)?generated)(/|$)")
    # 仅第一方 lib/bin/tools/test 下的 .cpp；排除 *build*/third_party/generated。
    set(NCNN_TIDY_SOURCE_FILTER
        "^(?!${CMAKE_SOURCE_DIR}/([^/]*build[^/]*|test/third_party|third_party|(.*/)?generated)(/|$))${CMAKE_SOURCE_DIR}/(lib|bin|tools|test)/.*[.]cpp$")
    set(NCNN_RUN_TIDY_SCRIPT "${CMAKE_SOURCE_DIR}/run-clang-tidy.py")
    set(NCNN_RUN_TIDY_ARGS
        "${NCNN_RUN_TIDY_SCRIPT}"
        "-p=${CMAKE_BINARY_DIR}"
        "-clang-tidy-binary=${CLANG_TIDY}"
        # run-clang-tidy.py 的长选项是单横线风格（与 clang-tidy 本体不同）。
        "-header-filter=${NCNN_TIDY_HEADER_FILTER}"
        "-exclude-header-filter=${NCNN_TIDY_EXCLUDE_HEADER_FILTER}"
        "-source-filter=${NCNN_TIDY_SOURCE_FILTER}")
    # 默认全核并行；COMPILER_TIDY_JOBS 可显式限速。
    if(COMPILER_TIDY_JOBS)
        list(APPEND NCNN_RUN_TIDY_ARGS "-j${COMPILER_TIDY_JOBS}")
    else()
        include(ProcessorCount)
        ProcessorCount(NCNN_TIDY_JOBS_DEFAULT)
        if(NCNN_TIDY_JOBS_DEFAULT GREATER 0)
            list(APPEND NCNN_RUN_TIDY_ARGS "-j${NCNN_TIDY_JOBS_DEFAULT}")
        endif()
    endif()

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

    # clang-tidy：静态检查（依赖 compile_commands.json）。
    # run-clang-tidy.py 并行运行，默认使用全部核心，可用 COMPILER_TIDY_JOBS 限速。
    add_custom_target(tidy
        COMMAND python3 ${NCNN_RUN_TIDY_ARGS}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy in parallel (Google)"
        VERBATIM)
    add_dependencies(tidy NCNNOpsIncGen NCNNPassesIncGen)
    add_custom_target(tidy_fix
        COMMAND python3 ${NCNN_RUN_TIDY_ARGS} -fix
                "-clang-apply-replacements-binary=${CLANG_APPLY_REPLACEMENTS}"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Auto-fixing with clang-tidy in parallel (Google)"
        VERBATIM)
    add_dependencies(tidy_fix NCNNOpsIncGen NCNNPassesIncGen)
endif()
