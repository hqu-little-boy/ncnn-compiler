cmake_minimum_required(VERSION 3.20)

# 用系统 clang-21（MLIR 21 同源工具链，版本一致避免 ABI 抖）。必须在 project() 前设置。
set(CMAKE_C_COMPILER /usr/bin/clang-21)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-21)

project(ncnn_graph LANGUAGES CXX)

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

# 库：计算图数据模型 + 解析器
add_library(ncnn_graph STATIC
    src/graph.cpp
    src/parser.cpp
)
target_include_directories(ncnn_graph PUBLIC include)

# 验收程序：读 squeezenet，dump + 自检
add_executable(load_squeezenet test/load_squeezenet.cpp)
target_link_libraries(load_squeezenet PRIVATE ncnn_graph)
target_compile_definitions(load_squeezenet PRIVATE
    NCNN_GRAPH_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(parser_test test/parser_test.cpp)
target_link_libraries(parser_test PRIVATE ncnn_graph)

add_executable(malformed_graph_test test/malformed_graph_test.cpp)
target_link_libraries(malformed_graph_test PRIVATE ncnn_graph)
target_compile_definitions(malformed_graph_test PRIVATE
    NCNN_GRAPH_TEST_TEMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/test-fixtures")

enable_testing()
add_test(NAME load_squeezenet_default
    COMMAND load_squeezenet
            ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/examples/squeezenet_v1.1.param
            ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/examples/squeezenet_v1.1.bin)
add_test(NAME parser_test COMMAND parser_test)
add_test(NAME malformed_graph_test COMMAND malformed_graph_test)

# ── 辅助目标：clang-format / clang-tidy ───────────────────
# 收集所有源/头文件（排除 build 目录）
file(GLOB_RECURSE NCNN_ALL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/test/*.cpp")

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
    set(CLANG_TIDY_FIX ${CMAKE_SOURCE_DIR}/.clang-tidy)
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
