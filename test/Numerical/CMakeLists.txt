find_package(GTest REQUIRED)
include(GoogleTest)

set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_THREADS OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
set(NCNN_RUNTIME_CPU OFF CACHE BOOL "" FORCE)
set(NCNN_GNU_INLINE_ASM OFF CACHE BOOL "" FORCE)
set(NCNN_SSE2 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX OFF CACHE BOOL "" FORCE)
set(NCNN_AVX2 OFF CACHE BOOL "" FORCE)
set(NCNN_FMA OFF CACHE BOOL "" FORCE)
set(NCNN_XOP OFF CACHE BOOL "" FORCE)
set(NCNN_F16C OFF CACHE BOOL "" FORCE)
set(NCNN_AVXVNNI OFF CACHE BOOL "" FORCE)
set(NCNN_AVXVNNIINT8 OFF CACHE BOOL "" FORCE)
set(NCNN_AVXVNNIINT16 OFF CACHE BOOL "" FORCE)
set(NCNN_AVXNECONVERT OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512VNNI OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512BF16 OFF CACHE BOOL "" FORCE)
set(NCNN_AVX512FP16 OFF CACHE BOOL "" FORCE)
set(NCNN_VFPV4 OFF CACHE BOOL "" FORCE)
set(NCNN_ARM82 OFF CACHE BOOL "" FORCE)
set(NCNN_ARM82DOT OFF CACHE BOOL "" FORCE)
set(NCNN_ARM82FP16FML OFF CACHE BOOL "" FORCE)
set(NCNN_INT8 OFF CACHE BOOL "" FORCE)
set(NCNN_BF16 OFF CACHE BOOL "" FORCE)
set(NCNN_WEIGHT_QUANT OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
set(NCNN_C_API OFF CACHE BOOL "" FORCE)
set(NCNN_PLATFORM_API OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION ON CACHE BOOL "" FORCE)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn
    ${CMAKE_CURRENT_BINARY_DIR}/ncnn
    EXCLUDE_FROM_ALL)

function(ncnn_add_compiled_fixture name param_file bin_file)
    set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/generated/${name})
    set(library ${output_dir}/lib${name}.so)
    add_custom_command(
        OUTPUT ${library} ${output_dir}/${name}.h ${output_dir}/${name}.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/compile_ncnn_model.py
            --driver $<TARGET_FILE:ncnn-mlir-driver>
            --opt $<TARGET_FILE:ncnn-mlir-opt>
            --translate ${MLIR_TRANSLATE}
            --clang ${CLANG}
            --nm ${LLVM_NM}
            --readelf ${LLVM_READELF}
            --param ${param_file}
            --bin ${bin_file}
            --model-name ${name}
            --output-dir ${output_dir}
        DEPENDS
            ncnn-mlir-driver
            ncnn-mlir-opt
            ${CMAKE_SOURCE_DIR}/tools/compile_ncnn_model.py
            ${param_file}
            ${bin_file}
        VERBATIM)
    add_custom_target(compile_${name} DEPENDS ${library})
    set(${name}_LIBRARY ${library} PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set(empty_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/empty.bin)
set(convolution_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution.bin)
file(WRITE ${empty_bin} "")
execute_process(
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_bins.py
            ${convolution_bin}
    COMMAND_ERROR_IS_FATAL ANY)
ncnn_add_compiled_fixture(
    convolution_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution.param
    ${convolution_bin})
ncnn_add_compiled_fixture(
    relu_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/relu.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    split_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/split.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    concat_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/concat.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    dropout_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dropout.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    softmax_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    squeezenet_v1_1
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.param
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.bin)
add_dependencies(compile_relu_scalar compile_convolution_scalar)
add_dependencies(compile_pooling_scalar compile_relu_scalar)
add_dependencies(compile_split_scalar compile_pooling_scalar)
add_dependencies(compile_concat_scalar compile_split_scalar)
add_dependencies(compile_dropout_scalar compile_concat_scalar)
add_dependencies(compile_softmax_scalar compile_dropout_scalar)
add_dependencies(compile_squeezenet_v1_1 compile_softmax_scalar)

add_library(numerical_test_support STATIC
    support/numerical_test_support.cpp)
target_include_directories(numerical_test_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(numerical_test_support PUBLIC ncnn GTest::gtest)

add_executable(numerical_tests
    operators/supported_ops_test.cpp
    models/squeezenet_test.cpp)
target_link_libraries(numerical_tests PRIVATE
    numerical_test_support
    GTest::gtest_main
    ${CMAKE_DL_LIBS})
target_compile_definitions(numerical_tests PRIVATE
    GTEST_HAS_CXXABI_H_=0
    NUMERICAL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    NUMERICAL_EMPTY_BIN_PATH="${empty_bin}"
    CONVOLUTION_BIN_PATH="${convolution_bin}"
    CONVOLUTION_LIBRARY_PATH="${convolution_scalar_LIBRARY}"
    RELU_PARAM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/relu.param"
    RELU_BIN_PATH="${empty_bin}"
    RELU_LIBRARY_PATH="${relu_scalar_LIBRARY}"
    POOLING_LIBRARY_PATH="${pooling_scalar_LIBRARY}"
    SPLIT_LIBRARY_PATH="${split_scalar_LIBRARY}"
    CONCAT_LIBRARY_PATH="${concat_scalar_LIBRARY}"
    DROPOUT_LIBRARY_PATH="${dropout_scalar_LIBRARY}"
    SOFTMAX_LIBRARY_PATH="${softmax_scalar_LIBRARY}"
    SQUEEZENET_PARAM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.param"
    SQUEEZENET_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.bin"
    SQUEEZENET_LIBRARY_PATH="${squeezenet_v1_1_LIBRARY}")
add_dependencies(numerical_tests compile_squeezenet_v1_1)

gtest_discover_tests(numerical_tests
    PROPERTIES LABELS "numerical;golden" TIMEOUT 900 RUN_SERIAL TRUE)
