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
    set(input_shape "${ARGV3}")
    set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/generated/${name})
    set(library ${output_dir}/lib${name}.so)
    set(input_shape_args)
    if(NOT "${input_shape}" STREQUAL "")
        list(APPEND input_shape_args --input-shape ${input_shape})
    endif()
    add_custom_command(
        OUTPUT ${library} ${output_dir}/${name}.h ${output_dir}/${name}.json
        COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
        COMMAND $<TARGET_FILE:ncnn-compile>
            --driver $<TARGET_FILE:ncnn-mlir-driver>
            --opt $<TARGET_FILE:ncnn-mlir-opt>
            --translate ${MLIR_TRANSLATE}
            --clang ${CLANG}
            --nm ${LLVM_NM}
            --readelf ${LLVM_READELF}
            --param ${param_file}
            --bin ${bin_file}
            --model-name ${name}
            ${input_shape_args}
            --output-dir ${output_dir}
            --emit-manifest
        DEPENDS
            ncnn-compile
            ncnn-mlir-driver
            ncnn-mlir-opt
            ${CMAKE_SOURCE_DIR}/tools/ncnn-compile.cpp
            ${param_file}
            ${bin_file}
        VERBATIM)
    add_custom_target(compile_${name} DEPENDS ${library})
    set(${name}_LIBRARY ${library} PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set(empty_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/empty.bin)
set(convolution_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution.bin)
set(convolution_no_bias_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_no_bias.bin)
set(convolution_dilated_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_dilated.bin)
set(convolution_asymmetric_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_asymmetric.bin)
set(convolution_same_upper_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_same_upper.bin)
set(convolution_same_lower_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_same_lower.bin)
file(WRITE ${empty_bin} "")
foreach(convolution_case IN ITEMS
        convolution
        convolution_no_bias
        convolution_dilated
        convolution_asymmetric
        convolution_same_upper
        convolution_same_lower)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_bins.py
                ${CMAKE_CURRENT_BINARY_DIR}/fixtures/${convolution_case}.bin
                ${convolution_case}
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
ncnn_add_compiled_fixture(
    convolution_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution.param
    ${convolution_bin})
ncnn_add_compiled_fixture(
    convolution_no_bias
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_no_bias.param
    ${convolution_no_bias_bin})
ncnn_add_compiled_fixture(
    convolution_dilated
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_dilated.param
    ${convolution_dilated_bin})
ncnn_add_compiled_fixture(
    convolution_asymmetric
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_asymmetric.param
    ${convolution_asymmetric_bin})
ncnn_add_compiled_fixture(
    convolution_same_upper
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_same_upper.param
    ${convolution_same_upper_bin})
ncnn_add_compiled_fixture(
    convolution_same_lower
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_same_lower.param
    ${convolution_same_lower_bin})
ncnn_add_compiled_fixture(
    relu_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/relu.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    relu_leaky
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/relu_leaky.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_average
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_average.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_global_max
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_global_max.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_global_average
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_global_average.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_same_upper
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_same_upper.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_same_lower
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_same_lower.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_tail
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_tail.param
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
    concat_height
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/concat_height.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    concat_width
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/concat_width.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    concat_negative_height
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/concat_negative_height.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    concat_negative_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/concat_negative_channel.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    concat_negative_width
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/concat_negative_width.param
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
    softmax_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_channel.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    softmax_height
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_height.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    softmax_width
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_width.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    softmax_negative_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_negative_channel.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    softmax_negative_height
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_negative_height.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    softmax_negative_width
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_negative_width.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    split_three
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/split_three.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    squeezenet_v1_1
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.param
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.bin)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_doc_ori
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.bin
    3x224x224)
set(numerical_fixture_targets
    compile_convolution_scalar
    compile_convolution_no_bias
    compile_convolution_dilated
    compile_convolution_asymmetric
    compile_convolution_same_upper
    compile_convolution_same_lower
    compile_relu_scalar
    compile_relu_leaky
    compile_pooling_scalar
    compile_pooling_average
    compile_pooling_global_max
    compile_pooling_global_average
    compile_pooling_same_upper
    compile_pooling_same_lower
    compile_pooling_tail
    compile_split_scalar
    compile_split_three
    compile_concat_scalar
    compile_concat_height
    compile_concat_width
    compile_concat_negative_channel
    compile_concat_negative_height
    compile_concat_negative_width
    compile_dropout_scalar
    compile_softmax_scalar
    compile_softmax_channel
    compile_softmax_height
    compile_softmax_width
    compile_softmax_negative_channel
    compile_softmax_negative_height
    compile_softmax_negative_width)
list(LENGTH numerical_fixture_targets numerical_fixture_target_count)
math(EXPR numerical_fixture_last_index "${numerical_fixture_target_count} - 1")
foreach(index RANGE ${numerical_fixture_last_index})
    math(EXPR next_index "${index} + 1")
    if(next_index LESS numerical_fixture_target_count)
        list(GET numerical_fixture_targets ${index} current_target)
        list(GET numerical_fixture_targets ${next_index} next_target)
        add_dependencies(${next_target} ${current_target})
    endif()
endforeach()
add_dependencies(compile_squeezenet_v1_1 compile_softmax_negative_width)
add_dependencies(compile_pp_lcnet_x1_0_doc_ori compile_squeezenet_v1_1)

add_library(numerical_test_support STATIC
    support/numerical_test_support.cpp)
target_include_directories(numerical_test_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(numerical_test_support PUBLIC ncnn GTest::gtest)

add_executable(numerical_tests
    operators/supported_ops_test.cpp
    models/squeezenet_test.cpp
    models/pp_lcnet_test.cpp)
target_link_libraries(numerical_tests PRIVATE
    numerical_test_support
    GTest::gtest_main
    ${CMAKE_DL_LIBS})
target_compile_definitions(numerical_tests PRIVATE
    GTEST_HAS_CXXABI_H_=0
    NUMERICAL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    NUMERICAL_EMPTY_BIN_PATH="${empty_bin}"
    CONVOLUTION_BIN_PATH="${convolution_bin}"
    CONVOLUTION_NO_BIAS_BIN_PATH="${convolution_no_bias_bin}"
    CONVOLUTION_DILATED_BIN_PATH="${convolution_dilated_bin}"
    CONVOLUTION_ASYMMETRIC_BIN_PATH="${convolution_asymmetric_bin}"
    CONVOLUTION_SAME_UPPER_BIN_PATH="${convolution_same_upper_bin}"
    CONVOLUTION_SAME_LOWER_BIN_PATH="${convolution_same_lower_bin}"
    CONVOLUTION_LIBRARY_PATH="${convolution_scalar_LIBRARY}"
    RELU_LIBRARY_PATH="${relu_scalar_LIBRARY}"
    RELU_LEAKY_LIBRARY_PATH="${relu_leaky_LIBRARY}"
    POOLING_LIBRARY_PATH="${pooling_scalar_LIBRARY}"
    POOLING_AVERAGE_LIBRARY_PATH="${pooling_average_LIBRARY}"
    POOLING_GLOBAL_MAX_LIBRARY_PATH="${pooling_global_max_LIBRARY}"
    POOLING_GLOBAL_AVERAGE_LIBRARY_PATH="${pooling_global_average_LIBRARY}"
    POOLING_SAME_UPPER_LIBRARY_PATH="${pooling_same_upper_LIBRARY}"
    POOLING_SAME_LOWER_LIBRARY_PATH="${pooling_same_lower_LIBRARY}"
    POOLING_TAIL_LIBRARY_PATH="${pooling_tail_LIBRARY}"
    SPLIT_LIBRARY_PATH="${split_scalar_LIBRARY}"
    SPLIT_THREE_LIBRARY_PATH="${split_three_LIBRARY}"
    CONCAT_LIBRARY_PATH="${concat_scalar_LIBRARY}"
    CONCAT_HEIGHT_LIBRARY_PATH="${concat_height_LIBRARY}"
    CONCAT_WIDTH_LIBRARY_PATH="${concat_width_LIBRARY}"
    CONCAT_NEGATIVE_CHANNEL_LIBRARY_PATH="${concat_negative_channel_LIBRARY}"
    CONCAT_NEGATIVE_HEIGHT_LIBRARY_PATH="${concat_negative_height_LIBRARY}"
    CONCAT_NEGATIVE_WIDTH_LIBRARY_PATH="${concat_negative_width_LIBRARY}"
    DROPOUT_LIBRARY_PATH="${dropout_scalar_LIBRARY}"
    SOFTMAX_LIBRARY_PATH="${softmax_scalar_LIBRARY}"
    SOFTMAX_CHANNEL_LIBRARY_PATH="${softmax_channel_LIBRARY}"
    SOFTMAX_HEIGHT_LIBRARY_PATH="${softmax_height_LIBRARY}"
    SOFTMAX_WIDTH_LIBRARY_PATH="${softmax_width_LIBRARY}"
    SOFTMAX_NEGATIVE_CHANNEL_LIBRARY_PATH="${softmax_negative_channel_LIBRARY}"
    SOFTMAX_NEGATIVE_HEIGHT_LIBRARY_PATH="${softmax_negative_height_LIBRARY}"
    SOFTMAX_NEGATIVE_WIDTH_LIBRARY_PATH="${softmax_negative_width_LIBRARY}"
    CONVOLUTION_NO_BIAS_LIBRARY_PATH="${convolution_no_bias_LIBRARY}"
    CONVOLUTION_DILATED_LIBRARY_PATH="${convolution_dilated_LIBRARY}"
    CONVOLUTION_ASYMMETRIC_LIBRARY_PATH="${convolution_asymmetric_LIBRARY}"
    CONVOLUTION_SAME_UPPER_LIBRARY_PATH="${convolution_same_upper_LIBRARY}"
    CONVOLUTION_SAME_LOWER_LIBRARY_PATH="${convolution_same_lower_LIBRARY}"
    SQUEEZENET_PARAM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.param"
    SQUEEZENET_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.bin"
    SQUEEZENET_LIBRARY_PATH="${squeezenet_v1_1_LIBRARY}"
    PP_LCNET_DOC_ORI_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.param"
    PP_LCNET_DOC_ORI_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.bin"
    PP_LCNET_DOC_ORI_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_LIBRARY}")
add_dependencies(numerical_tests compile_squeezenet_v1_1)
add_dependencies(numerical_tests compile_pp_lcnet_x1_0_doc_ori)
add_dependencies(numerical_tests compile_pp_lcnet_x1_0_doc_ori)

gtest_discover_tests(numerical_tests
    PROPERTIES LABELS "numerical;golden" TIMEOUT 900 RUN_SERIAL TRUE)
