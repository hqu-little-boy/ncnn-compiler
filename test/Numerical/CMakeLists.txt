find_package(GTest REQUIRED)
include(GoogleTest)

set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
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
if(NOT MSVC)
    target_compile_options(ncnn PRIVATE -fno-fast-math)
endif()

function(ncnn_add_compiled_fixture name param_file bin_file)
    set(input_shape "${ARGV3}")
    set(input_constraints ${ARGN})
    if(NOT "${input_shape}" STREQUAL "")
        list(REMOVE_AT input_constraints 0)
    endif()
    set(precision_args)
    set(clang_args)
    set(linker_args)
    set(filtered_constraints)
    set(additional_input_shapes)
    set(emit_ir OFF)
    set(emit_backend OFF)
    foreach(argument IN LISTS input_constraints)
        if(argument MATCHES "^PRECISION=(.+)$")
            list(APPEND precision_args --precision ${CMAKE_MATCH_1})
        elseif(argument MATCHES "^FP16_ACCUMULATOR=(.+)$")
            list(APPEND precision_args --fp16-accumulator ${CMAKE_MATCH_1})
        elseif(argument STREQUAL "ALLOW_FALLBACK")
            list(APPEND precision_args --allow-fallback)
        elseif(argument MATCHES "^TARGET_FEATURE=(.+)$")
            list(APPEND precision_args --target-feature ${CMAKE_MATCH_1})
        elseif(argument MATCHES "^CLANG_ARG=(.+)$")
            list(APPEND clang_args --clang-arg ${CMAKE_MATCH_1})
        elseif(argument MATCHES "^LINKER_ARG=(.+)$")
            list(APPEND linker_args --linker-arg ${CMAKE_MATCH_1})
        elseif(argument MATCHES "^INPUT_SHAPE=(.+)$")
            list(APPEND additional_input_shapes ${CMAKE_MATCH_1})
        elseif(argument STREQUAL "EMIT_IR")
            list(APPEND precision_args --emit ncnn --emit tosa --emit linalg)
            set(emit_ir ON)
        elseif(argument STREQUAL "EMIT_BACKEND")
            list(APPEND precision_args --emit linalg --emit llvm-ir
                                            --emit object --emit assembly)
            set(emit_backend ON)
        else()
            list(APPEND filtered_constraints ${argument})
        endif()
    endforeach()
    set(input_constraints ${filtered_constraints})
    set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/generated/${name})
    set(library ${output_dir}/lib${name}.so)
    set(generated_outputs
        ${library} ${output_dir}/${name}.h ${output_dir}/${name}.json)
    if(emit_ir)
        list(APPEND generated_outputs
             ${output_dir}/model.ncnn.mlir
             ${output_dir}/model.tosa.mlir
             ${output_dir}/model.linalg.mlir)
    endif()
    if(emit_backend)
        list(APPEND generated_outputs
             ${output_dir}/model.linalg.mlir
             ${output_dir}/model.ll
             ${output_dir}/model.o
             ${output_dir}/model.s)
    endif()
    list(REMOVE_DUPLICATES generated_outputs)
    set(input_shape_args)
    if(NOT "${input_shape}" STREQUAL "")
        list(APPEND input_shape_args --input-shape ${input_shape})
    endif()
    foreach(additional_input_shape IN LISTS additional_input_shapes)
        list(APPEND input_shape_args --input-shape ${additional_input_shape})
    endforeach()
    set(input_constraint_args)
    foreach(constraint IN LISTS input_constraints)
        list(APPEND input_constraint_args --input-dim-constraint ${constraint})
    endforeach()
    set(sanitizer_args)
    if(CMAKE_CXX_FLAGS MATCHES "fsanitize")
        list(APPEND sanitizer_args
             --clang-arg=-fsanitize=address,undefined
             --clang-arg=-fno-omit-frame-pointer
             --linker-arg=-fsanitize=address,undefined
             --linker-arg=-lasan
             --linker-arg=-lubsan)
    endif()
    add_custom_command(
        OUTPUT ${generated_outputs}
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
            ${input_constraint_args}
            ${precision_args}
            ${clang_args}
            ${linker_args}
            ${sanitizer_args}
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
    add_custom_target(compile_${name} DEPENDS ${generated_outputs})
    set(${name}_LIBRARY ${library} PARENT_SCOPE)
endfunction()

function(ncnn_add_compiled_shape_pair
         name param_file bin_file static_shape dynamic_shape)
    ncnn_add_compiled_fixture(
        ${name}_static ${param_file} ${bin_file} ${static_shape})
    ncnn_add_compiled_fixture(
        ${name}_dynamic ${param_file} ${bin_file} ${dynamic_shape} ${ARGN})
    set(${name}_static_LIBRARY ${${name}_static_LIBRARY} PARENT_SCOPE)
    set(${name}_dynamic_LIBRARY ${${name}_dynamic_LIBRARY} PARENT_SCOPE)
endfunction()

function(ncnn_add_complex_precision_fixtures name param_file bin_file)
    ncnn_add_compiled_fixture(
        ${name}_mixed_fp16 ${param_file} ${bin_file} ""
        PRECISION=fp16 FP16_ACCUMULATOR=f32 TARGET_FEATURE=+f16c EMIT_IR)
    ncnn_add_compiled_fixture(
        ${name}_mixed_bf16 ${param_file} ${bin_file} ""
        PRECISION=bf16 TARGET_FEATURE=+avx512bf16
        CLANG_ARG=-Xclang CLANG_ARG=-target-feature
        CLANG_ARG=-Xclang CLANG_ARG=-avx512bf16
        CLANG_ARG=-Xclang CLANG_ARG=-target-feature
        CLANG_ARG=-Xclang CLANG_ARG=-avx512f EMIT_IR)
    set(${name}_mixed_fp16_LIBRARY
        ${${name}_mixed_fp16_LIBRARY} PARENT_SCOPE)
    set(${name}_mixed_bf16_LIBRARY
        ${${name}_mixed_bf16_LIBRARY} PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
set(int8_math_shim ${CMAKE_CURRENT_BINARY_DIR}/fixtures/int8_math_shim.o)
execute_process(
    COMMAND ${CLANG} -c -fPIC
            ${CMAKE_CURRENT_SOURCE_DIR}/support/int8_math_shim.c
            -o ${int8_math_shim}
    COMMAND_ERROR_IS_FATAL ANY)
set(empty_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/empty.bin)
set(convolution_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution.bin)
set(convolution_no_bias_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_no_bias.bin)
set(convolution_dilated_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_dilated.bin)
set(convolution_asymmetric_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_asymmetric.bin)
set(convolution_asymmetric_stride_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_asymmetric_stride.bin)
set(convolution_same_upper_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_same_upper.bin)
set(convolution_same_lower_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_same_lower.bin)
set(convolution_fp16_storage_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_fp16_storage.bin)
set(convolution_depthwise_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise.bin)
set(convolution_depthwise_fp16_storage_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise_fp16_storage.bin)
set(convolution_depthwise_7x7_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise_7x7.bin)
set(convolution_depthwise_9x9_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise_9x9.bin)
set(convolution_depthwise_asymmetric_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise_asymmetric.bin)
set(convolution_depthwise_same_upper_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise_same_upper.bin)
set(convolution_depthwise_same_lower_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/convolution_depthwise_same_lower.bin)
set(inner_product_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/inner_product.bin)
set(inner_product_dynamic_m_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/inner_product_dynamic_m.bin)
set(inner_product_fused_relu_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/inner_product_fused_relu.bin)
set(batch_norm_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/batch_norm.bin)
set(gemm_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/gemm.bin)
set(deconvolution_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/deconvolution.bin)
set(deconvolution_no_bias_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/deconvolution_no_bias.bin)
set(deconvolution_tiny_head_bin ${CMAKE_CURRENT_BINARY_DIR}/fixtures/deconvolution_tiny_head.bin)
file(WRITE ${empty_bin} "")
foreach(convolution_case IN ITEMS
        convolution
        convolution_no_bias
        convolution_dilated
        convolution_asymmetric
        convolution_asymmetric_stride
        convolution_same_upper
        convolution_same_lower
        convolution_fp16_storage)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_bins.py
                ${CMAKE_CURRENT_BINARY_DIR}/fixtures/${convolution_case}.bin
                ${convolution_case}
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
foreach(weighted_case IN ITEMS
        convolution_depthwise
        convolution_depthwise_fp16_storage
        convolution_depthwise_7x7
        convolution_depthwise_9x9
        convolution_depthwise_asymmetric
        convolution_depthwise_same_upper
        convolution_depthwise_same_lower
        inner_product
        inner_product_dynamic_m
        inner_product_fused_relu
        batch_norm
        gemm
        deconvolution
        deconvolution_no_bias
        deconvolution_tiny_head)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_bins.py
                ${CMAKE_CURRENT_BINARY_DIR}/fixtures/${weighted_case}.bin
                ${weighted_case}
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
set(int8_cases
    convolution_int8_term1
    convolution_int8_term2
    convolution_int8_term101
    convolution_int8_term102
    depthwise_int8_term1
    depthwise_int8_term2
    depthwise_int8_term101
    depthwise_int8_term102
    inner_product_int8_term1
    inner_product_int8_term2
    gemm_int8_term2
    int8_complete_chain)
foreach(int8_case IN LISTS int8_cases)
    set(${int8_case}_bin
        ${CMAKE_CURRENT_BINARY_DIR}/fixtures/${int8_case}.bin)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_int8_bins.py
                ${${int8_case}_bin} ${int8_case}
        COMMAND_ERROR_IS_FATAL ANY)
    set(int8_fixture_options LINKER_ARG=${int8_math_shim})
    if(int8_case STREQUAL "int8_complete_chain")
        list(APPEND int8_fixture_options EMIT_IR)
    endif()
    if(int8_case MATCHES "^depthwise")
        set(int8_input_shape 3x3x4)
    elseif(int8_case MATCHES "^inner_product")
        set(int8_input_shape 1x1x4)
    elseif(int8_case MATCHES "^gemm")
        set(int8_input_shape 2x1x3)
    else()
        set(int8_input_shape 2x3x4)
    endif()
    ncnn_add_compiled_fixture(
        ${int8_case}
        ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${int8_case}.param
        ${${int8_case}_bin}
        ${int8_input_shape}
        ${int8_fixture_options})
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
    convolution_asymmetric_stride
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_asymmetric_stride.param
    ${convolution_asymmetric_stride_bin})
ncnn_add_compiled_fixture(
    convolution_same_upper
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_same_upper.param
    ${convolution_same_upper_bin})
ncnn_add_compiled_fixture(
    convolution_same_lower
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_same_lower.param
    ${convolution_same_lower_bin})
ncnn_add_compiled_fixture(
    convolution_fp16_storage
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_fp16_storage.param
    ${convolution_fp16_storage_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise.param
    ${convolution_depthwise_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise_fp16_storage
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise_fp16_storage.param
    ${convolution_depthwise_fp16_storage_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise_7x7
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise_7x7.param
    ${convolution_depthwise_7x7_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise_9x9
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise_9x9.param
    ${convolution_depthwise_9x9_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise_asymmetric
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise_asymmetric.param
    ${convolution_depthwise_asymmetric_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise_same_upper
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise_same_upper.param
    ${convolution_depthwise_same_upper_bin})
ncnn_add_compiled_fixture(
    convolution_depthwise_same_lower
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise_same_lower.param
    ${convolution_depthwise_same_lower_bin})
ncnn_add_compiled_fixture(
    relu_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/relu.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    hard_sigmoid
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hard_sigmoid.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    hard_swish
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hard_swish.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    gelu
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gelu.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    reshape
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reshape.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    reshape_copy_dimension
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reshape_copy_dimension.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    squeeze
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/squeeze.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    batch_norm
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/batch_norm.param
    ${batch_norm_bin})
ncnn_add_compiled_fixture(
    expand_dims
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/expand_dims.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    permute
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/permute.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gemm.param
    ${gemm_bin})
ncnn_add_compiled_fixture(
    binary_mul_scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/binary_mul_scalar.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    binary_mul_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/binary_mul_channel.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    binary_mul_reverse_broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/binary_mul_reverse_broadcast.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    binary_add
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/binary_add.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    inner_product
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/inner_product.param
    ${inner_product_bin})
ncnn_add_compiled_fixture(
    inner_product_fused_relu
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/inner_product_fused_relu.param
    ${inner_product_fused_relu_bin})
ncnn_add_compiled_fixture(
    shuffle_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shuffle_channel.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    shuffle_channel_reverse
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shuffle_channel_reverse.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    slice
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/slice.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    reduction_mean
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reduction_mean.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    reduction_mean_keepdims_coeff
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reduction_mean_keepdims_coeff.param
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
    pooling_asymmetric
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_asymmetric.param
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
    pooling_global_max_ignored_params
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_global_max_ignored_params.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pooling_global_average_ignored_params
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_global_average_ignored_params.param
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
    padding
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/padding.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    padding_four_sides
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/padding_four_sides.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    padding_identity
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/padding_identity.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    interp
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/interp.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    interp_asymmetric
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/interp_asymmetric.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    interp_explicit_target
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/interp_explicit_target.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    interp_identity
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/interp_identity.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    interp_eightfold
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/interp_eightfold.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    deconvolution
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/deconvolution.param
    ${deconvolution_bin})
ncnn_add_compiled_fixture(
    deconvolution_no_bias
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/deconvolution_no_bias.param
    ${deconvolution_no_bias_bin})
ncnn_add_compiled_fixture(
    deconvolution_tiny_head
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/deconvolution_tiny_head.param
    ${deconvolution_tiny_head_bin})
ncnn_add_compiled_fixture(
    sigmoid
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/sigmoid.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    relu_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/relu_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    sigmoid_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/sigmoid_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    padding_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/padding_dynamic.param
    ${empty_bin}
    2x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    interp_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/interp_dynamic.param
    ${empty_bin}
    4x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    hard_sigmoid_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hard_sigmoid_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    hard_swish_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hard_swish_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    gelu_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gelu_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    dropout_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dropout_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    softmax_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    batch_norm_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/batch_norm_dynamic.param
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/batch_norm.bin
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    shuffle_channel_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shuffle_channel_dynamic.param
    ${empty_bin}
    4x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    squeeze_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/squeeze_dynamic.param
    ${empty_bin}
    1x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    permute_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/permute_dynamic.param
    ${empty_bin}
    1x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    expand_dims_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/expand_dims_dynamic.param
    ${empty_bin}
    1x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    reduction_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reduction_dynamic.param
    ${empty_bin}
    ?x4x?
    0:0:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    reduction_dynamic_mean
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reduction_dynamic_mean.param
    ${empty_bin}
    4x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    gemm_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gemm_dynamic.param
    ${gemm_bin}
    2x1x?
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    slice_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/slice_dynamic.param
    ${empty_bin}
    4x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_padded_max_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_padded_max_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_padded_average_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_padded_average_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_padded_average_include_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_padded_average_include_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_global_max_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_global_max_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_global_average_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_global_average_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_adaptive_max_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_adaptive_max_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pooling_adaptive_average_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pooling_adaptive_average_dynamic.param
    ${empty_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    reshape_spec_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reshape_spec_dynamic.param
    ${empty_bin}
    1x?x6
    0:1:min=1,multiple=1)
ncnn_add_compiled_fixture(
    slice_ordered_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/slice_ordered_dynamic.param
    ${empty_bin}
    1x?x4
    0:1:min=5,multiple=1)
ncnn_add_compiled_fixture(
    inner_product_dynamic_m
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/inner_product_dynamic_m.param
    ${inner_product_dynamic_m_bin}
    1x?x4
    0:1:min=1,multiple=1)
ncnn_add_compiled_fixture(
    detection_output
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/detection_output.param
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
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_doc_ori_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.bin
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_doc_ori_int8
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori_int8.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori_int8.bin
    3x224x224)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_doc_ori_int8_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori_int8.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori_int8.bin
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1
    EMIT_IR)
set(pp_lcnet_doc_ori_int8_backbone_bin
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/pp_lcnet_doc_ori_int8_backbone.bin)
execute_process(
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_int8_bins.py
            ${pp_lcnet_doc_ori_int8_backbone_bin}
            pp_lcnet_doc_ori_int8_backbone
            ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori_int8.bin
    COMMAND_ERROR_IS_FATAL ANY)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_doc_ori_int8_backbone_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pp_lcnet_doc_ori_int8_backbone.param
    ${pp_lcnet_doc_ori_int8_backbone_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_textline_ori
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.bin
    3x80x160)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_textline_ori_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.bin
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_textline_ori_int8
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori_int8.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori_int8.bin
    3x80x160
    EMIT_BACKEND)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_textline_ori_int8_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori_int8.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori_int8.bin
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1
    EMIT_IR)
set(pp_lcnet_textline_ori_int8_backbone_bin
    ${CMAKE_CURRENT_BINARY_DIR}/fixtures/pp_lcnet_textline_ori_int8_backbone.bin)
execute_process(
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_int8_bins.py
            ${pp_lcnet_textline_ori_int8_backbone_bin}
            pp_lcnet_textline_ori_int8_backbone
            ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori_int8.bin
    COMMAND_ERROR_IS_FATAL ANY)
ncnn_add_compiled_fixture(
    pp_lcnet_x1_0_textline_ori_int8_backbone_dynamic
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pp_lcnet_textline_ori_int8_backbone.param
    ${pp_lcnet_textline_ori_int8_backbone_bin}
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1)
ncnn_add_compiled_fixture(
    chineseocr_lite_anglenet
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.bin
    3x32x192)
ncnn_add_compiled_fixture(
    chineseocr_lite_anglenet_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.bin
    3x?x?
    0:1:min=1,multiple=1
    0:2:min=1,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_rec
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.bin
    3x48x320)
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_rec_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.bin
    3x48x?
    0:2:min=5,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_rec_int8
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec_int8.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec_int8.bin
    3x48x320)
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_rec_int8_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec_int8.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec_int8.bin
    3x48x?
    0:2:min=5,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv5_mobile_rec
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.bin
    3x48x320
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv5_mobile_rec_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.bin
    3x48x?
    0:2:min=5,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv5_server_rec
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.bin
    3x48x320
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv5_server_rec_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.bin
    3x48x?
    0:2:min=5,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv6_medium_rec
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.bin
    3x48x320
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv6_medium_rec_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.bin
    3x48x?
    0:2:min=5,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_det
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin
    3x640x640)
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_det_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin
    3x?x?
    0:1:min=32,multiple=32
    0:2:min=32,multiple=32)
ncnn_add_compiled_fixture(
    pp_ocrv6_small_det
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.bin
    3x640x640)
ncnn_add_compiled_fixture(
    pp_ocrv6_small_det_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.bin
    3x?x?
    0:1:min=32,multiple=32
    0:2:min=32,multiple=32)
ncnn_add_compiled_fixture(
    pp_ocrv6_medium_det
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.bin
    3x640x640)
ncnn_add_compiled_fixture(
    pp_ocrv6_medium_det_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.bin
    3x?x?
    0:1:min=32,multiple=32
    0:2:min=32,multiple=32)
ncnn_add_compiled_shape_pair(
    pp_ocrv5_mobile_det
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.bin
    3x640x640
    3x?x?
    0:1:min=32,multiple=32
    0:2:min=32,multiple=32)
ncnn_add_compiled_shape_pair(
    pp_ocrv5_server_det
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_det.bin
    3x640x640
    3x?x?
    0:1:min=32,multiple=32
    0:2:min=32,multiple=32)
ncnn_add_compiled_fixture(
    pp_structrurev2_slanet_plus_cnn
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_cnn.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_cnn.bin
    3x488x488
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_structrurev2_slanet_plus_slahead
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_slahead.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_slahead.bin
    ""
    INPUT_SHAPE=1x256 INPUT_SHAPE=256x96 INPUT_SHAPE=1x50 EMIT_IR)
ncnn_add_compiled_fixture(
    pp_formulanet_plus_s_encoder
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_encoder.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_encoder.bin
    1x384x384
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_formulanet_plus_s_embed
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_embed.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_embed.bin
    ""
    INPUT_SHAPE=1 INPUT_SHAPE=1 EMIT_IR)
ncnn_add_compiled_fixture(
    pp_formulanet_plus_s_decoder
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_decoder.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_decoder.bin
    144x2048
    INPUT_SHAPE=1x384
    INPUT_SHAPE=1x?
    INPUT_SHAPE=16x?x24
    INPUT_SHAPE=16x?x24
    INPUT_SHAPE=16x?x24
    INPUT_SHAPE=16x?x24
    2:1:min=1,multiple=1
    3:1:min=1,multiple=1
    4:1:min=1,multiple=1
    5:1:min=1,multiple=1
    6:1:min=1,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    pp_uvdoc_dynamic
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-UVDoc.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-UVDoc.bin
    3x?x?
    0:1:min=2,multiple=1
    0:2:min=2,multiple=1
    EMIT_IR)
ncnn_add_compiled_fixture(
    convolution_mixed_fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution.param
    ${convolution_bin}
    1x4x4
    PRECISION=fp16 FP16_ACCUMULATOR=f32 TARGET_FEATURE=+f16c EMIT_IR)
ncnn_add_compiled_fixture(
    convolution_arithmetic_fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution.param
    ${convolution_bin}
    1x4x4
    PRECISION=fp16 FP16_ACCUMULATOR=f16 TARGET_FEATURE=+avx512fp16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512fp16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512f EMIT_BACKEND)
ncnn_add_compiled_fixture(
    convolution_codegen_fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution.param
    ${convolution_bin}
    1x4x4
    PRECISION=fp16 FP16_ACCUMULATOR=f16 TARGET_FEATURE=+avx512fp16
    EMIT_BACKEND)
ncnn_add_compiled_fixture(
    convolution_mixed_bf16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution.param
    ${convolution_bin}
    1x4x4
    PRECISION=bf16 TARGET_FEATURE=+avx512bf16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512bf16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512f EMIT_IR)
ncnn_add_compiled_fixture(
    convolution_depthwise_mixed_fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise.param
    ${convolution_depthwise_bin}
    2x5x5
    PRECISION=fp16 FP16_ACCUMULATOR=f32 TARGET_FEATURE=+f16c)
ncnn_add_compiled_fixture(
    convolution_depthwise_arithmetic_fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise.param
    ${convolution_depthwise_bin}
    2x5x5
    PRECISION=fp16 FP16_ACCUMULATOR=f16 TARGET_FEATURE=+avx512fp16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512fp16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512f)
ncnn_add_compiled_fixture(
    convolution_depthwise_mixed_bf16
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/convolution_depthwise.param
    ${convolution_depthwise_bin}
    2x5x5
    PRECISION=bf16 TARGET_FEATURE=+avx512bf16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512bf16
    CLANG_ARG=-Xclang CLANG_ARG=-target-feature
    CLANG_ARG=-Xclang CLANG_ARG=-avx512f)
ncnn_add_complex_precision_fixtures(
    sigmoid
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/sigmoid.param
    ${empty_bin})
ncnn_add_complex_precision_fixtures(
    hard_sigmoid
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hard_sigmoid.param
    ${empty_bin})
ncnn_add_complex_precision_fixtures(
    hard_swish
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hard_swish.param
    ${empty_bin})
ncnn_add_complex_precision_fixtures(
    gelu
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/gelu.param
    ${empty_bin})
ncnn_add_complex_precision_fixtures(
    softmax
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/softmax.param
    ${empty_bin})
ncnn_add_complex_precision_fixtures(
    batch_norm
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/batch_norm.param
    ${batch_norm_bin})
ncnn_add_complex_precision_fixtures(
    detection_output
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/detection_output.param
    ${empty_bin})
ncnn_add_compiled_fixture(
    pp_ocrv6_tiny_det_fp16
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin
    3x640x640
    PRECISION=fp16 FP16_ACCUMULATOR=f32 TARGET_FEATURE=+f16c)
ncnn_add_compiled_fixture(
    pp_structrurev2_slanet_plus_cnn_fp16
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_cnn.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_cnn.bin
    3x488x488
    PRECISION=fp16 FP16_ACCUMULATOR=f32 TARGET_FEATURE=+f16c)
ncnn_add_compiled_fixture(
    pp_formulanet_plus_s_encoder_fp16
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_encoder.param
    ${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_encoder.bin
    1x384x384
    PRECISION=fp16 FP16_ACCUMULATOR=f32 TARGET_FEATURE=+f16c EMIT_IR)
set(numerical_fixture_targets
    compile_convolution_int8_term1
    compile_convolution_int8_term2
    compile_convolution_int8_term101
    compile_convolution_int8_term102
    compile_depthwise_int8_term1
    compile_depthwise_int8_term2
    compile_depthwise_int8_term101
    compile_depthwise_int8_term102
    compile_inner_product_int8_term1
    compile_inner_product_int8_term2
    compile_gemm_int8_term2
    compile_int8_complete_chain
    compile_convolution_scalar
    compile_convolution_no_bias
    compile_convolution_dilated
    compile_convolution_asymmetric
    compile_convolution_asymmetric_stride
    compile_convolution_same_upper
    compile_convolution_same_lower
    compile_convolution_fp16_storage
    compile_convolution_depthwise
    compile_convolution_depthwise_fp16_storage
    compile_convolution_depthwise_7x7
    compile_convolution_depthwise_9x9
    compile_convolution_depthwise_asymmetric
    compile_convolution_depthwise_same_upper
    compile_convolution_depthwise_same_lower
    compile_hard_sigmoid
    compile_hard_swish
    compile_gelu
    compile_reshape
    compile_reshape_copy_dimension
    compile_squeeze
    compile_batch_norm
    compile_expand_dims
    compile_permute
    compile_gemm
    compile_binary_mul_scalar
    compile_binary_mul_channel
    compile_binary_mul_reverse_broadcast
    compile_binary_add
    compile_inner_product
    compile_inner_product_fused_relu
    compile_shuffle_channel
    compile_shuffle_channel_reverse
    compile_slice
    compile_reduction_mean
    compile_reduction_mean_keepdims_coeff
    compile_relu_scalar
    compile_relu_leaky
    compile_pooling_scalar
    compile_pooling_average
    compile_pooling_asymmetric
    compile_pooling_global_max
    compile_pooling_global_average
    compile_pooling_global_max_ignored_params
    compile_pooling_global_average_ignored_params
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
    compile_softmax_negative_width
    compile_padding
    compile_padding_four_sides
    compile_padding_identity
    compile_interp
    compile_interp_asymmetric
    compile_interp_explicit_target
    compile_interp_identity
    compile_interp_eightfold
    compile_deconvolution
    compile_deconvolution_no_bias
    compile_deconvolution_tiny_head
    compile_sigmoid
    compile_relu_dynamic
    compile_sigmoid_dynamic
    compile_padding_dynamic
    compile_interp_dynamic
    compile_hard_sigmoid_dynamic
    compile_hard_swish_dynamic
    compile_gelu_dynamic
    compile_dropout_dynamic
    compile_softmax_dynamic
    compile_batch_norm_dynamic
    compile_shuffle_channel_dynamic
    compile_squeeze_dynamic
    compile_permute_dynamic
    compile_expand_dims_dynamic
    compile_reduction_dynamic
    compile_reduction_dynamic_mean
    compile_gemm_dynamic
    compile_slice_dynamic
    compile_pooling_padded_max_dynamic
    compile_pooling_padded_average_dynamic
    compile_pooling_padded_average_include_dynamic
    compile_pooling_global_max_dynamic
    compile_pooling_global_average_dynamic
    compile_pooling_adaptive_max_dynamic
    compile_pooling_adaptive_average_dynamic
    compile_reshape_spec_dynamic
    compile_slice_ordered_dynamic
    compile_inner_product_dynamic_m
    compile_sigmoid_mixed_fp16
    compile_sigmoid_mixed_bf16
    compile_hard_sigmoid_mixed_fp16
    compile_hard_sigmoid_mixed_bf16
    compile_hard_swish_mixed_fp16
    compile_hard_swish_mixed_bf16
    compile_gelu_mixed_fp16
    compile_gelu_mixed_bf16
    compile_softmax_mixed_fp16
    compile_softmax_mixed_bf16
    compile_batch_norm_mixed_fp16
    compile_batch_norm_mixed_bf16
    compile_detection_output_mixed_fp16
    compile_detection_output_mixed_bf16
    compile_detection_output)
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
add_dependencies(compile_squeezenet_v1_1 compile_sigmoid)
add_dependencies(compile_pp_lcnet_x1_0_doc_ori compile_squeezenet_v1_1)
add_dependencies(compile_pp_lcnet_x1_0_doc_ori_dynamic
                 compile_pp_lcnet_x1_0_doc_ori)
add_dependencies(compile_pp_lcnet_x1_0_textline_ori
                 compile_pp_lcnet_x1_0_doc_ori_dynamic)
add_dependencies(compile_pp_lcnet_x1_0_textline_ori_dynamic
                 compile_pp_lcnet_x1_0_textline_ori)
add_dependencies(compile_chineseocr_lite_anglenet
                 compile_pp_lcnet_x1_0_textline_ori_dynamic)
add_dependencies(compile_chineseocr_lite_anglenet_dynamic
                 compile_chineseocr_lite_anglenet)
add_dependencies(compile_pp_ocrv6_tiny_rec
                 compile_chineseocr_lite_anglenet_dynamic)
add_dependencies(compile_pp_ocrv6_tiny_rec_dynamic
                 compile_pp_ocrv6_tiny_rec)
add_dependencies(compile_pp_ocrv6_tiny_rec_int8
                  compile_pp_ocrv6_tiny_rec_dynamic)
add_dependencies(compile_pp_ocrv6_tiny_rec_int8_dynamic
                 compile_pp_ocrv6_tiny_rec_int8)
add_dependencies(compile_pp_ocrv5_mobile_rec
                 compile_pp_ocrv6_tiny_rec_int8_dynamic)
add_dependencies(compile_pp_ocrv5_mobile_rec_dynamic
                 compile_pp_ocrv5_mobile_rec)
add_dependencies(compile_pp_ocrv6_tiny_det
                  compile_pp_ocrv5_mobile_rec_dynamic)
add_dependencies(compile_pp_ocrv6_tiny_det_dynamic compile_pp_ocrv6_tiny_det)
add_dependencies(compile_pp_ocrv6_small_det
                 compile_pp_ocrv6_tiny_det_dynamic)
add_dependencies(compile_pp_ocrv6_small_det_dynamic
                 compile_pp_ocrv6_small_det)
add_dependencies(compile_pp_ocrv6_medium_det
                 compile_pp_ocrv6_small_det_dynamic)
add_dependencies(compile_pp_ocrv6_medium_det_dynamic
                 compile_pp_ocrv6_medium_det)
add_dependencies(compile_pp_ocrv5_mobile_det_static
                 compile_pp_ocrv6_medium_det_dynamic)
add_dependencies(compile_pp_ocrv5_mobile_det_dynamic
                 compile_pp_ocrv5_mobile_det_static)
add_dependencies(compile_pp_ocrv5_server_det_static
                 compile_pp_ocrv5_mobile_det_dynamic)
add_dependencies(compile_pp_ocrv5_server_det_dynamic
                  compile_pp_ocrv5_server_det_static)
add_dependencies(compile_pp_structrurev2_slanet_plus_cnn
                  compile_pp_ocrv5_server_det_dynamic)
add_dependencies(compile_pp_formulanet_plus_s_encoder
                  compile_pp_structrurev2_slanet_plus_slahead)
add_dependencies(compile_pp_formulanet_plus_s_embed
                 compile_pp_formulanet_plus_s_encoder)
add_dependencies(compile_pp_formulanet_plus_s_decoder
                 compile_pp_formulanet_plus_s_embed)
add_dependencies(compile_pp_structrurev2_slanet_plus_slahead
                 compile_pp_structrurev2_slanet_plus_cnn)
add_dependencies(compile_pp_ocrv5_server_rec
                 compile_pp_formulanet_plus_s_decoder)
add_dependencies(compile_pp_ocrv5_server_rec_dynamic
                 compile_pp_ocrv5_server_rec)
add_dependencies(compile_pp_ocrv6_medium_rec
                 compile_pp_ocrv5_server_rec_dynamic)
add_dependencies(compile_pp_ocrv6_medium_rec_dynamic
                 compile_pp_ocrv6_medium_rec)
add_dependencies(compile_convolution_mixed_fp16
                 compile_pp_ocrv6_medium_rec_dynamic)
add_dependencies(compile_convolution_mixed_bf16
                 compile_convolution_codegen_fp16)
add_dependencies(compile_convolution_codegen_fp16
                 compile_convolution_arithmetic_fp16)
add_dependencies(compile_convolution_arithmetic_fp16
                 compile_convolution_mixed_fp16)
add_dependencies(compile_convolution_depthwise_mixed_fp16
                 compile_convolution_mixed_bf16)
add_dependencies(compile_convolution_depthwise_mixed_bf16
                 compile_convolution_depthwise_arithmetic_fp16)
add_dependencies(compile_convolution_depthwise_arithmetic_fp16
                 compile_convolution_depthwise_mixed_fp16)
add_dependencies(compile_pp_ocrv6_tiny_det_fp16
                 compile_convolution_depthwise_mixed_bf16)
add_dependencies(compile_pp_structrurev2_slanet_plus_cnn_fp16
                 compile_pp_ocrv6_tiny_det_fp16)
add_dependencies(compile_pp_formulanet_plus_s_encoder_fp16
                 compile_pp_structrurev2_slanet_plus_cnn_fp16)

add_library(numerical_test_support STATIC
    support/numerical_test_support.cpp)
target_include_directories(numerical_test_support PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(numerical_test_support PUBLIC ncnn GTest::gtest)

add_executable(numerical_tests
    operators/supported_ops_test.cpp
    operators/int8_ops_test.cpp
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
    CONVOLUTION_INT8_TERM1_BIN_PATH="${convolution_int8_term1_bin}"
    CONVOLUTION_INT8_TERM2_BIN_PATH="${convolution_int8_term2_bin}"
    CONVOLUTION_INT8_TERM101_BIN_PATH="${convolution_int8_term101_bin}"
    CONVOLUTION_INT8_TERM102_BIN_PATH="${convolution_int8_term102_bin}"
    DEPTHWISE_INT8_TERM1_BIN_PATH="${depthwise_int8_term1_bin}"
    DEPTHWISE_INT8_TERM2_BIN_PATH="${depthwise_int8_term2_bin}"
    DEPTHWISE_INT8_TERM101_BIN_PATH="${depthwise_int8_term101_bin}"
    DEPTHWISE_INT8_TERM102_BIN_PATH="${depthwise_int8_term102_bin}"
    INNER_PRODUCT_INT8_TERM1_BIN_PATH="${inner_product_int8_term1_bin}"
    INNER_PRODUCT_INT8_TERM2_BIN_PATH="${inner_product_int8_term2_bin}"
    GEMM_INT8_TERM2_BIN_PATH="${gemm_int8_term2_bin}"
    INT8_COMPLETE_CHAIN_BIN_PATH="${int8_complete_chain_bin}"
    CONVOLUTION_INT8_TERM1_LIBRARY_PATH="${convolution_int8_term1_LIBRARY}"
    CONVOLUTION_INT8_TERM2_LIBRARY_PATH="${convolution_int8_term2_LIBRARY}"
    CONVOLUTION_INT8_TERM101_LIBRARY_PATH="${convolution_int8_term101_LIBRARY}"
    CONVOLUTION_INT8_TERM102_LIBRARY_PATH="${convolution_int8_term102_LIBRARY}"
    DEPTHWISE_INT8_TERM1_LIBRARY_PATH="${depthwise_int8_term1_LIBRARY}"
    DEPTHWISE_INT8_TERM2_LIBRARY_PATH="${depthwise_int8_term2_LIBRARY}"
    DEPTHWISE_INT8_TERM101_LIBRARY_PATH="${depthwise_int8_term101_LIBRARY}"
    DEPTHWISE_INT8_TERM102_LIBRARY_PATH="${depthwise_int8_term102_LIBRARY}"
    INNER_PRODUCT_INT8_TERM1_LIBRARY_PATH="${inner_product_int8_term1_LIBRARY}"
    INNER_PRODUCT_INT8_TERM2_LIBRARY_PATH="${inner_product_int8_term2_LIBRARY}"
    GEMM_INT8_TERM2_LIBRARY_PATH="${gemm_int8_term2_LIBRARY}"
    INT8_COMPLETE_CHAIN_LIBRARY_PATH="${int8_complete_chain_LIBRARY}"
    INT8_COMPLETE_CHAIN_TOSA_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/int8_complete_chain/model.tosa.mlir"
    INT8_COMPLETE_CHAIN_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/int8_complete_chain/int8_complete_chain.json"
    INT8_COMPLETE_CHAIN_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/int8_complete_chain/int8_complete_chain.h"
    CONVOLUTION_BIN_PATH="${convolution_bin}"
    CONVOLUTION_NO_BIAS_BIN_PATH="${convolution_no_bias_bin}"
    CONVOLUTION_DILATED_BIN_PATH="${convolution_dilated_bin}"
    CONVOLUTION_ASYMMETRIC_BIN_PATH="${convolution_asymmetric_bin}"
    CONVOLUTION_ASYMMETRIC_STRIDE_BIN_PATH="${convolution_asymmetric_stride_bin}"
    CONVOLUTION_SAME_UPPER_BIN_PATH="${convolution_same_upper_bin}"
    CONVOLUTION_SAME_LOWER_BIN_PATH="${convolution_same_lower_bin}"
    CONVOLUTION_FP16_STORAGE_BIN_PATH="${convolution_fp16_storage_bin}"
    CONVOLUTION_MIXED_FP16_LIBRARY_PATH="${convolution_mixed_fp16_LIBRARY}"
    CONVOLUTION_ARITHMETIC_FP16_LIBRARY_PATH="${convolution_arithmetic_fp16_LIBRARY}"
    CONVOLUTION_ARITHMETIC_FP16_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/convolution_arithmetic_fp16/model.linalg.mlir"
    CONVOLUTION_ARITHMETIC_FP16_LLVM_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/convolution_arithmetic_fp16/model.ll"
    CONVOLUTION_MIXED_BF16_LIBRARY_PATH="${convolution_mixed_bf16_LIBRARY}"
    CONVOLUTION_DEPTHWISE_BIN_PATH="${convolution_depthwise_bin}"
    CONVOLUTION_DEPTHWISE_FP16_STORAGE_BIN_PATH="${convolution_depthwise_fp16_storage_bin}"
    CONVOLUTION_DEPTHWISE_MIXED_FP16_LIBRARY_PATH="${convolution_depthwise_mixed_fp16_LIBRARY}"
    CONVOLUTION_DEPTHWISE_ARITHMETIC_FP16_LIBRARY_PATH="${convolution_depthwise_arithmetic_fp16_LIBRARY}"
    CONVOLUTION_DEPTHWISE_MIXED_BF16_LIBRARY_PATH="${convolution_depthwise_mixed_bf16_LIBRARY}"
    CONVOLUTION_DEPTHWISE_7X7_BIN_PATH="${convolution_depthwise_7x7_bin}"
    CONVOLUTION_DEPTHWISE_9X9_BIN_PATH="${convolution_depthwise_9x9_bin}"
    CONVOLUTION_DEPTHWISE_ASYMMETRIC_BIN_PATH="${convolution_depthwise_asymmetric_bin}"
    CONVOLUTION_DEPTHWISE_SAME_UPPER_BIN_PATH="${convolution_depthwise_same_upper_bin}"
    CONVOLUTION_DEPTHWISE_SAME_LOWER_BIN_PATH="${convolution_depthwise_same_lower_bin}"
    INNER_PRODUCT_BIN_PATH="${inner_product_bin}"
    INNER_PRODUCT_FUSED_RELU_BIN_PATH="${inner_product_fused_relu_bin}"
    BATCH_NORM_BIN_PATH="${batch_norm_bin}"
    GEMM_BIN_PATH="${gemm_bin}"
    DECONVOLUTION_BIN_PATH="${deconvolution_bin}"
    DECONVOLUTION_NO_BIAS_BIN_PATH="${deconvolution_no_bias_bin}"
    DECONVOLUTION_TINY_HEAD_BIN_PATH="${deconvolution_tiny_head_bin}"
    CONVOLUTION_DEPTHWISE_LIBRARY_PATH="${convolution_depthwise_LIBRARY}"
    CONVOLUTION_DEPTHWISE_7X7_LIBRARY_PATH="${convolution_depthwise_7x7_LIBRARY}"
    CONVOLUTION_DEPTHWISE_9X9_LIBRARY_PATH="${convolution_depthwise_9x9_LIBRARY}"
    CONVOLUTION_DEPTHWISE_ASYMMETRIC_LIBRARY_PATH="${convolution_depthwise_asymmetric_LIBRARY}"
    CONVOLUTION_DEPTHWISE_SAME_UPPER_LIBRARY_PATH="${convolution_depthwise_same_upper_LIBRARY}"
    CONVOLUTION_DEPTHWISE_SAME_LOWER_LIBRARY_PATH="${convolution_depthwise_same_lower_LIBRARY}"
    HARD_SIGMOID_LIBRARY_PATH="${hard_sigmoid_LIBRARY}"
    HARD_SWISH_LIBRARY_PATH="${hard_swish_LIBRARY}"
    GELU_LIBRARY_PATH="${gelu_LIBRARY}"
    SIGMOID_MIXED_FP16_LIBRARY_PATH="${sigmoid_mixed_fp16_LIBRARY}"
    SIGMOID_MIXED_BF16_LIBRARY_PATH="${sigmoid_mixed_bf16_LIBRARY}"
    HARD_SIGMOID_MIXED_FP16_LIBRARY_PATH="${hard_sigmoid_mixed_fp16_LIBRARY}"
    HARD_SIGMOID_MIXED_BF16_LIBRARY_PATH="${hard_sigmoid_mixed_bf16_LIBRARY}"
    HARD_SWISH_MIXED_FP16_LIBRARY_PATH="${hard_swish_mixed_fp16_LIBRARY}"
    HARD_SWISH_MIXED_BF16_LIBRARY_PATH="${hard_swish_mixed_bf16_LIBRARY}"
    GELU_MIXED_FP16_LIBRARY_PATH="${gelu_mixed_fp16_LIBRARY}"
    GELU_MIXED_BF16_LIBRARY_PATH="${gelu_mixed_bf16_LIBRARY}"
    SOFTMAX_MIXED_FP16_LIBRARY_PATH="${softmax_mixed_fp16_LIBRARY}"
    SOFTMAX_MIXED_BF16_LIBRARY_PATH="${softmax_mixed_bf16_LIBRARY}"
    BATCH_NORM_MIXED_FP16_LIBRARY_PATH="${batch_norm_mixed_fp16_LIBRARY}"
    BATCH_NORM_MIXED_BF16_LIBRARY_PATH="${batch_norm_mixed_bf16_LIBRARY}"
    DETECTION_OUTPUT_MIXED_FP16_LIBRARY_PATH="${detection_output_mixed_fp16_LIBRARY}"
    DETECTION_OUTPUT_MIXED_BF16_LIBRARY_PATH="${detection_output_mixed_bf16_LIBRARY}"
    SIGMOID_MIXED_FP16_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/sigmoid_mixed_fp16/sigmoid_mixed_fp16.json"
    SIGMOID_MIXED_BF16_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/sigmoid_mixed_bf16/sigmoid_mixed_bf16.json"
    SIGMOID_MIXED_FP16_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/sigmoid_mixed_fp16/model.linalg.mlir"
    SIGMOID_MIXED_BF16_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/sigmoid_mixed_bf16/model.linalg.mlir"
    RESHAPE_LIBRARY_PATH="${reshape_LIBRARY}"
    RESHAPE_COPY_DIMENSION_LIBRARY_PATH="${reshape_copy_dimension_LIBRARY}"
    SQUEEZE_LIBRARY_PATH="${squeeze_LIBRARY}"
    BATCH_NORM_LIBRARY_PATH="${batch_norm_LIBRARY}"
    EXPAND_DIMS_LIBRARY_PATH="${expand_dims_LIBRARY}"
    PERMUTE_LIBRARY_PATH="${permute_LIBRARY}"
    GEMM_LIBRARY_PATH="${gemm_LIBRARY}"
    BINARY_MUL_SCALAR_LIBRARY_PATH="${binary_mul_scalar_LIBRARY}"
    BINARY_MUL_CHANNEL_LIBRARY_PATH="${binary_mul_channel_LIBRARY}"
    BINARY_MUL_REVERSE_BROADCAST_LIBRARY_PATH="${binary_mul_reverse_broadcast_LIBRARY}"
    BINARY_ADD_LIBRARY_PATH="${binary_add_LIBRARY}"
    INNER_PRODUCT_LIBRARY_PATH="${inner_product_LIBRARY}"
    INNER_PRODUCT_FUSED_RELU_LIBRARY_PATH="${inner_product_fused_relu_LIBRARY}"
    SHUFFLE_CHANNEL_LIBRARY_PATH="${shuffle_channel_LIBRARY}"
    SHUFFLE_CHANNEL_REVERSE_LIBRARY_PATH="${shuffle_channel_reverse_LIBRARY}"
    SLICE_LIBRARY_PATH="${slice_LIBRARY}"
    REDUCTION_MEAN_LIBRARY_PATH="${reduction_mean_LIBRARY}"
    REDUCTION_MEAN_KEEPDIMS_COEFF_LIBRARY_PATH="${reduction_mean_keepdims_coeff_LIBRARY}"
    CONVOLUTION_LIBRARY_PATH="${convolution_scalar_LIBRARY}"
    RELU_LIBRARY_PATH="${relu_scalar_LIBRARY}"
    RELU_LEAKY_LIBRARY_PATH="${relu_leaky_LIBRARY}"
    POOLING_LIBRARY_PATH="${pooling_scalar_LIBRARY}"
    POOLING_AVERAGE_LIBRARY_PATH="${pooling_average_LIBRARY}"
    POOLING_ASYMMETRIC_LIBRARY_PATH="${pooling_asymmetric_LIBRARY}"
    POOLING_GLOBAL_MAX_LIBRARY_PATH="${pooling_global_max_LIBRARY}"
    POOLING_GLOBAL_AVERAGE_LIBRARY_PATH="${pooling_global_average_LIBRARY}"
    POOLING_GLOBAL_MAX_IGNORED_PARAMS_LIBRARY_PATH="${pooling_global_max_ignored_params_LIBRARY}"
    POOLING_GLOBAL_AVERAGE_IGNORED_PARAMS_LIBRARY_PATH="${pooling_global_average_ignored_params_LIBRARY}"
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
    CONVOLUTION_ASYMMETRIC_STRIDE_LIBRARY_PATH="${convolution_asymmetric_stride_LIBRARY}"
    CONVOLUTION_SAME_UPPER_LIBRARY_PATH="${convolution_same_upper_LIBRARY}"
    CONVOLUTION_SAME_LOWER_LIBRARY_PATH="${convolution_same_lower_LIBRARY}"
    CONVOLUTION_FP16_STORAGE_LIBRARY_PATH="${convolution_fp16_storage_LIBRARY}"
    CONVOLUTION_DEPTHWISE_FP16_STORAGE_LIBRARY_PATH="${convolution_depthwise_fp16_storage_LIBRARY}"
    PADDING_LIBRARY_PATH="${padding_LIBRARY}"
    PADDING_FOUR_SIDES_LIBRARY_PATH="${padding_four_sides_LIBRARY}"
    PADDING_IDENTITY_LIBRARY_PATH="${padding_identity_LIBRARY}"
    INTERP_LIBRARY_PATH="${interp_LIBRARY}"
    INTERP_ASYMMETRIC_LIBRARY_PATH="${interp_asymmetric_LIBRARY}"
    INTERP_EXPLICIT_TARGET_LIBRARY_PATH="${interp_explicit_target_LIBRARY}"
    INTERP_IDENTITY_LIBRARY_PATH="${interp_identity_LIBRARY}"
    INTERP_EIGHTFOLD_LIBRARY_PATH="${interp_eightfold_LIBRARY}"
    DECONVOLUTION_LIBRARY_PATH="${deconvolution_LIBRARY}"
    DECONVOLUTION_NO_BIAS_LIBRARY_PATH="${deconvolution_no_bias_LIBRARY}"
    DECONVOLUTION_TINY_HEAD_LIBRARY_PATH="${deconvolution_tiny_head_LIBRARY}"
    SIGMOID_LIBRARY_PATH="${sigmoid_LIBRARY}"
    DETECTION_OUTPUT_LIBRARY_PATH="${detection_output_LIBRARY}"
    SQUEEZENET_PARAM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.param"
    SQUEEZENET_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/examples/squeezenet_v1.1.bin"
    SQUEEZENET_LIBRARY_PATH="${squeezenet_v1_1_LIBRARY}"
    PP_LCNET_DOC_ORI_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.param"
    PP_LCNET_DOC_ORI_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.bin"
    PP_LCNET_DOC_ORI_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_LIBRARY}"
    PP_LCNET_DOC_ORI_INT8_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_int8_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.param"
    PP_LCNET_TEXTLINE_ORI_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.bin"
    PP_LCNET_TEXTLINE_ORI_LIBRARY_PATH="${pp_lcnet_x1_0_textline_ori_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_INT8_LIBRARY_PATH="${pp_lcnet_x1_0_textline_ori_int8_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_INT8_LLVM_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_int8/model.ll"
    PP_LCNET_TEXTLINE_ORI_INT8_ASSEMBLY_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_int8/model.s"
    CHINESEOCR_LITE_ANGLENET_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.param"
    CHINESEOCR_LITE_ANGLENET_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.bin"
    CHINESEOCR_LITE_ANGLENET_LIBRARY_PATH="${chineseocr_lite_anglenet_LIBRARY}"
    PP_OCRV6_TINY_REC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.param"
    PP_OCRV6_TINY_REC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.bin"
    PP_OCRV6_TINY_REC_LIBRARY_PATH="${pp_ocrv6_tiny_rec_LIBRARY}"
    PP_OCRV6_TINY_REC_INT8_LIBRARY_PATH="${pp_ocrv6_tiny_rec_int8_LIBRARY}"
    PP_OCRV5_MOBILE_REC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.param"
    PP_OCRV5_MOBILE_REC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.bin"
    PP_OCRV5_MOBILE_REC_LIBRARY_PATH="${pp_ocrv5_mobile_rec_LIBRARY}"
    PP_OCRV5_MOBILE_REC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_mobile_rec/pp_ocrv5_mobile_rec.json"
    PP_OCRV5_MOBILE_REC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_mobile_rec/model.ncnn.mlir"
    PP_OCRV5_MOBILE_REC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_mobile_rec/model.linalg.mlir"
    PP_OCRV5_SERVER_REC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.param"
    PP_OCRV5_SERVER_REC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.bin"
    PP_OCRV5_SERVER_REC_LIBRARY_PATH="${pp_ocrv5_server_rec_LIBRARY}"
    PP_OCRV5_SERVER_REC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_server_rec/pp_ocrv5_server_rec.json"
    PP_OCRV5_SERVER_REC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_server_rec/model.ncnn.mlir"
    PP_OCRV5_SERVER_REC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_server_rec/model.linalg.mlir"
    PP_OCRV6_MEDIUM_REC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.param"
    PP_OCRV6_MEDIUM_REC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.bin"
    PP_OCRV6_MEDIUM_REC_LIBRARY_PATH="${pp_ocrv6_medium_rec_LIBRARY}"
    PP_OCRV6_MEDIUM_REC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_medium_rec/pp_ocrv6_medium_rec.json"
    PP_OCRV6_MEDIUM_REC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_medium_rec/model.ncnn.mlir"
    PP_OCRV6_MEDIUM_REC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_medium_rec/model.linalg.mlir"
    PP_OCRV6_TINY_DET_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param"
    PP_OCRV6_TINY_DET_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin"
    PP_OCRV6_TINY_DET_LIBRARY_PATH="${pp_ocrv6_tiny_det_LIBRARY}"
    PP_OCRV6_TINY_DET_FP16_LIBRARY_PATH="${pp_ocrv6_tiny_det_fp16_LIBRARY}"
    PP_OCRV6_SMALL_DET_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.param"
    PP_OCRV6_SMALL_DET_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.bin"
    PP_OCRV6_SMALL_DET_LIBRARY_PATH="${pp_ocrv6_small_det_LIBRARY}"
    PP_OCRV6_MEDIUM_DET_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.param"
    PP_OCRV6_MEDIUM_DET_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.bin"
    PP_OCRV6_MEDIUM_DET_LIBRARY_PATH="${pp_ocrv6_medium_det_LIBRARY}"
    PP_OCRV5_MOBILE_DET_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.param"
    PP_OCRV5_MOBILE_DET_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.bin"
    PP_OCRV5_MOBILE_DET_LIBRARY_PATH="${pp_ocrv5_mobile_det_static_LIBRARY}"
    PP_OCRV5_SERVER_DET_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_det.param"
    PP_OCRV5_SERVER_DET_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_det.bin"
    PP_OCRV5_SERVER_DET_STATIC_LIBRARY_PATH="${pp_ocrv5_server_det_static_LIBRARY}"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_cnn.param"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_cnn.bin"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_LIBRARY_PATH="${pp_structrurev2_slanet_plus_cnn_LIBRARY}"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_structrurev2_slanet_plus_cnn/pp_structrurev2_slanet_plus_cnn.json"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_structrurev2_slanet_plus_cnn/pp_structrurev2_slanet_plus_cnn.h"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_structrurev2_slanet_plus_cnn/model.linalg.mlir"
    PP_STRUCTRUREV2_SLANET_PLUS_CNN_FP16_LIBRARY_PATH="${pp_structrurev2_slanet_plus_cnn_fp16_LIBRARY}"
    PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_slahead.param"
    PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-StructrureV2_SLANet_plus_slahead.bin"
    PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_LIBRARY_PATH="${pp_structrurev2_slanet_plus_slahead_LIBRARY}"
    PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_structrurev2_slanet_plus_slahead/pp_structrurev2_slanet_plus_slahead.json"
    PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_structrurev2_slanet_plus_slahead/pp_structrurev2_slanet_plus_slahead.h"
    PP_STRUCTRUREV2_SLANET_PLUS_SLAHEAD_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_structrurev2_slanet_plus_slahead/model.linalg.mlir"
    PP_FORMULANET_PLUS_S_ENCODER_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_encoder.param"
    PP_FORMULANET_PLUS_S_ENCODER_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_encoder.bin"
    PP_FORMULANET_PLUS_S_ENCODER_LIBRARY_PATH="${pp_formulanet_plus_s_encoder_LIBRARY}"
    PP_FORMULANET_PLUS_S_ENCODER_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_encoder/pp_formulanet_plus_s_encoder.json"
    PP_FORMULANET_PLUS_S_ENCODER_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_encoder/pp_formulanet_plus_s_encoder.h"
    PP_FORMULANET_PLUS_S_ENCODER_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_encoder/model.linalg.mlir"
    PP_FORMULANET_PLUS_S_ENCODER_FP16_LIBRARY_PATH="${pp_formulanet_plus_s_encoder_fp16_LIBRARY}"
    PP_FORMULANET_PLUS_S_ENCODER_FP16_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_encoder_fp16/pp_formulanet_plus_s_encoder_fp16.json"
    PP_FORMULANET_PLUS_S_ENCODER_FP16_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_encoder_fp16/model.ncnn.mlir"
    PP_FORMULANET_PLUS_S_ENCODER_FP16_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_encoder_fp16/model.linalg.mlir"
    PP_FORMULANET_PLUS_S_EMBED_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_embed.param"
    PP_FORMULANET_PLUS_S_EMBED_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_embed.bin"
    PP_FORMULANET_PLUS_S_EMBED_LIBRARY_PATH="${pp_formulanet_plus_s_embed_LIBRARY}"
    PP_FORMULANET_PLUS_S_EMBED_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_embed/pp_formulanet_plus_s_embed.json"
    PP_FORMULANET_PLUS_S_EMBED_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_embed/pp_formulanet_plus_s_embed.h"
    PP_FORMULANET_PLUS_S_EMBED_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_embed/model.ncnn.mlir"
    PP_FORMULANET_PLUS_S_EMBED_TOSA_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_embed/model.tosa.mlir"
    PP_FORMULANET_PLUS_S_EMBED_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_embed/model.linalg.mlir"
    PP_FORMULANET_PLUS_S_DECODER_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_decoder.param"
    PP_FORMULANET_PLUS_S_DECODER_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-FormulaNet_plus_S_decoder.bin"
    PP_FORMULANET_PLUS_S_DECODER_LIBRARY_PATH="${pp_formulanet_plus_s_decoder_LIBRARY}"
    PP_FORMULANET_PLUS_S_DECODER_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_decoder/pp_formulanet_plus_s_decoder.json"
    PP_FORMULANET_PLUS_S_DECODER_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_decoder/pp_formulanet_plus_s_decoder.h"
    PP_FORMULANET_PLUS_S_DECODER_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_decoder/model.ncnn.mlir"
    PP_FORMULANET_PLUS_S_DECODER_TOSA_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_decoder/model.tosa.mlir"
    PP_FORMULANET_PLUS_S_DECODER_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_formulanet_plus_s_decoder/model.linalg.mlir")
add_dependencies(numerical_tests compile_squeezenet_v1_1)
add_dependencies(numerical_tests compile_detection_output)
add_dependencies(numerical_tests compile_pp_lcnet_x1_0_doc_ori)
add_dependencies(numerical_tests compile_pp_lcnet_x1_0_doc_ori_int8)
add_dependencies(numerical_tests
    compile_convolution_int8_term1
    compile_convolution_int8_term2
    compile_convolution_int8_term101
    compile_convolution_int8_term102
    compile_depthwise_int8_term1
    compile_depthwise_int8_term2
    compile_depthwise_int8_term101
    compile_depthwise_int8_term102
    compile_inner_product_int8_term1
    compile_inner_product_int8_term2
    compile_int8_complete_chain)
add_dependencies(numerical_tests compile_pp_lcnet_x1_0_textline_ori)
add_dependencies(numerical_tests compile_pp_lcnet_x1_0_textline_ori_int8)
add_dependencies(numerical_tests compile_chineseocr_lite_anglenet)
add_dependencies(numerical_tests compile_pp_ocrv6_tiny_rec)
add_dependencies(numerical_tests compile_pp_ocrv6_tiny_rec_int8)
add_dependencies(numerical_tests compile_pp_ocrv5_mobile_rec)
add_dependencies(numerical_tests compile_pp_ocrv5_server_rec)
add_dependencies(numerical_tests compile_pp_ocrv6_medium_rec)
add_dependencies(numerical_tests compile_pp_ocrv6_tiny_det)
add_dependencies(numerical_tests compile_pp_ocrv6_small_det)
add_dependencies(numerical_tests compile_pp_ocrv6_medium_det)
add_dependencies(numerical_tests compile_pp_ocrv5_mobile_det_static)
add_dependencies(numerical_tests compile_pp_ocrv5_server_det_static)
add_dependencies(numerical_tests compile_pp_structrurev2_slanet_plus_cnn)
add_dependencies(numerical_tests compile_pp_structrurev2_slanet_plus_slahead)
add_dependencies(numerical_tests compile_pp_formulanet_plus_s_encoder)
add_dependencies(numerical_tests compile_pp_formulanet_plus_s_embed)
add_dependencies(numerical_tests compile_pp_formulanet_plus_s_decoder)
add_dependencies(numerical_tests
    compile_convolution_mixed_fp16
    compile_convolution_arithmetic_fp16
    compile_convolution_mixed_bf16
    compile_convolution_depthwise_mixed_fp16
    compile_convolution_depthwise_arithmetic_fp16
    compile_convolution_depthwise_mixed_bf16
    compile_pp_ocrv6_tiny_det_fp16
    compile_pp_structrurev2_slanet_plus_cnn_fp16
    compile_pp_formulanet_plus_s_encoder_fp16)
add_dependencies(numerical_tests
    compile_sigmoid_mixed_fp16
    compile_sigmoid_mixed_bf16
    compile_hard_sigmoid_mixed_fp16
    compile_hard_sigmoid_mixed_bf16
    compile_hard_swish_mixed_fp16
    compile_hard_swish_mixed_bf16
    compile_gelu_mixed_fp16
    compile_gelu_mixed_bf16
    compile_softmax_mixed_fp16
    compile_softmax_mixed_bf16
    compile_batch_norm_mixed_fp16
    compile_batch_norm_mixed_bf16
    compile_detection_output_mixed_fp16
    compile_detection_output_mixed_bf16)

add_executable(numerical_dynamic_operator_tests
    operators/dynamic_ops_test.cpp)
target_link_libraries(numerical_dynamic_operator_tests PRIVATE
    numerical_test_support
    GTest::gtest_main
    ${CMAKE_DL_LIBS})
target_compile_definitions(numerical_dynamic_operator_tests PRIVATE
    GTEST_HAS_CXXABI_H_=0
    NUMERICAL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    NUMERICAL_EMPTY_BIN_PATH="${empty_bin}"
    RELU_DYNAMIC_LIBRARY_PATH="${relu_dynamic_LIBRARY}"
    SIGMOID_DYNAMIC_LIBRARY_PATH="${sigmoid_dynamic_LIBRARY}"
    PADDING_DYNAMIC_LIBRARY_PATH="${padding_dynamic_LIBRARY}"
    INTERP_DYNAMIC_LIBRARY_PATH="${interp_dynamic_LIBRARY}"
    HARD_SIGMOID_DYNAMIC_LIBRARY_PATH="${hard_sigmoid_dynamic_LIBRARY}"
    HARD_SWISH_DYNAMIC_LIBRARY_PATH="${hard_swish_dynamic_LIBRARY}"
    GELU_DYNAMIC_LIBRARY_PATH="${gelu_dynamic_LIBRARY}"
    DROPOUT_DYNAMIC_LIBRARY_PATH="${dropout_dynamic_LIBRARY}"
    SOFTMAX_DYNAMIC_LIBRARY_PATH="${softmax_dynamic_LIBRARY}"
    BATCH_NORM_DYNAMIC_LIBRARY_PATH="${batch_norm_dynamic_LIBRARY}"
    SHUFFLE_CHANNEL_DYNAMIC_LIBRARY_PATH="${shuffle_channel_dynamic_LIBRARY}"
    SQUEEZE_DYNAMIC_LIBRARY_PATH="${squeeze_dynamic_LIBRARY}"
    PERMUTE_DYNAMIC_LIBRARY_PATH="${permute_dynamic_LIBRARY}"
    EXPAND_DIMS_DYNAMIC_LIBRARY_PATH="${expand_dims_dynamic_LIBRARY}"
    REDUCTION_DYNAMIC_LIBRARY_PATH="${reduction_dynamic_LIBRARY}"
    REDUCTION_DYNAMIC_MEAN_LIBRARY_PATH="${reduction_dynamic_mean_LIBRARY}"
    GEMM_DYNAMIC_LIBRARY_PATH="${gemm_dynamic_LIBRARY}"
    SLICE_DYNAMIC_LIBRARY_PATH="${slice_dynamic_LIBRARY}"
    POOLING_PADDED_MAX_DYNAMIC_LIBRARY_PATH="${pooling_padded_max_dynamic_LIBRARY}"
    POOLING_PADDED_AVERAGE_DYNAMIC_LIBRARY_PATH="${pooling_padded_average_dynamic_LIBRARY}"
    POOLING_PADDED_AVERAGE_INCLUDE_DYNAMIC_LIBRARY_PATH="${pooling_padded_average_include_dynamic_LIBRARY}"
    POOLING_GLOBAL_MAX_DYNAMIC_LIBRARY_PATH="${pooling_global_max_dynamic_LIBRARY}"
    POOLING_GLOBAL_AVERAGE_DYNAMIC_LIBRARY_PATH="${pooling_global_average_dynamic_LIBRARY}"
    POOLING_ADAPTIVE_MAX_DYNAMIC_LIBRARY_PATH="${pooling_adaptive_max_dynamic_LIBRARY}"
    POOLING_ADAPTIVE_AVERAGE_DYNAMIC_LIBRARY_PATH="${pooling_adaptive_average_dynamic_LIBRARY}"
    RESHAPE_SPEC_DYNAMIC_LIBRARY_PATH="${reshape_spec_dynamic_LIBRARY}"
    SLICE_ORDERED_DYNAMIC_LIBRARY_PATH="${slice_ordered_dynamic_LIBRARY}"
    INNER_PRODUCT_DYNAMIC_M_LIBRARY_PATH="${inner_product_dynamic_m_LIBRARY}"
    INNER_PRODUCT_DYNAMIC_M_BIN_PATH="${inner_product_dynamic_m_bin}"
    GEMM_BIN_PATH="${gemm_bin}"
    BATCH_NORM_BIN_PATH="${CMAKE_CURRENT_BINARY_DIR}/fixtures/batch_norm.bin")
add_dependencies(numerical_dynamic_operator_tests
    compile_relu_dynamic
    compile_sigmoid_dynamic
    compile_padding_dynamic
    compile_interp_dynamic
    compile_hard_sigmoid_dynamic
    compile_hard_swish_dynamic
    compile_gelu_dynamic
    compile_dropout_dynamic
    compile_softmax_dynamic
    compile_batch_norm_dynamic
    compile_shuffle_channel_dynamic
    compile_squeeze_dynamic
    compile_permute_dynamic
    compile_expand_dims_dynamic
    compile_reduction_dynamic
    compile_reduction_dynamic_mean
    compile_gemm_dynamic
    compile_slice_dynamic
    compile_pooling_padded_max_dynamic
    compile_pooling_padded_average_dynamic
    compile_pooling_padded_average_include_dynamic
    compile_pooling_global_max_dynamic
    compile_pooling_global_average_dynamic
    compile_pooling_adaptive_max_dynamic
    compile_pooling_adaptive_average_dynamic
    compile_reshape_spec_dynamic
    compile_slice_ordered_dynamic
    compile_inner_product_dynamic_m)

add_executable(numerical_dynamic_tests
    models/dynamic_pp_ocr_test.cpp)
target_link_libraries(numerical_dynamic_tests PRIVATE
    numerical_test_support
    GTest::gtest_main
    ${CMAKE_DL_LIBS})
target_compile_definitions(numerical_dynamic_tests PRIVATE
    GTEST_HAS_CXXABI_H_=0
    PP_OCRV6_TINY_DET_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param"
    PP_OCRV6_TINY_DET_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin"
    PP_OCRV6_TINY_DET_DYNAMIC_LIBRARY_PATH="${pp_ocrv6_tiny_det_dynamic_LIBRARY}"
    PP_OCRV6_SMALL_DET_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.param"
    PP_OCRV6_SMALL_DET_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_small_det.bin"
    PP_OCRV6_SMALL_DET_DYNAMIC_LIBRARY_PATH="${pp_ocrv6_small_det_dynamic_LIBRARY}"
    PP_OCRV6_MEDIUM_DET_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.param"
    PP_OCRV6_MEDIUM_DET_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_det.bin"
    PP_OCRV6_MEDIUM_DET_DYNAMIC_LIBRARY_PATH="${pp_ocrv6_medium_det_dynamic_LIBRARY}"
    PP_LCNET_DOC_ORI_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.param"
    PP_LCNET_DOC_ORI_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_doc_ori.bin"
    PP_LCNET_DOC_ORI_DYNAMIC_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_dynamic_LIBRARY}"
    PP_LCNET_DOC_ORI_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_dynamic/pp_lcnet_x1_0_doc_ori_dynamic.json"
    PP_LCNET_DOC_ORI_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_dynamic/pp_lcnet_x1_0_doc_ori_dynamic.h"
    PP_LCNET_DOC_ORI_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_dynamic/model.linalg.mlir"
    PP_LCNET_DOC_ORI_INT8_STATIC_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_int8_LIBRARY}"
    PP_LCNET_DOC_ORI_INT8_DYNAMIC_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_int8_dynamic_LIBRARY}"
    PP_LCNET_DOC_ORI_INT8_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_int8_dynamic/pp_lcnet_x1_0_doc_ori_int8_dynamic.json"
    PP_LCNET_DOC_ORI_INT8_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_int8_dynamic/pp_lcnet_x1_0_doc_ori_int8_dynamic.h"
    PP_LCNET_DOC_ORI_INT8_DYNAMIC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_int8_dynamic/model.ncnn.mlir"
    PP_LCNET_DOC_ORI_INT8_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_doc_ori_int8_dynamic/model.linalg.mlir"
    PP_LCNET_DOC_ORI_INT8_BACKBONE_PARAM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pp_lcnet_doc_ori_int8_backbone.param"
    PP_LCNET_DOC_ORI_INT8_BACKBONE_BIN_PATH="${pp_lcnet_doc_ori_int8_backbone_bin}"
    PP_LCNET_DOC_ORI_INT8_BACKBONE_LIBRARY_PATH="${pp_lcnet_x1_0_doc_ori_int8_backbone_dynamic_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.param"
    PP_LCNET_TEXTLINE_ORI_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-LCNet_x1_0_textline_ori.bin"
    PP_LCNET_TEXTLINE_ORI_DYNAMIC_LIBRARY_PATH="${pp_lcnet_x1_0_textline_ori_dynamic_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_dynamic/pp_lcnet_x1_0_textline_ori_dynamic.json"
    PP_LCNET_TEXTLINE_ORI_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_dynamic/pp_lcnet_x1_0_textline_ori_dynamic.h"
    PP_LCNET_TEXTLINE_ORI_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_dynamic/model.linalg.mlir"
    PP_LCNET_TEXTLINE_ORI_INT8_STATIC_LIBRARY_PATH="${pp_lcnet_x1_0_textline_ori_int8_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_INT8_DYNAMIC_LIBRARY_PATH="${pp_lcnet_x1_0_textline_ori_int8_dynamic_LIBRARY}"
    PP_LCNET_TEXTLINE_ORI_INT8_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_int8_dynamic/pp_lcnet_x1_0_textline_ori_int8_dynamic.json"
    PP_LCNET_TEXTLINE_ORI_INT8_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_int8_dynamic/pp_lcnet_x1_0_textline_ori_int8_dynamic.h"
    PP_LCNET_TEXTLINE_ORI_INT8_DYNAMIC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_int8_dynamic/model.ncnn.mlir"
    PP_LCNET_TEXTLINE_ORI_INT8_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_lcnet_x1_0_textline_ori_int8_dynamic/model.linalg.mlir"
    PP_LCNET_TEXTLINE_ORI_INT8_BACKBONE_PARAM_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/pp_lcnet_textline_ori_int8_backbone.param"
    PP_LCNET_TEXTLINE_ORI_INT8_BACKBONE_BIN_PATH="${pp_lcnet_textline_ori_int8_backbone_bin}"
    PP_LCNET_TEXTLINE_ORI_INT8_BACKBONE_LIBRARY_PATH="${pp_lcnet_x1_0_textline_ori_int8_backbone_dynamic_LIBRARY}"
    CHINESEOCR_LITE_ANGLENET_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.param"
    CHINESEOCR_LITE_ANGLENET_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.bin"
    CHINESEOCR_LITE_ANGLENET_DYNAMIC_LIBRARY_PATH="${chineseocr_lite_anglenet_dynamic_LIBRARY}"
    CHINESEOCR_LITE_ANGLENET_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/chineseocr_lite_anglenet_dynamic/chineseocr_lite_anglenet_dynamic.json"
    CHINESEOCR_LITE_ANGLENET_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/chineseocr_lite_anglenet_dynamic/chineseocr_lite_anglenet_dynamic.h"
    CHINESEOCR_LITE_ANGLENET_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/chineseocr_lite_anglenet_dynamic/model.linalg.mlir"
    PP_OCRV6_TINY_REC_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.param"
    PP_OCRV6_TINY_REC_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.bin"
    PP_OCRV6_TINY_REC_DYNAMIC_LIBRARY_PATH="${pp_ocrv6_tiny_rec_dynamic_LIBRARY}"
    PP_OCRV6_TINY_REC_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_dynamic/pp_ocrv6_tiny_rec_dynamic.json"
    PP_OCRV6_TINY_REC_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_dynamic/pp_ocrv6_tiny_rec_dynamic.h"
    PP_OCRV6_TINY_REC_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_dynamic/model.linalg.mlir"
    PP_OCRV6_TINY_REC_INT8_STATIC_LIBRARY_PATH="${pp_ocrv6_tiny_rec_int8_LIBRARY}"
    PP_OCRV6_TINY_REC_INT8_DYNAMIC_LIBRARY_PATH="${pp_ocrv6_tiny_rec_int8_dynamic_LIBRARY}"
    PP_OCRV6_TINY_REC_INT8_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_int8_dynamic/pp_ocrv6_tiny_rec_int8_dynamic.json"
    PP_OCRV6_TINY_REC_INT8_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_int8_dynamic/pp_ocrv6_tiny_rec_int8_dynamic.h"
    PP_OCRV6_TINY_REC_INT8_DYNAMIC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_int8_dynamic/model.ncnn.mlir"
    PP_OCRV6_TINY_REC_INT8_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_tiny_rec_int8_dynamic/model.linalg.mlir"
    PP_OCRV5_MOBILE_REC_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.param"
    PP_OCRV5_MOBILE_REC_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_rec.bin"
    PP_OCRV5_MOBILE_REC_DYNAMIC_LIBRARY_PATH="${pp_ocrv5_mobile_rec_dynamic_LIBRARY}"
    PP_OCRV5_MOBILE_REC_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_mobile_rec_dynamic/pp_ocrv5_mobile_rec_dynamic.json"
    PP_OCRV5_MOBILE_REC_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_mobile_rec_dynamic/pp_ocrv5_mobile_rec_dynamic.h"
    PP_OCRV5_MOBILE_REC_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_mobile_rec_dynamic/model.linalg.mlir"
    PP_OCRV5_SERVER_REC_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.param"
    PP_OCRV5_SERVER_REC_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_rec.bin"
    PP_OCRV5_SERVER_REC_DYNAMIC_LIBRARY_PATH="${pp_ocrv5_server_rec_dynamic_LIBRARY}"
    PP_OCRV5_SERVER_REC_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_server_rec_dynamic/pp_ocrv5_server_rec_dynamic.json"
    PP_OCRV5_SERVER_REC_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_server_rec_dynamic/pp_ocrv5_server_rec_dynamic.h"
    PP_OCRV5_SERVER_REC_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv5_server_rec_dynamic/model.linalg.mlir"
    PP_OCRV6_MEDIUM_REC_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.param"
    PP_OCRV6_MEDIUM_REC_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv6_medium_rec.bin"
    PP_OCRV6_MEDIUM_REC_DYNAMIC_LIBRARY_PATH="${pp_ocrv6_medium_rec_dynamic_LIBRARY}"
    PP_OCRV6_MEDIUM_REC_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_medium_rec_dynamic/pp_ocrv6_medium_rec_dynamic.json"
    PP_OCRV6_MEDIUM_REC_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_medium_rec_dynamic/pp_ocrv6_medium_rec_dynamic.h"
    PP_OCRV6_MEDIUM_REC_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_ocrv6_medium_rec_dynamic/model.linalg.mlir"
    PP_OCRV5_MOBILE_DET_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.param"
    PP_OCRV5_MOBILE_DET_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.bin"
    PP_OCRV5_MOBILE_DET_DYNAMIC_LIBRARY_PATH="${pp_ocrv5_mobile_det_dynamic_LIBRARY}"
    PP_OCRV5_SERVER_DET_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_det.param"
    PP_OCRV5_SERVER_DET_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-OCRv5_server_det.bin"
    PP_OCRV5_SERVER_DET_DYNAMIC_LIBRARY_PATH="${pp_ocrv5_server_det_dynamic_LIBRARY}"
    PP_UVDOC_DYNAMIC_PARAM_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-UVDoc.param"
    PP_UVDOC_DYNAMIC_BIN_PATH="${CMAKE_SOURCE_DIR}/../ncnn_modelzoo/liteocr/PP-UVDoc.bin"
    PP_UVDOC_DYNAMIC_LIBRARY_PATH="${pp_uvdoc_dynamic_LIBRARY}"
    PP_UVDOC_DYNAMIC_MANIFEST_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_uvdoc_dynamic/pp_uvdoc_dynamic.json"
    PP_UVDOC_DYNAMIC_HEADER_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_uvdoc_dynamic/pp_uvdoc_dynamic.h"
    PP_UVDOC_DYNAMIC_NCNN_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_uvdoc_dynamic/model.ncnn.mlir"
    PP_UVDOC_DYNAMIC_LINALG_IR_PATH="${CMAKE_CURRENT_BINARY_DIR}/generated/pp_uvdoc_dynamic/model.linalg.mlir")
add_dependencies(numerical_dynamic_tests
    compile_pp_ocrv5_mobile_det_dynamic
    compile_pp_ocrv6_tiny_det_dynamic
    compile_pp_ocrv6_small_det_dynamic
    compile_pp_ocrv6_medium_det_dynamic
    compile_pp_lcnet_x1_0_doc_ori_dynamic
    compile_pp_lcnet_x1_0_doc_ori_int8_dynamic
    compile_pp_lcnet_x1_0_doc_ori_int8_backbone_dynamic
    compile_pp_lcnet_x1_0_textline_ori_dynamic
    compile_pp_lcnet_x1_0_textline_ori_int8_dynamic
    compile_pp_lcnet_x1_0_textline_ori_int8_backbone_dynamic
    compile_chineseocr_lite_anglenet_dynamic
    compile_pp_ocrv6_tiny_rec_dynamic
    compile_pp_ocrv6_tiny_rec_int8_dynamic
    compile_pp_ocrv5_mobile_rec_dynamic
    compile_pp_ocrv5_server_rec_dynamic
    compile_pp_ocrv6_medium_rec_dynamic
    compile_pp_ocrv5_server_det_dynamic
    compile_pp_uvdoc_dynamic)

set(numerical_test_timeout 900)
set(numerical_dynamic_test_timeout 3600)
if(CMAKE_CXX_FLAGS MATCHES "fsanitize")
    set(numerical_test_timeout 7200)
    set(numerical_dynamic_test_timeout 14400)
endif()

gtest_discover_tests(numerical_tests
    PROPERTIES
        LABELS numerical-static-baseline
        TIMEOUT ${numerical_test_timeout}
        RUN_SERIAL TRUE)
gtest_discover_tests(numerical_dynamic_tests
    PROPERTIES
        LABELS numerical-dynamic
        TIMEOUT ${numerical_dynamic_test_timeout}
        RUN_SERIAL TRUE)
gtest_discover_tests(numerical_dynamic_operator_tests
    PROPERTIES
        LABELS numerical-dynamic
        TIMEOUT 900
        RUN_SERIAL TRUE)
