function(ncnn_prepare_reference_source output_variable)
    set(original "${CMAKE_SOURCE_DIR}/test/third_party/ncnn")
    set(staged "${CMAKE_CURRENT_BINARY_DIR}/ncnn-reference-source")
    set(reference_patches
        "${CMAKE_SOURCE_DIR}/test/Native/patches/ncnn-x86-im2col-null-scratch.patch"
        "${CMAKE_SOURCE_DIR}/test/Native/patches/ncnn-x86-gemm-null-scratch.patch")
    find_program(NCNN_REFERENCE_PATCH_EXECUTABLE NAMES patch REQUIRED)

    # Keep the vendored snapshot immutable. Both Release and sanitizer tests
    # build the same fixed reference, without changing its GEMM arithmetic.
    file(GLOB_RECURSE reference_inputs CONFIGURE_DEPENDS
        LIST_DIRECTORIES false "${original}/*")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${reference_inputs} ${reference_patches})
    file(COPY "${original}/" DESTINATION "${staged}")
    foreach(input IN ITEMS
        "convolution_im2col_gemm.h|2fef1bb95292ae69d5b844c0e180cc8703068ea70f33b74e6eeb841a7f400cdb"
        "convolution_im2col_gemm_bf16s.h|d400edff0787f45281ca073d291ec1f3f0115f6da075dbc603e025d0981fe519"
        "gemm_x86.cpp|24cdedef156e165302355207046ed631d9f0e63359e9c8af4e4777c474d7ef9e")
        string(REPLACE "|" ";" fields "${input}")
        list(GET fields 0 name)
        list(GET fields 1 expected_hash)
        set(relative_path "src/layer/x86/${name}")
        file(SHA256 "${original}/${relative_path}" actual_hash)
        if(NOT actual_hash STREQUAL expected_hash)
            message(FATAL_ERROR
                "ncnn reference snapshot changed: review the null-scratch patch for ${name}")
        endif()
        # Always restore these headers before patching on reconfiguration.
        configure_file("${original}/${relative_path}"
            "${staged}/${relative_path}" COPYONLY)
    endforeach()
    foreach(reference_patch IN LISTS reference_patches)
        execute_process(
            COMMAND "${NCNN_REFERENCE_PATCH_EXECUTABLE}"
                --batch --forward --fuzz=0 -p1 --input "${reference_patch}"
            WORKING_DIRECTORY "${staged}"
            RESULT_VARIABLE patch_status
            OUTPUT_VARIABLE patch_output
            ERROR_VARIABLE patch_error)
        if(NOT patch_status EQUAL 0)
            message(FATAL_ERROR
                "Cannot apply the ncnn reference null-scratch patch ${reference_patch}: ${patch_output}${patch_error}")
        endif()
        file(SHA256 "${reference_patch}" patch_hash)
        message(STATUS "ncnn reference null-scratch patch SHA256: ${patch_hash}")
    endforeach()
    set(${output_variable} "${staged}" PARENT_SCOPE)
endfunction()
