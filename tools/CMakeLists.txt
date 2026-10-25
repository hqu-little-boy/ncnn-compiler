# ncnn-mlir-driver：.param/.bin -> ncnn 方言 MLIR 模块的端到端驱动。
llvm_map_components_to_libnames(NCNN_DRIVER_LLVM_LIBS support)
add_executable(ncnn-mlir-driver ncnn-mlir-driver.cpp)
target_compile_options(ncnn-mlir-driver PRIVATE -fno-rtti)
set(NCNN_EXEC_RPATH_PROPERTIES BUILD_RPATH "${LLVM_LIBRARY_DIR}")
if(COMPILER_INSTALL_RPATH)
    list(APPEND NCNN_EXEC_RPATH_PROPERTIES INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
endif()
set_target_properties(ncnn-mlir-driver PROPERTIES ${NCNN_EXEC_RPATH_PROPERTIES})
target_link_libraries(ncnn-mlir-driver PRIVATE
    NCNNImporter
    NCNNDialect
    NCNNGraph
    NCNNPrecisionSupport
    MLIRFuncDialect
    MLIRArithDialect
    ${NCNN_DRIVER_LLVM_LIBS})
install(TARGETS ncnn-mlir-driver RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(ncnn-compile ncnn-compile.cpp CompileSession.cpp)
target_compile_options(ncnn-compile PRIVATE -fno-rtti)
target_include_directories(ncnn-compile PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(ncnn-compile SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(ncnn-compile PRIVATE
    NCNNPrecisionSupport
    NCNNTargetVectorInfoSupport
    ${NCNN_DRIVER_LLVM_LIBS})
# 运行期按此目录定位 vendored SLEEF 静态档案（构建树内直接可用；安装布局
# 由 ../lib 相对定位兜底）。
target_compile_definitions(ncnn-compile PRIVATE
    NCNN_SLEEF_ARCHIVE_DIR="$<TARGET_FILE_DIR:sleef>"
    NCNN_PROFILE_RUNTIME_SOURCE="${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_runtime.c"
    NCNN_PROFILE_RUNTIME_WRITER_SOURCE="${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_json_writer.c"
    NCNN_PROFILE_RUNTIME_RELATIVE_PATH="../${CMAKE_INSTALL_LIBDIR}/ncnn-mlir/profile_runtime.c"
    NCNN_PROFILE_RUNTIME_WRITER_RELATIVE_PATH="../${CMAKE_INSTALL_LIBDIR}/ncnn-mlir/profile_json_writer.c")
add_dependencies(ncnn-compile sleef)
if(NCNN_ENABLE_VULKAN_PRODUCT)
    find_program(NCNN_MLIR_OPT_EXECUTABLE NAMES mlir-opt-21 REQUIRED)
    find_program(NCNN_SPIRV_VAL_EXECUTABLE NAMES spirv-val REQUIRED)
    find_program(NCNN_SPIRV_DIS_EXECUTABLE NAMES spirv-dis REQUIRED)
    find_program(NCNN_VULKAN_READELF NAMES llvm-readelf-${LLVM_VERSION_MAJOR} readelf REQUIRED)

    set(NCNN_VULKAN_USES_SHARED_LLVM TRUE)
    # VA5: the deployment model link never includes LLVM, shared or static.
    # The historical static-LLVM link-argument/SONAME walk existed only to
    # support the old model-side LLVM dependency and is intentionally gone;
    # the compiler binary itself keeps linking LLVM through its own targets.
    if(NOT LLVM_LINK_LLVM_DYLIB)
        set(NCNN_VULKAN_USES_SHARED_LLVM FALSE)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(NCNN_VULKAN_HOST_CXX_COMPILER_IS_CLANG true)
    else()
        set(NCNN_VULKAN_HOST_CXX_COMPILER_IS_CLANG false)
    endif()
    set(NCNN_VULKAN_HOST_CXX_FLAGS)
    separate_arguments(_ncnn_vulkan_host_cxx_flags NATIVE_COMMAND
        "${CMAKE_CXX_FLAGS}")
    list(APPEND NCNN_VULKAN_HOST_CXX_FLAGS
        ${_ncnn_vulkan_host_cxx_flags})
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _ncnn_vulkan_build_type)
    set(_ncnn_vulkan_host_config_cxx_var
        "CMAKE_CXX_FLAGS_${_ncnn_vulkan_build_type}")
    separate_arguments(_ncnn_vulkan_host_config_cxx_flags NATIVE_COMMAND
        "${${_ncnn_vulkan_host_config_cxx_var}}")
    list(APPEND NCNN_VULKAN_HOST_CXX_FLAGS
        ${_ncnn_vulkan_host_config_cxx_flags})
    if(CMAKE_CXX_COMPILER_ARG1)
        separate_arguments(_ncnn_vulkan_compiler_arg1 NATIVE_COMMAND
            "${CMAKE_CXX_COMPILER_ARG1}")
        list(APPEND NCNN_VULKAN_HOST_CXX_FLAGS
            ${_ncnn_vulkan_compiler_arg1})
    endif()
    if(CMAKE_CXX_COMPILER_TARGET)
        list(APPEND NCNN_VULKAN_HOST_CXX_FLAGS
            "--target=${CMAKE_CXX_COMPILER_TARGET}")
    endif()
    if(CMAKE_SYSROOT)
        list(APPEND NCNN_VULKAN_HOST_CXX_FLAGS
            "--sysroot=${CMAKE_SYSROOT}")
    endif()
    set(NCNN_VULKAN_HOST_CXX_RUNTIME_ARGS)
    foreach(_implicit_lib IN LISTS CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
        if(_implicit_lib STREQUAL "stdc++" OR _implicit_lib STREQUAL "c++")
            list(APPEND NCNN_VULKAN_HOST_CXX_RUNTIME_ARGS "-l${_implicit_lib}")
        endif()
    endforeach()
    if(NOT NCNN_VULKAN_HOST_CXX_RUNTIME_ARGS)
        message(FATAL_ERROR
            "Cannot determine the configured C++ runtime for Vulkan host linking")
    endif()

    set(NCNN_VULKAN_HOST_SANITIZER_ARGS)
    set(NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
    set(NCNN_VULKAN_SANITIZER_RUNTIME_LIBRARIES)
    set(NCNN_VULKAN_SANITIZER_RUNTIME_DIR "")
    set(NCNN_VULKAN_ASAN_RUNTIME_PATH "")
    if(COMPILER_SANITIZERS)
        set(_vulkan_sanitizers "${COMPILER_SANITIZERS}")
        list(REMOVE_DUPLICATES _vulkan_sanitizers)
        list(JOIN _vulkan_sanitizers "," _vulkan_sanitizer_string)
        list(APPEND NCNN_VULKAN_HOST_SANITIZER_ARGS
            "-fsanitize=${_vulkan_sanitizer_string}"
            "-fno-sanitize-recover=all" "-fno-omit-frame-pointer")
        if("address" IN_LIST _vulkan_sanitizers AND
           CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND NCNN_VULKAN_HOST_SANITIZER_ARGS "-shared-libasan")
        endif()

        set(_vulkan_sanitizer_source
            "${CMAKE_CURRENT_BINARY_DIR}/vulkan-sanitizer-link-probe.cpp")
        set(_vulkan_sanitizer_output
            "${CMAKE_CURRENT_BINARY_DIR}/vulkan-sanitizer-link-probe.so")
        file(WRITE "${_vulkan_sanitizer_source}"
            "extern \"C\" __attribute__((noinline)) int asan_probe(int* p, int i) { return p[i]; }\n"
            "extern \"C\" __attribute__((noinline)) int ubsan_probe(int a, int b) { return a + b; }\n")
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" ${NCNN_VULKAN_HOST_SANITIZER_ARGS}
                -shared -fPIC -O0 "${_vulkan_sanitizer_source}"
                -o "${_vulkan_sanitizer_output}"
            RESULT_VARIABLE _sanitizer_probe_status
            OUTPUT_VARIABLE _sanitizer_probe_stdout
            ERROR_VARIABLE _sanitizer_probe_stderr)
        if(NOT _sanitizer_probe_status EQUAL 0)
            message(FATAL_ERROR
                "Cannot link the configured sanitizer runtime into a shared probe: ${_sanitizer_probe_stderr}")
        endif()
        execute_process(
            COMMAND "${NCNN_VULKAN_READELF}" --needed-libs
                "${_vulkan_sanitizer_output}"
            RESULT_VARIABLE _sanitizer_readelf_status
            OUTPUT_VARIABLE _sanitizer_needed_output
            ERROR_VARIABLE _sanitizer_readelf_stderr)
        if(NOT _sanitizer_readelf_status EQUAL 0)
            message(FATAL_ERROR
                "Cannot inspect sanitizer probe dependencies: ${_sanitizer_readelf_stderr}")
        endif()
        string(REGEX MATCHALL "lib[A-Za-z0-9_.+-]+" _sanitizer_needed_lines
            "${_sanitizer_needed_output}")
        foreach(_dependency IN LISTS _sanitizer_needed_lines)
            if(_dependency MATCHES "(asan|ubsan)")
                list(APPEND NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES
                    "${_dependency}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
        if("address" IN_LIST _vulkan_sanitizers AND
           NOT NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
            message(FATAL_ERROR
                "AddressSanitizer is configured but its linked shared runtime is absent from the probe DSO")
        endif()

        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            execute_process(
                COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
                RESULT_VARIABLE _resource_status
                OUTPUT_VARIABLE _resource_dir
                ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT _resource_status EQUAL 0 OR NOT _resource_dir)
                message(FATAL_ERROR
                    "Cannot resolve the Clang sanitizer resource directory")
            endif()
            set(_candidate_runtime_dir "${_resource_dir}/lib/linux")
        else()
            set(_candidate_runtime_dir "")
            foreach(_runtime_name IN ITEMS libasan.so libubsan.so)
                execute_process(
                    COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${_runtime_name}"
                    OUTPUT_VARIABLE _runtime_path
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
                if(EXISTS "${_runtime_path}")
                    get_filename_component(_candidate_runtime_dir "${_runtime_path}" DIRECTORY)
                    break()
                endif()
            endforeach()
        endif()
        if(NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
            if(NOT IS_DIRECTORY "${_candidate_runtime_dir}")
                message(FATAL_ERROR
                    "Cannot resolve the configured sanitizer runtime directory")
            endif()
            set(NCNN_VULKAN_SANITIZER_RUNTIME_DIR "${_candidate_runtime_dir}")
            file(GLOB _runtime_candidates
                "${_candidate_runtime_dir}/libclang_rt.asan*.so"
                "${_candidate_runtime_dir}/libclang_rt.ubsan*.so"
                "${_candidate_runtime_dir}/libasan.so*"
                "${_candidate_runtime_dir}/libubsan.so*")
            foreach(_needed IN LISTS NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES)
                set(_found_runtime "")
                foreach(_candidate IN LISTS _runtime_candidates)
                    execute_process(
                        COMMAND "${NCNN_VULKAN_READELF}" -d "${_candidate}"
                        RESULT_VARIABLE _runtime_readelf_status
                        OUTPUT_VARIABLE _runtime_dynamic
                        ERROR_QUIET)
                    if(_runtime_readelf_status EQUAL 0 AND
                       _runtime_dynamic MATCHES "\\(SONAME\\)[^\\n]*\\[${_needed}\\]")
                        set(_found_runtime "${_candidate}")
                        break()
                    endif()
                    get_filename_component(_candidate_name "${_candidate}" NAME)
                    if(_candidate_name STREQUAL _needed)
                        set(_found_runtime "${_candidate}")
                        break()
                    endif()
                endforeach()
                if(NOT _found_runtime)
                    message(FATAL_ERROR
                        "Configured sanitizer runtime ${_needed} is absent from ${_candidate_runtime_dir}")
                endif()
                list(APPEND NCNN_VULKAN_SANITIZER_RUNTIME_LIBRARIES
                    "${_found_runtime}")
                if(_needed MATCHES "asan")
                    set(NCNN_VULKAN_ASAN_RUNTIME_PATH "${_found_runtime}")
                endif()
            endforeach()
        endif()
    endif()

    function(_ncnn_vulkan_cpp_array list_var count_var text_var)
        set(_items "${${list_var}}")
        set(_quoted_items)
        foreach(_item IN LISTS _items)
            string(REPLACE "\\" "\\\\" _cpp_item "${_item}")
            string(REPLACE "\"" "\\\"" _cpp_item "${_cpp_item}")
            string(REPLACE "\n" "\\n" _cpp_item "${_cpp_item}")
            list(APPEND _quoted_items "\"${_cpp_item}\"")
        endforeach()
        list(LENGTH _quoted_items _count)
        list(JOIN _quoted_items ", " _text)
        set(${count_var} "${_count}" PARENT_SCOPE)
        set(${text_var} "${_text}" PARENT_SCOPE)
    endfunction()
    function(_ncnn_vulkan_cpp_string value_var text_var)
        set(_value "${${value_var}}")
        string(REPLACE "\\" "\\\\" _cpp_value "${_value}")
        string(REPLACE "\"" "\\\"" _cpp_value "${_cpp_value}")
        string(REPLACE "\n" "\\n" _cpp_value "${_cpp_value}")
        set(${text_var} "\"${_cpp_value}\"" PARENT_SCOPE)
    endfunction()
    _ncnn_vulkan_cpp_array(NCNN_VULKAN_HOST_SANITIZER_ARGS
        NCNN_VULKAN_HOST_SANITIZER_ARGS_COUNT
        NCNN_VULKAN_HOST_SANITIZER_ARGS_CPP)
    _ncnn_vulkan_cpp_array(NCNN_VULKAN_HOST_CXX_FLAGS
        NCNN_VULKAN_HOST_CXX_FLAGS_COUNT
        NCNN_VULKAN_HOST_CXX_FLAGS_CPP)
    _ncnn_vulkan_cpp_array(NCNN_VULKAN_HOST_CXX_RUNTIME_ARGS
        NCNN_VULKAN_HOST_CXX_RUNTIME_ARGS_COUNT
        NCNN_VULKAN_HOST_CXX_RUNTIME_ARGS_CPP)
    _ncnn_vulkan_cpp_array(NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES
        NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES_COUNT
        NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES_CPP)
    _ncnn_vulkan_cpp_array(NCNN_VULKAN_SANITIZER_RUNTIME_LIBRARIES
        NCNN_VULKAN_SANITIZER_RUNTIME_LIBRARIES_COUNT
        NCNN_VULKAN_SANITIZER_RUNTIME_LIBRARIES_CPP)
    _ncnn_vulkan_cpp_string(NCNN_VULKAN_SANITIZER_RUNTIME_DIR
        NCNN_VULKAN_SANITIZER_RUNTIME_DIR_CPP)
    _ncnn_vulkan_cpp_string(NCNN_VULKAN_ASAN_RUNTIME_PATH
        NCNN_VULKAN_ASAN_RUNTIME_CPP)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/VulkanHostConfig.hpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/VulkanHostConfig.hpp
        @ONLY)

    # Forward only build-configured values to the Native CTest registration.
    set(NCNN_VULKAN_SANITIZER_RUNTIME_DIR "${NCNN_VULKAN_SANITIZER_RUNTIME_DIR}" PARENT_SCOPE)
    set(NCNN_VULKAN_ASAN_RUNTIME_PATH "${NCNN_VULKAN_ASAN_RUNTIME_PATH}" PARENT_SCOPE)
    set(NCNN_VULKAN_HOST_SANITIZER_ARGS "${NCNN_VULKAN_HOST_SANITIZER_ARGS}" PARENT_SCOPE)
    set(NCNN_VULKAN_USES_SHARED_LLVM "${NCNN_VULKAN_USES_SHARED_LLVM}" PARENT_SCOPE)
    set(NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES
        "${NCNN_VULKAN_SANITIZER_NEEDED_LIBRARIES}" PARENT_SCOPE)

    add_executable(ncnn-vulkan-spirv-extract
        ncnn-vulkan-spirv-extract.cpp)
    target_include_directories(ncnn-vulkan-spirv-extract SYSTEM PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS})
    target_compile_options(ncnn-vulkan-spirv-extract PRIVATE -fno-rtti)
    target_link_libraries(ncnn-vulkan-spirv-extract PRIVATE
        MLIRGPUDialect MLIRSPIRVDialect MLIRParser MLIRIR MLIRSupport
        LLVMSupport)
    set(NCNN_VULKAN_EXTRACTOR_RPATH_PROPERTIES
        BUILD_RPATH "${LLVM_LIBRARY_DIR}")
    if(COMPILER_INSTALL_RPATH)
        list(APPEND NCNN_VULKAN_EXTRACTOR_RPATH_PROPERTIES
            INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
    endif()
    set_target_properties(ncnn-vulkan-spirv-extract PROPERTIES
        ${NCNN_VULKAN_EXTRACTOR_RPATH_PROPERTIES})
    install(TARGETS ncnn-vulkan-spirv-extract
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES
        ${CMAKE_SOURCE_DIR}/include/ncnn-mlir/Vulkan/ModelABI.h
        ${CMAKE_SOURCE_DIR}/include/ncnn-mlir/Vulkan/VulkanRuntime.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ncnn-mlir/Vulkan)
    install(FILES
        ${CMAKE_SOURCE_DIR}/include/ncnn-mlir/Support/VulkanExecutionPlan.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ncnn-mlir/Support)

    add_library(NCNNVulkanProductSupport STATIC
        VulkanProduct.cpp VulkanLifecycleCodegen.cpp)
    target_include_directories(NCNNVulkanProductSupport PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(NCNNVulkanProductSupport PRIVATE -fno-rtti)
    target_link_libraries(NCNNVulkanProductSupport PUBLIC
        NCNNVulkanPlanSupport NCNNVulkanHostProgramSupport)
    target_link_libraries(ncnn-compile PRIVATE NCNNVulkanProductSupport)
    target_include_directories(ncnn-compile PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(ncnn-compile PRIVATE
        NCNN_HAS_VULKAN_PRODUCT=1
        NCNN_VULKAN_MLIR_OPT="${NCNN_MLIR_OPT_EXECUTABLE}"
        NCNN_VULKAN_SPIRV_VAL="${NCNN_SPIRV_VAL_EXECUTABLE}"
        NCNN_VULKAN_SPIRV_DIS="${NCNN_SPIRV_DIS_EXECUTABLE}"
        NCNN_VULKAN_SPIRV_EXTRACTOR="$<TARGET_FILE:ncnn-vulkan-spirv-extract>"
        NCNN_VULKAN_DEVICE_HELPER_ARCHIVE="$<TARGET_FILE:NCNNVulkanDeviceHelper>"
        NCNN_VULKAN_DEVICE_HELPER_RELATIVE_PATH="../${CMAKE_INSTALL_LIBDIR}/ncnn-mlir/$<TARGET_FILE_NAME:NCNNVulkanDeviceHelper>"
        NCNN_VULKAN_HOST_CXX="${CMAKE_CXX_COMPILER}"
        NCNN_VULKAN_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include"
        NCNN_VULKAN_INCLUDE_RELATIVE_PATH="../${CMAKE_INSTALL_INCLUDEDIR}")
    add_dependencies(ncnn-compile
        NCNNVulkanDeviceHelper ncnn-vulkan-spirv-extract)

    if(BUILD_TESTING)
        add_executable(ncnn-vulkan-fixedrun-capture
            ncnn-compile.cpp CompileSession.cpp)
        get_target_property(_ncnn_compile_includes ncnn-compile INCLUDE_DIRECTORIES)
        get_target_property(_ncnn_compile_definitions ncnn-compile COMPILE_DEFINITIONS)
        get_target_property(_ncnn_compile_options ncnn-compile COMPILE_OPTIONS)
        get_target_property(_ncnn_compile_links ncnn-compile LINK_LIBRARIES)
        target_include_directories(ncnn-vulkan-fixedrun-capture PRIVATE
            ${_ncnn_compile_includes})
        target_include_directories(ncnn-vulkan-fixedrun-capture SYSTEM PRIVATE
            ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(ncnn-vulkan-fixedrun-capture PRIVATE
            ${_ncnn_compile_definitions}
            NCNN_VULKAN_FIXEDRUN_CAPTURE_TOOL=1)
        target_compile_options(ncnn-vulkan-fixedrun-capture PRIVATE
            ${_ncnn_compile_options})
        target_link_libraries(ncnn-vulkan-fixedrun-capture PRIVATE
            ${_ncnn_compile_links})
        target_link_options(ncnn-vulkan-fixedrun-capture PRIVATE
            ${NCNN_VULKAN_HOST_SANITIZER_ARGS})
        add_dependencies(ncnn-vulkan-fixedrun-capture
            sleef NCNNVulkanRuntime NCNNVulkanDeviceHelper
            ncnn-vulkan-spirv-extract)
    endif()
endif()
install(TARGETS ncnn-compile RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES $<TARGET_FILE:sleef> DESTINATION ${CMAKE_INSTALL_LIBDIR})
# profile runtime 与其 JSON writer 都以 `.c` 源 install：ncnn-compile 在每个
# 模型编译期把它们编进产物 .so。writer 的头必须一并 install（源与头同目录，
# clang 按包含者所在目录解析 #include）。
install(FILES
    ${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_runtime.c
    ${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_json_writer.c
    ${CMAKE_SOURCE_DIR}/lib/ProfileRuntime/profile_json_writer.h
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/ncnn-mlir)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_ncnn_model.py
    ${CMAKE_CURRENT_BINARY_DIR}/compile_ncnn_model.py
    COPYONLY)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/compile_ncnn_model.py
    DESTINATION ${CMAKE_INSTALL_BINDIR})
