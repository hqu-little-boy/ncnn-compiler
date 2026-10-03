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

    add_library(NCNNVulkanProductSupport STATIC VulkanProduct.cpp)
    target_include_directories(NCNNVulkanProductSupport PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(NCNNVulkanProductSupport PRIVATE -fno-rtti)
    target_link_libraries(NCNNVulkanProductSupport PUBLIC
        NCNNVulkanPlanSupport)
    target_link_libraries(ncnn-compile PRIVATE NCNNVulkanProductSupport)
    target_compile_definitions(ncnn-compile PRIVATE
        NCNN_HAS_VULKAN_PRODUCT=1
        NCNN_VULKAN_MLIR_OPT="${NCNN_MLIR_OPT_EXECUTABLE}"
        NCNN_VULKAN_SPIRV_VAL="${NCNN_SPIRV_VAL_EXECUTABLE}"
        NCNN_VULKAN_SPIRV_DIS="${NCNN_SPIRV_DIS_EXECUTABLE}"
        NCNN_VULKAN_SPIRV_EXTRACTOR="$<TARGET_FILE:ncnn-vulkan-spirv-extract>"
        NCNN_VULKAN_RUNTIME_ARCHIVE="$<TARGET_FILE:NCNNVulkanRuntime>"
        NCNN_VULKAN_PLAN_ARCHIVE="$<TARGET_FILE:NCNNVulkanPlanSupport>"
        NCNN_VULKAN_RUNTIME_RELATIVE_PATH="../${CMAKE_INSTALL_LIBDIR}/ncnn-mlir/$<TARGET_FILE_NAME:NCNNVulkanRuntime>"
        NCNN_VULKAN_PLAN_RELATIVE_PATH="../${CMAKE_INSTALL_LIBDIR}/ncnn-mlir/$<TARGET_FILE_NAME:NCNNVulkanPlanSupport>"
        NCNN_VULKAN_HOST_CXX="${CMAKE_CXX_COMPILER}"
        NCNN_VULKAN_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include"
        NCNN_VULKAN_INCLUDE_RELATIVE_PATH="../${CMAKE_INSTALL_INCLUDEDIR}"
        NCNN_VULKAN_LLVM_LIBRARY_DIR="${LLVM_LIBRARY_DIR}"
        NCNN_VULKAN_LLVM_LIBRARY_NAME="LLVM-${LLVM_VERSION_MAJOR}")
    add_dependencies(ncnn-compile NCNNVulkanRuntime ncnn-vulkan-spirv-extract)
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
