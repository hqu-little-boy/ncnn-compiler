find_package(Vulkan REQUIRED)

# The model DSO embeds this runtime archive; it must remain PIC and static even
# when the rest of the project is configured with BUILD_SHARED_LIBS=ON.
add_library(NCNNVulkanRuntime STATIC VulkanRuntime.cpp)
target_include_directories(NCNNVulkanRuntime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNVulkanRuntime SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(NCNNVulkanRuntime PRIVATE ${LLVM_DEFINITIONS})
target_compile_options(NCNNVulkanRuntime PRIVATE
    -fno-rtti -Wno-missing-field-initializers)
set_target_properties(NCNNVulkanRuntime PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(NCNNVulkanRuntime PUBLIC
    NCNNVulkanDeviceHelper NCNNVulkanPlanSupport Vulkan::Vulkan LLVMSupport)
if(NCNN_ENABLE_VULKAN_PRODUCT)
    install(TARGETS NCNNVulkanRuntime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/ncnn-mlir)
endif()
