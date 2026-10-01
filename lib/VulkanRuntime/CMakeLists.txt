find_package(Vulkan REQUIRED)

add_library(NCNNVulkanRuntime VulkanRuntime.cpp)
target_include_directories(NCNNVulkanRuntime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNVulkanRuntime SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(NCNNVulkanRuntime PRIVATE ${LLVM_DEFINITIONS})
target_compile_options(NCNNVulkanRuntime PRIVATE
    -fno-rtti -Wno-missing-field-initializers)
target_link_libraries(NCNNVulkanRuntime PUBLIC
    NCNNVulkanPlanSupport Vulkan::Vulkan LLVMSupport)
