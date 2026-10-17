find_package(Vulkan REQUIRED)

# Keep the compiler's low-level Vulkan layer free of LLVM/MLIR dependencies.
add_library(NCNNVulkanDeviceHelper STATIC VulkanDeviceHelper.cpp)
target_include_directories(NCNNVulkanDeviceHelper PUBLIC
    ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(NCNNVulkanDeviceHelper PUBLIC Vulkan::Vulkan)
set_target_properties(NCNNVulkanDeviceHelper PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

if(NCNN_ENABLE_VULKAN_PRODUCT)
    install(TARGETS NCNNVulkanDeviceHelper
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/ncnn-mlir)
    install(FILES
        ${PROJECT_SOURCE_DIR}/include/ncnn-mlir/Vulkan/VulkanDeviceTypes.hpp
        ${PROJECT_SOURCE_DIR}/include/ncnn-mlir/Vulkan/VulkanDeviceHelper.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ncnn-mlir/Vulkan)
endif()
