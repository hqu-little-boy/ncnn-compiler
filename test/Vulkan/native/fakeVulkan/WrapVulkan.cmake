# Test-only ELF interposition; never attach this to a production helper target.
function(ncnn_wrap_vulkan_for_test target)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
       NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Fake Vulkan tests require Linux ELF --wrap support")
    endif()
    set(_functions
        vkCreateInstance
        vkEnumerateInstanceLayerProperties
        vkEnumerateInstanceExtensionProperties
        vkEnumeratePhysicalDevices
        vkGetPhysicalDeviceProperties
        vkGetPhysicalDeviceMemoryProperties
        vkGetPhysicalDeviceQueueFamilyProperties
        vkCreateDevice
        vkDestroyDevice
        vkGetDeviceQueue
        vkDeviceWaitIdle
        vkDestroyInstance
        vkCreateBuffer
        vkGetBufferMemoryRequirements
        vkAllocateMemory
        vkBindBufferMemory
        vkMapMemory
        vkUnmapMemory
        vkFlushMappedMemoryRanges
        vkInvalidateMappedMemoryRanges
        vkDestroyBuffer
        vkFreeMemory
        vkCreateCommandPool
        vkAllocateCommandBuffers
        vkCreateFence
        vkResetFences
        vkResetCommandPool
        vkBeginCommandBuffer
        vkEndCommandBuffer
        vkQueueSubmit
        vkWaitForFences
        vkDestroyFence
        vkDestroyCommandPool
        vkCreateDescriptorSetLayout
        vkCreateDescriptorPool
        vkAllocateDescriptorSets
        vkUpdateDescriptorSets
        vkDestroyDescriptorPool
        vkDestroyDescriptorSetLayout
        vkCreateShaderModule
        vkCreatePipelineLayout
        vkCreateComputePipelines
        vkDestroyPipeline
        vkDestroyPipelineLayout
        vkDestroyShaderModule)
    foreach(_function IN LISTS _functions)
        target_link_options(${target} PRIVATE "-Wl,--wrap=${_function}")
    endforeach()
endfunction()
