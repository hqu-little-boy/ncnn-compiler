add_library(NCNNPrecisionSupport Precision.cpp)
target_include_directories(NCNNPrecisionSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNPrecisionSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNPrecisionSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNPrecisionSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNPrecisionSupport PUBLIC MLIRIR)

add_library(NCNNConstantFoldSupport ConstantFold.cpp)
target_include_directories(NCNNConstantFoldSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNConstantFoldSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNConstantFoldSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNConstantFoldSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNConstantFoldSupport PUBLIC MLIRIR
    MLIRArithDialect MLIRTensorDialect)

add_library(NCNNTargetVectorInfoSupport TargetVectorInfo.cpp)
target_include_directories(NCNNTargetVectorInfoSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNTargetVectorInfoSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNTargetVectorInfoSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNTargetVectorInfoSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNTargetVectorInfoSupport PUBLIC LLVMSupport)

add_library(NCNNDiagnosticsSupport Diagnostics.cpp)
target_include_directories(NCNNDiagnosticsSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNDiagnosticsSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNDiagnosticsSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNDiagnosticsSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNDiagnosticsSupport PUBLIC MLIRIR MLIRSupport)

# Host planning and SPIR-V reflection are compiler-only. In particular, never
# append these sources or dependencies to the product's plan/runtime archives.
if(NCNN_ENABLE_VULKAN_PRODUCT)
    add_library(NCNNVulkanHostProgramSupport STATIC
        VulkanHostProgram.cpp
        VulkanShaderInterface.cpp)
    target_include_directories(NCNNVulkanHostProgramSupport PUBLIC
        ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
    target_include_directories(NCNNVulkanHostProgramSupport SYSTEM PRIVATE
        ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    target_compile_definitions(NCNNVulkanHostProgramSupport PRIVATE
        ${LLVM_DEFINITIONS})
    target_compile_options(NCNNVulkanHostProgramSupport PRIVATE -fno-rtti)
    target_link_libraries(NCNNVulkanHostProgramSupport PRIVATE
        NCNNVulkanPlanSupport MLIRSPIRVDeserialization MLIRSPIRVDialect
        MLIRIR LLVMTargetParser LLVMSupport)
endif()

# The product host DSO embeds the plan implementation with its runtime.
add_library(NCNNVulkanPlanSupport STATIC
    VulkanExecutionPlan.cpp
    VulkanModelPayload.cpp)
target_include_directories(NCNNVulkanPlanSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNVulkanPlanSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS})
target_compile_definitions(NCNNVulkanPlanSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNVulkanPlanSupport PRIVATE -fno-rtti
    -ffunction-sections)
set_target_properties(NCNNVulkanPlanSupport PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(NCNNVulkanPlanSupport PUBLIC LLVMSupport)
if(NCNN_ENABLE_VULKAN_PRODUCT)
    install(TARGETS NCNNVulkanPlanSupport
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/ncnn-mlir)
endif()
