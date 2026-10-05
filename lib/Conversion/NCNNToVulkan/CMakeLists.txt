add_library(NCNNToVulkan
    NCNNToVulkan.cpp
    NCNNVulkanGraph.cpp
    VulkanGraphKernel.cpp
    VulkanTensorUtils.cpp
)
add_dependencies(NCNNToVulkan NCNNPassesIncGen)
target_compile_options(NCNNToVulkan PRIVATE -fno-rtti)
target_include_directories(NCNNToVulkan PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(NCNNToVulkan PUBLIC
    NCNNVulkanPlanSupport
    LLVMSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSPIRVDialect
    MLIRTensorDialect
    MLIRTosaDialect)
