add_library(VerifyVulkanSupportedModel
    VerifyVulkanSupportedModel.cpp
)
add_dependencies(VerifyVulkanSupportedModel NCNNPassesIncGen)
target_compile_options(VerifyVulkanSupportedModel PRIVATE -fno-rtti)
target_include_directories(VerifyVulkanSupportedModel PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyVulkanSupportedModel PUBLIC
    NCNNToVulkan
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRTensorDialect)
