add_library(GenerateCAPI STATIC
    GenerateCAPI.cpp
)
add_dependencies(GenerateCAPI NCNNPassesIncGen)
target_compile_options(GenerateCAPI PRIVATE -fno-rtti)
target_include_directories(GenerateCAPI PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(GenerateCAPI PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRPass
    MLIRSupport)
