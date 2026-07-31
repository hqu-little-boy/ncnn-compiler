add_library(VerifyBufferizedModel STATIC
    VerifyBufferizedModel.cpp
)
target_compile_options(VerifyBufferizedModel PRIVATE -fno-rtti)
target_include_directories(VerifyBufferizedModel PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyBufferizedModel PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass)
