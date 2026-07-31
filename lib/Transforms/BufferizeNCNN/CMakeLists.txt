add_library(BufferizeNCNN STATIC
    BufferizeNCNN.cpp
)
target_compile_options(BufferizeNCNN PRIVATE -fno-rtti)
target_include_directories(BufferizeNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(BufferizeNCNN PUBLIC
    MLIRBufferizationTransforms
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgUtils
    MLIRMemRefDialect
    MLIRPass)
