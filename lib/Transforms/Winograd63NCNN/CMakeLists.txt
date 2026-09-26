add_library(Winograd63NCNN
    Winograd63NCNN.cpp
)
target_include_directories(Winograd63NCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(Winograd63NCNN SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(Winograd63NCNN PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(Winograd63NCNN PRIVATE -fno-rtti)
target_link_libraries(Winograd63NCNN PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRTensorDialect)
