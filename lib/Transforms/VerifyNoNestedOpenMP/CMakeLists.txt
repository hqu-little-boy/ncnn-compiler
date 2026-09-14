add_library(VerifyNoNestedOpenMP
    VerifyNoNestedOpenMP.cpp
)
add_dependencies(VerifyNoNestedOpenMP NCNNPassesIncGen)
target_include_directories(VerifyNoNestedOpenMP PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(VerifyNoNestedOpenMP SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(VerifyNoNestedOpenMP PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(VerifyNoNestedOpenMP PRIVATE -fno-rtti)
target_link_libraries(VerifyNoNestedOpenMP PUBLIC
    MLIRIR
    MLIRPass
    MLIROpenMPDialect)
