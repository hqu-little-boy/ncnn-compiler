add_library(VerifyNoSCFForall
    VerifyNoSCFForall.cpp
)
add_dependencies(VerifyNoSCFForall NCNNPassesIncGen)
target_compile_options(VerifyNoSCFForall PRIVATE -fno-rtti)
target_include_directories(VerifyNoSCFForall PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(VerifyNoSCFForall PUBLIC
    MLIRIR
    MLIRPass
    MLIRSCFDialect)
