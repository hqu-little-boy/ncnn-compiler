add_library(InstrumentNCNNProfile
    InstrumentNCNNProfile.cpp
)
add_dependencies(InstrumentNCNNProfile NCNNPassesIncGen)
target_compile_options(InstrumentNCNNProfile PRIVATE -fno-rtti)
target_include_directories(InstrumentNCNNProfile PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})
target_link_libraries(InstrumentNCNNProfile PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIROpenMPDialect)
