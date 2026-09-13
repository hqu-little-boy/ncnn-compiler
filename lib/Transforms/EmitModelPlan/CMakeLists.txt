add_library(EmitModelPlan
    EmitModelPlan.cpp
)
add_dependencies(EmitModelPlan NCNNPassesIncGen)
target_compile_options(EmitModelPlan PRIVATE -fno-rtti)
target_include_directories(EmitModelPlan PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})
target_link_libraries(EmitModelPlan PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIROpenMPDialect
    MLIRPass
    MLIRSCFDialect
    MLIRVectorDialect)
