add_library(NCNNPrecisionSupport Precision.cpp)
target_include_directories(NCNNPrecisionSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNPrecisionSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNPrecisionSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNPrecisionSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNPrecisionSupport PUBLIC MLIRIR)

add_library(NCNNConstantFoldSupport ConstantFold.cpp)
target_include_directories(NCNNConstantFoldSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNConstantFoldSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNConstantFoldSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNConstantFoldSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNConstantFoldSupport PUBLIC MLIRIR
    MLIRArithDialect MLIRTensorDialect)

add_library(NCNNTargetVectorInfoSupport TargetVectorInfo.cpp)
target_include_directories(NCNNTargetVectorInfoSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNTargetVectorInfoSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNTargetVectorInfoSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNTargetVectorInfoSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNTargetVectorInfoSupport PUBLIC LLVMSupport)

add_library(NCNNDiagnosticsSupport Diagnostics.cpp)
target_include_directories(NCNNDiagnosticsSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNDiagnosticsSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNDiagnosticsSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNDiagnosticsSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNDiagnosticsSupport PUBLIC MLIRIR MLIRSupport)
