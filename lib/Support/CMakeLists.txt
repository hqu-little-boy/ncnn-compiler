add_library(NCNNPrecisionSupport Precision.cpp)
target_include_directories(NCNNPrecisionSupport PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(NCNNConstantFoldSupport ConstantFold.cpp)
target_include_directories(NCNNConstantFoldSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNConstantFoldSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNConstantFoldSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNConstantFoldSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNConstantFoldSupport PUBLIC MLIRIR)

add_library(NCNNTargetVectorInfoSupport TargetVectorInfo.cpp)
target_include_directories(NCNNTargetVectorInfoSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNTargetVectorInfoSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNTargetVectorInfoSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNTargetVectorInfoSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNTargetVectorInfoSupport PUBLIC LLVMSupport)
