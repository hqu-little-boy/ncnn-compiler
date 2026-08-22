add_library(NCNNPrecisionSupport STATIC Precision.cpp)
target_include_directories(NCNNPrecisionSupport PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(NCNNConstantFoldSupport STATIC ConstantFold.cpp)
target_include_directories(NCNNConstantFoldSupport PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_include_directories(NCNNConstantFoldSupport SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(NCNNConstantFoldSupport PUBLIC ${LLVM_DEFINITIONS})
target_compile_options(NCNNConstantFoldSupport PRIVATE -fno-rtti)
target_link_libraries(NCNNConstantFoldSupport PUBLIC MLIRIR)
