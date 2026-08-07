add_library(NCNNToTosa STATIC
    NCNNToTosa.cpp
)
add_dependencies(NCNNToTosa NCNNPassesIncGen)
target_compile_options(NCNNToTosa PRIVATE -fno-rtti)
target_include_directories(NCNNToTosa PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(NCNNToTosa PUBLIC
    NCNNDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRTensorDialect
    MLIRTosaDialect
    MLIRTransforms)
