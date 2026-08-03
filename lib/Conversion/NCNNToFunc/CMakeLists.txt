add_library(NCNNToFunc STATIC
    NCNNToFunc.cpp
)
add_dependencies(NCNNToFunc NCNNPassesIncGen)
target_compile_options(NCNNToFunc PRIVATE -fno-rtti)
target_include_directories(NCNNToFunc PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(NCNNToFunc PUBLIC
    NCNNDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms)
