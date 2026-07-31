add_library(NCNNToFunc STATIC
    NCNNToFunc.cpp
)
target_compile_options(NCNNToFunc PRIVATE -fno-rtti)
target_link_libraries(NCNNToFunc PUBLIC
    NCNNDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass)
