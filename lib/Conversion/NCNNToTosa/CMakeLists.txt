add_library(NCNNToTosa STATIC
    NCNNToTosa.cpp
)
target_compile_options(NCNNToTosa PRIVATE -fno-rtti)
target_link_libraries(NCNNToTosa PUBLIC
    NCNNDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRTosaDialect
    MLIRTransforms)
