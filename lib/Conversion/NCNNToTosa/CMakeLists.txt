# NCNN -> TOSA conversion.
#
# The conversion is split by operator family so each family is a translation
# unit of its own (see NCNNToTosaPatterns.hpp for the populate entry points).
# NCNNToTosa.cpp keeps only the pass shell; TosaLoweringUtils.cpp holds the
# helpers shared across families.
add_library(NCNNToTosa
    NCNNToTosa.cpp
    TosaLoweringUtils.cpp
    ConvToTosa.cpp
    DeconvToTosa.cpp
    PoolingToTosa.cpp
    AttentionToTosa.cpp
    SDPAToTosa.cpp
    QuantToTosa.cpp
    GemmToTosa.cpp
    ElementwiseToTosa.cpp
    DetectionToTosa.cpp
    LayoutToTosa.cpp
    ReduceToTosa.cpp
)
add_dependencies(NCNNToTosa NCNNPassesIncGen)
target_compile_options(NCNNToTosa PRIVATE -fno-rtti)
target_include_directories(NCNNToTosa PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(NCNNToTosa PUBLIC
    NCNNDialect
    NCNNConstantFoldSupport
    NCNNPrecisionSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTosaDialect
    MLIRTransforms
    MLIRVectorDialect)
