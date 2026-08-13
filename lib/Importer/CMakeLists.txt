# importer：parsed-graph -> ncnn 方言 MLIR 模块。
add_library(NCNNImporter STATIC
    ImportActivation.cpp
    ImportConvolution.cpp
    ImportDetection.cpp
    ImportOps.cpp
    ImportInput.cpp
    ImportPooling.cpp
    ImportTensor.cpp
    NCNNImporter.cpp
)
target_compile_options(NCNNImporter PRIVATE -fno-rtti)
target_link_libraries(NCNNImporter PUBLIC
    NCNNGraph
    NCNNPrecisionSupport
    NCNNDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRParser)
