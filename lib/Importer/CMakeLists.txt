# importer：parsed-graph -> ncnn 方言 MLIR 模块。
add_library(NCNNImporter STATIC
    NCNNImporter.cpp
)
target_compile_options(NCNNImporter PRIVATE -fno-rtti)
target_link_libraries(NCNNImporter PUBLIC
    NCNNGraph
    NCNNDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRParser)
