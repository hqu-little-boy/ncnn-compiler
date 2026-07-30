# ncnn 模型解析器与计算图数据模型（纯 C++，不依赖 MLIR）。
add_library(NCNNGraph STATIC
    graph.cpp
    parser.cpp
)
target_include_directories(NCNNGraph PUBLIC ${CMAKE_SOURCE_DIR}/include)
