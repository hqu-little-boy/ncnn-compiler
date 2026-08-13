add_library(NCNNPrecisionSupport STATIC Precision.cpp)
target_include_directories(NCNNPrecisionSupport PUBLIC ${CMAKE_SOURCE_DIR}/include)
