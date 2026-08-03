add_library(NormalizeNCNN STATIC
    NormalizeNCNN.cpp
)
add_dependencies(NormalizeNCNN NCNNPassesIncGen)
target_compile_options(NormalizeNCNN PRIVATE -fno-rtti)
target_include_directories(NormalizeNCNN PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include)
target_link_libraries(NormalizeNCNN PUBLIC
    NCNNDialect
    MLIRFuncDialect
    MLIRPass)
