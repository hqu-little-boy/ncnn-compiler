add_library(NormalizeNCNN STATIC
    NormalizeNCNN.cpp
)
target_compile_options(NormalizeNCNN PRIVATE -fno-rtti)
target_link_libraries(NormalizeNCNN PUBLIC
    NCNNDialect
    MLIRFuncDialect
    MLIRPass)
