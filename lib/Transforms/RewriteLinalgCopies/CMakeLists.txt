add_library(RewriteLinalgCopies
    RewriteLinalgCopies.cpp
)
add_dependencies(RewriteLinalgCopies NCNNPassesIncGen)
target_compile_options(RewriteLinalgCopies PRIVATE -fno-rtti)
target_include_directories(RewriteLinalgCopies PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS})
target_link_libraries(RewriteLinalgCopies PUBLIC
    MLIRIR
    MLIRPass
    MLIRLinalgDialect
    MLIRMemRefDialect)
