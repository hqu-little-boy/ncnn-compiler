# ncnn-mlir-opt：基于 MlirOptMain 的 mlir-opt 克隆。
# 注册全部上游方言/扩展/pass（故支持所有标准 mlir-opt 选项与 pass）+ ncnn 方言。
# 链接全部 dialect/conversion/extension 库，registerAll* 引用的符号才齐全。
# Keep the executable on the monolithic shared MLIR/LLVM libraries.  Linking
# component archives together with MLIRMlirOptMain pulls in a second copy of
# MLIR's global registries and can crash during dialect/pattern construction.

add_executable(ncnn-mlir-opt ncnn-mlir-opt.cpp)
add_dependencies(ncnn-mlir-opt NCNNPassesIncGen)
# 本目录在 include 路径上，便于以 "RegisterNCNNDialects.h" 直接引用。
target_include_directories(ncnn-mlir-opt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include)
target_compile_options(ncnn-mlir-opt PRIVATE -fno-rtti)
set(NCNN_EXEC_RPATH_PROPERTIES BUILD_RPATH "${LLVM_LIBRARY_DIR}")
if(COMPILER_INSTALL_RPATH)
    list(APPEND NCNN_EXEC_RPATH_PROPERTIES INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
endif()
set_target_properties(ncnn-mlir-opt PROPERTIES ${NCNN_EXEC_RPATH_PROPERTIES})
target_link_libraries(ncnn-mlir-opt PRIVATE
    BufferizeNCNN
    RewriteLinalgCopies
    GenerateCAPI
    NCNNDialect
    NCNNToFunc
    NCNNToTosa
    NCNNPipelines
    NormalizeNCNN
    VerifyBufferizedModel
    VerifyModelShapeContracts
    VerifyNoNCNNOps
    VerifyNoTosaOps
    VerifyNoSCFForall
    MLIRMlirOptMain
    MLIR
    LLVM)
install(TARGETS ncnn-mlir-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
