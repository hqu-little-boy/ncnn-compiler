# ncnn-mlir-opt：基于 MlirOptMain 的 mlir-opt 克隆。
# 注册全部上游方言/扩展/pass（故支持所有标准 mlir-opt 选项与 pass）+ ncnn 方言。
# 链接全部 dialect/conversion/extension 库，registerAll* 引用的符号才齐全。
# Release links the monolithic shared MLIR/LLVM libraries. Sanitizer builds use
# the fully instrumented component archives instead; mixing those archives with
# an instrumented monolithic DSO registers duplicate LLVM/MLIR globals under ASan.

add_executable(ncnn-mlir-opt ncnn-mlir-opt.cpp)
add_dependencies(ncnn-mlir-opt NCNNPassesIncGen)
if(COMPILER_SANITIZERS OR CMAKE_CXX_FLAGS MATCHES "fsanitize")
    get_target_property(NCNN_MLIR_OPT_MAIN_LINKS
        MLIRMlirOptMain INTERFACE_LINK_LIBRARIES)
    if("LLVM" IN_LIST NCNN_MLIR_OPT_MAIN_LINKS OR
       "MLIR" IN_LIST NCNN_MLIR_OPT_MAIN_LINKS)
        message(FATAL_ERROR
            "Sanitizer builds require matching instrumented static LLVM/MLIR "
            "component libraries; MLIRMlirOptMain still links monolithic "
            "LLVM/MLIR shared libraries. Set LLVM_DIR and MLIR_DIR to the "
            "matching static sanitizer toolchain.")
    endif()
endif()
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
    FuseQuantChainNCNN
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
    TileMatmulForall
    ForallizeDisjointTileLoops
    VerifyNoSCFForall
    MLIRMlirOptMain)
if(NOT COMPILER_SANITIZERS AND NOT CMAKE_CXX_FLAGS MATCHES "fsanitize")
    target_link_libraries(ncnn-mlir-opt PRIVATE MLIR LLVM)
endif()
install(TARGETS ncnn-mlir-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
