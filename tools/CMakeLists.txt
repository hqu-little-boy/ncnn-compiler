# ncnn-mlir-driver：.param/.bin -> ncnn 方言 MLIR 模块的端到端驱动。
llvm_map_components_to_libnames(NCNN_DRIVER_LLVM_LIBS support)
add_executable(ncnn-mlir-driver ncnn-mlir-driver.cpp)
target_compile_options(ncnn-mlir-driver PRIVATE -fno-rtti)
set_target_properties(ncnn-mlir-driver PROPERTIES
    INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
target_link_libraries(ncnn-mlir-driver PRIVATE
    NCNNImporter
    NCNNDialect
    NCNNGraph
    MLIRFuncDialect
    MLIRArithDialect
    ${NCNN_DRIVER_LLVM_LIBS})
install(TARGETS ncnn-mlir-driver RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_ncnn_model.py
    ${CMAKE_CURRENT_BINARY_DIR}/compile_ncnn_model.py
    COPYONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn-compile
    ${CMAKE_CURRENT_BINARY_DIR}/ncnn-compile
    COPYONLY)
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/ncnn-compile
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/ncnn-compile
    DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/compile_ncnn_model.py
    DESTINATION ${CMAKE_INSTALL_BINDIR})
