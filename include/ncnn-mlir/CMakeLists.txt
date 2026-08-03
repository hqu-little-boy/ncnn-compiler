include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NCNN)
add_public_tablegen_target(NCNNPassesIncGen)

add_subdirectory(Dialect)
