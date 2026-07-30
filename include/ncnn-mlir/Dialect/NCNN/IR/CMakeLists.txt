# TableGen 生成 ncnn 方言的 dialect / op 增量代码。
# 产物落在 ${CMAKE_BINARY_DIR}/include/ncnn-mlir/Dialect/NCNN/IR/，与源码目录
# 结构镜像；把 ${CMAKE_BINARY_DIR}/include 加进 include 路径即可用
# "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.h.inc" 形式引用。
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS NCNNOps.td)
mlir_tablegen(NCNNDialect.h.inc -gen-dialect-decls -dialect=ncnn)
mlir_tablegen(NCNNDialect.cpp.inc -gen-dialect-defs -dialect=ncnn)
mlir_tablegen(NCNNOps.h.inc -gen-op-decls)
mlir_tablegen(NCNNOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NCNNOpsIncGen)
