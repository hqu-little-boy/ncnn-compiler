#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"

#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

// 生成的方言定义使用全限定名（::mlir::ncnn::NCNNDialect::...），在文件作用域包含。
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.cpp.inc"

namespace mlir::ncnn {

void NCNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.cpp.inc"
    >();
}

}  // namespace mlir::ncnn
