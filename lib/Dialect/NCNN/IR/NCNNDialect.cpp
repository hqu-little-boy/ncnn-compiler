#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNAttrs.hpp"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

// 生成的方言定义使用全限定名（::mlir::ncnn::NCNNDialect::...），在文件作用域包含。
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNAttrs.cpp.inc"

namespace mlir::ncnn {

LogicalResult DimConstraintAttr::verify(
  llvm::function_ref<InFlightDiagnostic()> emitError,
  uint32_t,
  uint32_t,
  int64_t min,
  int64_t multiple_of) {
  if (min <= 0) {
    return emitError() << "minimum extent must be positive";
  }
  if (multiple_of <= 0) {
    return emitError() << "multiple_of must be positive";
  }
  return success();
}

void NCNNDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNAttrs.cpp.inc"
    >();
  addOperations<
#define GET_OP_LIST
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.cpp.inc"
    >();
}

}  // namespace mlir::ncnn
