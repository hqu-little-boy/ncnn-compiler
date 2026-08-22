#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace ncnn_mlir {

bool is_foldable_element_type(mlir::Type elementType);

mlir::DenseElementsAttr reshape_dense_elements(
  mlir::ElementsAttr elements, mlir::RankedTensorType resultType);

mlir::DenseElementsAttr transpose_dense_elements(
  mlir::ElementsAttr elements,
  mlir::RankedTensorType sourceType,
  llvm::ArrayRef<int32_t> permutation,
  mlir::RankedTensorType resultType);

}  // namespace ncnn_mlir
