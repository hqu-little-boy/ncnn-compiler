#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace ncnn_mlir {

// Elements attribute of the `arith.constant` that produces `value`, or null
// when `value` does not trace back to an elements constant.
//
// Walks through `tensor.cast`, `tensor.collapse_shape` and
// `tensor.expand_shape` only.  Those are pure views: the result's elements
// *are* the source's elements, so handing the source attribute to a caller is
// correct.  Two neighbouring ops are deliberately excluded because they are not
// views and would hand callers the wrong data or a value that does not exist
// yet:
//   * `tensor.extract_slice` — a subset, not the same elements;
//   * `tensor.from_elements` — synthesizes a tensor from scalars.
// Every consumer needs the same answer, so this is the one lookup used by
// constant folding and by the packing rewrite.
mlir::ElementsAttr findConstantElements(mlir::Value value);

bool is_foldable_element_type(mlir::Type elementType);

mlir::DenseElementsAttr reshape_dense_elements(
  mlir::ElementsAttr elements, mlir::RankedTensorType resultType);

mlir::DenseElementsAttr transpose_dense_elements(
  mlir::ElementsAttr elements,
  mlir::RankedTensorType sourceType,
  llvm::ArrayRef<int32_t> permutation,
  mlir::RankedTensorType resultType);

}  // namespace ncnn_mlir
