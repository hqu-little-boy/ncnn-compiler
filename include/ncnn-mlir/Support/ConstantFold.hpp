#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace ncnn_mlir {

// Elements attribute of the ConstantLike constant that produces `value`, or
// null when `value` does not trace back to an elements constant.
//
// The terminal hop is `m_Constant`, which accepts any `OpTrait::ConstantLike`
// op whose fold yields an attribute — not only `arith.constant`.  In the
// product pipelines both callers only ever meet `arith.constant`:
// `ConvertNCNNModelToFunc` rewrites `ncnn.const` away before FoldNCNNBatchNorm
// runs, and PackStaticMatmulNCNN runs after `VerifyNoTosaOps`, so `tosa.const`
// cannot remain.  The widening is real for standalone `ncnn-mlir-opt` use and
// is recorded in docs/refactor/p3-2026-09-26/.
//
// Walks through `tensor.cast`, `tensor.collapse_shape` and
// `tensor.expand_shape` only.  Those are pure views: the result's elements
// *are* the source's elements, so handing the source attribute to a caller is
// correct.  Two neighbouring ops are deliberately excluded because they are not
// views and would hand callers the wrong data or a value that does not exist
// yet:
//   * `tensor.extract_slice` — a subset, not the same elements;
//   * `tensor.from_elements` — synthesizes a tensor from scalars.
// The view walk runs before the terminal hop, so a ConstantLike view would
// still be looked through as a view rather than folded.
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
