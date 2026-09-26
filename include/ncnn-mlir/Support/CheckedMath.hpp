#pragma once

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Support/LogicalResult.h"

namespace ncnn_mlir {

// Overflow-safe shape arithmetic: one home for extent, element-count and
// byte-count proofs across the compiler.
//
// Every operand is a non-negative extent or an offset derived from one.  That
// precondition is load-bearing: MLIR's `ShapedType::kDynamic` is -1, and a
// dynamic extent must fail the proof instead of multiplying or adding through
// to a plausible-looking size.  Call sites that need to tell "dynamic" apart
// from "too large" check for `kDynamic` themselves before calling in.
//
// `checkedAdd`/`checkedMul` return non-negative results.  `checkedSub` is the
// one signed result: padding arithmetic legitimately produces a negative
// offset (e.g. required input minus actual input), and rewriting those sites as
// `checkedAdd(x, -y)` would defeat the non-negative precondition above.
//
// Result type is `mlir::FailureOr` because every caller is MLIR code that
// already reports bounds failures as `failure()`; that keeps one error idiom
// instead of adding another `bool`/`std::optional` shape.

// Fails when either operand is negative or the sum overflows.
inline mlir::FailureOr<int64_t> checkedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0) {
    return mlir::failure();
  }
  int64_t result = 0;
  if (llvm::AddOverflow(lhs, rhs, result)) {
    return mlir::failure();
  }
  return result;
}

// Fails when either operand is negative.  The difference is signed and may be
// negative; for non-negative operands it cannot overflow.
inline mlir::FailureOr<int64_t> checkedSub(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0) {
    return mlir::failure();
  }
  int64_t result = 0;
  if (llvm::SubOverflow(lhs, rhs, result)) {
    return mlir::failure();
  }
  return result;
}

// Fails when either operand is negative or the product overflows.
inline mlir::FailureOr<int64_t> checkedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0) {
    return mlir::failure();
  }
  int64_t result = 0;
  if (llvm::MulOverflow(lhs, rhs, result)) {
    return mlir::failure();
  }
  return result;
}

// Product of `dims`, failing when a dim is negative, the running product
// overflows, or it would exceed `limit`.  `limit` is a caller-supplied budget
// (typically a per-pass element ceiling), not an overflow bound: the owning
// pass decides how large a rewrite may get, so no default is provided here.
inline mlir::FailureOr<int64_t> checkedProduct(llvm::ArrayRef<int64_t> dims,
                                               int64_t limit) {
  int64_t result = 1;
  for (int64_t extent : dims) {
    mlir::FailureOr<int64_t> next = checkedMul(result, extent);
    if (mlir::failed(next) || *next > limit) {
      return mlir::failure();
    }
    result = *next;
  }
  return result;
}

}  // namespace ncnn_mlir
