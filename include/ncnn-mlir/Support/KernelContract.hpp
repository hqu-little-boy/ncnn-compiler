#pragma once

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::ncnn::contract {

inline constexpr llvm::StringLiteral kLayout = "ncnn.layout";
inline constexpr llvm::StringLiteral kInputLayout = "ncnn.input_layout";
inline constexpr llvm::StringLiteral kWeightLayout = "ncnn.weight_layout";
inline constexpr llvm::StringLiteral kOutputLayout = "ncnn.output_layout";
inline constexpr llvm::StringLiteral kPacking = "ncnn.packing";
inline constexpr llvm::StringLiteral kPackFactor = "ncnn.pack_factor";
inline constexpr llvm::StringLiteral kPackBytes = "ncnn.pack_bytes";
inline constexpr llvm::StringLiteral kUnpackBytes = "ncnn.unpack_bytes";
inline constexpr llvm::StringLiteral kTileM = "ncnn.tile_m";
inline constexpr llvm::StringLiteral kTileN = "ncnn.tile_n";
inline constexpr llvm::StringLiteral kTileK = "ncnn.tile_k";
inline constexpr llvm::StringLiteral kKernel = "ncnn.kernel";
inline constexpr llvm::StringLiteral kParallel = "ncnn.parallel";
inline constexpr llvm::StringLiteral kSimdLanes = "ncnn.simd_lanes";
inline constexpr llvm::StringLiteral kSimdChunk = "ncnn.simd_chunk";
inline constexpr llvm::StringLiteral kFma = "ncnn.fma";
inline constexpr llvm::StringLiteral kTail = "ncnn.tail";
inline constexpr llvm::StringLiteral kAlignment = "ncnn.alignment";
inline constexpr llvm::StringLiteral kAlias = "ncnn.alias";
inline constexpr llvm::StringLiteral kContract = "ncnn.contract";
inline constexpr llvm::StringLiteral kFallback = "ncnn.fallback_reason";

inline void setString(Operation* operation, StringRef name, StringRef value) {
  operation->setAttr(name, StringAttr::get(operation->getContext(), value));
}

inline void setInteger(Operation* operation, StringRef name, int64_t value) {
  operation->setAttr(
    name,
    IntegerAttr::get(IntegerType::get(operation->getContext(), 64), value));
}

inline void annotateTile(Operation* operation,
                         StringRef kernel,
                         StringRef inputLayout,
                         StringRef weightLayout,
                         StringRef outputLayout,
                         int64_t tileM,
                         int64_t tileN,
                         int64_t tileK,
                         StringRef parallel,
                         StringRef tail) {
  setString(operation, kKernel, kernel);
  setString(operation, kInputLayout, inputLayout);
  setString(operation, kWeightLayout, weightLayout);
  setString(operation, kOutputLayout, outputLayout);
  setString(operation, kLayout, outputLayout);
  setInteger(operation, kTileM, tileM);
  setInteger(operation, kTileN, tileN);
  setInteger(operation, kTileK, tileK);
  setString(operation, kParallel, parallel);
  setString(operation, kTail, tail);
  setString(operation, kAlignment, "unknown");
  setString(operation, kAlias, "disjoint_output_proven");
  setString(operation, kContract, "selected");
}

inline void annotatePacking(Operation* operation,
                            StringRef packing,
                            int64_t packFactor,
                            int64_t packBytes,
                            int64_t unpackBytes) {
  setString(operation, kPacking, packing);
  setInteger(operation, kPackFactor, packFactor);
  setInteger(operation, kPackBytes, packBytes);
  setInteger(operation, kUnpackBytes, unpackBytes);
}

inline void annotateFallback(Operation* operation, StringRef reason) {
  setString(operation, kContract, "fallback");
  setString(operation, kFallback, reason);
}

}  // namespace mlir::ncnn::contract
