#pragma once

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::ncnn::contract {

inline constexpr llvm::StringLiteral kInt8Target = "ncnn.int8_target";
inline constexpr llvm::StringLiteral kInt8Kernel = "ncnn.int8_kernel";
inline constexpr llvm::StringLiteral kInt8Depthwise = "ncnn.int8_depthwise";
inline constexpr llvm::StringLiteral kInt8CastChain = "ncnn.int8_cast_chain";
inline constexpr llvm::StringLiteral kInt8Isa = "ncnn.int8_isa";
// Explicit reduction-tail element count for INT8 dot-product kernels (K % 32
// for VNNI).  Optional: the plan audit derives it from ncnn.tile_k when the
// producer does not record it.
inline constexpr llvm::StringLiteral kInt8ReductionTail =
  "ncnn.int8_reduction_tail";
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
inline constexpr llvm::StringLiteral kOperationFamily = "ncnn.operation_family";
inline constexpr llvm::StringLiteral kImplementation = "ncnn.implementation";
inline constexpr llvm::StringLiteral kKernelStatic = "ncnn.kernel_static";
inline constexpr llvm::StringLiteral kKernelHeight = "ncnn.kernel_h";
inline constexpr llvm::StringLiteral kKernelWidth = "ncnn.kernel_w";
inline constexpr llvm::StringLiteral kStrideHeight = "ncnn.stride_h";
inline constexpr llvm::StringLiteral kStrideWidth = "ncnn.stride_w";
inline constexpr llvm::StringLiteral kDilationHeight = "ncnn.dilation_h";
inline constexpr llvm::StringLiteral kDilationWidth = "ncnn.dilation_w";
inline constexpr llvm::StringLiteral kInputChannels = "ncnn.input_channels";
inline constexpr llvm::StringLiteral kOutputChannels = "ncnn.output_channels";
inline constexpr llvm::StringLiteral kMultiplier = "ncnn.multiplier";
inline constexpr llvm::StringLiteral kFusion = "ncnn.fusion";
inline constexpr llvm::StringLiteral kFusionKind = "ncnn.fusion_kind";
inline constexpr llvm::StringLiteral kFusionProducer = "ncnn.fusion_producer";
inline constexpr llvm::StringLiteral kFusionResidualInputs =
  "ncnn.fusion_residual_inputs";
inline constexpr llvm::StringLiteral kFusionTileWidth =
  "ncnn.fusion_tile_width";
inline constexpr llvm::StringLiteral kFusionIntermediateBytes =
  "ncnn.fusion_intermediate_bytes";
inline constexpr llvm::StringLiteral kFusionSavedBytes =
  "ncnn.fusion_saved_bytes";
inline constexpr llvm::StringLiteral kFusionEnabled = "ncnn.fusion_enabled";
inline constexpr llvm::StringLiteral kFusionSelectedCount =
  "ncnn.fusion_selected_count";
inline constexpr llvm::StringLiteral kFusionResidualCount =
  "ncnn.fusion_residual_count";
inline constexpr llvm::StringLiteral kFusionRejectedCount =
  "ncnn.fusion_rejected_count";
inline constexpr llvm::StringLiteral kFusionRejectionReasons =
  "ncnn.fusion_rejection_reasons";
inline constexpr llvm::StringLiteral kFusionRecords = "ncnn.fusion_records";
inline constexpr llvm::StringLiteral kAttentionSegments =
  "ncnn.attention_segments";
inline constexpr llvm::StringLiteral kAttentionOrdinal =
  "ncnn.attention_next_ordinal";
inline constexpr llvm::StringLiteral kAttentionRevision =
  "ncnn.attention_segment_revision";
inline constexpr llvm::StringLiteral kAttentionPhase = "ncnn.attention_phase";
inline constexpr llvm::StringLiteral kAttentionSegmentId =
  "ncnn.attention_segment_id";

inline void setString(Operation* operation, StringRef name, StringRef value) {
  operation->setAttr(name, StringAttr::get(operation->getContext(), value));
}

inline void setBool(Operation* operation, StringRef name, bool value) {
  operation->setAttr(name, BoolAttr::get(operation->getContext(), value));
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

inline void annotateOperationFamily(Operation* operation,
                                    StringRef family,
                                    StringRef implementation) {
  setString(operation, kOperationFamily, family);
  setString(operation, kImplementation, implementation);
}

inline void annotateGeometry(Operation* operation,
                             int64_t kernelHeight,
                             int64_t kernelWidth,
                             int64_t strideHeight,
                             int64_t strideWidth,
                             int64_t dilationHeight,
                             int64_t dilationWidth,
                             int64_t inputChannels,
                             int64_t outputChannels,
                             std::optional<int64_t> multiplier = std::nullopt) {
  setInteger(operation, kKernelHeight, kernelHeight);
  setInteger(operation, kKernelWidth, kernelWidth);
  setInteger(operation, kStrideHeight, strideHeight);
  setInteger(operation, kStrideWidth, strideWidth);
  setInteger(operation, kDilationHeight, dilationHeight);
  setInteger(operation, kDilationWidth, dilationWidth);
  setInteger(operation, kInputChannels, inputChannels);
  setInteger(operation, kOutputChannels, outputChannels);
  if (multiplier) {
    setInteger(operation, kMultiplier, *multiplier);
  }
  setBool(operation, kKernelStatic, true);
}

inline void annotateFusionFallback(Operation* operation, StringRef reason) {
  setString(operation, kFusion, "fallback");
  setString(operation, kFallback, reason);
}

inline void annotateFusion(Operation* operation,
                           StringRef kind,
                           StringRef producer,
                           int64_t residualInputs,
                           int64_t tileWidth,
                           int64_t intermediateBytes,
                           int64_t savedBytes) {
  setString(operation, kContract, "selected");
  setString(operation, kFusion, "selected");
  setString(operation, kFusionKind, kind);
  setString(operation, kFusionProducer, producer);
  setInteger(operation, kFusionResidualInputs, residualInputs);
  setInteger(operation, kFusionTileWidth, tileWidth);
  setInteger(operation, kFusionIntermediateBytes, intermediateBytes);
  setInteger(operation, kFusionSavedBytes, savedBytes);
}

inline void appendFusionRecord(Operation* operation,
                               StringRef function,
                               StringRef operationKind,
                               StringRef kind,
                               StringRef producer,
                               int64_t residualInputs,
                               int64_t tileWidth,
                               int64_t intermediateBytes,
                               int64_t savedBytes) {
  MLIRContext* context = operation->getContext();
  SmallVector<Attribute> records;
  if (auto existing = operation->getAttrOfType<ArrayAttr>(kFusionRecords)) {
    for (Attribute record : existing.getValue()) {
      records.push_back(record);
    }
  }
  NamedAttrList record;
  record.set("function", StringAttr::get(context, function));
  record.set("operation", StringAttr::get(context, operationKind));
  record.set("fusion_status", StringAttr::get(context, "selected"));
  record.set("fusion_kind", StringAttr::get(context, kind));
  record.set("fusion_producer", StringAttr::get(context, producer));
  record.set("fusion_residual_inputs",
             IntegerAttr::get(IntegerType::get(context, 64), residualInputs));
  record.set("fusion_tile_width",
             IntegerAttr::get(IntegerType::get(context, 64), tileWidth));
  record.set(
    "fusion_intermediate_bytes",
    IntegerAttr::get(IntegerType::get(context, 64), intermediateBytes));
  record.set("fusion_saved_bytes",
             IntegerAttr::get(IntegerType::get(context, 64), savedBytes));
  records.push_back(DictionaryAttr::get(context, record));
  operation->setAttr(kFusionRecords, ArrayAttr::get(context, records));
}

}  // namespace mlir::ncnn::contract
