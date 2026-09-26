// Pattern population entry points for the NCNN -> TOSA conversion.
//
// The conversion is split by operator family so each family lives in its
// own translation unit.  ConvertNCNNToTosaPass::runOnOperation calls these
// in family order; every family matches a disjoint set of operation names,
// so registration order does not select between competing patterns.
#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::ncnn {

// ConvToTosa family.
void populateConvToTosaPatterns(mlir::RewritePatternSet& patterns,
                                const mlir::TypeConverter& typeConverter,
                                mlir::MLIRContext* context);

// DeconvToTosa family.
void populateDeconvToTosaPatterns(mlir::RewritePatternSet& patterns,
                                  const mlir::TypeConverter& typeConverter,
                                  mlir::MLIRContext* context);

// PoolingToTosa family.
void populatePoolingToTosaPatterns(mlir::RewritePatternSet& patterns,
                                   const mlir::TypeConverter& typeConverter,
                                   mlir::MLIRContext* context);

// AttentionToTosa family.
void populateAttentionToTosaPatterns(mlir::RewritePatternSet& patterns,
                                     const mlir::TypeConverter& typeConverter,
                                     mlir::MLIRContext* context);

// SDPAToTosa family.
void populateSDPAToTosaPatterns(mlir::RewritePatternSet& patterns,
                                const mlir::TypeConverter& typeConverter,
                                mlir::MLIRContext* context);

// QuantToTosa family.
void populateQuantToTosaPatterns(mlir::RewritePatternSet& patterns,
                                 const mlir::TypeConverter& typeConverter,
                                 mlir::MLIRContext* context);

// GemmToTosa family.
void populateGemmToTosaPatterns(mlir::RewritePatternSet& patterns,
                                const mlir::TypeConverter& typeConverter,
                                mlir::MLIRContext* context);

// ElementwiseToTosa family.
void populateElementwiseToTosaPatterns(mlir::RewritePatternSet& patterns,
                                       const mlir::TypeConverter& typeConverter,
                                       mlir::MLIRContext* context);

// DetectionToTosa family.
void populateDetectionToTosaPatterns(mlir::RewritePatternSet& patterns,
                                     const mlir::TypeConverter& typeConverter,
                                     mlir::MLIRContext* context);

// LayoutToTosa family.
void populateLayoutToTosaPatterns(mlir::RewritePatternSet& patterns,
                                  const mlir::TypeConverter& typeConverter,
                                  mlir::MLIRContext* context);

// ReduceToTosa family.
void populateReduceToTosaPatterns(mlir::RewritePatternSet& patterns,
                                  const mlir::TypeConverter& typeConverter,
                                  mlir::MLIRContext* context);

}  // namespace mlir::ncnn
