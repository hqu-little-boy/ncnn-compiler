// Shared lowering helpers for the NCNN -> TOSA conversion patterns.
//
// These are the building blocks several operator families need: shape
// reshaping, layout conversion (CHW <-> NHWC), constant folding through
// views, low-precision boundaries and dynamic-size plumbing.  Each family
// TU keeps its own single-consumer helpers file-local; only genuinely
// shared entry points live here.
//
// Everything is defined in TosaLoweringUtils.cpp and called unqualified
// from the pattern TUs via per-file using-declarations (never
// `using namespace`, which clang-tidy rejects).
#pragma once

#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir::ncnn::tosa_lowering {

RankedTensorType getNHWCType(RankedTensorType chwType);

RankedTensorType getOHWIType(RankedTensorType oihwType);

bool usesFP16Arithmetic(Operation* operation);

Type getLowPrecisionStorageType(OpBuilder& builder, Operation* operation);

bool usesLowPrecisionBoundary(Operation* operation);

Value applyLowPrecisionBoundary(OpBuilder& builder,
                                Location location,
                                Operation* operation,
                                Value value);

Value createShape(OpBuilder& builder,
                  Location location,
                  ArrayRef<int64_t> dimensions);

Value createSplat(OpBuilder& builder,
                  Location location,
                  RankedTensorType type,
                  double value);

Value createI8Zero(OpBuilder& builder, Location location);

Value createIntegerZero(OpBuilder& builder,
                        Location location,
                        RankedTensorType type);

ElementsAttr getConstantTensorElements(Value value);

Value foldConstantReshape(OpBuilder& builder,
                          Location location,
                          Value input,
                          RankedTensorType outputType);

Value foldConstantTranspose(OpBuilder& builder,
                            Location location,
                            Value input,
                            ArrayRef<int32_t> permutation);

Value transposeOrFoldConstant(OpBuilder& builder,
                              Location location,
                              Value input,
                              RankedTensorType resultType,
                              ArrayRef<int32_t> permutation);

DenseElementsAttr quantizeConstantToI8(ElementsAttr elements,
                                       RankedTensorType sourceType,
                                       ElementsAttr scaleElements,
                                       std::optional<unsigned> scaleDimension);

Value foldConstantQuantizeI8(OpBuilder& builder,
                             Location location,
                             Value input,
                             Value scale,
                             std::optional<unsigned> scaleDimension);

Value quantizeSignedI8(OpBuilder& builder,
                       Location location,
                       Value input,
                       Value scale,
                       std::optional<unsigned> scaleDimension = std::nullopt);

Value convertI32ToF32(OpBuilder& builder, Location location, Value input);

Value dequantizeNcnn(OpBuilder& builder,
                     Location location,
                     Value input,
                     Value scale,
                     Value bias = {});

Value dequantizeAccumulator(OpBuilder& builder,
                            Location location,
                            Value accumulator,
                            Value weightScale,
                            Value inputScale,
                            unsigned channelDimension);

Value convertFloatingTensor(OpBuilder& builder,
                            Location location,
                            Value input,
                            Type targetElement);

Value roundStoragePrecision(OpBuilder& builder,
                            Location location,
                            Value input,
                            Type storageElement);

Value initializeConvolutionOutput(OpBuilder& builder,
                                  Location location,
                                  RankedTensorType outputType,
                                  Value bias);

DenseIntElementsAttr createI64PairAttr(OpBuilder& builder,
                                       ArrayRef<int64_t> values);

Value createIndexConstant(OpBuilder& builder, Location location, int64_t value);

SmallVector<Value> getDynamicSizeValues(OpBuilder& builder,
                                        Location location,
                                        Value source,
                                        RankedTensorType type);

Value getConvolutionOutputExtent(OpBuilder& builder,
                                 Location location,
                                 Value input,
                                 int64_t dimension,
                                 int64_t leadingPadding,
                                 int64_t trailingPadding,
                                 int64_t kernel,
                                 int64_t stride,
                                 int64_t dilation);

Value initializeDynamicConvolutionOutput(OpBuilder& builder,
                                         Location location,
                                         RankedTensorType outputType,
                                         Value input,
                                         ArrayRef<int64_t> padding,
                                         ArrayRef<int64_t> kernel,
                                         ArrayRef<int64_t> stride,
                                         ArrayRef<int64_t> dilation);

RankedTensorType getBroadcastScalarType(RankedTensorType type);

bool isStaticF32Tensor(Type type);

bool isRankedF32Tensor(Type type);

FailureOr<int64_t> getRequiredIntegerAttr(Operation* operation, StringRef name);

int64_t getIntegerAttrOr(Operation* operation,
                         StringRef name,
                         int64_t fallback);

double getFloatAttrOr(Operation* operation, StringRef name, double fallback);

Value reshapeValue(OpBuilder& builder,
                   Location location,
                   Value input,
                   RankedTensorType outputType);

Value reshapeValue(OpBuilder& builder,
                   Location location,
                   Value input,
                   RankedTensorType outputType,
                   ArrayRef<std::optional<unsigned>> sourceDimensions);

Value restoreNCNNLayout(OpBuilder& builder,
                        Location location,
                        Value input,
                        RankedTensorType sourceType);

Value convertNCNNLayout(OpBuilder& builder,
                        Location location,
                        Value input,
                        RankedTensorType sourceType);

Value convertCHWToNHWC(OpBuilder& builder, Location location, Value input);

Value convertNHWCToCHW(OpBuilder& builder,
                       Location location,
                       Value input,
                       RankedTensorType chwType);

uint32_t convertAxis(int64_t axis, int64_t sourceRank);

}  // namespace mlir::ncnn::tosa_lowering
