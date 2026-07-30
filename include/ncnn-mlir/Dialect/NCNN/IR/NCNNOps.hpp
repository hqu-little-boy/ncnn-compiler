#pragma once

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"

namespace mlir::ncnn {

// 池化算子的语义枚举。算子属性以 i64 存储（0/1/2），用本枚举解释，避免引入
// 方言级自定义属性（及其 parser/printer 机制）。取值与 ncnn 一致。
enum class PoolKind : int64_t { Maximum = 0, Average = 1 };
enum class PoolMode : int64_t { Regular = 0, Global = 1, Adaptive = 2 };

// 公开的形状推断入口：importer 在构建算子前调用以取得结果类型（与算子内部
// inferReturnTypeComponents / verify 共用同一套校验与公式）。失败时已发诊断。
FailureOr<RankedTensorType> inferConvResultType(
  MLIRContext* context,
  std::optional<Location> location,
  RankedTensorType input,
  RankedTensorType weight,
  ValueRange biasAndScales,
  bool hasBias,
  int64_t term,
  int64_t kernelH,
  int64_t kernelW,
  int64_t strideH,
  int64_t strideW,
  int64_t dilationH,
  int64_t dilationW,
  int64_t padTop,
  int64_t padBottom,
  int64_t padLeft,
  int64_t padRight);

FailureOr<RankedTensorType> inferPoolResultType(
  std::optional<Location> location,
  RankedTensorType input,
  int64_t kind,
  int64_t mode,
  int64_t kernelH,
  int64_t kernelW,
  int64_t strideH,
  int64_t strideW,
  int64_t padTop,
  int64_t padBottom,
  int64_t padLeft,
  int64_t padRight,
  int64_t padMode,
  bool includePad);

FailureOr<RankedTensorType> inferConcatResultType(
  std::optional<Location> location, ValueRange inputs, int64_t axis);

}  // namespace mlir::ncnn

#define GET_OP_CLASSES
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.h.inc"
