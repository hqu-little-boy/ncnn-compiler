#pragma once

#include <cstdint>

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNAttrs.hpp"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"

namespace mlir::ncnn {

// 池化算子的语义枚举。算子属性以 i64 存储（0/1/2），用本枚举解释，避免引入
// 方言级自定义属性（及其 parser/printer 机制）。取值与 ncnn 一致。
enum class PoolKind : int64_t { Maximum = 0, Average = 1 };
enum class PoolMode : int64_t { Regular = 0, Global = 1, Adaptive = 2 };

}  // namespace mlir::ncnn

#define GET_OP_CLASSES
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.h.inc"
