#pragma once

#include <cctype>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNAttrs.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/ModelLedger.hpp"
#include "ncnn-mlir/Support/ShapeProgram.hpp"

namespace mlir::ncnn::capi_detail {

struct InputDimRelation {
  unsigned lhsInput;
  unsigned lhsDim;
  unsigned rhsInput;
  unsigned rhsDim;
  int64_t offset;
};

// Shared state for FinalizeCAPIPass::runOnOperation (decode) and the
// free functions that emit the wrapper and _infer_output_shapes bodies.
// Member names mirror the original runOnOperation locals.
struct FinalizeContext {
  StringAttr exportName;
  StringAttr internalName;
  LLVM::LLVMFuncOp internal;
  SmallVector<MemRefType> argumentTypes;
  llvm::SmallDenseSet<unsigned> outputs;
  llvm::SmallDenseSet<unsigned> shapeCarriers;
  SmallVector<int32_t> shapeVersions;
  SmallVector<unsigned> inputIndices;
  SmallVector<unsigned> outputArgumentIndices;
  SmallVector<SmallVector<ShapeExpr>> shapePrograms;
  SmallVector<SmallVector<DimConstraintAttr>> constraintsByInput;
  SmallVector<InputDimRelation> inputDimRelations;
  SmallVector<unsigned> wrapperOrder;

  // Created while emitting the wrapper, reused by _infer_output_shapes.
  // OpBuilder has no default constructor; this null-context value is always
  // replaced by `builder = OpBuilder(internal)` before any use.
  OpBuilder builder{static_cast<MLIRContext*>(nullptr)};
  LLVM::LLVMFuncOp wrapper;
  Type pointerType;
  Type i64Type;
  Type overflowResultType;
  using LoadInputDimension = std::function<Value(unsigned, unsigned)>;
  std::function<Value(Value, Value)> mergeFlag;
  std::function<std::pair<Value, Value>(Value, Value)> checkedMultiply;
  std::function<std::pair<Value, Value>(
    const ShapeExpr&, const LoadInputDimension&, bool)>
    evaluateShapeExpression;
};

// Emit the exported wrapper function (original F12-F24 of runOnOperation).
void emitWrapper(ModuleOp module, FinalizeContext& ctx);

// Emit <export>_infer_output_shapes (original F25 of runOnOperation).
void emitInferOutputShapes(FinalizeContext& ctx);

}  // namespace mlir::ncnn::capi_detail
