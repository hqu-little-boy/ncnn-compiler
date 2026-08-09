#include "ncnn-mlir/Transforms/VerifyModelShapeContracts/VerifyModelShapeContracts.hpp"

#include <array>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ncnn-mlir/Support/ShapeMode.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYMODELSHAPECONTRACTSPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

bool hasDynamicTensorOrMemRef(Type type) {
  if (!isa<TensorType, BaseMemRefType>(type)) {
    return false;
  }
  auto shaped = cast<ShapedType>(type);
  return !shaped.hasRank() || !shaped.hasStaticShape();
}

bool hasDynamicExtent(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasRank() && !shaped.hasStaticShape();
}

LogicalResult verifyStaticFunction(func::FuncOp function) {
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes())) {
    if (hasDynamicTensorOrMemRef(type)) {
      return function.emitOpError()
             << "static shape route has dynamic argument " << index;
    }
  }
  for (Type type : function.getResultTypes()) {
    if (hasDynamicTensorOrMemRef(type)) {
      return function.emitOpError(
        "static shape route has a dynamic function result");
    }
  }

  WalkResult result = function.walk([&](Operation* operation) {
    if (llvm::any_of(operation->getOperandTypes(), hasDynamicTensorOrMemRef) ||
        llvm::any_of(operation->getResultTypes(), hasDynamicTensorOrMemRef)) {
      operation->emitOpError(
        "static shape route contains a dynamic tensor or memref type");
      return WalkResult::interrupt();
    }
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        if (llvm::any_of(block.getArgumentTypes(), hasDynamicTensorOrMemRef)) {
          operation->emitOpError(
            "static shape route contains a dynamic region argument");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

LogicalResult verifyShapeProgram(func::FuncOp function,
                                 unsigned outputIndex,
                                 MemRefType outputType,
                                 ArrayRef<unsigned> inputIndices) {
  auto source = function.getArgAttrOfType<IntegerAttr>(
    outputIndex, "ncnn.shape_source_input");
  if (!source || source.getInt() < 0 ||
      static_cast<std::size_t>(source.getInt()) >= inputIndices.size()) {
    return function.emitOpError() << "dynamic output " << outputIndex
                                  << " has no valid input shape source";
  }
  auto inputType = dyn_cast<MemRefType>(
    function.getArgumentTypes()[inputIndices[source.getInt()]]);
  if (!inputType || inputType.hasStaticShape() ||
      inputType.getRank() != outputType.getRank()) {
    return function.emitOpError()
           << "dynamic output " << outputIndex
           << " shape source must be a dynamic input of the same rank";
  }

  auto program =
    function.getArgAttrOfType<ArrayAttr>(outputIndex, "ncnn.shape_program");
  if (!program ||
      program.size() != static_cast<std::size_t>(outputType.getRank())) {
    return function.emitOpError() << "dynamic output " << outputIndex
                                  << " has no complete shape program";
  }
  for (Attribute dimension : program) {
    auto instructions = dyn_cast<DenseI64ArrayAttr>(dimension);
    if (!instructions || instructions.size() % 2 != 0) {
      return function.emitOpError() << "dynamic output " << outputIndex
                                    << " has an invalid shape program";
    }
    ArrayRef<int64_t> values = instructions.asArrayRef();
    for (unsigned index = 0; index < values.size(); index += 2) {
      if (values[index] < 0 || values[index] > 2 ||
          (values[index] == 2 && values[index + 1] <= 0)) {
        return function.emitOpError() << "dynamic output " << outputIndex
                                      << " has an invalid shape program";
      }
    }
  }
  return success();
}

LogicalResult verifyDataDependentOutput(func::FuncOp function,
                                        unsigned outputIndex,
                                        MemRefType outputType,
                                        uint64_t mask) {
  if (outputType.getRank() > 32) {
    return function.emitOpError()
           << "output " << outputIndex
           << " has an invalid data-dependent shape contract";
  }
  const uint64_t validMask = (UINT64_C(1) << outputType.getRank()) - 1;
  if (!outputType.hasStaticShape() || (mask & ~validMask) != 0 ||
      function.getArgAttr(outputIndex, "ncnn.shape_source_input") ||
      function.getArgAttr(outputIndex, "ncnn.shape_program")) {
    return function.emitOpError()
           << "output " << outputIndex
           << " has an invalid data-dependent shape contract";
  }
  if (outputIndex + 1 >= function.getNumArguments()) {
    return function.emitOpError() << "data-dependent output " << outputIndex
                                  << " has no shape carrier";
  }
  auto carrierType =
    dyn_cast<MemRefType>(function.getArgumentTypes()[outputIndex + 1]);
  if (!function.getArgAttr(outputIndex + 1, "bufferize.result") ||
      !function.getArgAttr(outputIndex + 1, "ncnn.shape_carrier") ||
      !carrierType || !carrierType.hasStaticShape() ||
      !carrierType.getElementType().isInteger(64) ||
      carrierType.getRank() != 1 ||
      carrierType.getShape()[0] != outputType.getRank()) {
    return function.emitOpError() << "data-dependent output " << outputIndex
                                  << " has an invalid shape carrier";
  }
  return success();
}

LogicalResult verifyDynamicRankFunction(func::FuncOp function,
                                        ArrayRef<unsigned> inputIndices,
                                        ArrayRef<unsigned> outputIndices) {
  auto rank = function->getAttrOfType<IntegerAttr>("ncnn.rank_variant");
  if (!rank || rank.getInt() < 1 || rank.getInt() > 4) {
    return function.emitOpError(
      "dynamic rank specialization requires rank_variant in [1, 4]");
  }
  if (inputIndices.size() != 1 || outputIndices.size() != 1) {
    return function.emitOpError(
      "dynamic rank specialization requires one input and one output");
  }
  auto input =
    dyn_cast<MemRefType>(function.getArgumentTypes()[inputIndices.front()]);
  auto output =
    dyn_cast<MemRefType>(function.getArgumentTypes()[outputIndices.front()]);
  if (!input || !output || input != output || input.hasStaticShape() ||
      input.getRank() != rank.getInt() ||
      !llvm::all_of(input.getShape(), ShapedType::isDynamic)) {
    return function.emitOpError(
      "dynamic rank specialization must preserve one fully dynamic ranked "
      "memref");
  }
  auto program = function.getArgAttrOfType<ArrayAttr>(outputIndices.front(),
                                                      "ncnn.shape_program");
  if (failed(verifyShapeProgram(
        function, outputIndices.front(), output, inputIndices))) {
    return failure();
  }
  if (!program || llvm::any_of(program, [](Attribute dimension) {
        auto instructions = dyn_cast<DenseI64ArrayAttr>(dimension);
        return !instructions || !instructions.empty();
      })) {
    return function.emitOpError(
      "dynamic rank output must preserve every input dimension");
  }
  return success();
}

LogicalResult verifyFunction(func::FuncOp function) {
  const bool dynamicRank = function->hasAttr("ncnn.dynamic_rank");
  const bool hasRankVariant = function->hasAttr("ncnn.rank_variant");
  if (dynamicRank != hasRankVariant) {
    return function.emitOpError(
      "dynamic_rank and rank_variant attributes must appear together");
  }

  SmallVector<unsigned> inputIndices;
  SmallVector<unsigned> outputIndices;
  bool hasDynamicInput = false;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (function.getArgAttr(index, "bufferize.result")) {
      if (!function.getArgAttr(index, "ncnn.shape_carrier")) {
        outputIndices.push_back(index);
      }
      continue;
    }
    inputIndices.push_back(index);
    hasDynamicInput |= hasDynamicExtent(function.getArgumentTypes()[index]);
  }
  ShapeMode mode = classifyShapeMode(dynamicRank, hasDynamicInput);
  if (mode == ShapeMode::Static && failed(verifyStaticFunction(function))) {
    return failure();
  }

  for (unsigned outputIndex : outputIndices) {
    auto outputType =
      dyn_cast<MemRefType>(function.getArgumentTypes()[outputIndex]);
    if (!outputType) {
      return function.emitOpError()
             << "output " << outputIndex << " must be a ranked memref";
    }
    auto maskAttr = function.getArgAttrOfType<IntegerAttr>(
      outputIndex, "ncnn.data_dependent_dim_mask");
    const uint64_t mask = maskAttr ? maskAttr.getUInt() : 0;
    if (mask != 0) {
      if (failed(verifyDataDependentOutput(
            function, outputIndex, outputType, mask))) {
        return failure();
      }
    } else if (!outputType.hasStaticShape() &&
               failed(verifyShapeProgram(
                 function, outputIndex, outputType, inputIndices))) {
      return failure();
    }
  }

  if (mode == ShapeMode::DynamicRankSpecialization) {
    return verifyDynamicRankFunction(function, inputIndices, outputIndices);
  }
  return success();
}

class VerifyModelShapeContractsPass final
  : public impl::VerifyModelShapeContractsPassBase<
      VerifyModelShapeContractsPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    SmallVector<func::FuncOp> dynamicRankFunctions;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (!function->hasAttr("ncnn.entry_point")) {
        continue;
      }
      if (failed(verifyFunction(function))) {
        signalPassFailure();
        return;
      }
      if (function->hasAttr("ncnn.dynamic_rank")) {
        dynamicRankFunctions.push_back(function);
      }
    }

    if (!dynamicRankFunctions.empty()) {
      std::array<bool, 4> ranks{};
      for (func::FuncOp function : dynamicRankFunctions) {
        int64_t rank =
          function->getAttrOfType<IntegerAttr>("ncnn.rank_variant").getInt();
        ranks[rank - 1] = true;
      }
      if (dynamicRankFunctions.size() != 4 ||
          !llvm::all_of(ranks, std::identity{})) {
        getOperation().emitError(
          "dynamic rank route requires unique specializations for ranks 1 "
          "through 4");
        signalPassFailure();
      }
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
