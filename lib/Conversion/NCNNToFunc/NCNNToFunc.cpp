#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace mlir::ncnn {
namespace {

class ConvertNCNNModelToFuncPass final
  : public PassWrapper<ConvertNCNNModelToFuncPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNCNNModelToFuncPass)

  StringRef getArgument() const final { return "convert-ncnn-model-to-func"; }
  StringRef getDescription() const final {
    return "Build a function boundary for each ncnn.model";
  }

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() final {
    SmallVector<ModelOp> models(getOperation().getOps<ModelOp>());
    for (ModelOp model : models) {
      if (failed(convertModel(model))) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  static LogicalResult convertModel(ModelOp model) {
    SmallVector<InputOp> inputs;
    SmallVector<OutputOp> outputs;
    for (Operation& operation : model.getBody().front()) {
      if (auto input = dyn_cast<InputOp>(operation)) {
        inputs.push_back(input);
      } else if (auto output = dyn_cast<OutputOp>(operation)) {
        outputs.push_back(output);
      }
    }
    if (outputs.empty()) {
      return model.emitOpError("has no outputs to form function results");
    }

    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes;
    for (InputOp input : inputs) {
      inputTypes.push_back(input.getOutput().getType());
    }
    for (OutputOp output : outputs) {
      outputTypes.push_back(output.getInput().getType());
    }

    OpBuilder builder(model);
    auto function = builder.create<func::FuncOp>(
      model.getLoc(),
      model.getSymName(),
      FunctionType::get(model.getContext(), inputTypes, outputTypes));
    function->setAttr("ncnn.entry_point", builder.getUnitAttr());
    function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
    Block* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    IRMapping mapping;
    for (auto [input, argument] : llvm::zip(inputs, entry->getArguments())) {
      mapping.map(input.getOutput(), argument);
    }

    for (Operation& operation : model.getBody().front()) {
      if (isa<InputOp, OutputOp>(operation)) {
        continue;
      }
      if (auto constant = dyn_cast<ConstOp>(operation)) {
        auto lowered = builder.create<arith::ConstantOp>(constant.getLoc(),
                                                         constant.getValue());
        mapping.map(constant.getOutput(), lowered.getResult());
        continue;
      }
      builder.clone(operation, mapping);
    }

    SmallVector<Value> results;
    for (OutputOp output : outputs) {
      Value mapped = mapping.lookupOrNull(output.getInput());
      if (!mapped) {
        function.erase();
        return output.emitOpError("input was not converted into the function");
      }
      results.push_back(mapped);
    }
    builder.create<func::ReturnOp>(model.getLoc(), results);
    model.erase();
    return success();
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertNCNNModelToFuncPass() {
  return std::make_unique<ConvertNCNNModelToFuncPass>();
}

void registerNCNNToFuncPasses() {
  static PassRegistration<ConvertNCNNModelToFuncPass> registration;
}

}  // namespace mlir::ncnn
