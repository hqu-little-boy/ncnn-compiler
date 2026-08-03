#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_CONVERTNCNNMODELTOFUNCPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class ConvertModel final : public OpConversionPattern<ModelOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ModelOp model, OpAdaptor, ConversionPatternRewriter& rewriter) const final {
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

    rewriter.setInsertionPoint(model);
    auto function = rewriter.create<func::FuncOp>(
      model.getLoc(),
      model.getSymName(),
      FunctionType::get(model.getContext(), inputTypes, outputTypes));
    function->setAttr("ncnn.entry_point", rewriter.getUnitAttr());
    function->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
    SmallVector<Location> argumentLocations(inputTypes.size(), model.getLoc());
    Block* entry = rewriter.createBlock(&function.getBody(),
                                        function.getBody().end(),
                                        inputTypes,
                                        argumentLocations);
    rewriter.inlineBlockBefore(&model.getBody().front(), entry, entry->end());

    for (auto [input, argument] : llvm::zip(inputs, entry->getArguments())) {
      rewriter.replaceOp(input, argument);
    }

    SmallVector<Value> results;
    for (OutputOp output : outputs) {
      Value mapped = rewriter.getRemappedValue(output.getInput());
      if (!mapped) {
        return output.emitOpError("input was not remapped into the function");
      }
      results.push_back(mapped);
      rewriter.eraseOp(output);
    }
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<func::ReturnOp>(model.getLoc(), results);
    rewriter.eraseOp(model);
    return success();
  }
};

class ConvertConst final : public OpConversionPattern<ConstOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    ConstOp operation,
    OpAdaptor,
    ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(operation,
                                                   operation.getValue());
    return success();
  }
};

class ConvertNCNNModelToFuncPass final
  : public impl::ConvertNCNNModelToFuncPassBase<ConvertNCNNModelToFuncPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    MLIRContext* context = getOperation().getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertModel, ConvertConst>(context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<ModelOp, InputOp, ConstOp, OutputOp>();
    target.markUnknownOpDynamicallyLegal(
      [](Operation*) -> std::optional<bool> { return true; });

    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyFullConversion(getOperation(), target, frozen))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
