#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"

#include <memory>

#include "llvm/ADT/DenseMap.h"
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

struct ShapeTransform {
  unsigned inputIndex;
  SmallVector<SmallVector<int64_t>> programs;
};

enum class ShapeOpcode : int64_t { Add = 0, Multiply = 1, Divide = 2 };

void appendInstruction(ShapeTransform& transform,
                       unsigned dimension,
                       ShapeOpcode opcode,
                       int64_t operand) {
  transform.programs[dimension].push_back(static_cast<int64_t>(opcode));
  transform.programs[dimension].push_back(operand);
}

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
    SmallVector<Value> functionResults;
    for (InputOp input : inputs) {
      inputTypes.push_back(input.getOutput().getType());
    }
    for (OutputOp output : outputs) {
      outputTypes.push_back(output.getInput().getType());
      functionResults.push_back(output.getInput());
      if (auto detection =
            output.getInput().getDefiningOp<DetectionOutputOp>()) {
        outputTypes.push_back(detection.getActualShape().getType());
        functionResults.push_back(detection.getActualShape());
      }
    }

    rewriter.setInsertionPoint(model);
    auto function = rewriter.create<func::FuncOp>(
      model.getLoc(),
      model.getSymName(),
      FunctionType::get(model.getContext(), inputTypes, outputTypes));
    function->setAttr("ncnn.entry_point", rewriter.getUnitAttr());
    function->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
    if (Attribute rank = model->getAttr("ncnn.rank_variant")) {
      function->setAttr("ncnn.rank_variant", rank);
      function->setAttr("ncnn.dynamic_rank", rewriter.getUnitAttr());
    }
    unsigned functionResultIndex = 0;
    for (OutputOp output : outputs) {
      if (output.getInput().getDefiningOp<DetectionOutputOp>()) {
        function.setResultAttr(functionResultIndex,
                               "ncnn.data_dependent_dim_mask",
                               rewriter.getI32IntegerAttr(1));
        function.setResultAttr(functionResultIndex + 1,
                               "ncnn.shape_carrier",
                               rewriter.getUnitAttr());
        functionResultIndex += 2;
      } else {
        ++functionResultIndex;
      }
    }
    DenseMap<Value, ShapeTransform> shapeTransforms;
    for (auto [inputIndex, input] : llvm::enumerate(inputs)) {
      auto type = cast<RankedTensorType>(input.getOutput().getType());
      shapeTransforms[input.getOutput()] = {
        .inputIndex = static_cast<unsigned>(inputIndex),
        .programs = SmallVector<SmallVector<int64_t>>(type.getRank())};
    }
    for (Operation& operation : model.getBody().front()) {
      if (operation.getNumOperands() == 0 || operation.getNumResults() == 0) {
        continue;
      }
      for (Value result : operation.getResults()) {
        auto inputType =
          dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!inputType || !resultType || inputType.getRank() != 3 ||
            resultType.getRank() != 3) {
          continue;
        }
        if (auto reshape = dyn_cast<ReshapeOp>(operation);
            reshape && reshape.getShapeSources()) {
          ArrayRef<int64_t> sources = *reshape.getShapeSources();
          int64_t sourceOperand = sources.front();
          bool identity = sources.size() ==
                          static_cast<std::size_t>(resultType.getRank()) * 2;
          for (int64_t dimension = 0;
               identity && dimension < resultType.getRank();
               ++dimension) {
            identity = sources[dimension * 2] == sourceOperand &&
                       sources[(dimension * 2) + 1] == dimension;
          }
          if (identity && sourceOperand >= 0 &&
              static_cast<unsigned>(sourceOperand) <
                operation.getNumOperands()) {
            auto reshapeSource =
              shapeTransforms.find(operation.getOperand(sourceOperand));
            if (reshapeSource != shapeTransforms.end()) {
              shapeTransforms[result] = reshapeSource->second;
            }
          }
          continue;
        }
        auto source = shapeTransforms.find(operation.getOperand(0));
        if (source == shapeTransforms.end()) {
          continue;
        }
        ShapeTransform transform = source->second;
        if (auto padding = dyn_cast<PaddingOp>(operation)) {
          appendInstruction(transform,
                            1,
                            ShapeOpcode::Add,
                            padding.getTop() + padding.getBottom());
          appendInstruction(transform,
                            2,
                            ShapeOpcode::Add,
                            padding.getLeft() + padding.getRight());
          shapeTransforms[result] = std::move(transform);
        } else if (auto interp = dyn_cast<InterpOp>(operation)) {
          appendInstruction(
            transform, 1, ShapeOpcode::Multiply, interp.getHeightScale());
          appendInstruction(
            transform, 2, ShapeOpcode::Multiply, interp.getWidthScale());
          shapeTransforms[result] = std::move(transform);
        } else if (auto convolution = dyn_cast<ConvolutionOp>(operation)) {
          if (convolution.getPadTopAttr().getInt() < 0 ||
              convolution.getPadBottomAttr().getInt() < 0 ||
              convolution.getPadLeftAttr().getInt() < 0 ||
              convolution.getPadRightAttr().getInt() < 0) {
            continue;
          }
          const int64_t effectiveH =
            (convolution.getDilationH() * (convolution.getKernelH() - 1)) + 1;
          const int64_t effectiveW =
            (convolution.getDilationW() * (convolution.getKernelW() - 1)) + 1;
          appendInstruction(
            transform,
            1,
            ShapeOpcode::Add,
            convolution.getPadTop() + convolution.getPadBottom() - effectiveH);
          appendInstruction(
            transform, 1, ShapeOpcode::Divide, convolution.getStrideH());
          appendInstruction(transform, 1, ShapeOpcode::Add, 1);
          appendInstruction(
            transform,
            2,
            ShapeOpcode::Add,
            convolution.getPadLeft() + convolution.getPadRight() - effectiveW);
          appendInstruction(
            transform, 2, ShapeOpcode::Divide, convolution.getStrideW());
          appendInstruction(transform, 2, ShapeOpcode::Add, 1);
          shapeTransforms[result] = std::move(transform);
        } else if (result.getType() == operation.getOperand(0).getType()) {
          shapeTransforms[result] = source->second;
        }
      }
    }
    for (auto [resultIndex, output] : llvm::enumerate(outputs)) {
      auto outputType = cast<RankedTensorType>(output.getInput().getType());
      if (outputType.hasStaticShape()) {
        continue;
      }
      auto transform = shapeTransforms.find(output.getInput());
      if (transform != shapeTransforms.end()) {
        function.setResultAttr(resultIndex,
                               "ncnn.shape_source_input",
                               rewriter.getI32IntegerAttr(static_cast<int32_t>(
                                 transform->second.inputIndex)));
        SmallVector<Attribute> programs;
        for (ArrayRef<int64_t> program : transform->second.programs) {
          programs.push_back(rewriter.getDenseI64ArrayAttr(program));
        }
        function.setResultAttr(
          resultIndex, "ncnn.shape_program", rewriter.getArrayAttr(programs));
      }
    }
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
    for (Value result : functionResults) {
      Value mapped = rewriter.getRemappedValue(result);
      if (!mapped) {
        return model.emitOpError("output was not remapped into the function");
      }
      results.push_back(mapped);
    }
    for (OutputOp output : outputs) {
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
