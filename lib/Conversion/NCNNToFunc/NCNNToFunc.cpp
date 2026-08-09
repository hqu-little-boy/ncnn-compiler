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
#include "ncnn-mlir/Support/ShapeProgram.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_CONVERTNCNNMODELTOFUNCPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

struct ShapeTransform {
  SmallVector<DimensionExpr> dimensions;
};

SmallVector<ShapeConstraint> getShapeConstraints(ModelOp model) {
  SmallVector<ShapeConstraint> result;
  auto constraints = model->getAttrOfType<ArrayAttr>("ncnn.shape_constraints");
  if (!constraints) {
    return result;
  }
  result.reserve(constraints.size());
  for (Attribute attribute : constraints) {
    auto constraint = cast<DimConstraintAttr>(attribute);
    result.push_back({.inputIndex = constraint.getInput(),
                      .inputDimension = constraint.getDim(),
                      .minimum = constraint.getMin(),
                      .multipleOf = constraint.getMultipleOf()});
  }
  return result;
}

void appendInstruction(ShapeTransform& transform,
                       unsigned dimension,
                       ShapeOpcode opcode,
                       int64_t operand) {
  transform.dimensions[dimension].append(opcode, operand);
}

LogicalResult requireSameDimensionExpr(Operation* operation,
                                       unsigned dimension,
                                       ArrayRef<ShapeConstraint> constraints,
                                       const DimensionExpr& lhs,
                                       const DimensionExpr& rhs) {
  if (lhs.equivalentUnder(constraints, rhs)) {
    return success();
  }
  return operation->emitOpError()
         << "cannot prove input dimension " << dimension
         << " equal under input shape constraints";
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
    if (Attribute constraints = model->getAttr("ncnn.shape_constraints")) {
      function->setAttr("ncnn.shape_constraints", constraints);
    }
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
    SmallVector<ShapeConstraint> shapeConstraints = getShapeConstraints(model);
    for (auto [inputIndex, input] : llvm::enumerate(inputs)) {
      auto type = cast<RankedTensorType>(input.getOutput().getType());
      SmallVector<DimensionExpr> dimensions;
      dimensions.reserve(type.getRank());
      for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
        dimensions.emplace_back(static_cast<unsigned>(inputIndex),
                                static_cast<unsigned>(dimension));
      }
      shapeTransforms[input.getOutput()] = {.dimensions =
                                              std::move(dimensions)};
    }
    for (Operation& operation : model.getBody().front()) {
      if (operation.getNumOperands() == 0 || operation.getNumResults() == 0) {
        continue;
      }
      for (Value result : operation.getResults()) {
        auto inputType =
          dyn_cast<RankedTensorType>(operation.getOperand(0).getType());
        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!inputType || !resultType) {
          continue;
        }
        auto source = shapeTransforms.find(operation.getOperand(0));
        if (inputType.getRank() == resultType.getRank() &&
            isa<ReluOp, SigmoidOp, SplitOp>(operation)) {
          if (source != shapeTransforms.end()) {
            ShapeTransform propagated = source->second;
            shapeTransforms[result] = std::move(propagated);
          }
          continue;
        }
        if (inputType.getRank() != 3 || resultType.getRank() != 3) {
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
              ShapeTransform reshapeTransform = reshapeSource->second;
              shapeTransforms[result] = std::move(reshapeTransform);
            }
          }
          continue;
        }
        source = shapeTransforms.find(operation.getOperand(0));
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
        } else if (isa<ConvolutionOp, ConvolutionDepthWiseOp>(operation)) {
          const int64_t kernelH =
            operation.getAttrOfType<IntegerAttr>("kernel_h").getInt();
          const int64_t kernelW =
            operation.getAttrOfType<IntegerAttr>("kernel_w").getInt();
          const int64_t dilationH =
            operation.getAttrOfType<IntegerAttr>("dilation_h").getInt();
          const int64_t dilationW =
            operation.getAttrOfType<IntegerAttr>("dilation_w").getInt();
          const int64_t strideH =
            operation.getAttrOfType<IntegerAttr>("stride_h").getInt();
          const int64_t strideW =
            operation.getAttrOfType<IntegerAttr>("stride_w").getInt();
          const int64_t padTop =
            operation.getAttrOfType<IntegerAttr>("pad_top").getInt();
          const int64_t padBottom =
            operation.getAttrOfType<IntegerAttr>("pad_bottom").getInt();
          const int64_t padLeft =
            operation.getAttrOfType<IntegerAttr>("pad_left").getInt();
          const int64_t padRight =
            operation.getAttrOfType<IntegerAttr>("pad_right").getInt();
          if (padTop < 0 || padBottom < 0 || padLeft < 0 || padRight < 0) {
            const bool same = (padTop == -233 || padTop == -234) &&
                              padBottom == padTop && padLeft == padTop &&
                              padRight == padTop;
            if (!same) {
              continue;
            }
            appendInstruction(transform, 1, ShapeOpcode::Add, -1);
            appendInstruction(transform, 1, ShapeOpcode::Divide, strideH);
            appendInstruction(transform, 1, ShapeOpcode::Add, 1);
            appendInstruction(transform, 2, ShapeOpcode::Add, -1);
            appendInstruction(transform, 2, ShapeOpcode::Divide, strideW);
            appendInstruction(transform, 2, ShapeOpcode::Add, 1);
            shapeTransforms[result] = std::move(transform);
            continue;
          }
          const int64_t effectiveH = (dilationH * (kernelH - 1)) + 1;
          const int64_t effectiveW = (dilationW * (kernelW - 1)) + 1;
          if (strideH != 1 || padTop + padBottom != effectiveH - 1) {
            appendInstruction(
              transform, 1, ShapeOpcode::Add, padTop + padBottom - effectiveH);
            appendInstruction(transform, 1, ShapeOpcode::Divide, strideH);
            appendInstruction(transform, 1, ShapeOpcode::Add, 1);
          }
          if (strideW != 1 || padLeft + padRight != effectiveW - 1) {
            appendInstruction(
              transform, 2, ShapeOpcode::Add, padLeft + padRight - effectiveW);
            appendInstruction(transform, 2, ShapeOpcode::Divide, strideW);
            appendInstruction(transform, 2, ShapeOpcode::Add, 1);
          }
          shapeTransforms[result] = std::move(transform);
        } else if (auto pooling = dyn_cast<PoolingOp>(operation)) {
          if (pooling.getMode() != static_cast<int64_t>(PoolMode::Regular)) {
            continue;
          }
          if (pooling.getPadMode() == 0 &&
              (pooling.getStrideH() != 1 || pooling.getStrideW() != 1)) {
            continue;
          }
          if (pooling.getPadMode() == 2 || pooling.getPadMode() == 3) {
            appendInstruction(transform, 1, ShapeOpcode::Add, -1);
            appendInstruction(
              transform, 1, ShapeOpcode::Divide, pooling.getStrideH());
            appendInstruction(transform, 1, ShapeOpcode::Add, 1);
            appendInstruction(transform, 2, ShapeOpcode::Add, -1);
            appendInstruction(
              transform, 2, ShapeOpcode::Divide, pooling.getStrideW());
            appendInstruction(transform, 2, ShapeOpcode::Add, 1);
          } else {
            appendInstruction(transform,
                              1,
                              ShapeOpcode::Add,
                              pooling.getPadTop() + pooling.getPadBottom() -
                                pooling.getKernelH());
            appendInstruction(
              transform, 1, ShapeOpcode::Divide, pooling.getStrideH());
            appendInstruction(transform, 1, ShapeOpcode::Add, 1);
            appendInstruction(transform,
                              2,
                              ShapeOpcode::Add,
                              pooling.getPadLeft() + pooling.getPadRight() -
                                pooling.getKernelW());
            appendInstruction(
              transform, 2, ShapeOpcode::Divide, pooling.getStrideW());
            appendInstruction(transform, 2, ShapeOpcode::Add, 1);
          }
          shapeTransforms[result] = std::move(transform);
        } else if (auto deconvolution = dyn_cast<DeconvolutionOp>(operation)) {
          const int64_t effectiveH =
            (deconvolution.getDilationH() * (deconvolution.getKernelH() - 1)) +
            1;
          const int64_t effectiveW =
            (deconvolution.getDilationW() * (deconvolution.getKernelW() - 1)) +
            1;
          appendInstruction(transform, 1, ShapeOpcode::Add, -1);
          appendInstruction(
            transform, 1, ShapeOpcode::Multiply, deconvolution.getStrideH());
          appendInstruction(transform,
                            1,
                            ShapeOpcode::Add,
                            effectiveH + deconvolution.getOutputPadBottom() -
                              deconvolution.getPadTop() -
                              deconvolution.getPadBottom());
          appendInstruction(transform, 2, ShapeOpcode::Add, -1);
          appendInstruction(
            transform, 2, ShapeOpcode::Multiply, deconvolution.getStrideW());
          appendInstruction(transform,
                            2,
                            ShapeOpcode::Add,
                            effectiveW + deconvolution.getOutputPadRight() -
                              deconvolution.getPadLeft() -
                              deconvolution.getPadRight());
          shapeTransforms[result] = std::move(transform);
        } else if (auto concat = dyn_cast<ConcatOp>(operation)) {
          if (concat.getAxis() != 0) {
            if (!resultType.hasStaticShape()) {
              return concat.emitOpError(
                "dynamic symbolic shape proof only supports channel concat");
            }
            continue;
          }
          for (Value input : concat.getInputs().drop_front()) {
            auto candidate = shapeTransforms.find(input);
            if (candidate == shapeTransforms.end()) {
              return concat.emitOpError(
                "cannot prove dynamic concat input shapes equal");
            }
            for (unsigned dimension : {1U, 2U}) {
              if (!ShapedType::isDynamic(resultType.getShape()[dimension])) {
                continue;
              }
              if (failed(requireSameDimensionExpr(
                    concat,
                    dimension,
                    shapeConstraints,
                    transform.dimensions[dimension],
                    candidate->second.dimensions[dimension]))) {
                return failure();
              }
            }
          }
          shapeTransforms[result] = std::move(transform);
        } else if (auto binary = dyn_cast<BinaryOp>(operation)) {
          if (resultType.hasStaticShape()) {
            continue;
          }
          if (binary.getWithScalar()) {
            shapeTransforms[result] = std::move(transform);
            continue;
          }
          auto candidate = shapeTransforms.find(binary.getInputs()[1]);
          if (candidate == shapeTransforms.end()) {
            return binary.emitOpError(
              "cannot prove dynamic binary input shapes broadcastable");
          }
          auto firstType =
            cast<RankedTensorType>(binary.getInputs()[0].getType());
          auto secondType =
            cast<RankedTensorType>(binary.getInputs()[1].getType());
          for (unsigned dimension = 0; dimension < 3; ++dimension) {
            if (firstType.getShape()[dimension] == 1) {
              transform.dimensions[dimension] =
                candidate->second.dimensions[dimension];
            } else if (secondType.getShape()[dimension] != 1 &&
                       ShapedType::isDynamic(
                         resultType.getShape()[dimension]) &&
                       failed(requireSameDimensionExpr(
                         binary,
                         dimension,
                         shapeConstraints,
                         transform.dimensions[dimension],
                         candidate->second.dimensions[dimension]))) {
              return failure();
            }
          }
          shapeTransforms[result] = std::move(transform);
        }
      }
    }
    functionResultIndex = 0;
    for (OutputOp output : outputs) {
      auto outputType = cast<RankedTensorType>(output.getInput().getType());
      if (outputType.hasStaticShape()) {
        functionResultIndex +=
          output.getInput().getDefiningOp<DetectionOutputOp>() ? 2 : 1;
        continue;
      }
      auto transform = shapeTransforms.find(output.getInput());
      if (transform != shapeTransforms.end()) {
        function.setResultAttr(
          functionResultIndex,
          "ncnn.shape_source_input",
          rewriter.getI32IntegerAttr(static_cast<int32_t>(
            transform->second.dimensions.front().getInputIndex())));
        SmallVector<Attribute> programs;
        for (const DimensionExpr& dimension : transform->second.dimensions) {
          if (dimension.getInputIndex() !=
              transform->second.dimensions.front().getInputIndex()) {
            return model.emitOpError(
              "output shape dimensions require multiple source inputs");
          }
          DimensionExpr identity(dimension.getInputIndex(),
                                 dimension.getInputDimension());
          programs.push_back(rewriter.getDenseI64ArrayAttr(
            dimension.equivalentUnder(shapeConstraints, identity)
              ? identity.serialize()
              : dimension.serialize()));
        }
        function.setResultAttr(functionResultIndex,
                               "ncnn.shape_program",
                               rewriter.getArrayAttr(programs));
      }
      functionResultIndex +=
        output.getInput().getDefiningOp<DetectionOutputOp>() ? 2 : 1;
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
