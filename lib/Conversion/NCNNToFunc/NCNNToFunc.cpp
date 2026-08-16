#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"

#include <memory>
#include <optional>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
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

struct ShapeDimension {
  ShapeExpr expression;
  std::optional<DimensionExpr> v1;
};

struct ShapeTransform {
  SmallVector<ShapeDimension> dimensions;
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
  if (transform.dimensions[dimension].v1) {
    transform.dimensions[dimension].v1->append(opcode, operand);
  }
  ShapeExprOpcode expressionOpcode;
  switch (opcode) {
    case ShapeOpcode::Add:
      expressionOpcode = ShapeExprOpcode::Add;
      break;
    case ShapeOpcode::Multiply:
      expressionOpcode = ShapeExprOpcode::Multiply;
      break;
    case ShapeOpcode::Divide:
      expressionOpcode = ShapeExprOpcode::FloorDivide;
      break;
    default:
      llvm_unreachable("unsupported shape instruction");
  }
  transform.dimensions[dimension].expression =
    ShapeExpr::binary(expressionOpcode,
                      std::move(transform.dimensions[dimension].expression),
                      ShapeExpr::constant(operand));
}

LogicalResult requireSameDimensionExpr(Operation* operation,
                                       unsigned dimension,
                                       ArrayRef<ShapeConstraint> constraints,
                                       const ShapeDimension& lhs,
                                       const ShapeDimension& rhs) {
  auto lhsProgram = lhs.expression.serializeChecked();
  auto rhsProgram = rhs.expression.serializeChecked();
  if ((lhs.v1 && rhs.v1 && lhs.v1->equivalentUnder(constraints, *rhs.v1)) ||
      (lhsProgram && rhsProgram && *lhsProgram == *rhsProgram)) {
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
    for (StringRef name : {"ncnn.precision",
                           "ncnn.fp16_accumulator",
                           "ncnn.precision_fallback"}) {
      if (Attribute value = model->getAttr(name)) {
        function->setAttr(name, value);
      }
    }
    for (StringRef name :
         {"ncnn.shape_constraints", "ncnn.input_dim_relations"}) {
      if (Attribute value = model->getAttr(name)) {
        function->setAttr(name, value);
      }
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
      SmallVector<ShapeDimension> dimensions;
      dimensions.reserve(type.getRank());
      for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
        DimensionExpr v1(static_cast<unsigned>(inputIndex),
                         static_cast<unsigned>(dimension));
        dimensions.push_back({.expression = v1.toV2(), .v1 = std::move(v1)});
      }
      shapeTransforms[input.getOutput()] = {.dimensions =
                                              std::move(dimensions)};
    }
    for (Operation& operation : model.getBody().front()) {
      if (operation.getNumOperands() == 0 || operation.getNumResults() == 0) {
        continue;
      }
      if (auto sdpa = dyn_cast<SDPAOp>(operation)) {
        auto query = shapeTransforms.find(sdpa.getQuery());
        auto key = shapeTransforms.find(sdpa.getKey());
        auto value = shapeTransforms.find(sdpa.getValue());
        if (query != shapeTransforms.end() && value != shapeTransforms.end()) {
          shapeTransforms[sdpa.getContext()] = {
            .dimensions = {query->second.dimensions[0],
                           query->second.dimensions[1],
                           value->second.dimensions[2]}};
        }
        if (sdpa.getKvCache()) {
          const unsigned cacheInput = sdpa.getHasMask() ? 1 : 0;
          auto pastKey =
            shapeTransforms.find(sdpa.getOptionalInputs()[cacheInput]);
          auto pastValue =
            shapeTransforms.find(sdpa.getOptionalInputs()[cacheInput + 1]);
          if (pastKey == shapeTransforms.end() ||
              pastValue == shapeTransforms.end()) {
            continue;
          }
          auto dimension = [&](
                             Value tensor,
                             DenseMap<Value, ShapeTransform>::iterator found,
                             unsigned index) -> std::optional<ShapeDimension> {
            auto type = cast<RankedTensorType>(tensor.getType());
            if (!type.isDynamicDim(index)) {
              return ShapeDimension{
                .expression = ShapeExpr::constant(type.getShape()[index]),
                .v1 = std::nullopt};
            }
            if (found == shapeTransforms.end()) {
              return std::nullopt;
            }
            return found->second.dimensions[index];
          };
          auto sumSequence = [](const ShapeDimension& past,
                                const ShapeDimension& current) {
            return ShapeDimension{
              .expression = ShapeExpr::binary(
                ShapeExprOpcode::Add, past.expression, current.expression),
              .v1 = std::nullopt};
          };
          auto keyHead = dimension(sdpa.getKey(), key, 0);
          auto keySequence = dimension(sdpa.getKey(), key, 1);
          auto keyFeature = dimension(sdpa.getKey(), key, 2);
          auto valueHead = dimension(sdpa.getValue(), value, 0);
          auto valueSequence = dimension(sdpa.getValue(), value, 1);
          auto valueFeature = dimension(sdpa.getValue(), value, 2);
          if (!keyHead || !keySequence || !keyFeature || !valueHead ||
              !valueSequence || !valueFeature) {
            continue;
          }
          shapeTransforms[sdpa.getCacheResults()[0]] = {
            .dimensions = {
              *keyHead,
              sumSequence(pastKey->second.dimensions[1], *keySequence),
              *keyFeature}};
          shapeTransforms[sdpa.getCacheResults()[1]] = {
            .dimensions = {
              *valueHead,
              sumSequence(pastValue->second.dimensions[1], *valueSequence),
              *valueFeature}};
        }
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
            isa<ReluOp,
                SigmoidOp,
                TanHOp,
                SwishOp,
                SplitOp,
                HardSigmoidOp,
                HardSwishOp,
                GELUOp,
                DropoutOp,
                SoftmaxOp,
                BatchNormOp,
                LayerNormOp,
                MultiHeadAttentionOp,
                ShuffleChannelOp>(operation)) {
          if (source != shapeTransforms.end()) {
            ShapeTransform propagated = source->second;
            shapeTransforms[result] = std::move(propagated);
          }
          continue;
        }
        if (auto reshape = dyn_cast<ReshapeOp>(operation);
            reshape && reshape.getShapeSources()) {
          ArrayRef<int64_t> sources = *reshape.getShapeSources();
          SmallVector<ShapeDimension> dimensions;
          dimensions.reserve(resultType.getRank());
          for (std::size_t index = 0; index < sources.size(); index += 2) {
            const int64_t sourceOperand = sources[index];
            const int64_t sourceDimension = sources[index + 1];
            if (sourceOperand < 0 || sourceDimension < 0 ||
                static_cast<unsigned>(sourceOperand) >=
                  operation.getNumOperands()) {
              dimensions.clear();
              break;
            }
            auto reshapeSource =
              shapeTransforms.find(operation.getOperand(sourceOperand));
            if (reshapeSource == shapeTransforms.end() ||
                static_cast<unsigned>(sourceDimension) >=
                  reshapeSource->second.dimensions.size()) {
              dimensions.clear();
              break;
            }
            dimensions.push_back(
              reshapeSource->second.dimensions[sourceDimension]);
          }
          if (dimensions.size() ==
              static_cast<std::size_t>(resultType.getRank())) {
            shapeTransforms[result] = {.dimensions = std::move(dimensions)};
          }
          continue;
        }
        if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
          auto shapeSpec =
            reshape->getAttrOfType<DenseI64ArrayAttr>("shape_spec");
          auto zeroSourceAttr =
            reshape->getAttrOfType<DenseI64ArrayAttr>("shape_zero_sources");
          ArrayRef<int64_t> shape =
            shapeSpec ? shapeSpec.asArrayRef() : reshape.getShape();
          ArrayRef<int64_t> zeroSources =
            zeroSourceAttr ? zeroSourceAttr.asArrayRef() : ArrayRef<int64_t>();
          if (shape.size() != static_cast<std::size_t>(resultType.getRank())) {
            continue;
          }
          auto reshapeSource = shapeTransforms.find(reshape.getInput());
          if (reshapeSource == shapeTransforms.end()) {
            continue;
          }
          SmallVector<ShapeDimension> dimensions;
          dimensions.reserve(resultType.getRank());
          std::optional<unsigned> inferredDimension;
          for (auto [dimension, extent] : llvm::enumerate(shape)) {
            if (extent == 0 && dimension < zeroSources.size() &&
                zeroSources[dimension] >= 0 &&
                static_cast<std::size_t>(zeroSources[dimension]) <
                  reshapeSource->second.dimensions.size()) {
              dimensions.push_back(
                reshapeSource->second.dimensions[zeroSources[dimension]]);
            } else if (extent == -1) {
              inferredDimension = dimension;
              dimensions.push_back(
                {.expression = ShapeExpr::constant(1), .v1 = std::nullopt});
            } else if (extent > 0) {
              dimensions.push_back({.expression = ShapeExpr::constant(extent),
                                    .v1 = std::nullopt});
            } else {
              dimensions.clear();
              break;
            }
          }
          if (dimensions.empty()) {
            continue;
          }
          if (inferredDimension) {
            std::optional<unsigned> dynamicSource;
            int64_t inputStaticElements = 1;
            bool canCopyDynamicSource = true;
            for (auto [dimension, extent] :
                 llvm::enumerate(inputType.getShape())) {
              if (ShapedType::isDynamic(extent)) {
                if (dynamicSource) {
                  canCopyDynamicSource = false;
                  break;
                }
                dynamicSource = dimension;
              } else if (llvm::MulOverflow(
                           inputStaticElements, extent, inputStaticElements)) {
                canCopyDynamicSource = false;
                break;
              }
            }
            int64_t outputStaticElements = 1;
            for (auto [dimension, extent] : llvm::enumerate(shape)) {
              if (dimension == *inferredDimension) {
                continue;
              }
              if (extent <= 0 ||
                  llvm::MulOverflow(
                    outputStaticElements, extent, outputStaticElements)) {
                canCopyDynamicSource = false;
                break;
              }
            }
            if (canCopyDynamicSource && dynamicSource &&
                inputStaticElements == outputStaticElements) {
              dimensions[*inferredDimension] =
                reshapeSource->second.dimensions[*dynamicSource];
              shapeTransforms[result] = {.dimensions = std::move(dimensions)};
              continue;
            }
            ShapeExpr elements = ShapeExpr::constant(1);
            for (const ShapeDimension& dimension :
                 reshapeSource->second.dimensions) {
              elements = ShapeExpr::binary(ShapeExprOpcode::Multiply,
                                           std::move(elements),
                                           dimension.expression);
            }
            ShapeExpr knownElements = ShapeExpr::constant(1);
            for (auto [dimension, expression] : llvm::enumerate(dimensions)) {
              if (dimension != *inferredDimension) {
                knownElements = ShapeExpr::binary(ShapeExprOpcode::Multiply,
                                                  std::move(knownElements),
                                                  expression.expression);
              }
            }
            dimensions[*inferredDimension].expression =
              ShapeExpr::binary(ShapeExprOpcode::FloorDivide,
                                std::move(elements),
                                std::move(knownElements));
          }
          shapeTransforms[result] = {.dimensions = std::move(dimensions)};
          continue;
        }
        source = shapeTransforms.find(operation.getOperand(0));
        if (source == shapeTransforms.end()) {
          continue;
        }
        ShapeTransform transform = source->second;
        if (transform.dimensions.empty()) {
          continue;
        }
        if (auto permute = dyn_cast<PermuteOp>(operation)) {
          SmallVector<ShapeDimension> dimensions;
          dimensions.reserve(resultType.getRank());
          for (int64_t axis : permute.getPermutation()) {
            dimensions.push_back(transform.dimensions[axis]);
          }
          shapeTransforms[result] = {.dimensions = std::move(dimensions)};
          continue;
        }
        if (auto expand = dyn_cast<ExpandDimsOp>(operation)) {
          std::set<int64_t> axes;
          for (int64_t axis : expand.getAxes()) {
            axes.insert(axis < 0 ? axis + resultType.getRank() : axis);
          }
          SmallVector<ShapeDimension> dimensions;
          dimensions.reserve(resultType.getRank());
          unsigned inputDimension = 0;
          for (int64_t axis = 0; axis < resultType.getRank(); ++axis) {
            dimensions.push_back(
              axes.contains(axis)
                ? ShapeDimension{.expression = ShapeExpr::constant(1),
                                 .v1 = transform.dimensions.front().v1}
                : transform.dimensions[inputDimension++]);
          }
          shapeTransforms[result] = {.dimensions = std::move(dimensions)};
          continue;
        }
        if (auto squeeze = dyn_cast<SqueezeOp>(operation)) {
          std::set<int64_t> axes;
          for (int64_t axis : squeeze.getAxes()) {
            axes.insert(axis < 0 ? axis + inputType.getRank() : axis);
          }
          SmallVector<ShapeDimension> dimensions;
          for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
            if (!axes.contains(axis)) {
              dimensions.push_back(transform.dimensions[axis]);
            }
          }
          if (dimensions.empty()) {
            dimensions.push_back(transform.dimensions.front());
          }
          shapeTransforms[result] = {.dimensions = std::move(dimensions)};
          continue;
        }
        if (auto reduction = dyn_cast<ReductionOp>(operation)) {
          std::set<int64_t> axes;
          if (reduction.getReduceAll()) {
            for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
              axes.insert(axis);
            }
          } else {
            for (int64_t axis : reduction.getAxes()) {
              axes.insert(axis < 0 ? axis + inputType.getRank() : axis);
            }
          }
          SmallVector<ShapeDimension> dimensions;
          for (int64_t axis = 0; axis < inputType.getRank(); ++axis) {
            if (!axes.contains(axis)) {
              dimensions.push_back(transform.dimensions[axis]);
            } else if (reduction.getKeepdims()) {
              dimensions.push_back({.expression = ShapeExpr::constant(1),
                                    .v1 = transform.dimensions.front().v1});
            }
          }
          if (dimensions.empty()) {
            dimensions.push_back(transform.dimensions.front());
          }
          shapeTransforms[result] = {.dimensions = std::move(dimensions)};
          continue;
        }
        if (isa<GemmOp>(operation)) {
          shapeTransforms[result] = {
            .dimensions = {
              transform.dimensions[0],
              {.expression = ShapeExpr::constant(resultType.getShape()[1]),
               .v1 = transform.dimensions[0].v1}}};
          continue;
        }
        if (isa<InnerProductOp>(operation) && inputType.getRank() == 2 &&
            resultType.getRank() == 2) {
          shapeTransforms[result] = {
            .dimensions = {
              transform.dimensions[0],
              {.expression = ShapeExpr::constant(resultType.getShape()[1]),
               .v1 = transform.dimensions[0].v1}}};
          continue;
        }
        if (auto slice = dyn_cast<SliceOp>(operation)) {
          int64_t axis = slice.getAxis();
          if (axis < 0) {
            axis += inputType.getRank();
          }
          ShapeExpr consumed = ShapeExpr::constant(0);
          for (auto [resultIndex, sliceResult] :
               llvm::enumerate(slice.getResults())) {
            ShapeExpr size;
            const int64_t requested = slice.getSlices()[resultIndex];
            if (requested == -233) {
              size = ShapeExpr::binary(
                ShapeExprOpcode::FloorDivide,
                ShapeExpr::binary(ShapeExprOpcode::Add,
                                  transform.dimensions[axis].expression,
                                  ShapeExpr::binary(ShapeExprOpcode::Multiply,
                                                    consumed,
                                                    ShapeExpr::constant(-1))),
                ShapeExpr::constant(operation.getNumResults() - resultIndex));
            } else {
              size = ShapeExpr::constant(requested);
            }
            SmallVector<ShapeDimension> dimensions = transform.dimensions;
            dimensions[axis] = {.expression = size,
                                .v1 = requested == -233
                                        ? std::nullopt
                                        : transform.dimensions.front().v1};
            shapeTransforms[sliceResult] = {.dimensions =
                                              std::move(dimensions)};
            consumed = ShapeExpr::binary(
              ShapeExprOpcode::Add, std::move(consumed), std::move(size));
          }
          continue;
        }
        if (auto pooling = dyn_cast<PoolingOp>(operation)) {
          if (pooling.getMode() == static_cast<int64_t>(PoolMode::Global)) {
            shapeTransforms[result] = {
              .dimensions = {transform.dimensions.front()}};
            continue;
          }
          if (pooling.getMode() == static_cast<int64_t>(PoolMode::Adaptive)) {
            if (resultType.getRank() != 3) {
              continue;
            }
            const int64_t adaptiveExtents[] = {
              static_cast<int64_t>(pooling.getKernelH()),
              static_cast<int64_t>(pooling.getKernelW())};
            for (auto [dimension, extent] :
                 llvm::zip_equal(ArrayRef<unsigned>{1, 2}, adaptiveExtents)) {
              if (extent != -233) {
                transform.dimensions[dimension] = {
                  .expression = ShapeExpr::constant(extent),
                  .v1 = std::nullopt};
              }
            }
            shapeTransforms[result] = std::move(transform);
            continue;
          }
        }
        if ((inputType.getRank() != 3 || resultType.getRank() != 3) &&
            !isa<ConcatOp, BinaryOp>(operation)) {
          continue;
        }
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
          if (interp.getOutputH() != 0) {
            transform.dimensions[1] = {
              .expression = ShapeExpr::constant(interp.getOutputH()),
              .v1 = std::nullopt};
          } else {
            appendInstruction(
              transform, 1, ShapeOpcode::Multiply, interp.getHeightScale());
          }
          if (interp.getOutputW() != 0) {
            transform.dimensions[2] = {
              .expression = ShapeExpr::constant(interp.getOutputW()),
              .v1 = std::nullopt};
          } else {
            appendInstruction(
              transform, 2, ShapeOpcode::Multiply, interp.getWidthScale());
          }
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
          int64_t axis = concat.getAxis();
          if (axis < 0) {
            axis += resultType.getRank();
          }
          for (Value input : concat.getInputs().drop_front()) {
            auto candidate = shapeTransforms.find(input);
            if (candidate == shapeTransforms.end()) {
              return concat.emitOpError(
                "cannot prove dynamic concat input shapes equal");
            }
            for (int64_t dimension = 0; dimension < resultType.getRank();
                 ++dimension) {
              if (dimension == axis) {
                if (ShapedType::isDynamic(resultType.getShape()[dimension])) {
                  transform.dimensions[dimension].expression =
                    ShapeExpr::binary(
                      ShapeExprOpcode::Add,
                      std::move(transform.dimensions[dimension].expression),
                      candidate->second.dimensions[dimension].expression);
                  transform.dimensions[dimension].v1 = std::nullopt;
                }
                continue;
              }
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
          auto firstType =
            cast<RankedTensorType>(binary.getInputs()[0].getType());
          auto secondType =
            cast<RankedTensorType>(binary.getInputs()[1].getType());
          auto candidate = shapeTransforms.find(binary.getInputs()[1]);
          if (candidate == shapeTransforms.end()) {
            bool staticUnitBroadcast = true;
            for (unsigned dimension = 0; dimension < resultType.getRank();
                 ++dimension) {
              if (ShapedType::isDynamic(resultType.getShape()[dimension]) &&
                  secondType.getShape()[dimension] != 1) {
                staticUnitBroadcast = false;
                break;
              }
            }
            if (!staticUnitBroadcast) {
              return binary.emitOpError(
                "cannot prove dynamic binary input shapes broadcastable");
            }
            shapeTransforms[result] = std::move(transform);
            continue;
          }
          for (unsigned dimension = 0; dimension < resultType.getRank();
               ++dimension) {
            if (!ShapedType::isDynamic(resultType.getShape()[dimension])) {
              continue;
            }
            if (firstType.getShape()[dimension] == 1) {
              transform.dimensions[dimension] =
                candidate->second.dimensions[dimension];
            } else if (secondType.getShape()[dimension] != 1) {
              if (failed(requireSameDimensionExpr(
                    binary,
                    dimension,
                    shapeConstraints,
                    transform.dimensions[dimension],
                    candidate->second.dimensions[dimension]))) {
                return failure();
              }
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
        const auto sourceInput =
          transform->second.dimensions.front().v1
            ? std::optional<unsigned>(
                transform->second.dimensions.front().v1->getInputIndex())
            : std::nullopt;
        const bool v2 =
          !sourceInput ||
          llvm::any_of(transform->second.dimensions,
                       [&](const ShapeDimension& dimension) {
                         return !dimension.v1 ||
                                dimension.v1->getInputIndex() != *sourceInput;
                       });
        SmallVector<Attribute> programs;
        for (auto [dimensionIndex, dimension] :
             llvm::enumerate(transform->second.dimensions)) {
          if (v2) {
            auto program = dimension.expression.serializeChecked();
            if (!program) {
              return output.emitOpError() << program.error();
            }
            programs.push_back(rewriter.getDenseI64ArrayAttr(*program));
            continue;
          }
          const DimensionExpr& v1 = *dimension.v1;
          DimensionExpr identity(v1.getInputIndex(), v1.getInputDimension());
          programs.push_back(rewriter.getDenseI64ArrayAttr(
            v1.equivalentUnder(shapeConstraints, identity)
              ? identity.serialize(dimensionIndex)
              : v1.serialize(dimensionIndex)));
        }
        function.setResultAttr(functionResultIndex,
                               "ncnn.shape_program",
                               rewriter.getArrayAttr(programs));
        if (v2) {
          function.setResultAttr(functionResultIndex,
                                 "ncnn.shape_program_version",
                                 rewriter.getI32IntegerAttr(2));
        } else {
          function.setResultAttr(
            functionResultIndex,
            "ncnn.shape_source_input",
            rewriter.getI32IntegerAttr(static_cast<int32_t>(*sourceInput)));
        }
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
