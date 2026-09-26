#include "GenerateCAPIInternal.hpp"

namespace mlir::ncnn::capi_detail {

void emitInferOutputShapes(FinalizeContext& ctx) {
  auto& exportName = ctx.exportName;
  auto& internal = ctx.internal;
  auto& argumentTypes = ctx.argumentTypes;
  auto& shapeVersions = ctx.shapeVersions;
  auto& inputIndices = ctx.inputIndices;
  auto& outputArgumentIndices = ctx.outputArgumentIndices;
  auto& shapePrograms = ctx.shapePrograms;
  auto& constraintsByInput = ctx.constraintsByInput;
  auto& inputDimRelations = ctx.inputDimRelations;
  auto& builder = ctx.builder;
  auto& wrapper = ctx.wrapper;
  auto& pointerType = ctx.pointerType;
  auto& i64Type = ctx.i64Type;
  auto& overflowResultType = ctx.overflowResultType;
  auto& mergeFlag = ctx.mergeFlag;
  auto& checkedMultiply = ctx.checkedMultiply;
  auto& evaluateShapeExpression = ctx.evaluateShapeExpression;

  const bool hasDynamicOutput = llvm::any_of(
    outputArgumentIndices,
    [&](unsigned index) { return !argumentTypes[index].hasStaticShape(); });
  if (hasDynamicOutput) {
    SmallVector<Type> shapeFunctionArguments;
    for (unsigned index : inputIndices) {
      if (!argumentTypes[index].hasStaticShape()) {
        shapeFunctionArguments.push_back(pointerType);
      }
    }
    for (unsigned index : outputArgumentIndices) {
      if (!argumentTypes[index].hasStaticShape()) {
        shapeFunctionArguments.push_back(pointerType);
      }
    }
    auto shapeFunctionType = LLVM::LLVMFunctionType::get(
      builder.getI32Type(), shapeFunctionArguments, false);
    builder.setInsertionPointAfter(wrapper);
    auto shapeFunction = builder.create<LLVM::LLVMFuncOp>(
      internal.getLoc(),
      (exportName.getValue() + "_infer_output_shapes").str(),
      shapeFunctionType,
      LLVM::Linkage::External);
    Block* shapeEntry = shapeFunction.addEntryBlock(builder);
    Block* shapeSuccess = &shapeFunction.getBody().emplaceBlock();
    Block* shapeError = &shapeFunction.getBody().emplaceBlock();
    Block* shapeInvalidError = &shapeFunction.getBody().emplaceBlock();
    Block* shapeConstraintError = &shapeFunction.getBody().emplaceBlock();
    Block* shapeOverflowError = &shapeFunction.getBody().emplaceBlock();
    Block* shapeValid = &shapeFunction.getBody().emplaceBlock();
    Block* shapeConstraintValid = &shapeFunction.getBody().emplaceBlock();
    Block* shapeArithmeticValid = &shapeFunction.getBody().emplaceBlock();
    Block* shapeRelationValid = &shapeFunction.getBody().emplaceBlock();
    Block* shapeReturn = &shapeFunction.getBody().emplaceBlock();
    builder.setInsertionPointToStart(shapeEntry);
    Value shapeNullPointer =
      builder.create<LLVM::ZeroOp>(internal.getLoc(), pointerType);
    Value shapeAnyNull;
    for (Value argument : shapeEntry->getArguments()) {
      Value isNull = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::eq, argument, shapeNullPointer);
      shapeAnyNull = shapeAnyNull ? builder.create<LLVM::OrOp>(
                                      internal.getLoc(), shapeAnyNull, isNull)
                                  : isNull;
    }
    builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                   shapeAnyNull,
                                   shapeError,
                                   ValueRange{},
                                   shapeSuccess,
                                   ValueRange{});
    builder.setInsertionPointToStart(shapeError);
    Value shapeFailure = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI32IntegerAttr(1));
    builder.create<LLVM::ReturnOp>(internal.getLoc(), shapeFailure);
    auto emitShapeStatus = [&](Block* block, int32_t status) {
      builder.setInsertionPointToStart(block);
      Value value = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI32IntegerAttr(status));
      builder.create<LLVM::ReturnOp>(internal.getLoc(), value);
    };
    emitShapeStatus(shapeInvalidError, 2);
    emitShapeStatus(shapeConstraintError, 3);
    emitShapeStatus(shapeOverflowError, 4);
    builder.setInsertionPointToStart(shapeSuccess);
    SmallVector<Value> shapeInputs(inputIndices.size());
    unsigned shapeArgumentIndex = 0;
    Value shapeInvalid;
    Value outputShapeInvalid;
    Value shapeConstraintInvalid;
    Value shapeArithmeticInvalid;
    Value shapeRelationInvalid;
    for (auto [inputIndex, functionIndex] : llvm::enumerate(inputIndices)) {
      if (!argumentTypes[functionIndex].hasStaticShape()) {
        Value inputShape = shapeEntry->getArgument(shapeArgumentIndex++);
        shapeInputs[inputIndex] = inputShape;
        Value inputElements = builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getI64IntegerAttr(1));
        for (auto [dimensionIndex, dimension] :
             llvm::enumerate(argumentTypes[functionIndex].getShape())) {
          Value address = builder.create<LLVM::GEPOp>(
            internal.getLoc(),
            pointerType,
            i64Type,
            inputShape,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimensionIndex)});
          Value extent =
            builder.create<LLVM::LoadOp>(internal.getLoc(), i64Type, address);
          Value expected = builder.create<LLVM::ConstantOp>(
            internal.getLoc(),
            builder.getI64IntegerAttr(
              ShapedType::isDynamic(dimension) ? 0 : dimension));
          Value valid = builder.create<LLVM::ICmpOp>(
            internal.getLoc(),
            ShapedType::isDynamic(dimension) ? LLVM::ICmpPredicate::sgt
                                             : LLVM::ICmpPredicate::eq,
            extent,
            expected);
          Value invalid = builder.create<LLVM::XOrOp>(
            internal.getLoc(),
            valid,
            builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                             builder.getBoolAttr(true)));
          shapeInvalid = shapeInvalid
                           ? builder.create<LLVM::OrOp>(
                               internal.getLoc(), shapeInvalid, invalid)
                           : invalid;
          auto multiplied = checkedMultiply(inputElements, extent);
          inputElements = multiplied.first;
          shapeArithmeticInvalid =
            mergeFlag(shapeArithmeticInvalid, multiplied.second);
          for (DimConstraintAttr constraint : constraintsByInput[inputIndex]) {
            if (constraint.getDim() != dimensionIndex) {
              continue;
            }
            Value minimum = builder.create<LLVM::ConstantOp>(
              internal.getLoc(),
              builder.getI64IntegerAttr(constraint.getMin()));
            Value aboveMinimum = builder.create<LLVM::ICmpOp>(
              internal.getLoc(), LLVM::ICmpPredicate::sge, extent, minimum);
            Value multiple = builder.create<LLVM::ConstantOp>(
              internal.getLoc(),
              builder.getI64IntegerAttr(constraint.getMultipleOf()));
            Value remainder =
              builder.create<LLVM::SRemOp>(internal.getLoc(), extent, multiple);
            Value zero = builder.create<LLVM::ConstantOp>(
              internal.getLoc(), builder.getI64IntegerAttr(0));
            Value divisible = builder.create<LLVM::ICmpOp>(
              internal.getLoc(), LLVM::ICmpPredicate::eq, remainder, zero);
            Value constraintValid = builder.create<LLVM::AndOp>(
              internal.getLoc(), aboveMinimum, divisible);
            Value constraintInvalid = builder.create<LLVM::XOrOp>(
              internal.getLoc(),
              constraintValid,
              builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                               builder.getBoolAttr(true)));
            shapeConstraintInvalid =
              mergeFlag(shapeConstraintInvalid, constraintInvalid);
          }
        }
        const unsigned elementBytes = argumentTypes[functionIndex]
                                        .getElementType()
                                        .getIntOrFloatBitWidth() /
                                      8;
        auto inputBytes = checkedMultiply(
          inputElements,
          builder.create<LLVM::ConstantOp>(
            internal.getLoc(), builder.getI64IntegerAttr(elementBytes)));
        shapeArithmeticInvalid =
          mergeFlag(shapeArithmeticInvalid, inputBytes.second);
      }
    }
    auto loadShapeInputDimension = [&](unsigned inputIndex,
                                       unsigned dimension) -> Value {
      unsigned functionIndex = inputIndices[inputIndex];
      MemRefType type = argumentTypes[functionIndex];
      if (!type.isDynamicDim(dimension)) {
        return builder.create<LLVM::ConstantOp>(
          internal.getLoc(),
          builder.getI64IntegerAttr(type.getShape()[dimension]));
      }
      Value address = builder.create<LLVM::GEPOp>(
        internal.getLoc(),
        pointerType,
        i64Type,
        shapeInputs[inputIndex],
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
      return builder.create<LLVM::LoadOp>(internal.getLoc(), i64Type, address);
    };
    for (const InputDimRelation& relation : inputDimRelations) {
      Value lhs = loadShapeInputDimension(relation.lhsInput, relation.lhsDim);
      Value rhs = loadShapeInputDimension(relation.rhsInput, relation.rhsDim);
      Value offset = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(relation.offset));
      Value sum = builder.create<LLVM::SAddWithOverflowOp>(
        internal.getLoc(), overflowResultType, rhs, offset);
      Value expected = builder.create<LLVM::ExtractValueOp>(
        internal.getLoc(), sum, ArrayRef<int64_t>{0});
      Value overflow = builder.create<LLVM::ExtractValueOp>(
        internal.getLoc(), sum, ArrayRef<int64_t>{1});
      shapeArithmeticInvalid = mergeFlag(shapeArithmeticInvalid, overflow);
      Value mismatch = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::ne, lhs, expected);
      shapeRelationInvalid = mergeFlag(shapeRelationInvalid, mismatch);
    }
    for (auto [outputIndex, functionIndex] :
         llvm::enumerate(outputArgumentIndices)) {
      MemRefType type = argumentTypes[functionIndex];
      if (type.hasStaticShape()) {
        continue;
      }
      Value destination = shapeEntry->getArgument(shapeArgumentIndex++);
      Value outputElements = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(1));
      for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
        Value destinationAddress = builder.create<LLVM::GEPOp>(
          internal.getLoc(),
          pointerType,
          i64Type,
          destination,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
        Value extent;
        if (ShapedType::isDynamic(type.getShape()[dimension])) {
          auto evaluated = evaluateShapeExpression(
            shapePrograms[outputIndex][dimension],
            [&](unsigned inputIndex, unsigned inputDimension) -> Value {
              unsigned inputFunctionIndex = inputIndices[inputIndex];
              MemRefType inputType = argumentTypes[inputFunctionIndex];
              if (!inputType.isDynamicDim(inputDimension)) {
                return builder.create<LLVM::ConstantOp>(
                  internal.getLoc(),
                  builder.getI64IntegerAttr(
                    inputType.getShape()[inputDimension]));
              }
              Value source = shapeInputs[inputIndex];
              Value sourceAddress = builder.create<LLVM::GEPOp>(
                internal.getLoc(),
                pointerType,
                i64Type,
                source,
                ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(inputDimension)});
              return builder.create<LLVM::LoadOp>(
                internal.getLoc(), i64Type, sourceAddress);
            },
            shapeVersions[outputIndex] == 2);
          extent = evaluated.first;
          shapeArithmeticInvalid =
            mergeFlag(shapeArithmeticInvalid, evaluated.second);
        } else {
          extent = builder.create<LLVM::ConstantOp>(
            internal.getLoc(),
            builder.getI64IntegerAttr(type.getShape()[dimension]));
        }
        builder.create<LLVM::StoreOp>(
          internal.getLoc(), extent, destinationAddress);
        Value positive = builder.create<LLVM::ICmpOp>(
          internal.getLoc(),
          LLVM::ICmpPredicate::sgt,
          extent,
          builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                           builder.getI64IntegerAttr(0)));
        Value invalid = builder.create<LLVM::XOrOp>(
          internal.getLoc(),
          positive,
          builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                           builder.getBoolAttr(true)));
        outputShapeInvalid = mergeFlag(outputShapeInvalid, invalid);
        auto multiplied = checkedMultiply(outputElements, extent);
        outputElements = multiplied.first;
        shapeArithmeticInvalid =
          mergeFlag(shapeArithmeticInvalid, multiplied.second);
      }
      const unsigned elementBytes =
        type.getElementType().getIntOrFloatBitWidth() / 8;
      auto outputBytes = checkedMultiply(
        outputElements,
        builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getI64IntegerAttr(elementBytes)));
      shapeArithmeticInvalid =
        mergeFlag(shapeArithmeticInvalid, outputBytes.second);
    }
    if (!shapeInvalid) {
      shapeInvalid = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getBoolAttr(false));
    }
    if (!shapeConstraintInvalid) {
      shapeConstraintInvalid = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getBoolAttr(false));
    }
    if (!shapeArithmeticInvalid) {
      shapeArithmeticInvalid = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getBoolAttr(false));
    }
    if (!shapeRelationInvalid) {
      shapeRelationInvalid = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getBoolAttr(false));
    }
    if (!outputShapeInvalid) {
      outputShapeInvalid = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getBoolAttr(false));
    }
    builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                   shapeInvalid,
                                   shapeInvalidError,
                                   ValueRange{},
                                   shapeValid,
                                   ValueRange{});
    builder.setInsertionPointToStart(shapeValid);
    builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                   shapeConstraintInvalid,
                                   shapeConstraintError,
                                   ValueRange{},
                                   shapeConstraintValid,
                                   ValueRange{});
    builder.setInsertionPointToStart(shapeConstraintValid);
    builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                   shapeArithmeticInvalid,
                                   shapeOverflowError,
                                   ValueRange{},
                                   shapeArithmeticValid,
                                   ValueRange{});
    builder.setInsertionPointToStart(shapeArithmeticValid);
    builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                   shapeRelationInvalid,
                                   shapeInvalidError,
                                   ValueRange{},
                                   shapeRelationValid,
                                   ValueRange{});
    builder.setInsertionPointToStart(shapeRelationValid);
    builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                   outputShapeInvalid,
                                   shapeInvalidError,
                                   ValueRange{},
                                   shapeReturn,
                                   ValueRange{});
    builder.setInsertionPointToStart(shapeReturn);
    Value shapeSuccessStatus = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI32IntegerAttr(0));
    builder.create<LLVM::ReturnOp>(internal.getLoc(), shapeSuccessStatus);
  }
}

}  // namespace mlir::ncnn::capi_detail
