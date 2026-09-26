#include "GenerateCAPIInternal.hpp"

namespace mlir::ncnn::capi_detail {

void emitWrapper(ModuleOp module, FinalizeContext& ctx) {
  auto& exportName = ctx.exportName;
  auto& internalName = ctx.internalName;
  auto& internal = ctx.internal;
  auto& argumentTypes = ctx.argumentTypes;
  auto& outputs = ctx.outputs;
  auto& shapeCarriers = ctx.shapeCarriers;
  auto& shapeVersions = ctx.shapeVersions;
  auto& inputIndices = ctx.inputIndices;
  auto& outputArgumentIndices = ctx.outputArgumentIndices;
  auto& shapePrograms = ctx.shapePrograms;
  auto& constraintsByInput = ctx.constraintsByInput;
  auto& inputDimRelations = ctx.inputDimRelations;
  auto& wrapperOrder = ctx.wrapperOrder;
  auto& builder = ctx.builder;
  auto& wrapper = ctx.wrapper;
  auto& pointerType = ctx.pointerType;
  auto& i64Type = ctx.i64Type;
  auto& overflowResultType = ctx.overflowResultType;
  auto& mergeFlag = ctx.mergeFlag;
  auto& checkedMultiply = ctx.checkedMultiply;
  auto& evaluateShapeExpression = ctx.evaluateShapeExpression;

  LLVMTypeConverter typeConverter(module.getContext());
  pointerType = LLVM::LLVMPointerType::get(module.getContext());
  SmallVector<Type> wrapperTypes;
  SmallVector<bool> wrapperPointers;
  SmallVector<unsigned> wrapperDataIndices(argumentTypes.size());
  SmallVector<unsigned> wrapperShapeIndices(argumentTypes.size());
  SmallVector<unsigned> wrapperCapacityIndices(argumentTypes.size());
  SmallVector<unsigned> wrapperRankIndices(argumentTypes.size());
  for (unsigned functionIndex : wrapperOrder) {
    wrapperDataIndices[functionIndex] = wrapperTypes.size();
    wrapperTypes.push_back(pointerType);
    wrapperPointers.push_back(true);
    if (!outputs.contains(functionIndex) &&
        !argumentTypes[functionIndex].hasStaticShape()) {
      wrapperShapeIndices[functionIndex] = wrapperTypes.size();
      wrapperTypes.push_back(pointerType);
      wrapperPointers.push_back(true);
    }
  }
  for (unsigned functionIndex : wrapperOrder) {
    if (outputs.contains(functionIndex) &&
        !argumentTypes[functionIndex].hasStaticShape() &&
        !shapeCarriers.contains(functionIndex)) {
      wrapperCapacityIndices[functionIndex] = wrapperTypes.size();
      wrapperTypes.push_back(IntegerType::get(module.getContext(), 64));
      wrapperPointers.push_back(false);
    }
  }
  for (unsigned functionIndex : wrapperOrder) {
    if (shapeCarriers.contains(functionIndex)) {
      wrapperCapacityIndices[functionIndex] = wrapperTypes.size();
      wrapperTypes.push_back(IntegerType::get(module.getContext(), 32));
      wrapperPointers.push_back(false);
      wrapperRankIndices[functionIndex] = wrapperTypes.size();
      wrapperTypes.push_back(pointerType);
      wrapperPointers.push_back(true);
    }
  }
  auto wrapperType = LLVM::LLVMFunctionType::get(
    IntegerType::get(module.getContext(), 32), wrapperTypes, false);
  builder = OpBuilder(internal);
  wrapper = builder.create<LLVM::LLVMFuncOp>(internal.getLoc(),
                                             exportName.getValue(),
                                             wrapperType,
                                             LLVM::Linkage::External);
  Block* entry = wrapper.addEntryBlock(builder);
  Block* nonNullBlock = &wrapper.getBody().emplaceBlock();
  Block* invokeBlock = &wrapper.getBody().emplaceBlock();
  Block* errorBlock = &wrapper.getBody().emplaceBlock();
  Block* shapeErrorBlock = &wrapper.getBody().emplaceBlock();
  Block* constraintErrorBlock = &wrapper.getBody().emplaceBlock();
  Block* overflowErrorBlock = &wrapper.getBody().emplaceBlock();
  Block* capacityErrorBlock = &wrapper.getBody().emplaceBlock();
  Block* shapeValidBlock = &wrapper.getBody().emplaceBlock();
  Block* constraintValidBlock = &wrapper.getBody().emplaceBlock();
  Block* arithmeticValidBlock = &wrapper.getBody().emplaceBlock();
  Block* relationValidBlock = &wrapper.getBody().emplaceBlock();
  Block* outputShapeValidBlock = &wrapper.getBody().emplaceBlock();

  builder.setInsertionPointToStart(entry);
  Value nullPointer =
    builder.create<LLVM::ZeroOp>(internal.getLoc(), pointerType);
  Value anyNull;
  for (auto [index, argument] : llvm::enumerate(entry->getArguments())) {
    if (!wrapperPointers[index]) {
      continue;
    }
    Value isNull = builder.create<LLVM::ICmpOp>(
      internal.getLoc(), LLVM::ICmpPredicate::eq, argument, nullPointer);
    anyNull = anyNull
                ? builder.create<LLVM::OrOp>(internal.getLoc(), anyNull, isNull)
                : isNull;
  }
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 anyNull,
                                 errorBlock,
                                 ValueRange{},
                                 nonNullBlock,
                                 ValueRange{});

  builder.setInsertionPointToStart(errorBlock);
  Value failure = builder.create<LLVM::ConstantOp>(
    internal.getLoc(), builder.getI32IntegerAttr(1));
  builder.create<LLVM::ReturnOp>(internal.getLoc(), failure);
  auto emitStatus = [&](Block* block, int32_t status) {
    builder.setInsertionPointToStart(block);
    Value value = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI32IntegerAttr(status));
    builder.create<LLVM::ReturnOp>(internal.getLoc(), value);
  };
  emitStatus(shapeErrorBlock, 2);
  emitStatus(constraintErrorBlock, 3);
  emitStatus(overflowErrorBlock, 4);
  builder.setInsertionPointToStart(capacityErrorBlock);
  Value capacityFailure = builder.create<LLVM::ConstantOp>(
    internal.getLoc(), builder.getI32IntegerAttr(5));
  builder.create<LLVM::ReturnOp>(internal.getLoc(), capacityFailure);

  builder.setInsertionPointToStart(nonNullBlock);
  Value invalidShape;
  Value invalidOutputShape;
  Value invalidConstraint;
  Value arithmeticInvalid;
  Value invalidRelation;
  i64Type = builder.getI64Type();
  overflowResultType = LLVM::LLVMStructType::getLiteral(
    module.getContext(), {i64Type, builder.getI1Type()});
  mergeFlag = [&](Value current, Value next) {
    return current
             ? builder.create<LLVM::OrOp>(internal.getLoc(), current, next)
             : next;
  };
  checkedMultiply = [&](Value lhs, Value rhs) {
    Value result = builder.create<LLVM::UMulWithOverflowOp>(
      internal.getLoc(), overflowResultType, lhs, rhs);
    Value value = builder.create<LLVM::ExtractValueOp>(
      internal.getLoc(), result, ArrayRef<int64_t>{0});
    Value overflow = builder.create<LLVM::ExtractValueOp>(
      internal.getLoc(), result, ArrayRef<int64_t>{1});
    return std::pair<Value, Value>{value, overflow};
  };
  using LoadInputDimension = std::function<Value(unsigned, unsigned)>;
  evaluateShapeExpression = [&](const ShapeExpr& expression,
                                const LoadInputDimension& loadInputDimension,
                                bool floorDivision) {
    std::function<std::pair<Value, Value>(const ShapeExpr&)> evaluate =
      [&](const ShapeExpr& node) -> std::pair<Value, Value> {
      const auto opcode = node.getOpcode();
      if (opcode == ShapeExprOpcode::Constant) {
        Value value = builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getI64IntegerAttr(node.getValue()));
        Value valid = builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getBoolAttr(false));
        return {value, valid};
      }
      if (opcode == ShapeExprOpcode::InputDimension) {
        Value value = loadInputDimension(node.getInput(), node.getValue());
        Value valid = builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getBoolAttr(false));
        return {value, valid};
      }
      auto lhs = evaluate(node.getLhs());
      auto rhs = evaluate(node.getRhs());
      Value invalid = mergeFlag(lhs.second, rhs.second);
      if (opcode == ShapeExprOpcode::Add ||
          opcode == ShapeExprOpcode::Multiply) {
        Value result;
        if (opcode == ShapeExprOpcode::Add) {
          result = builder.create<LLVM::SAddWithOverflowOp>(
            internal.getLoc(), overflowResultType, lhs.first, rhs.first);
        } else {
          result = builder.create<LLVM::SMulWithOverflowOp>(
            internal.getLoc(), overflowResultType, lhs.first, rhs.first);
        }
        Value value = builder.create<LLVM::ExtractValueOp>(
          internal.getLoc(), result, ArrayRef<int64_t>{0});
        invalid = mergeFlag(invalid,
                            builder.create<LLVM::ExtractValueOp>(
                              internal.getLoc(), result, ArrayRef<int64_t>{1}));
        return {value, invalid};
      }
      if (opcode == ShapeExprOpcode::Max) {
        Value lhsGreater = builder.create<LLVM::ICmpOp>(
          internal.getLoc(), LLVM::ICmpPredicate::sgt, lhs.first, rhs.first);
        Value value = builder.create<LLVM::SelectOp>(
          internal.getLoc(), lhsGreater, lhs.first, rhs.first);
        return {value, invalid};
      }
      Value zero = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(0));
      Value one = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(1));
      Value minusOne = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(-1));
      Value minimum = builder.create<LLVM::ConstantOp>(
        internal.getLoc(),
        builder.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
      Value divisorZero = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::eq, rhs.first, zero);
      Value minimumDividend = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::eq, lhs.first, minimum);
      Value negativeOneDivisor = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::eq, rhs.first, minusOne);
      Value divisionOverflow = builder.create<LLVM::AndOp>(
        internal.getLoc(), minimumDividend, negativeOneDivisor);
      Value divisionInvalid = builder.create<LLVM::OrOp>(
        internal.getLoc(), divisorZero, divisionOverflow);
      invalid = mergeFlag(invalid, divisionInvalid);
      Value safeDivisor = builder.create<LLVM::SelectOp>(
        internal.getLoc(), divisionInvalid, one, rhs.first);
      Value quotient =
        builder.create<LLVM::SDivOp>(internal.getLoc(), lhs.first, safeDivisor);
      Value remainder =
        builder.create<LLVM::SRemOp>(internal.getLoc(), lhs.first, safeDivisor);
      Value hasRemainder = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::ne, remainder, zero);
      Value lhsNegative = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::slt, lhs.first, zero);
      Value rhsNegative = builder.create<LLVM::ICmpOp>(
        internal.getLoc(), LLVM::ICmpPredicate::slt, safeDivisor, zero);
      Value signsDiffer = builder.create<LLVM::XOrOp>(
        internal.getLoc(), lhsNegative, rhsNegative);
      Value adjust;
      if (opcode == ShapeExprOpcode::FloorDivide && floorDivision) {
        adjust = builder.create<LLVM::AndOp>(
          internal.getLoc(), hasRemainder, signsDiffer);
      } else if (opcode == ShapeExprOpcode::CeilDivide) {
        Value sameSigns = builder.create<LLVM::XOrOp>(
          internal.getLoc(),
          signsDiffer,
          builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                           builder.getBoolAttr(true)));
        adjust = builder.create<LLVM::AndOp>(
          internal.getLoc(), hasRemainder, sameSigns);
      } else {
        adjust = builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                                  builder.getBoolAttr(false));
      }
      Value adjustment =
        builder.create<LLVM::SelectOp>(internal.getLoc(), adjust, one, zero);
      Value result;
      if (opcode == ShapeExprOpcode::FloorDivide) {
        result =
          builder.create<LLVM::SubOp>(internal.getLoc(), quotient, adjustment);
      } else {
        result =
          builder.create<LLVM::AddOp>(internal.getLoc(), quotient, adjustment);
      }
      return {result, invalid};
    };
    return evaluate(expression);
  };
  for (unsigned functionIndex : wrapperOrder) {
    MemRefType type = argumentTypes[functionIndex];
    if (outputs.contains(functionIndex) || type.hasStaticShape()) {
      continue;
    }
    Value shape = entry->getArgument(wrapperShapeIndices[functionIndex]);
    Value inputElements = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI64IntegerAttr(1));
    for (auto [dimensionIndex, dimension] : llvm::enumerate(type.getShape())) {
      Value address = builder.create<LLVM::GEPOp>(
        internal.getLoc(),
        pointerType,
        i64Type,
        shape,
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimensionIndex)});
      Value size =
        builder.create<LLVM::LoadOp>(internal.getLoc(), i64Type, address);
      Value expected = builder.create<LLVM::ConstantOp>(
        internal.getLoc(),
        builder.getI64IntegerAttr(
          ShapedType::isDynamic(dimension) ? 0 : dimension));
      Value valid = builder.create<LLVM::ICmpOp>(
        internal.getLoc(),
        ShapedType::isDynamic(dimension) ? LLVM::ICmpPredicate::sgt
                                         : LLVM::ICmpPredicate::eq,
        size,
        expected);
      Value invalid = builder.create<LLVM::XOrOp>(
        internal.getLoc(),
        valid,
        builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                         builder.getBoolAttr(true)));
      invalidShape = invalidShape ? builder.create<LLVM::OrOp>(
                                      internal.getLoc(), invalidShape, invalid)
                                  : invalid;
      auto multiplied = checkedMultiply(inputElements, size);
      inputElements = multiplied.first;
      arithmeticInvalid = mergeFlag(arithmeticInvalid, multiplied.second);
      const auto inputIndex = static_cast<unsigned>(
        llvm::find(inputIndices, functionIndex) - inputIndices.begin());
      for (DimConstraintAttr constraint : constraintsByInput[inputIndex]) {
        if (constraint.getDim() != dimensionIndex) {
          continue;
        }
        Value minimum = builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getI64IntegerAttr(constraint.getMin()));
        Value aboveMinimum = builder.create<LLVM::ICmpOp>(
          internal.getLoc(), LLVM::ICmpPredicate::sge, size, minimum);
        Value multiple = builder.create<LLVM::ConstantOp>(
          internal.getLoc(),
          builder.getI64IntegerAttr(constraint.getMultipleOf()));
        Value remainder =
          builder.create<LLVM::SRemOp>(internal.getLoc(), size, multiple);
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
        invalidConstraint = mergeFlag(invalidConstraint, constraintInvalid);
      }
    }
    const unsigned elementBytes =
      type.getElementType().getIntOrFloatBitWidth() / 8;
    auto inputBytes = checkedMultiply(
      inputElements,
      builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(elementBytes)));
    arithmeticInvalid = mergeFlag(arithmeticInvalid, inputBytes.second);
  }
  auto loadWrapperInputDimension = [&](unsigned inputIndex,
                                       unsigned dimension) -> Value {
    unsigned functionIndex = inputIndices[inputIndex];
    MemRefType type = argumentTypes[functionIndex];
    if (!type.isDynamicDim(dimension)) {
      return builder.create<LLVM::ConstantOp>(
        internal.getLoc(),
        builder.getI64IntegerAttr(type.getShape()[dimension]));
    }
    Value shape = entry->getArgument(wrapperShapeIndices[functionIndex]);
    Value address = builder.create<LLVM::GEPOp>(
      internal.getLoc(),
      pointerType,
      i64Type,
      shape,
      ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
    return builder.create<LLVM::LoadOp>(internal.getLoc(), i64Type, address);
  };
  for (const InputDimRelation& relation : inputDimRelations) {
    Value lhs = loadWrapperInputDimension(relation.lhsInput, relation.lhsDim);
    Value rhs = loadWrapperInputDimension(relation.rhsInput, relation.rhsDim);
    Value offset = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI64IntegerAttr(relation.offset));
    Value sum = builder.create<LLVM::SAddWithOverflowOp>(
      internal.getLoc(), overflowResultType, rhs, offset);
    Value expected = builder.create<LLVM::ExtractValueOp>(
      internal.getLoc(), sum, ArrayRef<int64_t>{0});
    Value overflow = builder.create<LLVM::ExtractValueOp>(
      internal.getLoc(), sum, ArrayRef<int64_t>{1});
    arithmeticInvalid = mergeFlag(arithmeticInvalid, overflow);
    Value mismatch = builder.create<LLVM::ICmpOp>(
      internal.getLoc(), LLVM::ICmpPredicate::ne, lhs, expected);
    invalidRelation = mergeFlag(invalidRelation, mismatch);
  }
  Value capacityInvalid;
  for (auto [outputIndex, functionIndex] :
       llvm::enumerate(outputArgumentIndices)) {
    MemRefType type = argumentTypes[functionIndex];
    if (type.hasStaticShape() || shapeCarriers.contains(functionIndex)) {
      continue;
    }
    Value required = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI64IntegerAttr(1));
    for (auto [dimensionIndex, dimension] : llvm::enumerate(type.getShape())) {
      Value extent;
      if (ShapedType::isDynamic(dimension)) {
        auto evaluated = evaluateShapeExpression(
          shapePrograms[outputIndex][dimensionIndex],
          [&](unsigned inputIndex, unsigned inputDimension) -> Value {
            unsigned inputFunctionIndex = inputIndices[inputIndex];
            MemRefType inputType = argumentTypes[inputFunctionIndex];
            if (!inputType.isDynamicDim(inputDimension)) {
              return builder.create<LLVM::ConstantOp>(
                internal.getLoc(),
                builder.getI64IntegerAttr(
                  inputType.getShape()[inputDimension]));
            }
            Value shape =
              entry->getArgument(wrapperShapeIndices[inputFunctionIndex]);
            Value address = builder.create<LLVM::GEPOp>(
              internal.getLoc(),
              pointerType,
              i64Type,
              shape,
              ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(inputDimension)});
            return builder.create<LLVM::LoadOp>(
              internal.getLoc(), i64Type, address);
          },
          shapeVersions[outputIndex] == 2);
        extent = evaluated.first;
        arithmeticInvalid = mergeFlag(arithmeticInvalid, evaluated.second);
      } else {
        extent = builder.create<LLVM::ConstantOp>(
          internal.getLoc(), builder.getI64IntegerAttr(dimension));
      }
      Value nonPositive = builder.create<LLVM::ICmpOp>(
        internal.getLoc(),
        LLVM::ICmpPredicate::sle,
        extent,
        builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                         builder.getI64IntegerAttr(0)));
      Value invalid = nonPositive;
      invalidOutputShape = mergeFlag(invalidOutputShape, invalid);
      auto multiplied = checkedMultiply(required, extent);
      required = multiplied.first;
      arithmeticInvalid = mergeFlag(arithmeticInvalid, multiplied.second);
    }
    Value capacity = entry->getArgument(wrapperCapacityIndices[functionIndex]);
    Value insufficient = builder.create<LLVM::ICmpOp>(
      internal.getLoc(), LLVM::ICmpPredicate::ugt, required, capacity);
    capacityInvalid = mergeFlag(capacityInvalid, insufficient);
    const unsigned elementBytes =
      argumentTypes[functionIndex].getElementType().getIntOrFloatBitWidth() / 8;
    auto byteCount = checkedMultiply(
      required,
      builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(elementBytes)));
    arithmeticInvalid = mergeFlag(arithmeticInvalid, byteCount.second);
  }
  for (unsigned carrierIndex : shapeCarriers) {
    Value capacity = entry->getArgument(wrapperCapacityIndices[carrierIndex]);
    Value required = builder.create<LLVM::ConstantOp>(
      internal.getLoc(),
      builder.getI32IntegerAttr(argumentTypes[carrierIndex].getShape()[0]));
    Value valid = builder.create<LLVM::ICmpOp>(
      internal.getLoc(), LLVM::ICmpPredicate::uge, capacity, required);
    Value invalid = builder.create<LLVM::XOrOp>(
      internal.getLoc(),
      valid,
      builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                       builder.getBoolAttr(true)));
    invalidShape = invalidShape ? builder.create<LLVM::OrOp>(
                                    internal.getLoc(), invalidShape, invalid)
                                : invalid;
  }
  if (!capacityInvalid) {
    capacityInvalid = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getBoolAttr(false));
  }
  if (!invalidShape) {
    invalidShape = builder.create<LLVM::ConstantOp>(internal.getLoc(),
                                                    builder.getBoolAttr(false));
  }
  if (!invalidConstraint) {
    invalidConstraint = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getBoolAttr(false));
  }
  if (!arithmeticInvalid) {
    arithmeticInvalid = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getBoolAttr(false));
  }
  if (!invalidRelation) {
    invalidRelation = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getBoolAttr(false));
  }
  if (!invalidOutputShape) {
    invalidOutputShape = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getBoolAttr(false));
  }
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 invalidShape,
                                 shapeErrorBlock,
                                 ValueRange{},
                                 shapeValidBlock,
                                 ValueRange{});
  builder.setInsertionPointToStart(shapeValidBlock);
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 invalidConstraint,
                                 constraintErrorBlock,
                                 ValueRange{},
                                 constraintValidBlock,
                                 ValueRange{});
  builder.setInsertionPointToStart(constraintValidBlock);
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 arithmeticInvalid,
                                 overflowErrorBlock,
                                 ValueRange{},
                                 arithmeticValidBlock,
                                 ValueRange{});
  builder.setInsertionPointToStart(arithmeticValidBlock);
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 invalidRelation,
                                 shapeErrorBlock,
                                 ValueRange{},
                                 relationValidBlock,
                                 ValueRange{});
  builder.setInsertionPointToStart(relationValidBlock);
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 invalidOutputShape,
                                 shapeErrorBlock,
                                 ValueRange{},
                                 outputShapeValidBlock,
                                 ValueRange{});
  builder.setInsertionPointToStart(outputShapeValidBlock);
  builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                 capacityInvalid,
                                 capacityErrorBlock,
                                 ValueRange{},
                                 invokeBlock,
                                 ValueRange{});

  builder.setInsertionPointToStart(invokeBlock);
  SmallVector<SmallVector<Value>> unpacked(argumentTypes.size());
  SmallVector<Value> inputShapes(inputIndices.size());
  for (unsigned functionIndex : wrapperOrder) {
    MemRefType type = argumentTypes[functionIndex];
    Value data = entry->getArgument(wrapperDataIndices[functionIndex]);
    MemRefDescriptor descriptor = [&] {
      if (type.hasStaticShape()) {
        return MemRefDescriptor::fromStaticShape(
          builder, internal.getLoc(), typeConverter, type, data);
      }

      std::optional<unsigned> outputIndex;
      if (outputs.contains(functionIndex)) {
        auto* outputPosition = llvm::find(outputArgumentIndices, functionIndex);
        outputIndex =
          static_cast<unsigned>(outputPosition - outputArgumentIndices.begin());
      } else {
        Value shape = entry->getArgument(wrapperShapeIndices[functionIndex]);
        auto* inputPosition = llvm::find(inputIndices, functionIndex);
        inputShapes[inputPosition - inputIndices.begin()] = shape;
      }
      auto result = MemRefDescriptor::poison(
        builder, internal.getLoc(), typeConverter.convertType(type));
      result.setAllocatedPtr(builder, internal.getLoc(), data);
      result.setAlignedPtr(builder, internal.getLoc(), data);
      result.setConstantOffset(builder, internal.getLoc(), 0);
      SmallVector<Value> sizes;
      sizes.reserve(type.getRank());
      for (auto [dimensionIndex, dimension] :
           llvm::enumerate(type.getShape())) {
        if (!ShapedType::isDynamic(dimension)) {
          sizes.push_back(builder.create<LLVM::ConstantOp>(
            internal.getLoc(), builder.getI64IntegerAttr(dimension)));
        } else if (!outputIndex) {
          Value shape = entry->getArgument(wrapperShapeIndices[functionIndex]);
          Value address = builder.create<LLVM::GEPOp>(
            internal.getLoc(),
            pointerType,
            i64Type,
            shape,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimensionIndex)});
          sizes.push_back(
            builder.create<LLVM::LoadOp>(internal.getLoc(), i64Type, address));
        } else {
          auto evaluated = evaluateShapeExpression(
            shapePrograms[*outputIndex][dimensionIndex],
            [&](unsigned inputIndex, unsigned inputDimension) -> Value {
              unsigned inputFunctionIndex = inputIndices[inputIndex];
              MemRefType inputType = argumentTypes[inputFunctionIndex];
              if (!inputType.isDynamicDim(inputDimension)) {
                return builder.create<LLVM::ConstantOp>(
                  internal.getLoc(),
                  builder.getI64IntegerAttr(
                    inputType.getShape()[inputDimension]));
              }
              Value shape = inputShapes[inputIndex];
              Value address = builder.create<LLVM::GEPOp>(
                internal.getLoc(),
                pointerType,
                i64Type,
                shape,
                ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(inputDimension)});
              return builder.create<LLVM::LoadOp>(
                internal.getLoc(), i64Type, address);
            },
            shapeVersions[*outputIndex] == 2);
          sizes.push_back(evaluated.first);
        }
        result.setSize(
          builder, internal.getLoc(), dimensionIndex, sizes.back());
      }
      Value stride = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI64IntegerAttr(1));
      for (unsigned reverseIndex = 0; reverseIndex < type.getRank();
           ++reverseIndex) {
        unsigned dimensionIndex = type.getRank() - reverseIndex - 1;
        result.setStride(builder, internal.getLoc(), dimensionIndex, stride);
        stride = checkedMultiply(stride, sizes[dimensionIndex]).first;
      }
      return result;
    }();
    MemRefDescriptor::unpack(builder,
                             internal.getLoc(),
                             descriptor,
                             argumentTypes[functionIndex],
                             unpacked[functionIndex]);
  }
  SmallVector<Value> callArguments;
  for (const auto& values : unpacked) {
    callArguments.append(values);
  }
  builder.create<LLVM::CallOp>(internal.getLoc(),
                               internal.getFunctionType(),
                               internalName.getValue(),
                               callArguments);
  for (unsigned carrierIndex : shapeCarriers) {
    Value rank = builder.create<LLVM::ConstantOp>(
      internal.getLoc(),
      builder.getI32IntegerAttr(argumentTypes[carrierIndex].getShape()[0]));
    builder.create<LLVM::StoreOp>(
      internal.getLoc(),
      rank,
      entry->getArgument(wrapperRankIndices[carrierIndex]));
  }
  Value success = builder.create<LLVM::ConstantOp>(
    internal.getLoc(), builder.getI32IntegerAttr(0));
  builder.create<LLVM::ReturnOp>(internal.getLoc(), success);
}

}  // namespace mlir::ncnn::capi_detail
