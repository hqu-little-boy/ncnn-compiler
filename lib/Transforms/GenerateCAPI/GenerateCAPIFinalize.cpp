#include "GenerateCAPIInternal.hpp"

#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_FINALIZECAPIPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class FinalizeCAPIPass final
  : public impl::FinalizeCAPIPassBase<FinalizeCAPIPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    auto exportName =
      getOperation()->getAttrOfType<StringAttr>(contract::kCApiExportName);
    if (!exportName) {
      return;
    }
    auto rankVariantNames =
      getOperation()->getAttrOfType<ArrayAttr>(contract::kCApiRankVariantNames);
    if (rankVariantNames) {
      auto rankVariantTypes = getOperation()->getAttrOfType<ArrayAttr>(
        contract::kCApiRankVariantTypes);
      if (!rankVariantTypes || rankVariantNames.size() != 4 ||
          rankVariantTypes.size() != 4 ||
          failed(finalizeDynamicRankABI(
            exportName, rankVariantNames, rankVariantTypes))) {
        getOperation().emitError("has invalid dynamic rank C ABI metadata");
        signalPassFailure();
        return;
      }
      getOperation()->removeAttr(contract::kCApiExportName);
      getOperation()->removeAttr(contract::kCApiRankVariantNames);
      getOperation()->removeAttr(contract::kCApiRankVariantTypes);
      return;
    }
    auto internalName =
      getOperation()->getAttrOfType<StringAttr>(contract::kCApiInternalName);
    auto argumentTypeAttrs =
      getOperation()->getAttrOfType<ArrayAttr>(contract::kCApiArgumentTypes);
    auto outputIndices = getOperation()->getAttrOfType<DenseI32ArrayAttr>(
      contract::kCApiOutputIndices);
    auto outputShapeSources = getOperation()->getAttrOfType<DenseI32ArrayAttr>(
      contract::kCApiOutputShapeSources);
    auto outputShapeVersions = getOperation()->getAttrOfType<DenseI32ArrayAttr>(
      contract::kCApiOutputShapeProgramVersions);
    auto outputShapePrograms = getOperation()->getAttrOfType<ArrayAttr>(
      contract::kCApiOutputShapePrograms);
    auto shapeCarrierIndices = getOperation()->getAttrOfType<DenseI32ArrayAttr>(
      contract::kCApiShapeCarrierIndices);
    auto inputShapeConstraints = getOperation()->getAttrOfType<ArrayAttr>(
      contract::kCApiInputShapeConstraints);
    auto inputDimRelationAttrs = getOperation()->getAttrOfType<ArrayAttr>(
      contract::kCApiInputDimRelations);
    if (!inputDimRelationAttrs) {
      inputDimRelationAttrs = ArrayAttr::get(getOperation().getContext(), {});
    }
    auto internal =
      internalName ? getOperation().lookupSymbol<LLVM::LLVMFuncOp>(internalName)
                   : LLVM::LLVMFuncOp();
    if (!internalName || !internal || !argumentTypeAttrs || !outputIndices ||
        !outputShapeSources || !outputShapeVersions || !outputShapePrograms ||
        !shapeCarrierIndices || !inputShapeConstraints) {
      getOperation().emitError("has incomplete prepared ncnn C ABI metadata");
      signalPassFailure();
      return;
    }

    SmallVector<MemRefType> argumentTypes;
    for (Attribute attribute : argumentTypeAttrs) {
      auto typeAttr = dyn_cast<TypeAttr>(attribute);
      auto type =
        typeAttr ? dyn_cast<MemRefType>(typeAttr.getValue()) : MemRefType();
      if (!type) {
        getOperation().emitError("has invalid prepared argument type metadata");
        signalPassFailure();
        return;
      }
      argumentTypes.push_back(type);
    }
    llvm::SmallDenseSet<unsigned> outputs;
    for (int32_t index : outputIndices.asArrayRef()) {
      if (index < 0 || std::cmp_greater_equal(index, argumentTypes.size()) ||
          !outputs.insert(static_cast<unsigned>(index)).second) {
        getOperation().emitError("has invalid prepared output index metadata");
        signalPassFailure();
        return;
      }
    }
    llvm::SmallDenseSet<unsigned> shapeCarriers;
    for (int32_t index : shapeCarrierIndices.asArrayRef()) {
      if (index < 0 || std::cmp_greater_equal(index, argumentTypes.size()) ||
          !outputs.contains(static_cast<unsigned>(index)) ||
          !shapeCarriers.insert(static_cast<unsigned>(index)).second) {
        getOperation().emitError(
          "has invalid prepared shape carrier index metadata");
        signalPassFailure();
        return;
      }
    }
    SmallVector<int32_t> shapeSources(outputShapeSources.asArrayRef());
    SmallVector<int32_t> shapeVersions(outputShapeVersions.asArrayRef());
    SmallVector<unsigned> inputIndices;
    SmallVector<unsigned> outputArgumentIndices;
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      (outputs.contains(index) ? outputArgumentIndices : inputIndices)
        .push_back(index);
    }
    if (shapeSources.size() != outputArgumentIndices.size() ||
        shapeVersions.size() != outputArgumentIndices.size() ||
        outputShapePrograms.size() != outputArgumentIndices.size()) {
      getOperation().emitError("has inconsistent prepared output metadata");
      signalPassFailure();
      return;
    }
    SmallVector<unsigned> inputRanks;
    llvm::transform(
      inputIndices, std::back_inserter(inputRanks), [&](unsigned index) {
        return argumentTypes[index].getRank();
      });
    SmallVector<SmallVector<ShapeExpr>> shapePrograms;
    for (auto [outputIndex, output] : llvm::enumerate(outputShapePrograms)) {
      auto dimensions = dyn_cast<ArrayAttr>(output);
      const MemRefType outputType =
        argumentTypes[outputArgumentIndices[outputIndex]];
      const bool shapeCarrier =
        shapeCarriers.contains(outputArgumentIndices[outputIndex]);
      if (!dimensions ||
          (!shapeCarrier && shapeVersions[outputIndex] != 1 &&
           shapeVersions[outputIndex] != 2) ||
          (shapeCarrier && shapeVersions[outputIndex] != 0) ||
          (outputType.hasStaticShape() && !dimensions.empty()) ||
          (!outputType.hasStaticShape() &&
           dimensions.size() !=
             static_cast<std::size_t>(outputType.getRank()))) {
        getOperation().emitError("has invalid prepared output shape metadata");
        signalPassFailure();
        return;
      }
      SmallVector<ShapeExpr> expressions;
      if (shapeCarrier) {
        const unsigned carrierIndex = outputArgumentIndices[outputIndex];
        const bool paired =
          outputIndex > 0 && carrierIndex > 0 &&
          outputArgumentIndices[outputIndex - 1] == carrierIndex - 1 &&
          !shapeCarriers.contains(carrierIndex - 1);
        const MemRefType dataType =
          paired ? argumentTypes[carrierIndex - 1] : MemRefType();
        if (shapeSources[outputIndex] != -1 || !dimensions.empty() ||
            !outputType.hasStaticShape() || outputType.getRank() != 1 ||
            !outputType.getElementType().isInteger(64) || !dataType ||
            !dataType.hasStaticShape() || shapeSources[outputIndex - 1] != -1 ||
            shapeVersions[outputIndex - 1] != 1 ||
            outputType.getShape()[0] != dataType.getRank()) {
          getOperation().emitError("has invalid shape carrier metadata");
          signalPassFailure();
          return;
        }
        shapePrograms.push_back(std::move(expressions));
        continue;
      }
      if (outputType.hasStaticShape()) {
        if (shapeSources[outputIndex] != -1 ||
            shapeVersions[outputIndex] != 1) {
          getOperation().emitError("has invalid static output shape metadata");
          signalPassFailure();
          return;
        }
        shapePrograms.push_back(std::move(expressions));
        continue;
      }
      if (shapeVersions[outputIndex] == 2) {
        if (shapeSources[outputIndex] != -1) {
          getOperation().emitError("has invalid V2 shape source metadata");
          signalPassFailure();
          return;
        }
        for (Attribute dimension : dimensions) {
          auto values = dyn_cast<DenseI64ArrayAttr>(dimension);
          if (!values) {
            getOperation().emitError("has invalid serialized V2 shape program");
            signalPassFailure();
            return;
          }
          auto expression = ShapeExpr::deserialize(values.asArrayRef());
          if (!expression || !expression->validateInputRanks(inputRanks)) {
            getOperation().emitError("has invalid serialized V2 shape program");
            signalPassFailure();
            return;
          }
          expressions.push_back(std::move(*expression));
        }
        shapePrograms.push_back(std::move(expressions));
        continue;
      }
      if (shapeSources[outputIndex] < 0 ||
          std::cmp_greater_equal(shapeSources[outputIndex],
                                 inputIndices.size())) {
        getOperation().emitError("has out-of-range shape source input");
        signalPassFailure();
        return;
      }
      auto sourceInput = static_cast<unsigned>(shapeSources[outputIndex]);
      for (auto [dimensionIndex, dimension] : llvm::enumerate(dimensions)) {
        auto values = dyn_cast<DenseI64ArrayAttr>(dimension);
        if (!values) {
          getOperation().emitError("has invalid serialized shape program");
          signalPassFailure();
          return;
        }
        auto expression =
          DimensionExpr::deserialize(sourceInput,
                                     static_cast<unsigned>(dimensionIndex),
                                     values.asArrayRef());
        if (!expression) {
          getOperation().emitError("has invalid serialized shape program");
          signalPassFailure();
          return;
        }
        if (expression->getInputDimension() >=
            static_cast<unsigned>(
              argumentTypes[inputIndices[sourceInput]].getRank())) {
          getOperation().emitError("has out-of-range shape source dimension");
          signalPassFailure();
          return;
        }
        expressions.push_back(expression->toV2());
      }
      shapePrograms.push_back(std::move(expressions));
    }
    SmallVector<SmallVector<DimConstraintAttr>> constraintsByInput(
      inputIndices.size());
    for (Attribute attribute : inputShapeConstraints) {
      auto constraint = dyn_cast<DimConstraintAttr>(attribute);
      if (!constraint || constraint.getInput() >= inputIndices.size() ||
          constraint.getDim() >=
            static_cast<uint32_t>(
              argumentTypes[inputIndices[constraint.getInput()]].getRank())) {
        getOperation().emitError(
          "has invalid prepared input constraint metadata");
        signalPassFailure();
        return;
      }
      constraintsByInput[constraint.getInput()].push_back(constraint);
    }
    SmallVector<capi_detail::InputDimRelation> inputDimRelations;
    inputDimRelations.reserve(inputDimRelationAttrs.size());
    for (Attribute attribute : inputDimRelationAttrs) {
      auto values = dyn_cast<DenseI64ArrayAttr>(attribute);
      if (!values || values.size() != 5) {
        getOperation().emitError(
          "has invalid prepared input dimension relation metadata");
        signalPassFailure();
        return;
      }
      ArrayRef<int64_t> relation = values.asArrayRef();
      if (relation[0] < 0 || relation[1] < 0 || relation[2] < 0 ||
          relation[3] < 0 ||
          std::cmp_greater_equal(relation[0], inputIndices.size()) ||
          std::cmp_greater_equal(relation[2], inputIndices.size()) ||
          relation[1] >= argumentTypes[inputIndices[relation[0]]].getRank() ||
          relation[3] >= argumentTypes[inputIndices[relation[2]]].getRank()) {
        getOperation().emitError(
          "has invalid prepared input dimension relation metadata");
        signalPassFailure();
        return;
      }
      inputDimRelations.push_back(
        {.lhsInput = static_cast<unsigned>(relation[0]),
         .lhsDim = static_cast<unsigned>(relation[1]),
         .rhsInput = static_cast<unsigned>(relation[2]),
         .rhsDim = static_cast<unsigned>(relation[3]),
         .offset = relation[4]});
    }
    SmallVector<unsigned> wrapperOrder;
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      if (!outputs.contains(index)) {
        wrapperOrder.push_back(index);
      }
    }
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      if (outputs.contains(index) && !shapeCarriers.contains(index)) {
        wrapperOrder.push_back(index);
      }
    }
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      if (shapeCarriers.contains(index)) {
        wrapperOrder.push_back(index);
      }
    }


    capi_detail::FinalizeContext ctx;
    ctx.exportName = exportName;
    ctx.internalName = internalName;
    ctx.internal = internal;
    ctx.argumentTypes = std::move(argumentTypes);
    ctx.outputs = std::move(outputs);
    ctx.shapeCarriers = std::move(shapeCarriers);
    ctx.shapeVersions = std::move(shapeVersions);
    ctx.inputIndices = std::move(inputIndices);
    ctx.outputArgumentIndices = std::move(outputArgumentIndices);
    ctx.shapePrograms = std::move(shapePrograms);
    ctx.constraintsByInput = std::move(constraintsByInput);
    ctx.inputDimRelations = std::move(inputDimRelations);
    ctx.wrapperOrder = std::move(wrapperOrder);
    capi_detail::emitWrapper(getOperation(), ctx);
    capi_detail::emitInferOutputShapes(ctx);

    getOperation()->removeAttr(contract::kCApiExportName);
    getOperation()->removeAttr(contract::kCApiInternalName);
    getOperation()->removeAttr(contract::kCApiArgumentTypes);
    getOperation()->removeAttr(contract::kCApiOutputIndices);
    getOperation()->removeAttr(contract::kCApiOutputShapeSources);
    getOperation()->removeAttr(contract::kCApiOutputShapePrograms);
    getOperation()->removeAttr(contract::kCApiOutputShapeProgramVersions);
    getOperation()->removeAttr(contract::kCApiShapeCarrierIndices);
    getOperation()->removeAttr(contract::kCApiInputShapeConstraints);
    // Keep relation metadata visible in the generated artifact IR.
  }

 private:
  LogicalResult finalizeDynamicRankABI(StringAttr exportName,
                                       ArrayAttr nameAttrs,
                                       ArrayAttr typeAttrs) {
    SmallVector<LLVM::LLVMFuncOp, 4> internals;
    SmallVector<MemRefType, 4> types;
    for (auto [nameAttr, typeAttr] : llvm::zip(nameAttrs, typeAttrs)) {
      auto name = dyn_cast<StringAttr>(nameAttr);
      auto pair = dyn_cast<ArrayAttr>(typeAttr);
      if (!name || !pair || pair.size() != 2) {
        return failure();
      }
      auto internal = getOperation().lookupSymbol<LLVM::LLVMFuncOp>(name);
      auto input = dyn_cast<MemRefType>(cast<TypeAttr>(pair[0]).getValue());
      auto output = dyn_cast<MemRefType>(cast<TypeAttr>(pair[1]).getValue());
      if (!internal || !input || input != output) {
        return failure();
      }
      internals.push_back(internal);
      types.push_back(input);
    }

    MLIRContext* context = getOperation().getContext();
    OpBuilder builder(internals.front());
    Location location = internals.front().getLoc();
    auto pointerType = LLVM::LLVMPointerType::get(context);
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto statusType = LLVM::LLVMFunctionType::get(
      i32Type,
      {pointerType, pointerType, i32Type, pointerType, i64Type},
      false);
    auto wrapper = builder.create<LLVM::LLVMFuncOp>(
      location, exportName.getValue(), statusType, LLVM::Linkage::External);
    Block* entry = wrapper.addEntryBlock(builder);
    Block* error = &wrapper.getBody().emplaceBlock();
    Block* invalid = &wrapper.getBody().emplaceBlock();
    Block* overflowError = &wrapper.getBody().emplaceBlock();
    Block* capacityError = &wrapper.getBody().emplaceBlock();
    SmallVector<Block*, 4> dispatch;
    SmallVector<Block*, 4> invoke;
    for (unsigned rank = 0; rank < 4; ++rank) {
      dispatch.push_back(&wrapper.getBody().emplaceBlock());
      invoke.push_back(&wrapper.getBody().emplaceBlock());
    }
    auto constantI32 = [&](int32_t value) {
      return builder.create<LLVM::ConstantOp>(location,
                                              builder.getI32IntegerAttr(value));
    };
    auto constantI64 = [&](int64_t value) {
      return builder.create<LLVM::ConstantOp>(location,
                                              builder.getI64IntegerAttr(value));
    };
    builder.setInsertionPointToStart(entry);
    Value null = builder.create<LLVM::ZeroOp>(location, pointerType);
    Value anyNull;
    for (unsigned index : {0u, 1u, 3u}) {
      Value isNull = builder.create<LLVM::ICmpOp>(
        location, LLVM::ICmpPredicate::eq, entry->getArgument(index), null);
      anyNull = anyNull ? builder.create<LLVM::OrOp>(location, anyNull, isNull)
                        : isNull;
    }
    builder.create<LLVM::CondBrOp>(
      location, anyNull, error, ValueRange{}, dispatch.front(), ValueRange{});
    builder.setInsertionPointToStart(error);
    builder.create<LLVM::ReturnOp>(location, constantI32(1));
    builder.setInsertionPointToStart(invalid);
    builder.create<LLVM::ReturnOp>(location, constantI32(2));
    builder.setInsertionPointToStart(overflowError);
    builder.create<LLVM::ReturnOp>(location, constantI32(4));
    builder.setInsertionPointToStart(capacityError);
    builder.create<LLVM::ReturnOp>(location, constantI32(5));

    auto overflowResultType =
      LLVM::LLVMStructType::getLiteral(context, {i64Type, builder.getI1Type()});
    auto checkedMultiply = [&](Value lhs, Value rhs) {
      Value result = builder.create<LLVM::UMulWithOverflowOp>(
        location, overflowResultType, lhs, rhs);
      Value value = builder.create<LLVM::ExtractValueOp>(
        location, result, ArrayRef<int64_t>{0});
      Value overflow = builder.create<LLVM::ExtractValueOp>(
        location, result, ArrayRef<int64_t>{1});
      return std::pair<Value, Value>{value, overflow};
    };

    LLVMTypeConverter converter(context);
    auto makeDescriptor = [&](MemRefType type, Value data, Value shape) {
      auto descriptor = MemRefDescriptor::poison(
        builder, location, converter.convertType(type));
      descriptor.setAllocatedPtr(builder, location, data);
      descriptor.setAlignedPtr(builder, location, data);
      descriptor.setConstantOffset(builder, location, 0);
      SmallVector<Value> sizes;
      for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
        Value address = builder.create<LLVM::GEPOp>(
          location,
          pointerType,
          i64Type,
          shape,
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
        Value extent = builder.create<LLVM::LoadOp>(location, i64Type, address);
        descriptor.setSize(builder, location, dimension, extent);
        sizes.push_back(extent);
      }
      Value stride = constantI64(1);
      for (unsigned reverse = 0; reverse < type.getRank(); ++reverse) {
        unsigned dimension = type.getRank() - reverse - 1;
        descriptor.setStride(builder, location, dimension, stride);
        stride =
          builder.create<LLVM::MulOp>(location, stride, sizes[dimension]);
      }
      SmallVector<Value> unpacked;
      MemRefDescriptor::unpack(builder, location, descriptor, type, unpacked);
      return unpacked;
    };

    for (unsigned index = 0; index < 4; ++index) {
      const unsigned rank = index + 1;
      builder.setInsertionPointToStart(dispatch[index]);
      Value matches =
        builder.create<LLVM::ICmpOp>(location,
                                     LLVM::ICmpPredicate::eq,
                                     entry->getArgument(2),
                                     constantI32(static_cast<int32_t>(rank)));
      builder.create<LLVM::CondBrOp>(location,
                                     matches,
                                     invoke[index],
                                     ValueRange{},
                                     index == 3 ? invalid : dispatch[index + 1],
                                     ValueRange{});
      builder.setInsertionPointToStart(invoke[index]);
      Value invalidExtent;
      Value arithmeticOverflow;
      Value required = constantI64(1);
      for (unsigned dimension = 0; dimension < rank; ++dimension) {
        Value address = builder.create<LLVM::GEPOp>(
          location,
          pointerType,
          i64Type,
          entry->getArgument(1),
          ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
        Value extent = builder.create<LLVM::LoadOp>(location, i64Type, address);
        Value invalid = builder.create<LLVM::ICmpOp>(
          location, LLVM::ICmpPredicate::sle, extent, constantI64(0));
        invalidExtent = invalidExtent ? builder.create<LLVM::OrOp>(
                                          location, invalidExtent, invalid)
                                      : invalid;
        auto multiplied = checkedMultiply(required, extent);
        required = multiplied.first;
        arithmeticOverflow =
          arithmeticOverflow
            ? builder.create<LLVM::OrOp>(
                location, arithmeticOverflow, multiplied.second)
            : multiplied.second;
      }
      auto byteCount = checkedMultiply(
        required,
        constantI64(types[index].getElementType().getIntOrFloatBitWidth() / 8));
      arithmeticOverflow = builder.create<LLVM::OrOp>(
        location, arithmeticOverflow, byteCount.second);
      Block* call = &wrapper.getBody().emplaceBlock();
      Block* extentValid = &wrapper.getBody().emplaceBlock();
      Block* arithmeticValid = &wrapper.getBody().emplaceBlock();
      builder.create<LLVM::CondBrOp>(location,
                                     invalidExtent,
                                     invalid,
                                     ValueRange{},
                                     extentValid,
                                     ValueRange{});
      builder.setInsertionPointToStart(extentValid);
      builder.create<LLVM::CondBrOp>(location,
                                     arithmeticOverflow,
                                     overflowError,
                                     ValueRange{},
                                     arithmeticValid,
                                     ValueRange{});
      builder.setInsertionPointToStart(arithmeticValid);
      Value insufficient = builder.create<LLVM::ICmpOp>(
        location, LLVM::ICmpPredicate::ugt, required, entry->getArgument(4));
      builder.create<LLVM::CondBrOp>(location,
                                     insufficient,
                                     capacityError,
                                     ValueRange{},
                                     call,
                                     ValueRange{});
      builder.setInsertionPointToStart(call);
      SmallVector<Value> arguments = makeDescriptor(
        types[index], entry->getArgument(0), entry->getArgument(1));
      SmallVector<Value> output = makeDescriptor(
        types[index], entry->getArgument(3), entry->getArgument(1));
      arguments.append(output);
      builder.create<LLVM::CallOp>(location,
                                   internals[index].getFunctionType(),
                                   internals[index].getName(),
                                   arguments);
      builder.create<LLVM::ReturnOp>(location, constantI32(0));
    }

    auto inferType = LLVM::LLVMFunctionType::get(
      i32Type,
      {pointerType, i32Type, pointerType, i32Type, pointerType},
      false);
    builder.setInsertionPointAfter(wrapper);
    auto infer = builder.create<LLVM::LLVMFuncOp>(
      location,
      (exportName.getValue() + "_infer_output_shapes").str(),
      inferType,
      LLVM::Linkage::External);
    Block* inferEntry = infer.addEntryBlock(builder);
    Block* inferError = &infer.getBody().emplaceBlock();
    Block* inferInvalidShape = &infer.getBody().emplaceBlock();
    Block* inferRankError = &infer.getBody().emplaceBlock();
    Block* inferCopy = &infer.getBody().emplaceBlock();
    builder.setInsertionPointToStart(inferEntry);
    Value inferNull = builder.create<LLVM::ZeroOp>(location, pointerType);
    Value inferInvalid;
    for (unsigned index : {0u, 2u, 4u}) {
      Value isNull =
        builder.create<LLVM::ICmpOp>(location,
                                     LLVM::ICmpPredicate::eq,
                                     inferEntry->getArgument(index),
                                     inferNull);
      inferInvalid = inferInvalid ? builder.create<LLVM::OrOp>(
                                      location, inferInvalid, isNull)
                                  : isNull;
    }
    Value rankLow = builder.create<LLVM::ICmpOp>(location,
                                                 LLVM::ICmpPredicate::ult,
                                                 inferEntry->getArgument(1),
                                                 constantI32(1));
    Value rankHigh = builder.create<LLVM::ICmpOp>(location,
                                                  LLVM::ICmpPredicate::ugt,
                                                  inferEntry->getArgument(1),
                                                  constantI32(4));
    Value capacitySmall =
      builder.create<LLVM::ICmpOp>(location,
                                   LLVM::ICmpPredicate::ult,
                                   inferEntry->getArgument(3),
                                   inferEntry->getArgument(1));
    Value invalidRank = builder.create<LLVM::OrOp>(location, rankLow, rankHigh);
    invalidRank =
      builder.create<LLVM::OrOp>(location, invalidRank, capacitySmall);
    Value nullInvalid = inferInvalid;
    builder.create<LLVM::CondBrOp>(location,
                                   nullInvalid,
                                   inferError,
                                   ValueRange{},
                                   inferInvalidShape,
                                   ValueRange{});
    builder.setInsertionPointToStart(inferError);
    builder.create<LLVM::ReturnOp>(location, constantI32(1));
    builder.setInsertionPointToStart(inferInvalidShape);
    builder.create<LLVM::CondBrOp>(location,
                                   invalidRank,
                                   inferRankError,
                                   ValueRange{},
                                   inferCopy,
                                   ValueRange{});
    builder.setInsertionPointToStart(inferRankError);
    builder.create<LLVM::ReturnOp>(location, constantI32(2));
    builder.setInsertionPointToStart(inferCopy);
    for (unsigned dimension = 0; dimension < 4; ++dimension) {
      Value active = builder.create<LLVM::ICmpOp>(
        location,
        LLVM::ICmpPredicate::ugt,
        inferEntry->getArgument(1),
        constantI32(static_cast<int32_t>(dimension)));
      Block* store = &infer.getBody().emplaceBlock();
      Block* next = &infer.getBody().emplaceBlock();
      builder.create<LLVM::CondBrOp>(
        location, active, store, ValueRange{}, next, ValueRange{});
      builder.setInsertionPointToStart(store);
      Value source = builder.create<LLVM::GEPOp>(
        location,
        pointerType,
        i64Type,
        inferEntry->getArgument(0),
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
      Value destination = builder.create<LLVM::GEPOp>(
        location,
        pointerType,
        i64Type,
        inferEntry->getArgument(2),
        ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
      Value extent = builder.create<LLVM::LoadOp>(location, i64Type, source);
      builder.create<LLVM::StoreOp>(location, extent, destination);
      Value invalid = builder.create<LLVM::ICmpOp>(
        location, LLVM::ICmpPredicate::sle, extent, constantI64(0));
      builder.create<LLVM::CondBrOp>(
        location, invalid, inferRankError, ValueRange{}, next, ValueRange{});
      builder.setInsertionPointToStart(next);
    }
    builder.create<LLVM::StoreOp>(
      location, inferEntry->getArgument(1), inferEntry->getArgument(4));
    builder.create<LLVM::ReturnOp>(location, constantI32(0));
    return success();
  }
};

}  // namespace

}  // namespace mlir::ncnn
