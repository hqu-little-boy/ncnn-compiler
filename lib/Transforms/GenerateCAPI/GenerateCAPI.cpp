#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"

#include <cctype>
#include <cstdint>
#include <format>
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

namespace mlir::ncnn {

#define GEN_PASS_DEF_FINALIZECAPIPASS
#define GEN_PASS_DEF_GENERATECAPIPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

struct ArgumentInfo {
  unsigned functionIndex;
  MemRefType type;
  bool output;
  bool shapeCarrier;
  uint32_t dataDependentDimMask;
};

std::optional<StringRef> abiElementType(Type type) {
  if (type.isF16()) {
    return "f16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (type.isF32()) {
    return "f32";
  }
  if (type.isF64()) {
    return "f64";
  }
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer ||
      !llvm::is_contained({8u, 16u, 32u, 64u}, integer.getWidth())) {
    return std::nullopt;
  }
  switch (integer.getWidth()) {
    case 8:
      return integer.isUnsigned() ? "ui8" : "i8";
    case 16:
      return integer.isUnsigned() ? "ui16" : "i16";
    case 32:
      return integer.isUnsigned() ? "ui32" : "i32";
    case 64:
      return integer.isUnsigned() ? "ui64" : "i64";
    default:
      llvm_unreachable("validated integer width");
  }
}

constexpr StringLiteral kExportNameAttr = "ncnn.c_api.export_name";
constexpr StringLiteral kInternalNameAttr = "ncnn.c_api.internal_name";
constexpr StringLiteral kArgumentTypesAttr = "ncnn.c_api.argument_types";
constexpr StringLiteral kOutputIndicesAttr = "ncnn.c_api.output_indices";
constexpr StringLiteral kOutputShapeSourcesAttr =
  "ncnn.c_api.output_shape_sources";
constexpr StringLiteral kOutputShapeProgramsAttr =
  "ncnn.c_api.output_shape_programs";
constexpr StringLiteral kShapeCarrierIndicesAttr =
  "ncnn.c_api.shape_carrier_indices";
constexpr StringLiteral kRankVariantNamesAttr = "ncnn.c_api.rank_variant_names";
constexpr StringLiteral kRankVariantTypesAttr = "ncnn.c_api.rank_variant_types";

bool isCIdentifier(StringRef name) {
  if (name.empty() ||
      !(std::isalpha(static_cast<unsigned char>(name.front())) ||
        name.front() == '_')) {
    return false;
  }
  return llvm::all_of(name.drop_front(), [](char character) {
    return std::isalnum(static_cast<unsigned char>(character)) ||
           character == '_';
  });
}

class GenerateCAPIPass final
  : public impl::GenerateCAPIPassBase<GenerateCAPIPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    func::FuncOp nestedEntry;
    getOperation().walk([&](func::FuncOp function) {
      if (function->hasAttr("ncnn.entry_point") &&
          function->getParentOp() != getOperation()) {
        nestedEntry = function;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (nestedEntry) {
      nestedEntry.emitOpError(
        "must be a top-level function in the pass module");
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> entries;
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function->hasAttr("ncnn.entry_point")) {
        entries.push_back(function);
      }
    }
    if (entries.size() == 4 && llvm::all_of(entries, [](func::FuncOp function) {
          return function->hasAttr("ncnn.dynamic_rank");
        })) {
      if (!isCIdentifier(exportName) ||
          failed(prepareDynamicRankABI(entries))) {
        signalPassFailure();
      }
      return;
    }
    if (entries.size() != 1) {
      getOperation().emitError()
        << "expected exactly one ncnn.entry_point, got " << entries.size();
      signalPassFailure();
      return;
    }
    if (!isCIdentifier(exportName)) {
      entries.front().emitOpError(
        "requires export-name to be a valid C identifier");
      signalPassFailure();
      return;
    }

    if (failed(prepareABI(entries.front()))) {
      signalPassFailure();
    }
  }

 private:
  LogicalResult prepareDynamicRankABI(ArrayRef<func::FuncOp> functions) {
    SmallVector<func::FuncOp, 4> variants(4);
    Type elementType;
    for (func::FuncOp function : functions) {
      auto rankAttr = function->getAttrOfType<IntegerAttr>("ncnn.rank_variant");
      if (!rankAttr || rankAttr.getInt() < 1 || rankAttr.getInt() > 4 ||
          variants[rankAttr.getInt() - 1]) {
        return function.emitOpError(
          "requires unique rank variants 1 through 4");
      }
      if (function.getNumArguments() != 2 || function.getNumResults() != 0 ||
          !function.getArgAttr(1, "bufferize.result")) {
        return function.emitOpError(
          "dynamic rank currently requires one input and one output");
      }
      auto input = dyn_cast<MemRefType>(function.getArgumentTypes()[0]);
      auto output = dyn_cast<MemRefType>(function.getArgumentTypes()[1]);
      const auto rank = static_cast<unsigned>(rankAttr.getInt());
      if (!input || !output || input.getRank() != rank ||
          output.getRank() != rank || input != output ||
          input.hasStaticShape() || !input.getLayout().isIdentity() ||
          !llvm::all_of(input.getShape(), ShapedType::isDynamic) ||
          !abiElementType(input.getElementType())) {
        return function.emitOpError(
          "dynamic rank variant must be a shape-preserving, fully dynamic, "
          "identity-layout ranked memref specialization");
      }
      auto source =
        function.getArgAttrOfType<IntegerAttr>(1, "ncnn.shape_source_input");
      auto program =
        function.getArgAttrOfType<ArrayAttr>(1, "ncnn.shape_program");
      if (!source || source.getInt() != 0 || !program ||
          program.size() != rank || llvm::any_of(program, [](Attribute attr) {
            auto values = dyn_cast<DenseI64ArrayAttr>(attr);
            return !values || !values.empty();
          })) {
        return function.emitOpError(
          "dynamic rank output must preserve input rank and shape");
      }
      if (elementType && elementType != input.getElementType()) {
        return function.emitOpError(
          "dynamic rank variants must use one element type");
      }
      elementType = input.getElementType();
      variants[rank - 1] = function;
    }

    llvm::json::Object input;
    input["name"] = "input1";
    input["shape"] = llvm::json::Array{};
    input["element_type"] = *abiElementType(elementType);
    input["dynamic_dim_mask"] = 0;
    input["dynamic_rank"] = true;
    input["rank_min"] = 1;
    input["rank_max"] = 4;
    llvm::json::Object output;
    output["name"] = "output1";
    output["shape"] = llvm::json::Array{};
    output["element_type"] = *abiElementType(elementType);
    output["dynamic_dim_mask"] = 0;
    output["dynamic_rank"] = true;
    output["rank_min"] = 1;
    output["rank_max"] = 4;
    output["shape_depends_on_data"] = false;
    output["shape_source_input"] = 0;
    SmallVector<llvm::json::Object> inputs;
    SmallVector<llvm::json::Object> outputs;
    inputs.push_back(std::move(input));
    outputs.push_back(std::move(output));
    if (failed(writeManifest(std::move(inputs), std::move(outputs)))) {
      return failure();
    }

    SymbolTable symbols(getOperation());
    Builder builder(getOperation().getContext());
    SmallVector<Attribute> names;
    SmallVector<Attribute> types;
    for (auto [index, function] : llvm::enumerate(variants)) {
      const std::string name = std::format(
        "__ncnn_internal_{}_rank{}", exportName.getValue(), index + 1);
      if (failed(symbols.rename(function, name))) {
        return function.emitOpError(
          "cannot rename dynamic rank specialization");
      }
      function.setPrivate();
      function->removeAttr("llvm.emit_c_interface");
      function->removeAttr("ncnn.entry_point");
      names.push_back(builder.getStringAttr(name));
      types.push_back(
        builder.getArrayAttr({TypeAttr::get(function.getArgumentTypes()[0]),
                              TypeAttr::get(function.getArgumentTypes()[1])}));
    }
    getOperation()->setAttr(kExportNameAttr, builder.getStringAttr(exportName));
    getOperation()->setAttr(kRankVariantNamesAttr, builder.getArrayAttr(names));
    getOperation()->setAttr(kRankVariantTypesAttr, builder.getArrayAttr(types));
    return success();
  }

  LogicalResult prepareABI(func::FuncOp function) {
    if (function.getNumResults() != 0) {
      return function.emitOpError("must have no results before C ABI wrapping");
    }
    Operation* existing =
      SymbolTable::lookupSymbolIn(getOperation(), exportName);
    if (existing && existing != function.getOperation()) {
      return function.emitOpError()
             << "cannot export duplicate symbol '" << exportName << "'";
    }

    SmallVector<ArgumentInfo> inputs;
    SmallVector<ArgumentInfo> outputs;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      auto type = dyn_cast<MemRefType>(function.getArgumentTypes()[index]);
      if (!type || !type.getLayout().isIdentity() ||
          !abiElementType(type.getElementType())) {
        return function.emitOpError()
               << "argument " << index
               << " must be a ranked identity-layout memref with a supported "
                  "C ABI element type";
      }
      if (type.getRank() > 32) {
        return function.emitOpError()
               << "argument " << index
               << " rank exceeds the C ABI dynamic-dimension mask capacity";
      }
      auto dataDependentMask = function.getArgAttrOfType<IntegerAttr>(
        index, "ncnn.data_dependent_dim_mask");
      ArgumentInfo info{
        .functionIndex = index,
        .type = type,
        .output =
          static_cast<bool>(function.getArgAttr(index, "bufferize.result")),
        .shapeCarrier =
          static_cast<bool>(function.getArgAttr(index, "ncnn.shape_carrier")),
        .dataDependentDimMask =
          dataDependentMask ? static_cast<uint32_t>(dataDependentMask.getInt())
                            : 0};
      if (info.shapeCarrier && !info.output) {
        return function.emitOpError()
               << "argument " << index
               << " shape carrier must be a bufferize.result output";
      }
      (info.output ? outputs : inputs).push_back(info);
    }
    if (inputs.empty() || outputs.empty()) {
      return function.emitOpError("requires at least one input and one output");
    }
    SmallVector<int32_t> outputShapeSources;
    SmallVector<Attribute> outputShapePrograms;
    for (const ArgumentInfo& output : outputs) {
      if (output.shapeCarrier) {
        outputShapeSources.push_back(-1);
        Builder builder(function.getContext());
        outputShapePrograms.push_back(builder.getArrayAttr({}));
        continue;
      }
      if (output.dataDependentDimMask != 0) {
        const uint32_t validMask =
          output.type.getRank() == 32
            ? UINT32_MAX
            : (UINT32_C(1) << output.type.getRank()) - 1;
        if (!output.type.hasStaticShape() ||
            (output.dataDependentDimMask & ~validMask) != 0 ||
            output.functionIndex + 1 >= function.getNumArguments()) {
          return function.emitOpError()
                 << "output " << output.functionIndex
                 << " has an invalid data-dependent shape contract";
        }
        unsigned carrierIndex = output.functionIndex + 1;
        auto carrierType =
          dyn_cast<MemRefType>(function.getArgumentTypes()[carrierIndex]);
        if (!function.getArgAttr(carrierIndex, "bufferize.result") ||
            !function.getArgAttr(carrierIndex, "ncnn.shape_carrier") ||
            !carrierType || !carrierType.hasStaticShape() ||
            !carrierType.getElementType().isInteger(64) ||
            carrierType.getRank() != 1 ||
            carrierType.getShape()[0] != output.type.getRank()) {
          return function.emitOpError()
                 << "output " << output.functionIndex
                 << " must be followed by an i64 shape carrier of its rank";
        }
      }
      int32_t source = -1;
      if (auto attribute = function.getArgAttrOfType<IntegerAttr>(
            output.functionIndex, "ncnn.shape_source_input")) {
        source = static_cast<int32_t>(attribute.getInt());
      }
      if (!output.type.hasStaticShape() &&
          (source < 0 || static_cast<std::size_t>(source) >= inputs.size() ||
           inputs[source].type.getRank() != output.type.getRank() ||
           inputs[source].type.hasStaticShape())) {
        return function.emitOpError()
               << "output " << output.functionIndex
               << " has dynamic extents without a valid input shape source";
      }
      outputShapeSources.push_back(source);
      auto program = function.getArgAttrOfType<ArrayAttr>(output.functionIndex,
                                                          "ncnn.shape_program");
      if (!output.type.hasStaticShape() &&
          (!program ||
           program.size() != static_cast<std::size_t>(output.type.getRank()) ||
           llvm::any_of(program, [](Attribute dimension) {
             auto instructions = dyn_cast<DenseI64ArrayAttr>(dimension);
             if (!instructions || instructions.size() % 2 != 0) {
               return true;
             }
             ArrayRef<int64_t> values = instructions.asArrayRef();
             for (unsigned index = 0; index < values.size(); index += 2) {
               if (values[index] < 0 || values[index] > 2 ||
                   (values[index] == 2 && values[index + 1] <= 0)) {
                 return true;
               }
             }
             return false;
           }))) {
        return function.emitOpError()
               << "output " << output.functionIndex
               << " has an invalid dynamic shape program";
      }
      Builder builder(function.getContext());
      outputShapePrograms.push_back(program ? static_cast<Attribute>(program)
                                            : builder.getArrayAttr({}));
    }

    SmallVector<Attribute> argumentTypes;
    SmallVector<int32_t> outputIndices;
    SmallVector<int32_t> shapeCarrierIndices;
    SmallVector<llvm::json::Object> inputManifest;
    SmallVector<llvm::json::Object> outputManifest;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      argumentTypes.push_back(
        TypeAttr::get(function.getArgumentTypes()[index]));
    }
    SmallVector<ArgumentInfo> wrapperArguments(inputs);
    wrapperArguments.append(outputs);
    unsigned outputInfoIndex = 0;
    for (const ArgumentInfo& info : wrapperArguments) {
      if (info.output) {
        outputIndices.push_back(static_cast<int32_t>(info.functionIndex));
      }
      if (info.shapeCarrier) {
        shapeCarrierIndices.push_back(static_cast<int32_t>(info.functionIndex));
        ++outputInfoIndex;
        continue;
      }
      llvm::json::Array shape;
      llvm::json::Array maximumShape;
      for (auto [dimensionIndex, dimension] :
           llvm::enumerate(info.type.getShape())) {
        maximumShape.push_back(dimension);
        shape.push_back(
          (info.dataDependentDimMask & (UINT32_C(1) << dimensionIndex)) != 0
            ? -1
          : ShapedType::isDynamic(dimension) ? -1
                                             : dimension);
      }
      llvm::json::Object argument;
      argument["name"] = std::format(
        "{}{}",
        info.output ? "output" : "input",
        info.output ? outputManifest.size() + 1 : inputManifest.size() + 1);
      argument["shape"] = std::move(shape);
      argument["element_type"] = *abiElementType(info.type.getElementType());
      uint32_t dynamicDimMask = 0;
      for (auto [index, dimension] : llvm::enumerate(info.type.getShape())) {
        if (ShapedType::isDynamic(dimension) ||
            (info.dataDependentDimMask & (UINT32_C(1) << index)) != 0) {
          dynamicDimMask |= UINT32_C(1) << index;
        }
      }
      argument["dynamic_dim_mask"] = dynamicDimMask;
      if (info.output) {
        const bool dataDependent = info.dataDependentDimMask != 0;
        argument["shape_depends_on_data"] = dataDependent;
        if (dataDependent) {
          argument["maximum_shape"] = std::move(maximumShape);
        }
        if (outputShapeSources[outputInfoIndex] >= 0) {
          argument["shape_source_input"] = outputShapeSources[outputInfoIndex];
        }
        ++outputInfoIndex;
      }
      (info.output ? outputManifest : inputManifest)
        .push_back(std::move(argument));
    }

    const std::string internalName =
      std::format("__ncnn_internal_{}", exportName.getValue());
    if (SymbolTable::lookupSymbolIn(getOperation(), internalName)) {
      return function.emitOpError()
             << "cannot create duplicate internal symbol '" << internalName
             << "'";
    }
    if (failed(
          writeManifest(std::move(inputManifest), std::move(outputManifest)))) {
      return failure();
    }

    SymbolTable symbolTable(getOperation());
    if (failed(symbolTable.rename(function, internalName))) {
      return function.emitOpError(
        "cannot update all symbol uses for internal C ABI name");
    }
    function.setPrivate();
    function->removeAttr("llvm.emit_c_interface");
    function->removeAttr("ncnn.entry_point");
    Builder builder(function.getContext());
    getOperation()->setAttr(kExportNameAttr, builder.getStringAttr(exportName));
    getOperation()->setAttr(kInternalNameAttr,
                            builder.getStringAttr(internalName));
    getOperation()->setAttr(kArgumentTypesAttr,
                            builder.getArrayAttr(argumentTypes));
    getOperation()->setAttr(kOutputIndicesAttr,
                            builder.getDenseI32ArrayAttr(outputIndices));
    getOperation()->setAttr(kOutputShapeSourcesAttr,
                            builder.getDenseI32ArrayAttr(outputShapeSources));
    getOperation()->setAttr(kOutputShapeProgramsAttr,
                            builder.getArrayAttr(outputShapePrograms));
    getOperation()->setAttr(kShapeCarrierIndicesAttr,
                            builder.getDenseI32ArrayAttr(shapeCarrierIndices));
    return success();
  }

  LogicalResult writeManifest(SmallVector<llvm::json::Object> inputs,
                              SmallVector<llvm::json::Object> outputs) {
    if (manifestPath.empty()) {
      return success();
    }
    std::string error;
    auto stream = openOutputFile(manifestPath, &error);
    if (!stream) {
      getOperation().emitError()
        << "cannot open ABI manifest '" << manifestPath << "': " << error;
      return failure();
    }
    llvm::json::Array inputArray;
    llvm::json::Array outputArray;
    for (auto& input : inputs) {
      inputArray.push_back(std::move(input));
    }
    for (auto& output : outputs) {
      outputArray.push_back(std::move(output));
    }
    llvm::json::Object manifest;
    manifest["function"] = exportName;
    manifest["inputs"] = std::move(inputArray);
    manifest["outputs"] = std::move(outputArray);
    stream->os() << llvm::formatv("{0:2}\n",
                                  llvm::json::Value(std::move(manifest)));
    stream->os().close();
    if (stream->os().has_error()) {
      std::error_code error = stream->os().error();
      stream->os().clear_error();
      getOperation().emitError() << "cannot write ABI manifest '"
                                 << manifestPath << "': " << error.message();
      return failure();
    }
    stream->keep();
    return success();
  }
};

class FinalizeCAPIPass final
  : public impl::FinalizeCAPIPassBase<FinalizeCAPIPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    auto exportName =
      getOperation()->getAttrOfType<StringAttr>(kExportNameAttr);
    if (!exportName) {
      return;
    }
    auto rankVariantNames =
      getOperation()->getAttrOfType<ArrayAttr>(kRankVariantNamesAttr);
    if (rankVariantNames) {
      auto rankVariantTypes =
        getOperation()->getAttrOfType<ArrayAttr>(kRankVariantTypesAttr);
      if (!rankVariantTypes || rankVariantNames.size() != 4 ||
          rankVariantTypes.size() != 4 ||
          failed(finalizeDynamicRankABI(
            exportName, rankVariantNames, rankVariantTypes))) {
        getOperation().emitError("has invalid dynamic rank C ABI metadata");
        signalPassFailure();
        return;
      }
      getOperation()->removeAttr(kExportNameAttr);
      getOperation()->removeAttr(kRankVariantNamesAttr);
      getOperation()->removeAttr(kRankVariantTypesAttr);
      return;
    }
    auto internalName =
      getOperation()->getAttrOfType<StringAttr>(kInternalNameAttr);
    auto argumentTypeAttrs =
      getOperation()->getAttrOfType<ArrayAttr>(kArgumentTypesAttr);
    auto outputIndices =
      getOperation()->getAttrOfType<DenseI32ArrayAttr>(kOutputIndicesAttr);
    auto outputShapeSources =
      getOperation()->getAttrOfType<DenseI32ArrayAttr>(kOutputShapeSourcesAttr);
    auto outputShapePrograms =
      getOperation()->getAttrOfType<ArrayAttr>(kOutputShapeProgramsAttr);
    auto shapeCarrierIndices = getOperation()->getAttrOfType<DenseI32ArrayAttr>(
      kShapeCarrierIndicesAttr);
    auto internal = getOperation().lookupSymbol<LLVM::LLVMFuncOp>(internalName);
    if (!internal || !argumentTypeAttrs || !outputIndices ||
        !outputShapeSources || !outputShapePrograms || !shapeCarrierIndices) {
      getOperation().emitError("has incomplete prepared ncnn C ABI metadata");
      signalPassFailure();
      return;
    }

    SmallVector<MemRefType> argumentTypes;
    for (Attribute attribute : argumentTypeAttrs) {
      argumentTypes.push_back(
        cast<MemRefType>(cast<TypeAttr>(attribute).getValue()));
    }
    llvm::SmallDenseSet<unsigned> outputs;
    for (int32_t index : outputIndices.asArrayRef()) {
      outputs.insert(static_cast<unsigned>(index));
    }
    llvm::SmallDenseSet<unsigned> shapeCarriers;
    for (int32_t index : shapeCarrierIndices.asArrayRef()) {
      shapeCarriers.insert(static_cast<unsigned>(index));
    }
    SmallVector<int32_t> shapeSources(outputShapeSources.asArrayRef());
    SmallVector<SmallVector<SmallVector<int64_t>>> shapePrograms;
    for (Attribute output : outputShapePrograms) {
      SmallVector<SmallVector<int64_t>> dimensions;
      for (Attribute dimension : cast<ArrayAttr>(output)) {
        dimensions.emplace_back(
          cast<DenseI64ArrayAttr>(dimension).asArrayRef());
      }
      shapePrograms.push_back(std::move(dimensions));
    }
    SmallVector<unsigned> inputIndices;
    SmallVector<unsigned> outputArgumentIndices;
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      (outputs.contains(index) ? outputArgumentIndices : inputIndices)
        .push_back(index);
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

    LLVMTypeConverter typeConverter(getOperation().getContext());
    auto pointerType = LLVM::LLVMPointerType::get(getOperation().getContext());
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
      if (shapeCarriers.contains(functionIndex)) {
        wrapperCapacityIndices[functionIndex] = wrapperTypes.size();
        wrapperTypes.push_back(
          IntegerType::get(getOperation().getContext(), 32));
        wrapperPointers.push_back(false);
        wrapperRankIndices[functionIndex] = wrapperTypes.size();
        wrapperTypes.push_back(pointerType);
        wrapperPointers.push_back(true);
      }
    }
    auto wrapperType = LLVM::LLVMFunctionType::get(
      IntegerType::get(getOperation().getContext(), 32), wrapperTypes, false);
    OpBuilder builder(internal);
    auto wrapper = builder.create<LLVM::LLVMFuncOp>(internal.getLoc(),
                                                    exportName.getValue(),
                                                    wrapperType,
                                                    LLVM::Linkage::External);
    Block* entry = wrapper.addEntryBlock(builder);
    Block* nonNullBlock = &wrapper.getBody().emplaceBlock();
    Block* invokeBlock = &wrapper.getBody().emplaceBlock();
    Block* errorBlock = &wrapper.getBody().emplaceBlock();

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
      anyNull =
        anyNull ? builder.create<LLVM::OrOp>(internal.getLoc(), anyNull, isNull)
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

    builder.setInsertionPointToStart(nonNullBlock);
    Value invalidShape;
    auto i64Type = builder.getI64Type();
    auto applyShapeTransform =
      [&](Value extent, unsigned outputIndex, unsigned dimension) {
        ArrayRef<int64_t> program = shapePrograms[outputIndex][dimension];
        for (unsigned index = 0; index < program.size(); index += 2) {
          const int64_t opcode = program[index];
          const int64_t operand = program[index + 1];
          Value value = builder.create<LLVM::ConstantOp>(
            internal.getLoc(), builder.getI64IntegerAttr(operand));
          if (opcode == 0) {
            extent =
              builder.create<LLVM::AddOp>(internal.getLoc(), extent, value);
          } else if (opcode == 1) {
            extent =
              builder.create<LLVM::MulOp>(internal.getLoc(), extent, value);
          } else {
            extent =
              builder.create<LLVM::SDivOp>(internal.getLoc(), extent, value);
          }
        }
        return extent;
      };
    for (unsigned functionIndex : wrapperOrder) {
      MemRefType type = argumentTypes[functionIndex];
      if (outputs.contains(functionIndex) || type.hasStaticShape()) {
        continue;
      }
      Value shape = entry->getArgument(wrapperShapeIndices[functionIndex]);
      for (auto [dimensionIndex, dimension] :
           llvm::enumerate(type.getShape())) {
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
        invalidShape = invalidShape
                         ? builder.create<LLVM::OrOp>(
                             internal.getLoc(), invalidShape, invalid)
                         : invalid;
      }
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
    if (invalidShape) {
      builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                     invalidShape,
                                     errorBlock,
                                     ValueRange{},
                                     invokeBlock,
                                     ValueRange{});
    } else {
      builder.create<LLVM::BrOp>(internal.getLoc(), ValueRange{}, invokeBlock);
    }

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

        Value shape;
        if (outputs.contains(functionIndex)) {
          auto* outputPosition =
            llvm::find(outputArgumentIndices, functionIndex);
          auto outputIndex = static_cast<unsigned>(
            outputPosition - outputArgumentIndices.begin());
          shape = inputShapes[shapeSources[outputIndex]];
        } else {
          shape = entry->getArgument(wrapperShapeIndices[functionIndex]);
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
          } else {
            Value address = builder.create<LLVM::GEPOp>(
              internal.getLoc(),
              pointerType,
              i64Type,
              shape,
              ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimensionIndex)});
            Value size =
              builder.create<LLVM::LoadOp>(internal.getLoc(), i64Type, address);
            if (outputs.contains(functionIndex)) {
              auto* outputPosition =
                llvm::find(outputArgumentIndices, functionIndex);
              auto outputIndex = static_cast<unsigned>(
                outputPosition - outputArgumentIndices.begin());
              size = applyShapeTransform(size, outputIndex, dimensionIndex);
            }
            sizes.push_back(size);
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
          stride = builder.create<LLVM::MulOp>(
            internal.getLoc(), stride, sizes[dimensionIndex]);
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
      Block* shapeReturn = &shapeFunction.getBody().emplaceBlock();
      builder.setInsertionPointToStart(shapeEntry);
      Value shapeNullPointer =
        builder.create<LLVM::ZeroOp>(internal.getLoc(), pointerType);
      Value shapeAnyNull;
      for (Value argument : shapeEntry->getArguments()) {
        Value isNull = builder.create<LLVM::ICmpOp>(internal.getLoc(),
                                                    LLVM::ICmpPredicate::eq,
                                                    argument,
                                                    shapeNullPointer);
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
      builder.setInsertionPointToStart(shapeSuccess);
      SmallVector<Value> shapeInputs(inputIndices.size());
      unsigned shapeArgumentIndex = 0;
      Value shapeInvalid;
      for (auto [inputIndex, functionIndex] : llvm::enumerate(inputIndices)) {
        if (!argumentTypes[functionIndex].hasStaticShape()) {
          Value inputShape = shapeEntry->getArgument(shapeArgumentIndex++);
          shapeInputs[inputIndex] = inputShape;
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
          }
        }
      }
      for (auto [outputIndex, functionIndex] :
           llvm::enumerate(outputArgumentIndices)) {
        MemRefType type = argumentTypes[functionIndex];
        if (type.hasStaticShape()) {
          continue;
        }
        Value destination = shapeEntry->getArgument(shapeArgumentIndex++);
        Value source = shapeInputs[shapeSources[outputIndex]];
        for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
          Value sourceAddress = builder.create<LLVM::GEPOp>(
            internal.getLoc(),
            pointerType,
            i64Type,
            source,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
          Value destinationAddress = builder.create<LLVM::GEPOp>(
            internal.getLoc(),
            pointerType,
            i64Type,
            destination,
            ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(dimension)});
          Value extent;
          if (ShapedType::isDynamic(type.getShape()[dimension])) {
            extent = builder.create<LLVM::LoadOp>(
              internal.getLoc(), i64Type, sourceAddress);
            extent = applyShapeTransform(extent, outputIndex, dimension);
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
          shapeInvalid = shapeInvalid
                           ? builder.create<LLVM::OrOp>(
                               internal.getLoc(), shapeInvalid, invalid)
                           : invalid;
        }
      }
      builder.create<LLVM::CondBrOp>(internal.getLoc(),
                                     shapeInvalid,
                                     shapeError,
                                     ValueRange{},
                                     shapeReturn,
                                     ValueRange{});
      builder.setInsertionPointToStart(shapeReturn);
      Value shapeSuccessStatus = builder.create<LLVM::ConstantOp>(
        internal.getLoc(), builder.getI32IntegerAttr(0));
      builder.create<LLVM::ReturnOp>(internal.getLoc(), shapeSuccessStatus);
    }

    getOperation()->removeAttr(kExportNameAttr);
    getOperation()->removeAttr(kInternalNameAttr);
    getOperation()->removeAttr(kArgumentTypesAttr);
    getOperation()->removeAttr(kOutputIndicesAttr);
    getOperation()->removeAttr(kOutputShapeSourcesAttr);
    getOperation()->removeAttr(kOutputShapeProgramsAttr);
    getOperation()->removeAttr(kShapeCarrierIndicesAttr);
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
      i32Type, {pointerType, pointerType, i32Type, pointerType}, false);
    auto wrapper = builder.create<LLVM::LLVMFuncOp>(
      location, exportName.getValue(), statusType, LLVM::Linkage::External);
    Block* entry = wrapper.addEntryBlock(builder);
    Block* error = &wrapper.getBody().emplaceBlock();
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
                                     index == 3 ? error : dispatch[index + 1],
                                     ValueRange{});
      builder.setInsertionPointToStart(invoke[index]);
      Value invalidExtent;
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
      }
      Block* call = &wrapper.getBody().emplaceBlock();
      builder.create<LLVM::CondBrOp>(
        location, invalidExtent, error, ValueRange{}, call, ValueRange{});
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
    inferInvalid = builder.create<LLVM::OrOp>(location, inferInvalid, rankLow);
    inferInvalid = builder.create<LLVM::OrOp>(location, inferInvalid, rankHigh);
    inferInvalid =
      builder.create<LLVM::OrOp>(location, inferInvalid, capacitySmall);
    builder.create<LLVM::CondBrOp>(location,
                                   inferInvalid,
                                   inferError,
                                   ValueRange{},
                                   inferCopy,
                                   ValueRange{});
    builder.setInsertionPointToStart(inferError);
    builder.create<LLVM::ReturnOp>(location, constantI32(1));
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
        location, invalid, inferError, ValueRange{}, next, ValueRange{});
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
