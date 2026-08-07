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
      ArgumentInfo info{.functionIndex = index,
                        .type = type,
                        .output = static_cast<bool>(
                          function.getArgAttr(index, "bufferize.result"))};
      (info.output ? outputs : inputs).push_back(info);
    }
    if (inputs.empty() || outputs.empty()) {
      return function.emitOpError("requires at least one input and one output");
    }
    for (const ArgumentInfo& output : outputs) {
      if (!output.type.hasStaticShape()) {
        return function.emitOpError()
               << "output " << output.functionIndex
               << " has dynamic extents; output shape inference is required "
                  "before this ABI can be generated";
      }
    }

    SmallVector<Attribute> argumentTypes;
    SmallVector<int32_t> outputIndices;
    SmallVector<llvm::json::Object> inputManifest;
    SmallVector<llvm::json::Object> outputManifest;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      argumentTypes.push_back(
        TypeAttr::get(function.getArgumentTypes()[index]));
    }
    SmallVector<ArgumentInfo> wrapperArguments(inputs);
    wrapperArguments.append(outputs);
    for (const ArgumentInfo& info : wrapperArguments) {
      if (info.output) {
        outputIndices.push_back(static_cast<int32_t>(info.functionIndex));
      }
      llvm::json::Array shape;
      for (int64_t dimension : info.type.getShape()) {
        shape.push_back(ShapedType::isDynamic(dimension) ? -1 : dimension);
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
        if (ShapedType::isDynamic(dimension)) {
          dynamicDimMask |= UINT32_C(1) << index;
        }
      }
      argument["dynamic_dim_mask"] = dynamicDimMask;
      if (info.output) {
        argument["shape_depends_on_data"] = false;
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
    auto internalName =
      getOperation()->getAttrOfType<StringAttr>(kInternalNameAttr);
    auto argumentTypeAttrs =
      getOperation()->getAttrOfType<ArrayAttr>(kArgumentTypesAttr);
    auto outputIndices =
      getOperation()->getAttrOfType<DenseI32ArrayAttr>(kOutputIndicesAttr);
    auto internal = getOperation().lookupSymbol<LLVM::LLVMFuncOp>(internalName);
    if (!internal || !argumentTypeAttrs || !outputIndices) {
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
    SmallVector<unsigned> wrapperOrder;
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      if (!outputs.contains(index)) {
        wrapperOrder.push_back(index);
      }
    }
    for (unsigned index = 0; index < argumentTypes.size(); ++index) {
      if (outputs.contains(index)) {
        wrapperOrder.push_back(index);
      }
    }

    LLVMTypeConverter typeConverter(getOperation().getContext());
    auto pointerType = LLVM::LLVMPointerType::get(getOperation().getContext());
    SmallVector<Type> wrapperTypes;
    for (unsigned functionIndex : wrapperOrder) {
      wrapperTypes.push_back(pointerType);
      if (!outputs.contains(functionIndex) &&
          !argumentTypes[functionIndex].hasStaticShape()) {
        wrapperTypes.push_back(pointerType);
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
    for (Value argument : entry->getArguments()) {
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
    unsigned validationIndex = 0;
    auto i64Type = builder.getI64Type();
    for (unsigned functionIndex : wrapperOrder) {
      ++validationIndex;
      MemRefType type = argumentTypes[functionIndex];
      if (outputs.contains(functionIndex) || type.hasStaticShape()) {
        continue;
      }
      Value shape = entry->getArgument(validationIndex++);
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
    unsigned wrapperIndex = 0;
    for (unsigned functionIndex : wrapperOrder) {
      MemRefType type = argumentTypes[functionIndex];
      Value data = entry->getArgument(wrapperIndex++);
      MemRefDescriptor descriptor = [&] {
        if (type.hasStaticShape()) {
          return MemRefDescriptor::fromStaticShape(
            builder, internal.getLoc(), typeConverter, type, data);
        }

        Value shape = entry->getArgument(wrapperIndex++);
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
            sizes.push_back(builder.create<LLVM::LoadOp>(
              internal.getLoc(), i64Type, address));
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
    Value success = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI32IntegerAttr(0));
    builder.create<LLVM::ReturnOp>(internal.getLoc(), success);

    getOperation()->removeAttr(kExportNameAttr);
    getOperation()->removeAttr(kInternalNameAttr);
    getOperation()->removeAttr(kArgumentTypesAttr);
    getOperation()->removeAttr(kOutputIndicesAttr);
  }
};

}  // namespace

}  // namespace mlir::ncnn
