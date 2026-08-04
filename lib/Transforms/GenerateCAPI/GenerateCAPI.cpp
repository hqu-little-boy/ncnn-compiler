#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"

#include <cctype>
#include <cstdint>
#include <format>
#include <memory>
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
      if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
          !type.getElementType().isF32()) {
        return function.emitOpError()
               << "argument " << index
               << " must be a static identity-layout f32 memref";
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
        shape.push_back(dimension);
      }
      llvm::json::Object argument;
      argument["name"] = std::format(
        "{}{}",
        info.output ? "output" : "input",
        info.output ? outputManifest.size() + 1 : inputManifest.size() + 1);
      argument["shape"] = std::move(shape);
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
    SmallVector<Type> pointerTypes(wrapperOrder.size(), pointerType);
    auto wrapperType = LLVM::LLVMFunctionType::get(
      IntegerType::get(getOperation().getContext(), 32), pointerTypes, false);
    OpBuilder builder(internal);
    auto wrapper = builder.create<LLVM::LLVMFuncOp>(internal.getLoc(),
                                                    exportName.getValue(),
                                                    wrapperType,
                                                    LLVM::Linkage::External);
    Block* entry = wrapper.addEntryBlock(builder);
    Block* successBlock = &wrapper.getBody().emplaceBlock();
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
                                   successBlock,
                                   ValueRange{});

    builder.setInsertionPointToStart(errorBlock);
    Value failure = builder.create<LLVM::ConstantOp>(
      internal.getLoc(), builder.getI32IntegerAttr(1));
    builder.create<LLVM::ReturnOp>(internal.getLoc(), failure);

    builder.setInsertionPointToStart(successBlock);
    SmallVector<SmallVector<Value>> unpacked(argumentTypes.size());
    for (auto [wrapperIndex, functionIndex] : llvm::enumerate(wrapperOrder)) {
      auto descriptor =
        MemRefDescriptor::fromStaticShape(builder,
                                          internal.getLoc(),
                                          typeConverter,
                                          argumentTypes[functionIndex],
                                          entry->getArgument(wrapperIndex));
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
