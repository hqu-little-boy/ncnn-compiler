// GenerateCAPI：ABI 构造准备（prepareABI）。
//
// 职责
//   校验输入签名、构造 C ABI 类型与函数签名、发出前置声明。
//
// 不变量
//   * 签名一旦生成即不可变，后续 TU 只读；
//   * 动态 rank 的符号名由 finalizeDynamicRankABI 统一派生，本 TU 不改名；
//   * 失败必须事务性——不留半成品声明。
//
// 顺序依赖
//   * 必须在 EmitModelPlan 之后（plan 与 ABI 同源）；
//   * 必须在 GenerateCAPIEmit* 之前（签名先于定义）。
//
// 明确不做
//   * 不写 manifest（GenerateCAPIFinalize）；
//   * 不写实现体（GenerateCAPIEmitInfer / EmitWrapper）。

#include <cctype>
#include <cstdint>
#include <format>
#include <functional>
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
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNAttrs.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/ModelLedger.hpp"
#include "ncnn-mlir/Support/ShapeProgram.hpp"
#include "ncnn-mlir/Transforms/GenerateCAPI/GenerateCAPI.hpp"

namespace mlir::ncnn {

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
      if (function->hasAttr(contract::kEntryPoint) &&
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
      if (function->hasAttr(contract::kEntryPoint)) {
        entries.push_back(function);
      }
    }
    if (entries.size() == 4 && llvm::all_of(entries, [](func::FuncOp function) {
          return contract::ModelLedger::read(function).dynamicRank != nullptr;
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
      auto rankAttr = contract::ModelLedger::read(function).rankVariant;
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
        function.getArgAttrOfType<IntegerAttr>(1, contract::kShapeSourceInput);
      auto program =
        function.getArgAttrOfType<ArrayAttr>(1, contract::kShapeProgram);
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
    if (failed(writeManifest(
          variants.front(), std::move(inputs), std::move(outputs)))) {
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
      function->removeAttr(contract::kEntryPoint);
      names.push_back(builder.getStringAttr(name));
      types.push_back(
        builder.getArrayAttr({TypeAttr::get(function.getArgumentTypes()[0]),
                              TypeAttr::get(function.getArgumentTypes()[1])}));
    }
    getOperation()->setAttr(contract::kCApiExportName,
                            builder.getStringAttr(exportName));
    getOperation()->setAttr(contract::kCApiRankVariantNames,
                            builder.getArrayAttr(names));
    getOperation()->setAttr(contract::kCApiRankVariantTypes,
                            builder.getArrayAttr(types));
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
        index, contract::kDataDependentDimMask);
      ArgumentInfo info{
        .functionIndex = index,
        .type = type,
        .output =
          static_cast<bool>(function.getArgAttr(index, "bufferize.result")),
        .shapeCarrier = static_cast<bool>(
          function.getArgAttr(index, contract::kShapeCarrier)),
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
    SmallVector<int32_t> outputShapeVersions;
    SmallVector<Attribute> outputShapePrograms;
    auto shapeConstraints =
      contract::ModelLedger::read(function).shapeConstraints;
    SmallVector<SmallVector<DimConstraintAttr>> constraintsByInput(
      inputs.size());
    if (shapeConstraints) {
      llvm::SmallDenseSet<std::pair<uint32_t, uint32_t>> constrained;
      for (Attribute attribute : shapeConstraints) {
        auto constraint = dyn_cast<DimConstraintAttr>(attribute);
        if (!constraint || constraint.getInput() >= inputs.size() ||
            constraint.getDim() >=
              static_cast<uint32_t>(
                inputs[constraint.getInput()].type.getRank()) ||
            !inputs[constraint.getInput()].type.isDynamicDim(
              constraint.getDim()) ||
            !constrained.insert({constraint.getInput(), constraint.getDim()})
               .second) {
          return function.emitOpError("has invalid input shape constraints");
        }
        constraintsByInput[constraint.getInput()].push_back(constraint);
      }
    }
    auto relationAttrs =
      contract::ModelLedger::read(function).inputDimRelations;
    if (relationAttrs) {
      for (Attribute attribute : relationAttrs) {
        auto values = dyn_cast<DenseI64ArrayAttr>(attribute);
        if (!values || values.size() != 5) {
          return function.emitOpError("has invalid input dimension relations");
        }
        ArrayRef<int64_t> relation = values.asArrayRef();
        if (relation[0] < 0 || relation[1] < 0 || relation[2] < 0 ||
            relation[3] < 0 ||
            std::cmp_greater_equal(relation[0], inputs.size()) ||
            std::cmp_greater_equal(relation[2], inputs.size()) ||
            relation[1] >= inputs[relation[0]].type.getRank() ||
            relation[3] >= inputs[relation[2]].type.getRank()) {
          return function.emitOpError("has invalid input dimension relations");
        }
      }
    }
    for (const ArgumentInfo& output : outputs) {
      if (output.shapeCarrier) {
        if (output.functionIndex == 0 || !output.type.hasStaticShape() ||
            output.type.getRank() != 1 ||
            !output.type.getElementType().isInteger(64)) {
          return function.emitOpError() << "argument " << output.functionIndex
                                        << " has an invalid shape carrier type";
        }
        const unsigned dataIndex = output.functionIndex - 1;
        auto dataType =
          dyn_cast<MemRefType>(function.getArgumentTypes()[dataIndex]);
        auto dataMask = function.getArgAttrOfType<IntegerAttr>(
          dataIndex, contract::kDataDependentDimMask);
        if (!function.getArgAttr(dataIndex, "bufferize.result") ||
            function.getArgAttr(dataIndex, contract::kShapeCarrier) ||
            !dataType || !dataMask || dataMask.getInt() == 0 ||
            output.type.getShape()[0] != dataType.getRank()) {
          return function.emitOpError()
                 << "argument " << output.functionIndex
                 << " is not paired with a data-dependent output";
        }
        outputShapeSources.push_back(-1);
        outputShapeVersions.push_back(0);
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
            !function.getArgAttr(carrierIndex, contract::kShapeCarrier) ||
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
            output.functionIndex, contract::kShapeSourceInput)) {
        source = static_cast<int32_t>(attribute.getInt());
      }
      outputShapeSources.push_back(source);
      auto version = function.getArgAttrOfType<IntegerAttr>(
        output.functionIndex, contract::kShapeProgramVersion);
      if (version && version.getInt() != 2) {
        return function.emitOpError("has an unsupported shape program version");
      }
      const bool v2 = version && version.getInt() == 2;
      if (!output.type.hasStaticShape() && !v2 &&
          (source < 0 || static_cast<std::size_t>(source) >= inputs.size() ||
           inputs[source].type.hasStaticShape())) {
        return function.emitOpError()
               << "output " << output.functionIndex
               << " has dynamic extents without a valid input shape source";
      }
      if (!output.type.hasStaticShape() && v2 && source >= 0) {
        return function.emitOpError()
               << "output " << output.functionIndex
               << " V2 program must not have a shape source";
      }
      outputShapeVersions.push_back(
        version ? static_cast<int32_t>(version.getInt()) : 1);
      auto program = function.getArgAttrOfType<ArrayAttr>(
        output.functionIndex, contract::kShapeProgram);
      SmallVector<unsigned> inputRanks;
      for (const ArgumentInfo& input : inputs) {
        inputRanks.push_back(input.type.getRank());
      }
      if (!output.type.hasStaticShape() &&
          (!program ||
           program.size() != static_cast<std::size_t>(output.type.getRank()) ||
           llvm::any_of(program, [&](Attribute dimension) {
             auto instructions = dyn_cast<DenseI64ArrayAttr>(dimension);
             if (!instructions) {
               return true;
             }
             if (v2) {
               auto expression =
                 ShapeExpr::deserialize(instructions.asArrayRef());
               return !expression ||
                      !expression->validateInputRanks(inputRanks);
             }
             if (instructions.size() % 2 != 0) {
               return true;
             }
             ArrayRef<int64_t> values = instructions.asArrayRef();
             unsigned start = 0;
             if (!values.empty() && values[0] == 3) {
               if (values[1] < 0 ||
                   values[1] >= inputs[source].type.getRank()) {
                 return true;
               }
               start = 2;
             }
             for (unsigned index = start; index < values.size(); index += 2) {
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
      if (!info.output) {
        llvm::json::Array constraints;
        for (DimConstraintAttr constraint :
             constraintsByInput[inputManifest.size()]) {
          llvm::json::Object object;
          object["dimension"] = constraint.getDim();
          object["minimum"] = constraint.getMin();
          object["multiple_of"] = constraint.getMultipleOf();
          constraints.push_back(std::move(object));
        }
        if (!constraints.empty()) {
          argument["dimension_constraints"] = std::move(constraints);
        }
      }
      if (info.output) {
        const bool dataDependent = info.dataDependentDimMask != 0;
        argument["shape_depends_on_data"] = dataDependent;
        if (dataDependent) {
          argument["maximum_shape"] = std::move(maximumShape);
        }
        if (outputShapeSources[outputInfoIndex] >= 0) {
          argument["shape_source_input"] = outputShapeSources[outputInfoIndex];
        }
        if (outputShapeVersions[outputInfoIndex] == 2) {
          argument["shape_program_version"] = 2;
        }
        if (!info.type.hasStaticShape() && !dataDependent) {
          auto programs =
            dyn_cast<ArrayAttr>(outputShapePrograms[outputInfoIndex]);
          if (!programs) {
            return function.emitOpError(
              "has no serialized dynamic output shape program");
          }
          llvm::json::Array serializedPrograms;
          for (Attribute program : programs) {
            auto values = dyn_cast<DenseI64ArrayAttr>(program);
            if (!values) {
              return function.emitOpError(
                "has an invalid serialized output shape program");
            }
            llvm::json::Array serialized;
            for (int64_t value : values.asArrayRef()) {
              serialized.push_back(value);
            }
            serializedPrograms.push_back(std::move(serialized));
          }
          argument["shape_program"] = std::move(serializedPrograms);
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
    if (failed(writeManifest(
          function, std::move(inputManifest), std::move(outputManifest)))) {
      return failure();
    }

    SymbolTable symbolTable(getOperation());
    if (failed(symbolTable.rename(function, internalName))) {
      return function.emitOpError(
        "cannot update all symbol uses for internal C ABI name");
    }
    function.setPrivate();
    function->removeAttr("llvm.emit_c_interface");
    function->removeAttr(contract::kEntryPoint);
    contract::ModelLedger::clearDimensionConstraints(function);
    Builder builder(function.getContext());
    getOperation()->setAttr(contract::kCApiExportName,
                            builder.getStringAttr(exportName));
    getOperation()->setAttr(contract::kCApiInternalName,
                            builder.getStringAttr(internalName));
    getOperation()->setAttr(contract::kCApiArgumentTypes,
                            builder.getArrayAttr(argumentTypes));
    getOperation()->setAttr(contract::kCApiOutputIndices,
                            builder.getDenseI32ArrayAttr(outputIndices));
    getOperation()->setAttr(contract::kCApiOutputShapeSources,
                            builder.getDenseI32ArrayAttr(outputShapeSources));
    getOperation()->setAttr(contract::kCApiOutputShapeProgramVersions,
                            builder.getDenseI32ArrayAttr(outputShapeVersions));
    getOperation()->setAttr(contract::kCApiOutputShapePrograms,
                            builder.getArrayAttr(outputShapePrograms));
    getOperation()->setAttr(contract::kCApiShapeCarrierIndices,
                            builder.getDenseI32ArrayAttr(shapeCarrierIndices));
    getOperation()->setAttr(contract::kCApiInputShapeConstraints,
                            shapeConstraints
                              ? static_cast<Attribute>(shapeConstraints)
                              : builder.getArrayAttr({}));
    getOperation()->setAttr(contract::kCApiInputDimRelations,
                            relationAttrs
                              ? static_cast<Attribute>(relationAttrs)
                              : builder.getArrayAttr({}));
    return success();
  }

  LogicalResult writeManifest(func::FuncOp function,
                              SmallVector<llvm::json::Object> inputs,
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
    if (auto relations =
          contract::ModelLedger::read(function).inputDimRelations) {
      llvm::json::Array serializedRelations;
      for (Attribute attribute : relations) {
        ArrayRef<int64_t> relation =
          cast<DenseI64ArrayAttr>(attribute).asArrayRef();
        llvm::json::Object object;
        object["lhs_input"] = relation[0];
        object["lhs_dimension"] = relation[1];
        object["rhs_input"] = relation[2];
        object["rhs_dimension"] = relation[3];
        object["offset"] = relation[4];
        serializedRelations.push_back(std::move(object));
      }
      manifest["input_dimension_relations"] = std::move(serializedRelations);
    }
    auto precision = function->getAttrOfType<StringAttr>(contract::kPrecision);
    if (precision &&
        (precision.getValue() == "fp16" || precision.getValue() == "bf16")) {
      llvm::json::Object policy;
      policy["storage"] = precision.getValue();
      policy["complex_math"] = "f32";
      policy["complex_accumulator"] = "f32";
      if (auto accumulator =
            function->getAttrOfType<StringAttr>(contract::kFp16Accumulator)) {
        policy["fp16_accumulator"] = accumulator.getValue();
      }
      policy["fallback"] = function->hasAttr(contract::kPrecisionFallback);
      manifest["precision_policy"] = std::move(policy);
    }
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

}  // namespace

}  // namespace mlir::ncnn
