#include "ncnn-mlir/Transforms/VerifyVulkanSupportedModel/VerifyVulkanSupportedModel.hpp"

#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYVULKANSUPPORTEDMODELPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

struct SourceProvenance final {
  std::optional<std::int64_t> layer;
  std::optional<std::string> name;
};

SourceProvenance provenanceFromLocation(Location location) {
  if (auto named = dyn_cast<NameLoc>(location)) {
    if (auto file = dyn_cast<FileLineColLoc>(named.getChildLoc());
        file && file.getFilename().getValue() == "ncnn-layer") {
      SourceProvenance provenance;
      provenance.layer = static_cast<std::int64_t>(file.getLine());
      if (!named.getName().getValue().empty()) {
        provenance.name = named.getName().getValue().str();
      }
      return provenance;
    }
  }
  if (auto fused = dyn_cast<FusedLoc>(location)) {
    for (Location child : fused.getLocations()) {
      SourceProvenance provenance = provenanceFromLocation(child);
      if (provenance.layer || provenance.name) {
        return provenance;
      }
    }
  }
  return {};
}

SourceProvenance sourceProvenance(Operation& operation) {
  SourceProvenance provenance = provenanceFromLocation(operation.getLoc());
  if (!provenance.layer) {
    if (auto source =
          operation.getAttrOfType<IntegerAttr>(contract::kSourceLayer)) {
      provenance.layer = source.getInt();
    }
  }
  if (!provenance.name) {
    if (auto name = operation.getAttrOfType<StringAttr>(contract::kName)) {
      provenance.name = name.getValue().str();
    }
  }
  return provenance;
}

std::expected<std::uint64_t, std::string> checkedTensorBytes(Type type) {
  auto shaped = dyn_cast<RankedTensorType>(type);
  if (!shaped) {
    return std::unexpected("requires a ranked tensor");
  }
  if (!shaped.getElementType().isF32()) {
    return std::unexpected("requires f32 elements");
  }
  if (shaped.getRank() < 1 || shaped.getRank() > 4) {
    return std::unexpected("supports tensor ranks 1 through 4");
  }

  std::uint64_t elements = 1;
  for (std::int64_t extent : shaped.getShape()) {
    if (ShapedType::isDynamic(extent)) {
      return std::unexpected("requires static tensor dimensions");
    }
    if (extent <= 0) {
      return std::unexpected("requires positive tensor dimensions");
    }
    const auto unsignedExtent = static_cast<std::uint64_t>(extent);
    if (elements > std::numeric_limits<std::uint64_t>::max() / unsignedExtent) {
      return std::unexpected("tensor element count overflows uint64");
    }
    elements *= unsignedExtent;
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    return std::unexpected("tensor byte size overflows uint64");
  }
  const std::uint64_t bytes = elements * sizeof(float);
  if (bytes > ncnn_mlir::vulkan::kMinimumStorageBufferRange) {
    return std::unexpected(
      "tensor exceeds the Vulkan 1.1 minimum storage-buffer range");
  }
  return bytes;
}

bool isSameStaticF32Tensor(Type lhs, Type rhs) {
  return lhs == rhs && static_cast<bool>(checkedTensorBytes(lhs));
}

LogicalResult reject(Operation& operation, StringRef reason) {
  InFlightDiagnostic diagnostic =
    operation.emitOpError() << "is not supported by Vulkan V1: " << reason;
  const SourceProvenance provenance = sourceProvenance(operation);
  diagnostic << "; ncnn.name=";
  if (provenance.name) {
    diagnostic << '"' << *provenance.name << '"';
  } else {
    diagnostic << "<missing>";
  }
  diagnostic << ", ncnn.source_layer=";
  if (provenance.layer) {
    diagnostic << *provenance.layer;
  } else {
    diagnostic << "<missing>";
  }
  return failure();
}

LogicalResult verifyTensorType(Operation& operation, Type type) {
  auto bytes = checkedTensorBytes(type);
  if (!bytes) {
    return reject(operation, bytes.error());
  }
  return success();
}

LogicalResult verifyTypes(Operation& operation) {
  if (isa<func::ReturnOp>(operation)) {
    return success();
  }
  for (Type type : operation.getOperandTypes()) {
    if (isa<ShapedType>(type)) {
      if (failed(verifyTensorType(operation, type))) {
        return failure();
      }
    } else if (!type.isF32() && !type.isIndex() && !isa<IntegerType>(type)) {
      return reject(operation, "supports only f32, integer, and index values");
    }
  }
  for (Type type : operation.getResultTypes()) {
    if (isa<ShapedType>(type)) {
      if (failed(verifyTensorType(operation, type))) {
        return failure();
      }
    } else if (!type.isF32() && !type.isIndex() && !isa<IntegerType>(type)) {
      return reject(operation, "supports only f32, integer, and index values");
    }
  }
  return success();
}

bool isZeroConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant) {
    return false;
  }
  auto attribute = dyn_cast<FloatAttr>(constant.getValue());
  return attribute && attribute.getValue().isZero() &&
         !attribute.getValue().isNegative();
}

LogicalResult verifyPointwiseGeneric(linalg::GenericOp operation) {
  const std::size_t inputCount = operation.getInputs().size();
  if ((inputCount != 1 && inputCount != 2) ||
      operation.getOutputs().size() != 1) {
    return reject(
      *operation,
      "elementwise operation requires one or two inputs and one output");
  }
  const Type resultType = operation.getOutputs().front().getType();
  const auto resultBytes = checkedTensorBytes(resultType);
  if (!resultBytes || !llvm::all_of(operation.getInputs(), [&](Value input) {
        return isSameStaticF32Tensor(input.getType(), resultType);
      })) {
    return reject(
      *operation,
      "elementwise operation requires identical static f32 tensor types");
  }
  const auto maps = operation.getIndexingMapsArray();
  if (maps.size() != inputCount + 1 ||
      !llvm::all_of(maps, [](AffineMap map) { return map.isIdentity(); })) {
    return reject(*operation,
                  "elementwise operation requires identity indexing maps");
  }
  if (llvm::any_of(operation.getIteratorTypesArray(),
                   [](utils::IteratorType type) {
                     return type != utils::IteratorType::parallel;
                   })) {
    return reject(*operation,
                  "elementwise operation requires parallel-only iteration");
  }

  Region& region = operation.getRegion();
  if (!llvm::hasSingleElement(region)) {
    return reject(*operation,
                  "elementwise operation requires one scalar region");
  }
  Block& body = region.front();
  const std::size_t expectedArguments = inputCount + 1;
  if (body.getNumArguments() != expectedArguments ||
      !llvm::all_of(body.getArgumentTypes(),
                    [](Type type) { return type.isF32(); })) {
    return reject(*operation,
                  "elementwise operation requires f32 scalar block arguments");
  }
  if (body.getOperations().size() != 2) {
    return reject(*operation,
                  "elementwise region must contain one arithmetic operation");
  }
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1) {
    return reject(*operation, "elementwise region must yield one value");
  }

  Value yieldedValue = yield.getValues().front();
  if (inputCount == 1) {
    auto maximum = yieldedValue.getDefiningOp<arith::MaximumFOp>();
    if (!maximum || maximum.getLhs() != body.getArgument(0) ||
        !isZeroConstant(maximum.getRhs())) {
      return reject(*operation, "ReLU must compute max(input, 0.0)");
    }
    return success();
  }

  auto add = yieldedValue.getDefiningOp<arith::AddFOp>();
  if (!add || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1)) {
    return reject(*operation, "Add must compute lhs + rhs");
  }
  return success();
}

LogicalResult verifyFunction(func::FuncOp operation) {
  if (operation.isExternal()) {
    return reject(*operation, "function declarations are not supported");
  }
  FunctionType type = operation.getFunctionType();
  for (Type input : type.getInputs()) {
    if (failed(verifyTensorType(*operation, input))) {
      return failure();
    }
  }
  for (Type result : type.getResults()) {
    if (failed(verifyTensorType(*operation, result))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyCopy(linalg::CopyOp operation) {
  if (operation.getInputs().size() != 1 || operation.getOutputs().size() != 1) {
    return reject(*operation, "copy requires one input and one output");
  }
  if (!isSameStaticF32Tensor(operation.getInputs().front().getType(),
                             operation.getOutputs().front().getType())) {
    return reject(*operation,
                  "copy requires identical static f32 tensor types");
  }
  return success();
}

LogicalResult verifyTranspose(linalg::TransposeOp operation) {
  auto input = dyn_cast<RankedTensorType>(operation.getInput().getType());
  auto output = dyn_cast<RankedTensorType>(operation.getInit().getType());
  if (operation->getNumResults() != 1 ||
      operation->getResult(0).getType() != operation.getInit().getType() ||
      !input || !output || input.getRank() != output.getRank()) {
    return reject(*operation,
                  "layout transpose requires equal input/output rank");
  }
  ArrayRef<int64_t> permutation = operation.getPermutation();
  if (permutation.size() != static_cast<std::size_t>(input.getRank())) {
    return reject(*operation, "layout transpose permutation rank is invalid");
  }
  SmallVector<bool, 4> seen(permutation.size(), false);
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    const std::int64_t axis = permutation[index];
    if (axis < 0 || static_cast<std::size_t>(axis) >= permutation.size()) {
      return reject(*operation, "layout transpose permutation is invalid");
    }
    const auto unsignedAxis = static_cast<std::size_t>(axis);
    if (seen[unsignedAxis] ||
        output.getDimSize(index) != input.getDimSize(unsignedAxis)) {
      return reject(*operation, "layout transpose permutation is invalid");
    }
    seen[unsignedAxis] = true;
  }
  return success();
}

LogicalResult verifyAdd(linalg::AddOp operation) {
  if (operation.getInputs().size() != 2 || operation.getOutputs().size() != 1) {
    return reject(*operation, "Add requires two inputs and one output");
  }
  const Type resultType = operation.getOutputs().front().getType();
  if (!isSameStaticF32Tensor(operation.getInputs()[0].getType(), resultType) ||
      !isSameStaticF32Tensor(operation.getInputs()[1].getType(), resultType)) {
    return reject(*operation, "Add requires identical static f32 tensor types");
  }
  return success();
}

LogicalResult verifyReshape(Operation& operation) {
  if (operation.getNumOperands() != 1 || operation.getNumResults() != 1) {
    return reject(operation,
                  "layout reshape requires one input and one result");
  }
  auto inputBytes = checkedTensorBytes(operation.getOperand(0).getType());
  auto resultBytes = checkedTensorBytes(operation.getResult(0).getType());
  if (!inputBytes || !resultBytes) {
    return reject(operation,
                  !inputBytes ? StringRef(inputBytes.error())
                              : StringRef(resultBytes.error()));
  }
  if (*inputBytes != *resultBytes) {
    return reject(operation,
                  "layout reshape must preserve the static byte size");
  }
  return success();
}

LogicalResult verifyEmpty(tensor::EmptyOp operation) {
  for (OpOperand& use : operation.getResult().getUses()) {
    auto destination = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
    if (!destination || !destination.isDpsInit(&use)) {
      return reject(*operation,
                    "tensor.empty may only initialize a written output");
    }
  }
  return success();
}

LogicalResult verifyOperation(Operation& operation) {
  if (failed(verifyTypes(operation))) {
    return failure();
  }

  if (isa<ModuleOp,
          func::ReturnOp,
          arith::ConstantOp,
          tensor::ExpandShapeOp,
          tensor::CollapseShapeOp>(operation)) {
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(operation)) {
      return verifyReshape(operation);
    }
    return success();
  }
  if (auto empty = dyn_cast<tensor::EmptyOp>(operation)) {
    return verifyEmpty(empty);
  }
  if (auto function = dyn_cast<func::FuncOp>(operation)) {
    return verifyFunction(function);
  }
  if (isa<linalg::YieldOp>(operation)) {
    return success();
  }
  if (isa<arith::AddFOp, arith::MaximumFOp>(operation) &&
      isa_and_nonnull<linalg::GenericOp>(operation.getParentOp())) {
    return success();
  }
  if (auto transpose = dyn_cast<linalg::TransposeOp>(operation)) {
    return verifyTranspose(transpose);
  }
  if (auto add = dyn_cast<linalg::AddOp>(operation)) {
    return verifyAdd(add);
  }
  if (auto copy = dyn_cast<linalg::CopyOp>(operation)) {
    return verifyCopy(copy);
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
    return verifyPointwiseGeneric(generic);
  }
  return reject(operation,
                "operation is outside the static FP32 copy/ReLU/Add subset");
}

class VerifyVulkanSupportedModelPass final
  : public impl::VerifyVulkanSupportedModelPassBase<
      VerifyVulkanSupportedModelPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    bool invalid = false;
    bool hasModelEntry = false;
    detail::walk<ForwardIterator>(
      getOperation().getOperation(),
      [&](Operation* operation) {
        if (auto function = dyn_cast<func::FuncOp>(operation)) {
          FunctionType type = function.getFunctionType();
          hasModelEntry |= !function.isExternal() &&
                           !type.getInputs().empty() &&
                           !type.getResults().empty();
        }
        if (!invalid && failed(verifyOperation(*operation))) {
          invalid = true;
        }
      },
      WalkOrder::PreOrder);
    if (!hasModelEntry) {
      getOperation().emitError(
        "Vulkan V1 legality requires a model function with inputs and outputs");
      invalid = true;
    }
    if (invalid) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
