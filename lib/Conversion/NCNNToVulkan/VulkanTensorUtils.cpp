#include "ncnn-mlir/Conversion/NCNNToVulkan/VulkanTensorUtils.hpp"

#include <limits>
#include <optional>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace mlir::ncnn::vulkan {
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

}  // namespace

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

LogicalResult rejectSupportedOperation(Operation& operation,
                                       llvm::StringRef subset,
                                       llvm::StringRef reason) {
  InFlightDiagnostic diagnostic = operation.emitOpError()
                                  << "is not supported by Vulkan " << subset
                                  << ": " << reason;
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

}  // namespace mlir::ncnn::vulkan
