#include "ncnn-mlir/Support/ConstantFold.hpp"

#include <cstring>
#include <vector>

#include "llvm/ADT/SmallVector.h"

namespace ncnn_mlir {

bool is_foldable_element_type(mlir::Type elementType) {
  return elementType.isIntOrFloat() &&
         elementType.getIntOrFloatBitWidth() % 8 == 0;
}

mlir::DenseElementsAttr reshape_dense_elements(
  mlir::ElementsAttr elements, mlir::RankedTensorType resultType) {
  if (!elements ||
      (!elements.isSplat() && !mlir::isa<mlir::DenseElementsAttr>(elements)) ||
      !is_foldable_element_type(resultType.getElementType())) {
    return {};
  }
  if (elements.isSplat()) {
    return mlir::DenseElementsAttr::get(
      resultType, elements.getSplatValue<mlir::Attribute>());
  }
  return mlir::DenseElementsAttr::getFromRawBuffer(
    resultType, mlir::cast<mlir::DenseElementsAttr>(elements).getRawData());
}

mlir::DenseElementsAttr transpose_dense_elements(
  mlir::ElementsAttr elements,
  mlir::RankedTensorType sourceType,
  llvm::ArrayRef<int32_t> permutation,
  mlir::RankedTensorType resultType) {
  if (!elements || !is_foldable_element_type(sourceType.getElementType()) ||
      permutation.size() != static_cast<size_t>(sourceType.getRank())) {
    return {};
  }
  for (int32_t dimension : permutation) {
    if (dimension < 0 || dimension >= sourceType.getRank()) {
      return {};
    }
  }
  if (elements.isSplat()) {
    return mlir::DenseElementsAttr::get(
      resultType, elements.getSplatValue<mlir::Attribute>());
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(elements);
  if (!dense) {
    return {};
  }
  const llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();
  const int64_t rank = sourceType.getRank();
  llvm::SmallVector<int64_t> sourceStrides(rank, 1);
  llvm::SmallVector<int64_t> resultStrides(rank, 1);
  for (int64_t dimension = rank - 2; dimension >= 0; --dimension) {
    const int64_t next = dimension + 1;
    sourceStrides[dimension] = sourceStrides[next] * (sourceShape[next]);
    resultStrides[dimension] =
      resultStrides[next] * (resultType.getShape()[next]);
  }
  llvm::ArrayRef<char> sourceData = dense.getRawData();
  const int64_t count = sourceType.getNumElements();
  const int64_t elementSize = static_cast<int64_t>(sourceData.size()) / count;
  std::vector<char> resultData(sourceData.size());
  const char* source = sourceData.data();
  char* destination = resultData.data();
  llvm::SmallVector<int64_t> index(rank, 0);
  for (int64_t offset = 0; offset < count; ++offset) {
    int64_t sourceOffset = 0;
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      sourceOffset +=
        index[dimension] * (sourceStrides[permutation[dimension]]);
    }
    std::memcpy(destination + (offset * elementSize),
                source + (sourceOffset * elementSize),
                elementSize);
    for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
      if (++index[dimension] < resultType.getShape()[dimension]) {
        break;
      }
      index[dimension] = 0;
    }
  }
  return mlir::DenseElementsAttr::getFromRawBuffer(resultType, resultData);
}

}  // namespace ncnn_mlir
