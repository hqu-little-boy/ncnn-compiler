#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ncnn::vulkan {

[[nodiscard]] std::expected<std::uint64_t, std::string> checkedTensorBytes(
  Type type);
[[nodiscard]] LogicalResult rejectSupportedOperation(Operation& operation,
                                                     llvm::StringRef subset,
                                                     llvm::StringRef reason);

}  // namespace mlir::ncnn::vulkan
