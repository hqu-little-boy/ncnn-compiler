#pragma once

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace mlir::ncnn::vulkan_graph {

struct CopyKernel final {};
struct ReluKernel final {};
struct AddKernel final {};
struct TransposeKernel final {
  std::vector<std::int64_t> permutation;
};
struct SliceKernel final {
  std::vector<std::int64_t> offsets;
};
struct ConvKernel final {
  std::array<std::int64_t, 4> padding;
  std::array<std::int64_t, 2> strides;
};
struct ConcatKernel final {
  std::int64_t axis;
};
struct MaxPoolKernel final {
  std::array<std::int64_t, 2> kernel;
  std::array<std::int64_t, 2> strides;
  std::array<std::int64_t, 4> padding;
};
struct GlobalAveragePoolKernel final {};
struct ReduceMaxKernel final {
  std::uint32_t axis;
};
struct ReduceSumKernel final {
  std::uint32_t axis;
};
struct SubKernel final {};
struct ExpKernel final {};
struct ReciprocalKernel final {};
struct MulKernel final {};

using KernelBody = std::variant<CopyKernel,
                                ReluKernel,
                                AddKernel,
                                TransposeKernel,
                                SliceKernel,
                                ConvKernel,
                                ConcatKernel,
                                MaxPoolKernel,
                                GlobalAveragePoolKernel,
                                ReduceMaxKernel,
                                ReduceSumKernel,
                                SubKernel,
                                ExpKernel,
                                ReciprocalKernel,
                                MulKernel>;

// A fully checked, owning description shared by legality, planning and
// emission. Operand IDs preserve SSA operand order; execution.reads
// deduplicates bindings.
struct GraphKernel final {
  KernelBody body;
  Location location;
  std::vector<std::uint32_t> operands;
  std::vector<RankedTensorType> input_types;
  RankedTensorType output_type;
  ncnn_mlir::vulkan::KernelExecution execution;
};

[[nodiscard]] LogicalResult emitGraphKernel(
  ModuleOp module,
  const GraphKernel& kernel,
  const ncnn_mlir::vulkan::VulkanExecutionPlan& plan);

}  // namespace mlir::ncnn::vulkan_graph
