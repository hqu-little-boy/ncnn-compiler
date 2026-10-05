#include "VulkanGraphKernel.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "ncnn-mlir/Conversion/NCNNToVulkan/NCNNToVulkan.hpp"
#include "ncnn-mlir/Conversion/NCNNToVulkan/VulkanTensorUtils.hpp"
#include "ncnn-mlir/Support/StableHash.hpp"
#include "ncnn-mlir/Support/VulkanModelPayload.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_LOWERNCNNVULKANGRAPHPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

using ncnn_mlir::vulkan::DescriptorBinding;
using ncnn_mlir::vulkan::KernelExecution;
using ncnn_mlir::vulkan::ResourceAccess;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;
using ncnn_mlir::vulkan::VulkanModelWeight;
using vulkan_graph::AddKernel;
using vulkan_graph::ConcatKernel;
using vulkan_graph::ConvKernel;
using vulkan_graph::CopyKernel;
using vulkan_graph::ExpKernel;
using vulkan_graph::GlobalAveragePoolKernel;
using vulkan_graph::GraphKernel;
using vulkan_graph::KernelBody;
using vulkan_graph::MaxPoolKernel;
using vulkan_graph::MulKernel;
using vulkan_graph::ReciprocalKernel;
using vulkan_graph::ReduceMaxKernel;
using vulkan_graph::ReduceSumKernel;
using vulkan_graph::ReluKernel;
using vulkan_graph::SliceKernel;
using vulkan_graph::SubKernel;
using vulkan_graph::TransposeKernel;

LogicalResult reject(Operation& operation, StringRef reason) {
  return vulkan::rejectSupportedOperation(operation, "graph", reason);
}

DenseElementsAttr constantElements(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    return dyn_cast<DenseElementsAttr>(constant.getValue());
  }
  if (auto constant = value.getDefiningOp<tosa::ConstOp>()) {
    return dyn_cast<DenseElementsAttr>(constant.getValues());
  }
  return {};
}

bool isZeroMetadata(Value value) {
  const auto elements = constantElements(value);
  if (!elements || elements.getNumElements() != 1) {
    return false;
  }
  if (isa<IntegerType>(elements.getElementType())) {
    return (*elements.getValues<APInt>().begin()).isZero();
  }
  if (elements.getElementType().isF32()) {
    const auto& scalar = *elements.getValues<APFloat>().begin();
    return scalar.isZero() && !scalar.isNegative();
  }
  return false;
}

FailureOr<std::vector<std::int64_t>> shapeValues(Value value) {
  auto constant = value.getDefiningOp<tosa::ConstShapeOp>();
  if (!constant) {
    return failure();
  }
  std::vector<std::int64_t> result;
  for (const APInt& extent : constant.getValues().getValues<APInt>()) {
    if (!extent.isSignedIntN(64)) {
      return failure();
    }
    result.push_back(extent.getSExtValue());
  }
  return result;
}

std::vector<std::uint64_t> planShape(RankedTensorType type) {
  std::vector<std::uint64_t> result;
  for (auto extent : type.getShape()) {
    result.push_back(static_cast<std::uint64_t>(extent));
  }
  return result;
}

std::vector<std::uint8_t> constantBytes(DenseElementsAttr elements) {
  std::vector<std::uint8_t> result;
  result.reserve(static_cast<std::size_t>(elements.getNumElements()) *
                 sizeof(float));
  for (const APFloat& value : elements.getValues<APFloat>()) {
    const auto bits = value.bitcastToAPInt().getZExtValue();
    for (unsigned byte = 0; byte < sizeof(float); ++byte) {
      result.push_back(static_cast<std::uint8_t>((bits >> (byte * 8)) & 0xffU));
    }
  }
  return result;
}

struct GraphPlan final {
  VulkanExecutionPlan plan;
  std::vector<VulkanModelWeight> weights;
  std::vector<GraphKernel> kernels;
};

class GraphPlanner final {
 public:
  explicit GraphPlanner(func::FuncOp function)
    : function_(function),
      graph_{.plan = {.schema_version =
                        ncnn_mlir::vulkan::kExecutionPlanSchemaVersion,
                      .model = function.getSymName().str(),
                      .build_identity = "pending",
                      .host_triple = llvm::sys::getDefaultTargetTriple(),
                      .device_target = "vulkan1.1-spirv1.0",
                      .resources = {},
                      .kernels = {},
                      .inputs = {},
                      .outputs = {},
                      .plan_hash = ""},
             .weights = {},
             .kernels = {}},
      values_() {}

  [[nodiscard]] FailureOr<GraphPlan> analyze() {
    if (function_.isExternal() ||
        !llvm::hasSingleElement(function_.getBody()) ||
        function_.getNumArguments() == 0 || function_.getNumResults() == 0) {
      return reject(
        *function_,
        "requires one straight-line model with static tensor inputs/outputs");
    }
    for (auto [index, argument] : llvm::enumerate(function_.getArguments())) {
      auto resource = addResource(argument.getType(),
                                  ResourceKind::Input,
                                  std::format("input{}", index + 1));
      if (failed(resource)) {
        return failure();
      }
      values_[argument] = *resource;
      graph_.plan.inputs.push_back(*resource);
    }
    for (Operation& operation :
         function_.getBody().front().without_terminator()) {
      if (failed(analyzeOperation(operation))) {
        return failure();
      }
    }
    auto returned =
      dyn_cast<func::ReturnOp>(function_.getBody().front().getTerminator());
    if (!returned || returned.getNumOperands() != function_.getNumResults()) {
      return reject(*function_, "requires a complete tensor return");
    }
    for (auto [index, value] : llvm::enumerate(returned.getOperands())) {
      if (value.getType() != function_.getResultTypes()[index]) {
        return reject(*returned, "return type must match the public output");
      }
      auto output = addResource(value.getType(),
                                ResourceKind::Output,
                                std::format("output{}", index + 1));
      if (failed(output) ||
          failed(appendKernel(*returned,
                              CopyKernel{},
                              {value},
                              *output,
                              cast<RankedTensorType>(value.getType()),
                              "copy"))) {
        return failure();
      }
      graph_.plan.outputs.push_back(*output);
    }
    if (failed(finalize())) {
      return failure();
    }
    return std::move(graph_);
  }

 private:
  FailureOr<std::uint32_t> addResource(Type type,
                                       ResourceKind kind,
                                       std::string name) {
    auto bytes = vulkan::checkedTensorBytes(type);
    if (!bytes) {
      return reject(*function_, bytes.error());
    }
    if (graph_.plan.resources.size() >=
        std::numeric_limits<std::uint32_t>::max()) {
      return reject(*function_, "resource ID range is exhausted");
    }
    const auto id =
      static_cast<std::uint32_t>(graph_.plan.resources.size() + 1);
    const bool publicIO =
      kind == ResourceKind::Input || kind == ResourceKind::Output;
    const auto tensor = cast<RankedTensorType>(type);
    const std::string layout =
      publicIO && tensor.getRank() == 3 ? "CHW" : "row_major";
    graph_.plan.resources.push_back(
      {.id = id,
       .kind = kind,
       .name = std::move(name),
       .shape = planShape(tensor),
       .logical_layout = layout,
       .physical_layout = layout,
       .host_layout = publicIO ? layout : "",
       .layout_transform = std::nullopt,
       .weight_sha256 = "",
       .byte_size = *bytes,
       .first_kernel = std::numeric_limits<std::uint32_t>::max(),
       .last_kernel = std::numeric_limits<std::uint32_t>::max()});
    return id;
  }

  FailureOr<std::uint32_t> weightResource(Value value, bool convolutionFilter) {
    const auto elements = constantElements(value);
    auto bytes = vulkan::checkedTensorBytes(value.getType());
    if (!elements || !bytes || elements.getType() != value.getType()) {
      return reject(*function_, "requires a static FP32 constant payload");
    }
    auto id = addResource(value.getType(),
                          ResourceKind::Weight,
                          std::format("constant{}", graph_.weights.size()));
    if (failed(id)) {
      return failure();
    }
    TensorResource& resource = graph_.plan.resources.back();
    if (convolutionFilter) {
      const auto type = cast<RankedTensorType>(value.getType());
      resource.shape = {static_cast<std::uint64_t>(type.getDimSize(0)),
                        static_cast<std::uint64_t>(type.getDimSize(3)),
                        static_cast<std::uint64_t>(type.getDimSize(1)),
                        static_cast<std::uint64_t>(type.getDimSize(2))};
      resource.logical_layout = "OIHW";
      resource.physical_layout = "OHWI";
    }
    auto payload = constantBytes(elements);
    auto hash = ncnn_mlir::vulkan::vulkanWeightSHA256(resource, payload);
    if (!hash) {
      return reject(*function_, hash.error());
    }
    resource.weight_sha256 = *hash;
    graph_.weights.push_back({.resource_id = *id, .bytes = std::move(payload)});
    if (!convolutionFilter) {
      values_[value] = *id;
    }
    return id;
  }

  FailureOr<std::uint32_t> resourceFor(Value value, bool convolutionFilter) {
    if (!convolutionFilter) {
      if (auto iterator = values_.find(value); iterator != values_.end()) {
        return iterator->second;
      }
    }
    if (constantElements(value)) {
      return weightResource(value, convolutionFilter);
    }
    return reject(*function_, "tensor operand has no supported producer");
  }

  LogicalResult appendKernel(Operation& operation,
                             KernelBody body,
                             ValueRange operands,
                             std::uint32_t output,
                             RankedTensorType outputType,
                             StringRef name) {
    if (graph_.kernels.size() >= std::numeric_limits<std::uint32_t>::max()) {
      return reject(operation, "kernel ID range is exhausted");
    }
    const auto id = static_cast<std::uint32_t>(graph_.kernels.size());
    GraphKernel kernel{
      .body = std::move(body),
      .location = operation.getLoc(),
      .operands = {},
      .input_types = {},
      .output_type = outputType,
      .execution = {.id = id,
                    .operation = name.str(),
                    .entry_point = std::format("kernel_{}", id),
                    .local_size = {64, 1, 1},
                    .reads = {},
                    .writes = {output},
                    .bindings = {},
                    .push_constants = {}}};
    for (auto [index, operand] : llvm::enumerate(operands)) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || !vulkan::checkedTensorBytes(type)) {
        return reject(
          operation,
          "kernel operands require positive static FP32 rank 1-4 tensors");
      }
      auto resource = resourceFor(
        operand, std::holds_alternative<ConvKernel>(kernel.body) && index == 1);
      if (failed(resource)) {
        return failure();
      }
      kernel.operands.push_back(*resource);
      kernel.input_types.push_back(type);
      if (std::ranges::find(kernel.execution.reads, *resource) ==
          kernel.execution.reads.end()) {
        kernel.execution.reads.push_back(*resource);
      }
    }
    for (auto [binding, resource] : llvm::enumerate(kernel.execution.reads)) {
      kernel.execution.bindings.push_back(
        {.set = 0,
         .binding = static_cast<std::uint32_t>(binding),
         .resource_id = resource,
         .access = ResourceAccess::Read});
    }
    kernel.execution.bindings.push_back(
      {.set = 0,
       .binding = static_cast<std::uint32_t>(kernel.execution.reads.size()),
       .resource_id = output,
       .access = ResourceAccess::Write});
    graph_.plan.kernels.push_back(kernel.execution);
    graph_.kernels.push_back(std::move(kernel));
    return success();
  }

  LogicalResult compute(Operation& operation,
                        KernelBody body,
                        ValueRange operands,
                        StringRef name) {
    auto output = addResource(operation.getResult(0).getType(),
                              ResourceKind::Intermediate,
                              std::format("tensor{}", graph_.kernels.size()));
    if (failed(output) ||
        failed(
          appendKernel(operation,
                       std::move(body),
                       operands,
                       *output,
                       cast<RankedTensorType>(operation.getResult(0).getType()),
                       name))) {
      return failure();
    }
    values_[operation.getResult(0)] = *output;
    return success();
  }

  LogicalResult analyzeConstant(Operation& operation) {
    auto elements = constantElements(operation.getResult(0));
    if (!elements) {
      return reject(operation,
                    "only dense FP32 data constants or zero integer metadata "
                    "are supported");
    }
    if (elements.getElementType().isF32()) {
      if (auto bytes =
            vulkan::checkedTensorBytes(operation.getResult(0).getType());
          !bytes) {
        return reject(operation, bytes.error());
      }
      return success();
    }
    if (!isa<IntegerType>(elements.getElementType()) ||
        !isZeroMetadata(operation.getResult(0))) {
      return reject(operation,
                    "non-FP32 constants must be scalar zero metadata");
    }
    for (OpOperand& use : operation.getResult(0).getUses()) {
      const bool convolutionZeroPoint =
        isa<tosa::Conv2DOp>(use.getOwner()) && use.getOperandNumber() >= 3;
      const bool averagePoolZeroPoint =
        isa<tosa::AvgPool2dOp>(use.getOwner()) &&
        (use.getOperandNumber() == 1 || use.getOperandNumber() == 2);
      const bool multiplyShift =
        isa<tosa::MulOp>(use.getOwner()) && use.getOperandNumber() == 2;
      if (!convolutionZeroPoint && !averagePoolZeroPoint && !multiplyShift) {
        return reject(operation,
                      "integer metadata cannot be used as tensor data");
      }
    }
    return success();
  }

  LogicalResult analyzeReshape(tosa::ReshapeOp reshape) {
    auto input = vulkan::checkedTensorBytes(reshape.getInput1().getType());
    auto output = vulkan::checkedTensorBytes(reshape.getOutput().getType());
    auto shape = shapeValues(reshape.getShape());
    const auto type = cast<RankedTensorType>(reshape.getOutput().getType());
    if (!input || !output || *input != *output || failed(shape) ||
        !std::ranges::equal(*shape, type.getShape())) {
      return reject(*reshape,
                    "reshape requires an exact static shape and unchanged "
                    "linear byte size");
    }
    auto resource = resourceFor(reshape.getInput1(), false);
    if (failed(resource)) {
      return failure();
    }
    values_[reshape.getOutput()] = *resource;
    return success();
  }

  LogicalResult analyzeTranspose(tosa::TransposeOp transpose) {
    const auto input = cast<RankedTensorType>(transpose.getInput1().getType());
    const auto output = cast<RankedTensorType>(transpose.getOutput().getType());
    const auto permutation = transpose.getPerms();
    if (permutation.size() != static_cast<std::size_t>(input.getRank()) ||
        input.getRank() != output.getRank()) {
      return reject(*transpose,
                    "transpose requires an exact rank-preserving permutation");
    }
    std::vector<bool> seen(permutation.size(), false);
    std::vector<std::int64_t> indices;
    for (auto [dimension, source] : llvm::enumerate(permutation)) {
      if (source < 0 || source >= input.getRank() || seen[source] ||
          output.getDimSize(dimension) != input.getDimSize(source)) {
        return reject(*transpose, "transpose permutation/shape is invalid");
      }
      seen[source] = true;
      indices.push_back(source);
    }
    return compute(*transpose,
                   TransposeKernel{std::move(indices)},
                   {transpose.getInput1()},
                   "transpose");
  }

  LogicalResult analyzeSlice(tosa::SliceOp slice) {
    auto start = shapeValues(slice.getStart());
    auto size = shapeValues(slice.getSize());
    const auto input = cast<RankedTensorType>(slice.getInput1().getType());
    const auto output = cast<RankedTensorType>(slice.getOutput().getType());
    if (failed(start) || failed(size) ||
        start->size() != static_cast<std::size_t>(input.getRank()) ||
        !std::ranges::equal(*size, output.getShape()) ||
        input.getRank() != output.getRank()) {
      return reject(
        *slice, "slice requires static rank-preserving start/size metadata");
    }
    for (auto [dimension, offset] : llvm::enumerate(*start)) {
      if (offset < 0 || offset > input.getDimSize(dimension) ||
          output.getDimSize(dimension) > input.getDimSize(dimension) - offset) {
        return reject(*slice, "slice is outside the source tensor");
      }
    }
    return compute(
      *slice, SliceKernel{std::move(*start)}, {slice.getInput1()}, "slice");
  }

  LogicalResult analyzeConv(tosa::Conv2DOp convolution) {
    const auto input = cast<RankedTensorType>(convolution.getInput().getType());
    const auto filter =
      cast<RankedTensorType>(convolution.getWeight().getType());
    const auto bias = cast<RankedTensorType>(convolution.getBias().getType());
    const auto output =
      cast<RankedTensorType>(convolution.getOutput().getType());
    if (input.getRank() != 4 || filter.getRank() != 4 ||
        output.getRank() != 4 || bias.getRank() != 1 ||
        input.getDimSize(0) != 1 || output.getDimSize(0) != 1 ||
        input.getDimSize(3) != filter.getDimSize(3) ||
        output.getDimSize(3) != filter.getDimSize(0) ||
        bias.getDimSize(0) != filter.getDimSize(0) ||
        !convolution.getAccType().isF32()) {
      return reject(*convolution,
                    "conv2d requires batch-one NHWC, constant OHWI filters and "
                    "channel bias");
    }
    if (!constantElements(convolution.getWeight()) ||
        !constantElements(convolution.getBias()) ||
        !isZeroMetadata(convolution.getInputZp()) ||
        !isZeroMetadata(convolution.getWeightZp())) {
      return reject(
        *convolution,
        "conv2d weights/bias must be constant and zero points must be zero");
    }
    const auto kernelHeight = filter.getDimSize(1);
    const auto kernelWidth = filter.getDimSize(2);
    if (!((kernelHeight == 1 && kernelWidth == 1) ||
          (kernelHeight == 3 && kernelWidth == 3))) {
      return reject(*convolution, "conv2d supports only 1x1 and 3x3 kernels");
    }
    if (convolution.getPad().size() != 4 ||
        convolution.getStride().size() != 2 ||
        convolution.getDilation() != ArrayRef<std::int64_t>({1, 1})) {
      return reject(*convolution,
                    "conv2d requires static padding/stride and dilation one");
    }
    ConvKernel parameters{};
    std::ranges::copy(convolution.getPad(), parameters.padding.begin());
    std::ranges::copy(convolution.getStride(), parameters.strides.begin());
    for (auto [dimension, stride] : llvm::enumerate(parameters.strides)) {
      if (stride != 1 && stride != 2) {
        return reject(*convolution, "conv2d stride must be one or two");
      }
      const auto low = parameters.padding[dimension * 2];
      const auto high = parameters.padding[(dimension * 2) + 1];
      constexpr std::int64_t maximumPadding = 1;
      // Static tail padding can add up to stride-1 elements for 3x3 kernels.
      const std::int64_t maximumHighPadding =
        kernelHeight == 1 ? maximumPadding : maximumPadding + stride - 1;
      if (low < 0 || high < 0 || low > maximumPadding ||
          high > maximumHighPadding) {
        return reject(*convolution,
                      "conv2d padding is outside the supported 1x1/3x3 subset");
      }
      const auto extent = input.getDimSize(dimension + 1) + low + high;
      const auto kernel = filter.getDimSize(dimension + 1);
      if (extent < kernel || output.getDimSize(dimension + 1) !=
                               ((extent - kernel) / stride) + 1) {
        return reject(*convolution,
                      "conv2d output shape does not match padding/stride");
      }
    }
    return compute(
      *convolution,
      parameters,
      {convolution.getInput(), convolution.getWeight(), convolution.getBias()},
      "conv2d");
  }

  LogicalResult analyzeConcat(tosa::ConcatOp concat) {
    const auto output = cast<RankedTensorType>(concat.getOutput().getType());
    const std::int64_t axis = concat.getAxis();
    if (axis < 0 || axis >= output.getRank() || concat.getInput1().empty()) {
      return reject(*concat,
                    "concat requires a valid static axis and tensor inputs");
    }
    std::int64_t extent = 0;
    for (Value value : concat.getInput1()) {
      const auto input = cast<RankedTensorType>(value.getType());
      if (input.getRank() != output.getRank()) {
        return reject(*concat, "concat input/output ranks must match");
      }
      for (std::int64_t dimension = 0; dimension < output.getRank();
           ++dimension) {
        if (dimension != axis &&
            input.getDimSize(dimension) != output.getDimSize(dimension)) {
          return reject(*concat, "concat non-axis extents must match");
        }
      }
      if (input.getDimSize(axis) > output.getDimSize(axis) - extent) {
        return reject(*concat, "concat axis size exceeds the declared output");
      }
      extent += input.getDimSize(axis);
    }
    if (extent != output.getDimSize(axis)) {
      return reject(*concat,
                    "concat output axis size must equal the input sum");
    }
    return compute(*concat, ConcatKernel{axis}, concat.getInput1(), "concat");
  }

  LogicalResult analyzeBroadcastBinary(Operation& operation,
                                       Value lhs,
                                       Value rhs,
                                       RankedTensorType output,
                                       KernelBody body,
                                       StringRef name) {
    const auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    const auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType || lhsType.getRank() != output.getRank() ||
        rhsType.getRank() != output.getRank() ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() || !output.getElementType().isF32()) {
      return reject(operation,
                    "broadcast binary requires equal-rank FP32 tensors");
    }
    std::size_t broadcastDimensions = 0;
    for (std::int64_t dimension = 0; dimension < output.getRank();
         ++dimension) {
      const auto outExtent = output.getDimSize(dimension);
      const auto lhsExtent = lhsType.getDimSize(dimension);
      const auto rhsExtent = rhsType.getDimSize(dimension);
      if ((lhsExtent != outExtent && lhsExtent != 1) ||
          (rhsExtent != outExtent && rhsExtent != 1) ||
          outExtent != std::max(lhsExtent, rhsExtent)) {
        return reject(
          operation, "broadcast dimensions must be exact equal-or-one extents");
      }
      if (lhsExtent != rhsExtent && ++broadcastDimensions > 1) {
        return reject(operation,
                      "broadcast may expand at most one tensor axis");
      }
    }
    return compute(operation, std::move(body), {lhs, rhs}, name);
  }

  LogicalResult analyzeMaxPool(tosa::MaxPool2dOp pooling) {
    const auto input = cast<RankedTensorType>(pooling.getInput().getType());
    const auto output = cast<RankedTensorType>(pooling.getOutput().getType());
    const auto kernel = pooling.getKernel();
    const auto stride = pooling.getStride();
    const auto padding = pooling.getPad();
    if (input.getRank() != 4 || output.getRank() != 4) {
      return reject(*pooling, "max_pool2d requires rank-four NHWC tensors");
    }
    if (input.getDimSize(0) != 1 || output.getDimSize(0) != 1) {
      return reject(*pooling, "max_pool2d requires batch size one");
    }
    if (input.getDimSize(3) != output.getDimSize(3)) {
      return reject(*pooling, "max_pool2d must preserve the channel extent");
    }
    if (kernel.size() != 2 || stride.size() != 2 || padding.size() != 4) {
      return reject(*pooling,
                    "max_pool2d requires two kernel/stride and four padding "
                    "values");
    }
    if (pooling.getNanMode() != "PROPAGATE") {
      return reject(*pooling, "max_pool2d requires propagating NaN semantics");
    }
    MaxPoolKernel parameters{};
    std::ranges::copy(kernel, parameters.kernel.begin());
    std::ranges::copy(stride, parameters.strides.begin());
    std::ranges::copy(padding, parameters.padding.begin());
    if (parameters.kernel != std::array<std::int64_t, 2>{3, 3}) {
      return reject(*pooling, "max_pool2d supports only a 3x3 kernel");
    }
    if (parameters.strides != std::array<std::int64_t, 2>{2, 2}) {
      return reject(*pooling, "max_pool2d supports only stride 2x2");
    }
    if (parameters.padding[0] != 0 || parameters.padding[2] != 0) {
      return reject(*pooling, "max_pool2d requires zero top and left padding");
    }
    if (parameters.padding[1] < 0 || parameters.padding[1] > 1 ||
        parameters.padding[3] < 0 || parameters.padding[3] > 1) {
      return reject(*pooling,
                    "max_pool2d bottom/right tail padding must be zero or one");
    }
    for (std::int64_t dimension = 0; dimension < 2; ++dimension) {
      const auto inputExtent = input.getDimSize(dimension + 1);
      const auto paddedExtent =
        inputExtent + parameters.padding[(dimension * 2) + 1];
      const auto kernelExtent = parameters.kernel[dimension];
      const auto strideExtent = parameters.strides[dimension];
      const auto outputExtent = output.getDimSize(dimension + 1);
      if (paddedExtent < kernelExtent ||
          outputExtent != ((paddedExtent - kernelExtent) / strideExtent) + 1) {
        return reject(
          *pooling,
          "max_pool2d output shape must match kernel, stride and padding");
      }
    }
    return compute(*pooling, parameters, {pooling.getInput()}, "max_pool2d");
  }

  LogicalResult analyzeGlobalAveragePool(tosa::AvgPool2dOp pooling) {
    const auto input = cast<RankedTensorType>(pooling.getInput().getType());
    const auto output = cast<RankedTensorType>(pooling.getOutput().getType());
    const auto kernel = pooling.getKernel();
    const auto stride = pooling.getStride();
    const auto padding = pooling.getPad();
    if (input.getRank() != 4 || output.getRank() != 4) {
      return reject(*pooling, "avg_pool2d requires rank-four NHWC tensors");
    }
    if (input.getDimSize(0) != 1 || output.getDimSize(0) != 1) {
      return reject(*pooling, "avg_pool2d requires batch size one");
    }
    if (input.getDimSize(3) != output.getDimSize(3)) {
      return reject(*pooling, "avg_pool2d must preserve the channel extent");
    }
    if (kernel.size() != 2 || stride.size() != 2 || padding.size() != 4) {
      return reject(*pooling,
                    "avg_pool2d requires two kernel/stride and four padding "
                    "values");
    }
    if (kernel[0] != input.getDimSize(1) || kernel[1] != input.getDimSize(2) ||
        output.getDimSize(1) != 1 || output.getDimSize(2) != 1) {
      return reject(*pooling,
                    "avg_pool2d requires a full-spatial kernel and 1x1 output");
    }
    if (stride[0] != 1 || stride[1] != 1) {
      return reject(*pooling, "avg_pool2d requires unit spatial strides");
    }
    if (std::ranges::any_of(padding,
                            [](std::int64_t value) { return value != 0; })) {
      return reject(*pooling, "avg_pool2d requires zero padding");
    }
    if (!pooling.getAccType().isF32()) {
      return reject(*pooling, "avg_pool2d requires FP32 accumulation");
    }
    if (!isZeroMetadata(pooling.getInputZp()) ||
        !isZeroMetadata(pooling.getOutputZp())) {
      return reject(*pooling,
                    "avg_pool2d requires zero input/output zero-points");
    }
    return compute(*pooling,
                   GlobalAveragePoolKernel{},
                   {pooling.getInput()},
                   "global_avg_pool2d");
  }

  template <typename ReduceOp, typename Kernel>
  LogicalResult analyzeReduce(ReduceOp reduction,
                              Kernel kernel,
                              StringRef name) {
    const auto input = cast<RankedTensorType>(reduction.getInput().getType());
    const auto output = cast<RankedTensorType>(reduction.getOutput().getType());
    const std::uint32_t axis = reduction.getAxis();
    if (!input.getElementType().isF32() || !output.getElementType().isF32() ||
        input.getRank() != output.getRank() || axis >= input.getRank()) {
      return reject(
        *reduction,
        "reduction requires same-rank static FP32 and a valid axis");
    }
    if constexpr (std::is_same_v<ReduceOp, tosa::ReduceMaxOp>) {
      if (reduction.getNanMode() != "PROPAGATE") {
        return reject(*reduction,
                      "reduce_max requires propagating NaN semantics");
      }
    }
    for (std::int64_t dimension = 0; dimension < input.getRank(); ++dimension) {
      const auto expected =
        std::cmp_equal(dimension, axis) ? 1 : input.getDimSize(dimension);
      if (output.getDimSize(dimension) != expected) {
        return reject(*reduction,
                      "reduction output must preserve every non-axis extent "
                      "and set the reduced extent to one");
      }
    }
    return compute(*reduction, kernel, {reduction.getInput()}, name);
  }

  LogicalResult analyzeExp(tosa::ExpOp exponential) {
    const auto input =
      cast<RankedTensorType>(exponential.getInput1().getType());
    const auto output =
      cast<RankedTensorType>(exponential.getOutput().getType());
    if (input != output || !input.getElementType().isF32()) {
      return reject(*exponential, "exp requires an identical FP32 tensor type");
    }
    return compute(*exponential, ExpKernel{}, {exponential.getInput1()}, "exp");
  }

  LogicalResult analyzeReciprocal(tosa::ReciprocalOp reciprocal) {
    const auto input = cast<RankedTensorType>(reciprocal.getInput1().getType());
    const auto output =
      cast<RankedTensorType>(reciprocal.getOutput().getType());
    if (input != output || !input.getElementType().isF32()) {
      return reject(*reciprocal,
                    "reciprocal requires an identical FP32 tensor type");
    }
    return compute(
      *reciprocal, ReciprocalKernel{}, {reciprocal.getInput1()}, "reciprocal");
  }

  LogicalResult analyzeOperation(Operation& operation) {
    if (isa<arith::ConstantOp, tosa::ConstOp>(operation)) {
      return analyzeConstant(operation);
    }
    if (isa<tosa::ConstShapeOp>(operation)) {
      return success();
    }
    if (operation.getNumRegions() != 0 || operation.getNumResults() != 1) {
      return reject(operation, "unsupported operation or control-flow region");
    }
    for (Type type : operation.getResultTypes()) {
      if (auto bytes = vulkan::checkedTensorBytes(type); !bytes) {
        return reject(operation, bytes.error());
      }
    }
    for (auto [index, operand] : llvm::enumerate(operation.getOperands())) {
      Type type = operand.getType();
      if (isa<tosa::shapeType>(type)) {
        continue;
      }
      if (auto bytes = vulkan::checkedTensorBytes(type); !bytes) {
        const bool zeroMetadata =
          (isa<tosa::AvgPool2dOp>(operation) && index > 0) ||
          (isa<tosa::MulOp>(operation) && index == 2);
        if (zeroMetadata && isZeroMetadata(operand)) {
          continue;
        }
        if (!isa<tosa::Conv2DOp>(operation)) {
          return reject(operation, bytes.error());
        }
      }
    }
    if (auto reshape = dyn_cast<tosa::ReshapeOp>(operation)) {
      return analyzeReshape(reshape);
    }
    if (auto identity = dyn_cast<tosa::IdentityOp>(operation)) {
      return compute(operation, CopyKernel{}, {identity.getInput1()}, "copy");
    }
    if (auto transpose = dyn_cast<tosa::TransposeOp>(operation)) {
      return analyzeTranspose(transpose);
    }
    if (auto slice = dyn_cast<tosa::SliceOp>(operation)) {
      return analyzeSlice(slice);
    }
    if (auto relu = dyn_cast<tosa::ClampOp>(operation)) {
      auto minimum = dyn_cast<FloatAttr>(relu.getMinVal());
      auto maximum = dyn_cast<FloatAttr>(relu.getMaxVal());
      if (!minimum || !maximum || !minimum.getValue().isZero() ||
          minimum.getValue().isNegative() || !maximum.getValue().isInfinity() ||
          maximum.getValue().isNegative() || relu.getNanMode() != "PROPAGATE" ||
          relu.getInput().getType() != relu.getOutput().getType()) {
        return reject(
          operation,
          "only FP32 ReLU clamp [0,+inf] with propagating NaNs is supported");
      }
      return compute(operation, ReluKernel{}, {relu.getInput()}, "relu");
    }
    if (auto add = dyn_cast<tosa::AddOp>(operation)) {
      if (add.getInput1().getType() != add.getOutput().getType() ||
          add.getInput2().getType() != add.getOutput().getType()) {
        return reject(operation, "Add requires identical static tensor shapes");
      }
      return compute(
        operation, AddKernel{}, {add.getInput1(), add.getInput2()}, "add");
    }
    if (auto convolution = dyn_cast<tosa::Conv2DOp>(operation)) {
      // Metadata is exempt, but all actual data operands still need full
      // checks.
      for (Value value : {convolution.getInput(),
                          convolution.getWeight(),
                          convolution.getBias()}) {
        if (auto bytes = vulkan::checkedTensorBytes(value.getType()); !bytes) {
          return reject(operation, bytes.error());
        }
      }
      return analyzeConv(convolution);
    }
    if (auto concat = dyn_cast<tosa::ConcatOp>(operation)) {
      return analyzeConcat(concat);
    }
    if (auto maxPool = dyn_cast<tosa::MaxPool2dOp>(operation)) {
      return analyzeMaxPool(maxPool);
    }
    if (auto averagePool = dyn_cast<tosa::AvgPool2dOp>(operation)) {
      return analyzeGlobalAveragePool(averagePool);
    }
    if (auto reduceMax = dyn_cast<tosa::ReduceMaxOp>(operation)) {
      return analyzeReduce(
        reduceMax, ReduceMaxKernel{reduceMax.getAxis()}, "reduce_max");
    }
    if (auto reduceSum = dyn_cast<tosa::ReduceSumOp>(operation)) {
      return analyzeReduce(
        reduceSum, ReduceSumKernel{reduceSum.getAxis()}, "reduce_sum");
    }
    if (auto sub = dyn_cast<tosa::SubOp>(operation)) {
      return analyzeBroadcastBinary(
        *sub,
        sub.getInput1(),
        sub.getInput2(),
        cast<RankedTensorType>(sub.getOutput().getType()),
        SubKernel{},
        "sub");
    }
    if (auto exponential = dyn_cast<tosa::ExpOp>(operation)) {
      return analyzeExp(exponential);
    }
    if (auto reciprocal = dyn_cast<tosa::ReciprocalOp>(operation)) {
      return analyzeReciprocal(reciprocal);
    }
    if (auto multiply = dyn_cast<tosa::MulOp>(operation)) {
      if (!isZeroMetadata(multiply.getShift())) {
        return reject(*multiply, "mul requires scalar zero shift");
      }
      return analyzeBroadcastBinary(
        *multiply,
        multiply.getInput1(),
        multiply.getInput2(),
        cast<RankedTensorType>(multiply.getOutput().getType()),
        MulKernel{},
        "mul");
    }
    return reject(operation,
                  "unsupported operation in the static FP32 graph subset");
  }

  LogicalResult finalize() {
    for (auto [index, kernel] : llvm::enumerate(graph_.plan.kernels)) {
      for (const auto& binding : kernel.bindings) {
        TensorResource& resource =
          graph_.plan.resources[binding.resource_id - 1];
        resource.first_kernel =
          std::min(resource.first_kernel, static_cast<std::uint32_t>(index));
        resource.last_kernel = static_cast<std::uint32_t>(index);
      }
    }
    std::string identity;
    llvm::raw_string_ostream stream(identity);
    function_.print(stream);
    graph_.plan.build_identity =
      std::format("vulkan-graph-v1-{:016x}", ncnn_mlir::stableHash64(identity));
    if (auto valid = ncnn_mlir::vulkan::verifyExecutionPlan(graph_.plan);
        !valid) {
      return reject(*function_, valid.error());
    }
    if (auto valid = ncnn_mlir::vulkan::verifyVulkanModelWeights(
          graph_.plan, graph_.weights);
        !valid) {
      return reject(*function_, valid.error());
    }
    return success();
  }

  func::FuncOp function_;
  GraphPlan graph_;
  llvm::DenseMap<Value, std::uint32_t> values_;
};

LogicalResult writeArtifact(ModuleOp module,
                            StringRef path,
                            std::string_view bytes) {
  if (path.empty()) {
    return success();
  }
  std::ofstream output(path.str(), std::ios::binary | std::ios::trunc);
  if (!output || !(output << bytes << '\n')) {
    return module.emitError()
           << "cannot write Vulkan graph artifact '" << path << "'";
  }
  return success();
}

class LowerNCNNVulkanGraphPass final
  : public impl::LowerNCNNVulkanGraphPassBase<LowerNCNNVulkanGraphPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!planPath.getValue().empty() && !weightsPath.getValue().empty()) {
      std::error_code pathError;
      const std::filesystem::path planFile = std::filesystem::weakly_canonical(
        std::filesystem::path(planPath.getValue()), pathError);
      if (pathError) {
        module.emitError() << "cannot resolve Vulkan plan output path: "
                           << pathError.message();
        signalPassFailure();
        return;
      }
      const std::filesystem::path weightsFile =
        std::filesystem::weakly_canonical(
          std::filesystem::path(weightsPath.getValue()), pathError);
      if (pathError) {
        module.emitError() << "cannot resolve Vulkan weights output path: "
                           << pathError.message();
        signalPassFailure();
        return;
      }
      if (planFile == weightsFile) {
        module.emitError(
          "Vulkan plan and weights artifacts must use distinct output paths");
        signalPassFailure();
        return;
      }
    }
    if (!llvm::hasSingleElement(module.getBody()->getOperations())) {
      module.emitError(
        "Vulkan graph lowering requires exactly one model function");
      signalPassFailure();
      return;
    }
    auto function = dyn_cast<func::FuncOp>(module.getBody()->front());
    if (!function) {
      module.emitError(
        "Vulkan graph lowering requires a strict NCNN-to-TOSA function "
        "boundary");
      signalPassFailure();
      return;
    }
    auto graph = GraphPlanner(function).analyze();
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(graph->plan);
    auto weights =
      ncnn_mlir::vulkan::serializeVulkanModelWeights(graph->weights);
    if (!serialized || !weights) {
      module.emitError(
        "cannot serialize the checked Vulkan graph plan/payload");
      signalPassFailure();
      return;
    }
    for (const auto& kernel : graph->kernels) {
      if (failed(vulkan_graph::emitGraphKernel(module, kernel, graph->plan))) {
        signalPassFailure();
        return;
      }
    }
    function.erase();
    if (failed(verify(module)) ||
        failed(writeArtifact(module, planPath, *serialized)) ||
        failed(writeArtifact(module, weightsPath, *weights))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::ncnn
