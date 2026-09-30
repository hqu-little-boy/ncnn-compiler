#include "ncnn-mlir/Conversion/NCNNToVulkan/NCNNToVulkan.hpp"

#include <cstdint>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "ncnn-mlir/Support/StableHash.hpp"
#include "ncnn-mlir/Support/VulkanExecutionPlan.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_LOWERNCNNVULKANMODELPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

using ncnn_mlir::vulkan::DescriptorBinding;
using ncnn_mlir::vulkan::KernelExecution;
using ncnn_mlir::vulkan::ResourceAccess;
using ncnn_mlir::vulkan::ResourceKind;
using ncnn_mlir::vulkan::TensorResource;
using ncnn_mlir::vulkan::VulkanExecutionPlan;

constexpr std::uint32_t kLocalSize = 64;
constexpr llvm::StringLiteral kTargetEnvironment =
  "#spirv.target_env<#spirv.vce<v1.0, [Shader], "
  "[SPV_KHR_storage_buffer_storage_class]>, api=Vulkan, "
  "#spirv.resource_limits<>>";
constexpr llvm::StringLiteral kEntryPointABI =
  "#spirv.entry_point_abi<workgroup_size = [64, 1, 1]>";

FailureOr<std::uint64_t> elementCount(RankedTensorType type) {
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getRank() < 1 || type.getRank() > 4) {
    return failure();
  }
  std::uint64_t count = 1;
  for (std::int64_t extent : type.getShape()) {
    if (extent <= 0 || count > std::numeric_limits<std::uint64_t>::max() /
                                 static_cast<std::uint64_t>(extent)) {
      return failure();
    }
    count *= static_cast<std::uint64_t>(extent);
  }
  if (count > ncnn_mlir::vulkan::kMinimumStorageBufferRange / sizeof(float) ||
      count >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return failure();
  }
  return count;
}

Value stripLayoutViews(Value value) {
  while (Operation* definition = value.getDefiningOp()) {
    if (!isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
          definition)) {
      break;
    }
    auto inputType =
      dyn_cast<RankedTensorType>(definition->getOperand(0).getType());
    auto outputType =
      dyn_cast<RankedTensorType>(definition->getResult(0).getType());
    auto inputCount =
      inputType ? elementCount(inputType) : FailureOr<std::uint64_t>(failure());
    auto outputCount = outputType ? elementCount(outputType)
                                  : FailureOr<std::uint64_t>(failure());
    if (failed(inputCount) || failed(outputCount) ||
        *inputCount != *outputCount) {
      break;
    }
    value = definition->getOperand(0);
  }
  return value;
}

bool isLayoutViewOf(Value value, Value source) {
  return stripLayoutViews(value) == source;
}

std::vector<std::uint64_t> planShape(RankedTensorType type) {
  std::vector<std::uint64_t> shape;
  shape.reserve(type.getRank());
  for (std::int64_t extent : type.getShape()) {
    shape.push_back(static_cast<std::uint64_t>(extent));
  }
  return shape;
}

std::string sha256Bytes(StringRef bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  const auto digest = hash.final();
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (std::uint8_t byte : digest) {
    result.push_back(kHexDigits[byte >> 4]);
    result.push_back(kHexDigits[byte & 0x0f]);
  }
  return result;
}

std::string sha256Elements(DenseElementsAttr elements) {
  std::string bytes;
  bytes.reserve(static_cast<std::size_t>(elements.getNumElements()) *
                sizeof(float));
  for (float value : elements.getValues<float>()) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  return sha256Bytes(bytes);
}

std::string sha256ConvWeightsOIHW(DenseElementsAttr hwcf,
                                  std::int64_t kernelHeight,
                                  std::int64_t kernelWidth,
                                  std::int64_t inputChannels,
                                  std::int64_t outputChannels) {
  const auto values = hwcf.getValues<float>();
  std::vector<float> deviceValues(values.begin(), values.end());
  std::string logicalBytes;
  logicalBytes.reserve(deviceValues.size() * sizeof(float));
  for (std::int64_t output = 0; output < outputChannels; ++output) {
    for (std::int64_t input = 0; input < inputChannels; ++input) {
      for (std::int64_t y = 0; y < kernelHeight; ++y) {
        for (std::int64_t x = 0; x < kernelWidth; ++x) {
          const auto deviceIndex = static_cast<std::size_t>(
            (((y * kernelWidth + x) * inputChannels + input) * outputChannels) +
            output);
          const float value = deviceValues[deviceIndex];
          logicalBytes.append(reinterpret_cast<const char*>(&value),
                              sizeof(value));
        }
      }
    }
  }
  return sha256Bytes(logicalBytes);
}

std::string sha256ZeroBytes(std::size_t size) {
  const std::string zeroBytes(size, '\0');
  return sha256Bytes(zeroBytes);
}

LogicalResult writePlan(const VulkanExecutionPlan& plan, StringRef path) {
  auto serialized = ncnn_mlir::vulkan::serializeExecutionPlan(plan);
  if (!serialized) {
    return failure();
  }
  if (path.empty()) {
    return success();
  }
  std::ofstream output(path.str(), std::ios::binary | std::ios::trunc);
  if (!output || !(output << *serialized << '\n')) {
    return failure();
  }
  return success();
}

std::string operationName(Operation* operation) {
  if (!operation) {
    return "identity";
  }
  if (isa<linalg::CopyOp>(operation)) {
    return "copy";
  }
  if (isa<linalg::TransposeOp>(operation)) {
    return "transpose";
  }
  if (isa<linalg::Conv2DNhwcHwcfOp>(operation)) {
    return "conv2d";
  }
  if (isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
        operation)) {
    return "reshape";
  }
  if (isa<linalg::GenericOp>(operation)) {
    auto generic = cast<linalg::GenericOp>(operation);
    if (generic.getInputs().size() == 1) {
      return "relu";
    }
    if (generic.getInputs().size() == 2) {
      return "add";
    }
  }
  return "unsupported";
}

LogicalResult verifySingleOperation(func::FuncOp function,
                                    Operation* operation,
                                    RankedTensorType outputType) {
  auto outputCount = elementCount(outputType);
  if (function.isExternal() || function.getNumResults() != 1 ||
      failed(outputCount)) {
    return function.emitError(
      "Vulkan V2 single-op lowering requires one static f32 tensor result");
  }
  if (!operation) {
    if (function.getNumArguments() != 1 ||
        !isLayoutViewOf(
          function.getBody().front().getTerminator()->getOperand(0),
          function.getArgument(0))) {
      return function.emitError(
        "Vulkan V2 identity requires one input returned unchanged");
    }
    auto inputType =
      dyn_cast<RankedTensorType>(function.getArgument(0).getType());
    auto inputCount =
      inputType ? elementCount(inputType) : FailureOr<std::uint64_t>(failure());
    if (failed(inputCount) || *inputCount != *outputCount) {
      return function.emitError(
        "Vulkan V2 identity must preserve the static element count");
    }
    return success();
  }
  Value returnedValue =
    function.getBody().front().getTerminator()->getOperand(0);
  const bool isLayoutOnlyOp =
    isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
      operation);
  if (operation->getNumResults() != 1 ||
      (isLayoutOnlyOp
         ? returnedValue != operation->getResult(0)
         : !isLayoutViewOf(returnedValue, operation->getResult(0)))) {
    return operation->emitError(
      "Vulkan V2 single-op lowering requires one tensor result with only "
      "layout views");
  }
  auto operationOutputType =
    dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto operationOutputCount = operationOutputType
                                ? elementCount(operationOutputType)
                                : FailureOr<std::uint64_t>(failure());
  if (failed(operationOutputCount) || *operationOutputCount != *outputCount) {
    return operation->emitError(
      "Vulkan V2 operation and returned tensor must have equal element counts");
  }
  const std::string kind = operationName(operation);
  if (kind == "reshape") {
    if (function.getNumArguments() != 1 || operation->getNumOperands() < 1 ||
        operation->getOperand(0) != function.getArgument(0)) {
      return operation->emitError(
        "Vulkan V2 reshape requires one direct function input");
    }
    auto inputType =
      dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    auto inputCount =
      inputType ? elementCount(inputType) : FailureOr<std::uint64_t>(failure());
    if (failed(inputCount) || *inputCount != *outputCount) {
      return operation->emitError(
        "Vulkan V2 reshape must preserve the static element count");
    }
    if (auto reshape = dyn_cast<tensor::ReshapeOp>(operation)) {
      auto shape = reshape.getShape().getDefiningOp<arith::ConstantOp>();
      auto values = shape ? dyn_cast<DenseIntElementsAttr>(shape.getValue())
                          : DenseIntElementsAttr();
      if (!values || values.getNumElements() !=
                       static_cast<std::int64_t>(outputType.getRank())) {
        return operation->emitError(
          "Vulkan V2 reshape requires a constant output shape");
      }
      std::size_t axis = 0;
      for (const APInt& extent : values.getValues<APInt>()) {
        if (extent.getSExtValue() != outputType.getDimSize(axis++)) {
          return operation->emitError(
            "Vulkan V2 reshape dimensions must match the static result type");
        }
      }
    }
    return success();
  }
  if (kind == "unsupported") {
    return operation->emitError(
      "operation is outside the Vulkan V2.1 copy/ReLU/Add/transpose/reshape "
      "subset");
  }
  if (operation->getNumOperands() == 0) {
    return operation->emitError("Vulkan V2 operation requires tensor inputs");
  }
  for (Value operand : operation->getOperands()) {
    auto inputType = dyn_cast<RankedTensorType>(operand.getType());
    auto inputCount =
      inputType ? elementCount(inputType) : FailureOr<std::uint64_t>(failure());
    if (failed(inputCount) || *inputCount != *outputCount) {
      return operation->emitError(
        "Vulkan V2 operation requires equal-size static f32 tensors");
    }
  }
  if (operation->getNumOperands() != function.getNumArguments() + 1) {
    return operation->emitError(
      "Vulkan V2 single-op lowering requires function inputs and one init "
      "tensor");
  }
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    if (!isLayoutViewOf(operation->getOperand(index),
                        function.getArgument(index))) {
      return operation->emitError(
        "Vulkan V2 single-op lowering requires function inputs with only "
        "layout views");
    }
  }
  if (operation->getOperand(function.getNumArguments())
        .getDefiningOp<tensor::EmptyOp>() == nullptr) {
    return operation->emitError(
      "Vulkan V2 single-op lowering requires tensor.empty as the output init");
  }
  if (kind == "add" || kind == "relu") {
    auto generic = cast<linalg::GenericOp>(operation);
    auto yield = dyn_cast<linalg::YieldOp>(generic.getBody()->getTerminator());
    if (!yield || generic.getBody()->getOperations().size() != 2) {
      return operation->emitError(
        "Vulkan V2 pointwise lowering requires one arithmetic scalar "
        "operation");
    }
    Value yielded = yield.getValues().front();
    if (kind == "add") {
      auto add = yielded.getDefiningOp<arith::AddFOp>();
      if (!add || add.getLhs() != generic.getBody()->getArgument(0) ||
          add.getRhs() != generic.getBody()->getArgument(1)) {
        return operation->emitError("Vulkan V2 Add must compute lhs + rhs");
      }
    } else {
      auto maximum = yielded.getDefiningOp<arith::MaximumFOp>();
      auto zero = maximum ? maximum.getRhs().getDefiningOp<arith::ConstantOp>()
                          : arith::ConstantOp();
      auto zeroValue =
        zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
      if (!maximum || maximum.getLhs() != generic.getBody()->getArgument(0) ||
          !zeroValue || !zeroValue.getValue().isZero() ||
          zeroValue.getValue().isNegative()) {
        return operation->emitError(
          "Vulkan V2 ReLU must compute max(input, 0.0)");
      }
    }
  }
  if (auto transpose = dyn_cast<linalg::TransposeOp>(operation)) {
    auto inputType = cast<RankedTensorType>(transpose.getInput().getType());
    ArrayRef<std::int64_t> permutation = transpose.getPermutation();
    if (permutation.size() != static_cast<std::size_t>(inputType.getRank())) {
      return operation->emitError("Vulkan V2 transpose permutation is invalid");
    }
    SmallVector<bool> seen(permutation.size(), false);
    for (std::size_t axis = 0; axis < permutation.size(); ++axis) {
      const std::int64_t sourceAxis = permutation[axis];
      if (sourceAxis < 0 ||
          static_cast<std::size_t>(sourceAxis) >= permutation.size() ||
          seen[static_cast<std::size_t>(sourceAxis)] ||
          outputType.getDimSize(axis) != inputType.getDimSize(sourceAxis)) {
        return operation->emitError(
          "Vulkan V2 transpose permutation is invalid");
      }
      seen[static_cast<std::size_t>(sourceAxis)] = true;
    }
  }
  return success();
}

Value constantIndex(OpBuilder& builder, Location location, std::int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

void emitCopyBody(OpBuilder& builder,
                  Location location,
                  Block& entry,
                  RankedTensorType inputType,
                  RankedTensorType outputType,
                  ArrayRef<std::int64_t> permutation) {
  Value index = builder.create<gpu::GlobalIdOp>(location, gpu::Dimension::x);
  Value outputCount = constantIndex(
    builder, location, static_cast<std::int64_t>(*elementCount(outputType)));
  Value inBounds = builder.create<arith::CmpIOp>(
    location, arith::CmpIPredicate::ult, index, outputCount);
  builder.create<scf::IfOp>(
    location, inBounds, [&](OpBuilder& thenBuilder, Location thenLocation) {
      Value sourceIndex = index;
      if (!permutation.empty()) {
        SmallVector<Value> outputCoordinates(outputType.getRank());
        Value quotient = index;
        for (int axis = outputType.getRank() - 1; axis >= 0; --axis) {
          Value extent = constantIndex(
            thenBuilder, thenLocation, outputType.getDimSize(axis));
          outputCoordinates[axis] =
            thenBuilder.create<arith::RemUIOp>(thenLocation, quotient, extent);
          quotient =
            thenBuilder.create<arith::DivUIOp>(thenLocation, quotient, extent);
        }
        SmallVector<Value> inputCoordinates(inputType.getRank());
        for (std::size_t axis = 0; axis < permutation.size(); ++axis) {
          inputCoordinates[static_cast<std::size_t>(permutation[axis])] =
            outputCoordinates[axis];
        }
        sourceIndex = constantIndex(thenBuilder, thenLocation, 0);
        for (int axis = 0; axis < inputType.getRank(); ++axis) {
          Value extent = constantIndex(
            thenBuilder, thenLocation, inputType.getDimSize(axis));
          sourceIndex = thenBuilder.create<arith::AddIOp>(
            thenLocation,
            thenBuilder.create<arith::MulIOp>(
              thenLocation, sourceIndex, extent),
            inputCoordinates[axis]);
        }
      }
      Value value = thenBuilder.create<memref::LoadOp>(
        thenLocation, entry.getArgument(0), ValueRange{sourceIndex});
      thenBuilder.create<memref::StoreOp>(
        thenLocation, value, entry.getArgument(1), ValueRange{index});
      thenBuilder.create<scf::YieldOp>(thenLocation);
    });
}

struct Conv2DSpec final {
  RankedTensorType input_type;
  RankedTensorType filter_type;
  RankedTensorType conv_output_type;
  RankedTensorType public_output_type;
  DenseElementsAttr weights;
  DenseElementsAttr bias;
  std::int64_t input_channels;
  std::int64_t input_height;
  std::int64_t input_width;
  std::int64_t output_channels;
  std::int64_t output_height;
  std::int64_t output_width;
  std::int64_t kernel_height;
  std::int64_t kernel_width;
  std::int64_t stride_height;
  std::int64_t stride_width;
  std::int64_t pad_top;
  std::int64_t pad_left;
  std::int64_t pad_bottom;
  std::int64_t pad_right;
  Operation* bias_generic = nullptr;
  Operation* input_transpose = nullptr;
  Operation* output_transpose = nullptr;
  Operation* output_slice = nullptr;
  Operation* padding = nullptr;
  std::string weights_sha256;
  std::string bias_sha256;
};

LogicalResult verifyConvSubgraph(func::FuncOp function,
                                 linalg::Conv2DNhwcHwcfOp convolution,
                                 RankedTensorType publicOutputType,
                                 Conv2DSpec& spec) {
  auto rejectConv = [&](StringRef reason) {
    return convolution.emitOpError()
           << "is not supported by Vulkan V2.2: " << reason;
  };
  if (function.getNumArguments() != 1 || function.getNumResults() != 1 ||
      function.isExternal()) {
    return function.emitError(
      "Vulkan V2.2 Conv2D requires one model input and one output");
  }
  spec.input_type =
    dyn_cast<RankedTensorType>(function.getArgument(0).getType());
  spec.public_output_type = publicOutputType;
  auto inputBytes = spec.input_type ? elementCount(spec.input_type)
                                    : FailureOr<std::uint64_t>(failure());
  auto publicBytes = elementCount(publicOutputType);
  if (failed(inputBytes) || failed(publicBytes) ||
      spec.input_type.getRank() != 3 || publicOutputType.getRank() != 3) {
    return rejectConv("requires static rank-three FP32 CHW input/output");
  }
  spec.input_channels = spec.input_type.getDimSize(0);
  spec.input_height = spec.input_type.getDimSize(1);
  spec.input_width = spec.input_type.getDimSize(2);

  if (convolution.getInputs().size() != 2 ||
      convolution.getOutputs().size() != 1 ||
      convolution->getNumResults() != 1) {
    return rejectConv("requires two inputs, one output, and one result");
  }
  spec.filter_type =
    dyn_cast<RankedTensorType>(convolution.getInputs()[1].getType());
  spec.conv_output_type =
    dyn_cast<RankedTensorType>(convolution->getResult(0).getType());
  if (!spec.filter_type || !spec.conv_output_type ||
      failed(elementCount(spec.filter_type)) ||
      failed(elementCount(spec.conv_output_type)) ||
      spec.filter_type.getRank() != 4 || spec.conv_output_type.getRank() != 4) {
    return rejectConv("requires static rank-four FP32 filter/output tensors");
  }
  spec.kernel_height = spec.filter_type.getDimSize(0);
  spec.kernel_width = spec.filter_type.getDimSize(1);
  if (!((spec.kernel_height == 1 && spec.kernel_width == 1) ||
        (spec.kernel_height == 3 && spec.kernel_width == 3))) {
    return rejectConv("supports only 1x1 and 3x3 kernels");
  }
  if (spec.filter_type.getDimSize(2) != spec.input_channels) {
    return rejectConv("filter input channels must match the model input");
  }
  spec.output_channels = spec.filter_type.getDimSize(3);

  auto readPair = [&](DenseIntElementsAttr attribute,
                      StringRef name,
                      std::int64_t& first,
                      std::int64_t& second) -> LogicalResult {
    if (!attribute || attribute.getNumElements() != 2) {
      return convolution.emitOpError()
             << "is not supported by Vulkan V2.2: " << name
             << " must contain two static values";
    }
    auto values = attribute.getValues<APInt>();
    auto iterator = values.begin();
    first = (*iterator++).getSExtValue();
    second = (*iterator).getSExtValue();
    return success();
  };
  if (failed(readPair(convolution.getStrides(),
                      "stride",
                      spec.stride_height,
                      spec.stride_width))) {
    return failure();
  }
  std::int64_t dilationHeight = 0;
  std::int64_t dilationWidth = 0;
  if (failed(readPair(convolution.getDilations(),
                      "dilation",
                      dilationHeight,
                      dilationWidth))) {
    return failure();
  }
  if (dilationHeight != 1 || dilationWidth != 1) {
    return rejectConv("dilation must be one");
  }
  if ((spec.stride_height != 1 && spec.stride_height != 2) ||
      (spec.stride_width != 1 && spec.stride_width != 2)) {
    return rejectConv("stride must be one or two on each axis");
  }

  auto convInput = convolution.getInputs()[0];
  auto pad = convInput.getDefiningOp<tensor::PadOp>();
  spec.pad_top = 0;
  spec.pad_left = 0;
  spec.pad_bottom = 0;
  spec.pad_right = 0;
  std::int64_t rawPadBottom = 0;
  std::int64_t rawPadRight = 0;
  if (pad) {
    spec.padding = pad;
    const ArrayRef<std::int64_t> low = pad.getStaticLow();
    const ArrayRef<std::int64_t> high = pad.getStaticHigh();
    if (!pad.getLow().empty() || !pad.getHigh().empty() || low.size() != 4 ||
        high.size() != 4 || low[0] != 0 || low[3] != 0 || high[0] != 0 ||
        high[3] != 0) {
      return rejectConv("padding must be static and limited to spatial axes");
    }
    spec.pad_top = low[1];
    spec.pad_left = low[2];
    rawPadBottom = high[1];
    rawPadRight = high[2];
    spec.pad_bottom =
      std::max<std::int64_t>(0, high[1] - (spec.stride_height - 1));
    spec.pad_right =
      std::max<std::int64_t>(0, high[2] - (spec.stride_width - 1));
    auto yield =
      dyn_cast<tensor::YieldOp>(pad.getRegion().front().getTerminator());
    auto zero = yield ? yield.getValue().getDefiningOp<arith::ConstantOp>()
                      : arith::ConstantOp();
    auto zeroValue = zero ? dyn_cast<FloatAttr>(zero.getValue()) : FloatAttr();
    if (!zeroValue || !zeroValue.getValue().isZero() ||
        zeroValue.getValue().isNegative()) {
      return rejectConv("padding value must be positive zero");
    }
    convInput = pad.getSource();
  }
  if ((spec.kernel_height == 1 &&
       (spec.pad_top != 0 || spec.pad_bottom != 0 || spec.pad_left != 0 ||
        spec.pad_right != 0)) ||
      (spec.kernel_height == 3 && (spec.pad_top > 1 || spec.pad_bottom > 1 ||
                                   spec.pad_left > 1 || spec.pad_right > 1))) {
    return rejectConv("padding is outside the supported 1x1/3x3 subset");
  }

  Value inputBase = convInput;
  while (Operation* definition = inputBase.getDefiningOp()) {
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ReshapeOp>(
          definition)) {
      inputBase = definition->getOperand(0);
      continue;
    }
    if (auto transpose = dyn_cast<linalg::TransposeOp>(definition)) {
      const ArrayRef<std::int64_t> permutation = transpose.getPermutation();
      if (permutation != ArrayRef<std::int64_t>({1, 2, 0})) {
        return rejectConv("input transpose must map CHW to HWC");
      }
      spec.input_transpose = definition;
      inputBase = transpose.getInput();
      continue;
    }
    break;
  }
  if (inputBase != function.getArgument(0)) {
    return rejectConv("input layout path must originate at the model input");
  }

  auto inputLayoutType =
    dyn_cast<RankedTensorType>(convolution.getInputs()[0].getType());
  if (!inputLayoutType || inputLayoutType.getRank() != 4 ||
      inputLayoutType.getDimSize(0) != 1 ||
      inputLayoutType.getDimSize(1) !=
        spec.input_height + spec.pad_top + rawPadBottom ||
      inputLayoutType.getDimSize(2) !=
        spec.input_width + spec.pad_left + rawPadRight ||
      inputLayoutType.getDimSize(3) != spec.input_channels) {
    return rejectConv(
      "input layout/padding does not match static CHW dimensions");
  }

  const std::int64_t paddedHeight = inputLayoutType.getDimSize(1);
  const std::int64_t paddedWidth = inputLayoutType.getDimSize(2);
  const std::int64_t effectiveHeight = spec.kernel_height;
  const std::int64_t effectiveWidth = spec.kernel_width;
  const std::int64_t semanticHeight =
    spec.input_height + spec.pad_top + spec.pad_bottom;
  const std::int64_t semanticWidth =
    spec.input_width + spec.pad_left + spec.pad_right;
  if (paddedHeight < effectiveHeight || paddedWidth < effectiveWidth ||
      semanticHeight < effectiveHeight || semanticWidth < effectiveWidth) {
    return rejectConv("kernel exceeds the padded input dimensions");
  }
  const auto convolutionHeight =
    ((paddedHeight - effectiveHeight) / spec.stride_height) + 1;
  const auto convolutionWidth =
    ((paddedWidth - effectiveWidth) / spec.stride_width) + 1;
  spec.output_height =
    ((semanticHeight - effectiveHeight) / spec.stride_height) + 1;
  spec.output_width =
    ((semanticWidth - effectiveWidth) / spec.stride_width) + 1;
  if (publicOutputType.getDimSize(0) != spec.output_channels ||
      publicOutputType.getDimSize(1) != spec.output_height ||
      publicOutputType.getDimSize(2) != spec.output_width ||
      spec.conv_output_type.getDimSize(0) != 1 ||
      spec.conv_output_type.getDimSize(1) != convolutionHeight ||
      spec.conv_output_type.getDimSize(2) != convolutionWidth ||
      spec.conv_output_type.getDimSize(3) != spec.output_channels) {
    return rejectConv("output type does not match checked convolution shape");
  }

  auto weightConstant =
    convolution.getInputs()[1].getDefiningOp<arith::ConstantOp>();
  spec.weights = weightConstant
                   ? dyn_cast<DenseElementsAttr>(weightConstant.getValue())
                   : DenseElementsAttr();
  if (!spec.weights || !spec.weights.getElementType().isF32() ||
      spec.weights.getNumElements() != spec.kernel_height * spec.kernel_width *
                                         spec.input_channels *
                                         spec.output_channels) {
    return rejectConv("filter must be a static FP32 HWCF constant");
  }
  spec.weights_sha256 = sha256ConvWeightsOIHW(spec.weights,
                                              spec.kernel_height,
                                              spec.kernel_width,
                                              spec.input_channels,
                                              spec.output_channels);

  Value init = convolution.getOutputs().front();
  if (auto generic = init.getDefiningOp<linalg::GenericOp>()) {
    if (generic.getInputs().size() != 1 || generic.getOutputs().size() != 1 ||
        generic.getResult(0) != init) {
      return rejectConv(
        "bias initialization must broadcast a static FP32 vector");
    }
    auto biasType =
      dyn_cast<RankedTensorType>(generic.getInputs().front().getType());
    if (!biasType || biasType.getRank() != 1 ||
        biasType.getDimSize(0) != spec.output_channels ||
        !biasType.getElementType().isF32()) {
      return rejectConv(
        "bias initialization must broadcast a static FP32 vector");
    }
    const auto maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity() ||
        maps[0].getNumResults() != 1 ||
        !isa<AffineDimExpr>(maps[0].getResult(0)) ||
        cast<AffineDimExpr>(maps[0].getResult(0)).getPosition() != 3 ||
        !llvm::all_of(generic.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return type == utils::IteratorType::parallel;
                      })) {
      return rejectConv("bias initialization must index by the output channel");
    }
    auto yield = dyn_cast<linalg::YieldOp>(generic.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1 ||
        yield.getValues().front() != generic.getBody()->getArgument(0)) {
      return rejectConv(
        "bias initialization must copy the broadcast bias value");
    }
    auto biasConstant =
      generic.getInputs()[0].getDefiningOp<arith::ConstantOp>();
    spec.bias = biasConstant
                  ? dyn_cast<DenseElementsAttr>(biasConstant.getValue())
                  : DenseElementsAttr();
    if (!spec.bias || !spec.bias.getElementType().isF32() ||
        spec.bias.getNumElements() != spec.output_channels) {
      return rejectConv(
        "bias must be a static FP32 vector matching output channels");
    }
    spec.bias_generic = generic;
    spec.bias_sha256 = sha256Elements(spec.bias);
  } else if (init.getDefiningOp<tensor::EmptyOp>()) {
    spec.bias_sha256 = sha256ZeroBytes(
      static_cast<std::size_t>(spec.output_channels) * sizeof(float));
  } else {
    return rejectConv("output init must be empty or a broadcast FP32 bias");
  }

  Value returned = function.getBody().front().getTerminator()->getOperand(0);
  auto outputTranspose = returned.getDefiningOp<linalg::TransposeOp>();
  if (!outputTranspose ||
      outputTranspose.getPermutation() != ArrayRef<std::int64_t>({2, 0, 1})) {
    return rejectConv(
      "model output must transpose NHWC convolution output to CHW");
  }
  spec.output_transpose = outputTranspose;
  Value outputSource = outputTranspose.getInput();
  while (Operation* definition = outputSource.getDefiningOp()) {
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ReshapeOp>(
          definition)) {
      outputSource = definition->getOperand(0);
      continue;
    }
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(definition)) {
      const ArrayRef<std::int64_t> offsets = slice.getStaticOffsets();
      const ArrayRef<std::int64_t> sizes = slice.getStaticSizes();
      const ArrayRef<std::int64_t> strides = slice.getStaticStrides();
      if (slice->getNumOperands() != 1 || offsets.size() != 4 ||
          sizes.size() != 4 || strides.size() != 4 || offsets[0] != 0 ||
          offsets[1] != 0 || offsets[2] != 0 || offsets[3] != 0 ||
          sizes[0] != 1 || sizes[1] != spec.output_height ||
          sizes[2] != spec.output_width || sizes[3] != spec.output_channels ||
          !llvm::all_of(strides,
                        [](std::int64_t stride) { return stride == 1; })) {
        return rejectConv(
          "output slice must crop only the padded trailing region");
      }
      spec.output_slice = definition;
      outputSource = slice.getSource();
      continue;
    }
    break;
  }
  outputSource = stripLayoutViews(outputSource);
  if (outputSource != convolution->getResult(0)) {
    return rejectConv(
      "output layout path must originate at the convolution result");
  }

  bool invalidOperation = false;
  function.walk([&](Operation* operation) {
    if (isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
          operation)) {
      auto inputType =
        dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
      auto resultType =
        dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      auto inputCount = inputType ? elementCount(inputType)
                                  : FailureOr<std::uint64_t>(failure());
      auto resultCount = resultType ? elementCount(resultType)
                                    : FailureOr<std::uint64_t>(failure());
      bool valid = succeeded(inputCount) && succeeded(resultCount) &&
                   *inputCount == *resultCount;
      if (auto reshape = dyn_cast<tensor::ReshapeOp>(operation)) {
        auto shapeType =
          dyn_cast<RankedTensorType>(reshape.getShape().getType());
        auto constant = reshape.getShape().getDefiningOp<arith::ConstantOp>();
        auto values = constant
                        ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                        : DenseIntElementsAttr();
        valid &= shapeType && shapeType.getElementType().isIndex() &&
                 shapeType.getRank() == 1 && resultType && values &&
                 static_cast<std::int64_t>(values.getNumElements()) ==
                   resultType.getRank();
        if (valid) {
          std::size_t dimension = 0;
          for (const APInt& extent : values.getValues<APInt>()) {
            valid &=
              extent.getSExtValue() == resultType.getDimSize(dimension++);
          }
        }
      }
      if (!valid) {
        operation->emitError(
          "Vulkan V2.2 layout view must preserve a static FP32 element count");
        invalidOperation = true;
        return;
      }
    }
    const bool convolutionBodyArithmetic =
      isa<arith::MulFOp, arith::AddFOp>(operation) &&
      operation->getParentOfType<linalg::Conv2DNhwcHwcfOp>() == convolution;
    const bool allowed =
      isa<func::FuncOp,
          func::ReturnOp,
          arith::ConstantOp,
          tensor::EmptyOp,
          tensor::ExpandShapeOp,
          tensor::CollapseShapeOp,
          tensor::ReshapeOp,
          tensor::YieldOp,
          linalg::YieldOp>(operation) ||
      operation == convolution.getOperation() ||
      operation == spec.bias_generic || operation == spec.input_transpose ||
      operation == spec.output_transpose || operation == spec.output_slice ||
      operation == spec.padding || convolutionBodyArithmetic;
    if (!allowed) {
      operation->emitError("is not supported in the V2.2 Conv2D subgraph");
      invalidOperation = true;
    }
  });
  if (invalidOperation) {
    return failure();
  }
  return success();
}

LogicalResult createConvKernel(ModuleOp module,
                               func::FuncOp function,
                               linalg::Conv2DNhwcHwcfOp convolution,
                               RankedTensorType publicOutputType,
                               StringRef planPath) {
  Conv2DSpec spec;
  if (failed(
        verifyConvSubgraph(function, convolution, publicOutputType, spec))) {
    return failure();
  }
  MLIRContext* context = module.getContext();
  OpBuilder builder(context);
  Location location = convolution.getLoc();
  auto targetEnv = parseAttribute(kTargetEnvironment, context);
  auto entryABI = parseAttribute(kEntryPointABI, context);
  if (!targetEnv || !entryABI) {
    return module.emitError(
      "failed to construct Vulkan SPIR-V target attributes");
  }
  module->setAttr("gpu.container_module", builder.getUnitAttr());
  module->setAttr("spirv.target_env", targetEnv);

  const auto inputCount = static_cast<std::uint64_t>(
    spec.input_channels * spec.input_height * spec.input_width);
  const auto weightCount =
    static_cast<std::uint64_t>(spec.kernel_height * spec.kernel_width *
                               spec.input_channels * spec.output_channels);
  const auto outputCount = *elementCount(publicOutputType);
  const auto storageClass =
    spirv::StorageClassAttr::get(context, spirv::StorageClass::StorageBuffer);
  SmallVector<Type> argumentTypes;
  for (std::uint64_t count : {inputCount,
                              weightCount,
                              static_cast<std::uint64_t>(spec.output_channels),
                              outputCount}) {
    const auto extent = static_cast<std::int64_t>(count);
    argumentTypes.push_back(MemRefType::get(ArrayRef<std::int64_t>(&extent, 1),
                                            builder.getF32Type(),
                                            AffineMap(),
                                            storageClass));
  }
  FunctionType kernelType = FunctionType::get(context, argumentTypes, {});
  SmallVector<NamedAttribute> kernelAttrs{
    builder.getNamedAttr("gpu.kernel", builder.getUnitAttr()),
    builder.getNamedAttr("spirv.entry_point_abi",
                         parseAttribute(kEntryPointABI, context)),
  };
  const std::string moduleName = function.getSymName().str() + "_vulkan";
  builder.setInsertionPointToEnd(module.getBody());
  auto gpuModule = gpu::GPUModuleOp::create(
    builder, location, moduleName, ArrayRef<Attribute>{});
  Block* gpuModuleBody = gpuModule.getBody();
  if (!gpuModuleBody) {
    gpuModule->getRegion(0).push_back(new Block());
    gpuModuleBody = gpuModule.getBody();
  }
  builder.setInsertionPointToEnd(gpuModuleBody);
  auto gpuFunction = gpu::GPUFuncOp::create(
    builder, location, "conv2d", kernelType, {}, {}, kernelAttrs);
  Block* entry = &gpuFunction.getBody().front();
  builder.setInsertionPointToStart(entry);

  Value globalIndex =
    builder.create<gpu::GlobalIdOp>(location, gpu::Dimension::x);
  Value total =
    constantIndex(builder, location, static_cast<std::int64_t>(outputCount));
  Value inBounds = builder.create<arith::CmpIOp>(
    location, arith::CmpIPredicate::ult, globalIndex, total);
  builder.create<scf::IfOp>(
    location, inBounds, [&](OpBuilder& body, Location bodyLoc) {
      Value outputHeight = constantIndex(body, bodyLoc, spec.output_height);
      Value outputWidth = constantIndex(body, bodyLoc, spec.output_width);
      Value spatial =
        body.create<arith::MulIOp>(bodyLoc, outputHeight, outputWidth);
      Value outputChannel =
        body.create<arith::DivUIOp>(bodyLoc, globalIndex, spatial);
      Value spatialIndex =
        body.create<arith::RemUIOp>(bodyLoc, globalIndex, spatial);
      Value outputY =
        body.create<arith::DivUIOp>(bodyLoc, spatialIndex, outputWidth);
      Value outputX =
        body.create<arith::RemUIOp>(bodyLoc, spatialIndex, outputWidth);
      Value accumulator = body.create<memref::LoadOp>(
        bodyLoc, entry->getArgument(2), ValueRange{outputChannel});
      Value strideY = constantIndex(body, bodyLoc, spec.stride_height);
      Value strideX = constantIndex(body, bodyLoc, spec.stride_width);
      Value padTop = constantIndex(body, bodyLoc, spec.pad_top);
      Value padLeft = constantIndex(body, bodyLoc, spec.pad_left);
      Value paddedHeight =
        constantIndex(body, bodyLoc, spec.input_height + spec.pad_top);
      Value paddedWidth =
        constantIndex(body, bodyLoc, spec.input_width + spec.pad_left);
      for (std::int64_t kernelY = 0; kernelY < spec.kernel_height; ++kernelY) {
        for (std::int64_t kernelX = 0; kernelX < spec.kernel_width; ++kernelX) {
          Value paddedY = body.create<arith::AddIOp>(
            bodyLoc,
            body.create<arith::MulIOp>(bodyLoc, outputY, strideY),
            constantIndex(body, bodyLoc, kernelY));
          Value paddedX = body.create<arith::AddIOp>(
            bodyLoc,
            body.create<arith::MulIOp>(bodyLoc, outputX, strideX),
            constantIndex(body, bodyLoc, kernelX));
          Value yAfterLowPad = body.create<arith::CmpIOp>(
            bodyLoc, arith::CmpIPredicate::uge, paddedY, padTop);
          Value yBeforeHighPad = body.create<arith::CmpIOp>(
            bodyLoc, arith::CmpIPredicate::ult, paddedY, paddedHeight);
          Value xAfterLowPad = body.create<arith::CmpIOp>(
            bodyLoc, arith::CmpIPredicate::uge, paddedX, padLeft);
          Value xBeforeHighPad = body.create<arith::CmpIOp>(
            bodyLoc, arith::CmpIPredicate::ult, paddedX, paddedWidth);
          Value yInRange =
            body.create<arith::AndIOp>(bodyLoc, yAfterLowPad, yBeforeHighPad);
          Value xInRange =
            body.create<arith::AndIOp>(bodyLoc, xAfterLowPad, xBeforeHighPad);
          Value inRange =
            body.create<arith::AndIOp>(bodyLoc, yInRange, xInRange);
          Value inputY = body.create<arith::SubIOp>(bodyLoc, paddedY, padTop);
          Value inputX = body.create<arith::SubIOp>(bodyLoc, paddedX, padLeft);
          for (std::int64_t inputChannel = 0;
               inputChannel < spec.input_channels;
               ++inputChannel) {
            auto conditional = body.create<scf::IfOp>(
              bodyLoc, TypeRange{builder.getF32Type()}, inRange, true, true);
            OpBuilder thenBuilder = conditional.getThenBodyBuilder();
            Value inputChannelValue =
              constantIndex(thenBuilder, bodyLoc, inputChannel);
            Value inputHeightValue =
              constantIndex(thenBuilder, bodyLoc, spec.input_height);
            Value inputWidthValue =
              constantIndex(thenBuilder, bodyLoc, spec.input_width);
            Value inputOffset = thenBuilder.create<arith::AddIOp>(
              bodyLoc,
              thenBuilder.create<arith::MulIOp>(
                bodyLoc,
                thenBuilder.create<arith::AddIOp>(
                  bodyLoc,
                  thenBuilder.create<arith::MulIOp>(
                    bodyLoc, inputChannelValue, inputHeightValue),
                  inputY),
                inputWidthValue),
              inputX);
            Value weightOffset = constantIndex(
              thenBuilder,
              bodyLoc,
              ((kernelY * spec.kernel_width + kernelX) * spec.input_channels +
               inputChannel) *
                spec.output_channels);
            weightOffset = thenBuilder.create<arith::AddIOp>(
              bodyLoc, weightOffset, outputChannel);
            Value input = thenBuilder.create<memref::LoadOp>(
              bodyLoc, entry->getArgument(0), ValueRange{inputOffset});
            Value weight = thenBuilder.create<memref::LoadOp>(
              bodyLoc, entry->getArgument(1), ValueRange{weightOffset});
            Value product =
              thenBuilder.create<arith::MulFOp>(bodyLoc, input, weight);
            Value sum =
              thenBuilder.create<arith::AddFOp>(bodyLoc, accumulator, product);
            thenBuilder.create<scf::YieldOp>(bodyLoc, sum);
            OpBuilder elseBuilder = conditional.getElseBodyBuilder();
            elseBuilder.create<scf::YieldOp>(bodyLoc, accumulator);
            accumulator = conditional.getResult(0);
          }
        }
      }
      body.create<memref::StoreOp>(
        bodyLoc, accumulator, entry->getArgument(3), ValueRange{globalIndex});
      body.create<scf::YieldOp>(bodyLoc);
    });
  builder.setInsertionPointToEnd(entry);
  builder.create<gpu::ReturnOp>(location);

  const std::string zeroBiasHash = spec.bias_sha256;
  const std::uint64_t inputBytes = inputCount * sizeof(float);
  const std::uint64_t weightBytes = weightCount * sizeof(float);
  const std::uint64_t biasBytes = spec.output_channels * sizeof(float);
  const std::uint64_t outputBytes = outputCount * sizeof(float);
  const std::string identity =
    function.getSymName().str() + spec.weights_sha256 + zeroBiasHash;
  VulkanExecutionPlan plan{
    .schema_version = ncnn_mlir::vulkan::kExecutionPlanSchemaVersion,
    .model = function.getSymName().str(),
    .build_identity =
      std::format("vulkan-v2-{:016x}", ncnn_mlir::stableHash64(identity)),
    .host_triple = llvm::sys::getDefaultTargetTriple(),
    .device_target = "vulkan1.1-spirv1.0",
    .resources = {},
    .kernels = {},
    .inputs = {1},
    .outputs = {4},
    .plan_hash = "",
  };
  plan.resources.push_back(TensorResource{
    .id = 1,
    .kind = ResourceKind::Input,
    .name = "input",
    .shape = planShape(spec.input_type),
    .logical_layout = "CHW",
    .physical_layout = "CHW",
    .host_layout = "CHW",
    .layout_transform = std::nullopt,
    .weight_sha256 = "",
    .byte_size = inputBytes,
    .first_kernel = 0,
    .last_kernel = 0,
  });
  plan.resources.push_back(TensorResource{
    .id = 2,
    .kind = ResourceKind::Weight,
    .name = "weights",
    .shape = {static_cast<std::uint64_t>(spec.output_channels),
              static_cast<std::uint64_t>(spec.input_channels),
              static_cast<std::uint64_t>(spec.kernel_height),
              static_cast<std::uint64_t>(spec.kernel_width)},
    .logical_layout = "OIHW",
    .physical_layout = "HWCF",
    .host_layout = "",
    .layout_transform = std::nullopt,
    .weight_sha256 = spec.weights_sha256,
    .byte_size = weightBytes,
    .first_kernel = 0,
    .last_kernel = 0,
  });
  plan.resources.push_back(TensorResource{
    .id = 3,
    .kind = ResourceKind::Weight,
    .name = "bias",
    .shape = {static_cast<std::uint64_t>(spec.output_channels)},
    .logical_layout = "O",
    .physical_layout = "O",
    .host_layout = "",
    .layout_transform = std::nullopt,
    .weight_sha256 = spec.bias_sha256,
    .byte_size = biasBytes,
    .first_kernel = 0,
    .last_kernel = 0,
  });
  plan.resources.push_back(TensorResource{
    .id = 4,
    .kind = ResourceKind::Output,
    .name = "output",
    .shape = planShape(spec.public_output_type),
    .logical_layout = "CHW",
    .physical_layout = "CHW",
    .host_layout = "CHW",
    .layout_transform = std::nullopt,
    .weight_sha256 = "",
    .byte_size = outputBytes,
    .first_kernel = 0,
    .last_kernel = 0,
  });
  plan.kernels.push_back(KernelExecution{
    .id = 0,
    .operation = "conv2d",
    .entry_point = "conv2d",
    .local_size = {kLocalSize, 1, 1},
    .reads = {1, 2, 3},
    .writes = {4},
    .bindings = {{.set = 0,
                  .binding = 0,
                  .resource_id = 1,
                  .access = ResourceAccess::Read},
                 {.set = 0,
                  .binding = 1,
                  .resource_id = 2,
                  .access = ResourceAccess::Read},
                 {.set = 0,
                  .binding = 2,
                  .resource_id = 3,
                  .access = ResourceAccess::Read},
                 {.set = 0,
                  .binding = 3,
                  .resource_id = 4,
                  .access = ResourceAccess::Write}},
    .push_constants = {},
  });
  if (auto valid = ncnn_mlir::vulkan::verifyExecutionPlan(plan); !valid) {
    return module.emitError()
           << "invalid generated Vulkan Conv2D execution plan: "
           << valid.error();
  }
  if (failed(writePlan(plan, planPath))) {
    return module.emitError()
           << "failed to write Vulkan execution plan to '" << planPath << "'";
  }
  return success();
}

LogicalResult createKernel(ModuleOp module,
                           func::FuncOp function,
                           Operation* operation,
                           RankedTensorType outputType,
                           StringRef planPath) {
  MLIRContext* context = module.getContext();
  const std::string kind = operationName(operation);
  const bool isBinary = kind == "add";
  const unsigned inputCount = isBinary ? 2U : 1U;
  SmallVector<RankedTensorType> inputTypes;
  inputTypes.reserve(inputCount);
  for (unsigned index = 0; index < inputCount; ++index) {
    inputTypes.push_back(
      cast<RankedTensorType>(function.getArgument(index).getType()));
  }
  auto count = elementCount(outputType);
  if (failed(count)) {
    return failure();
  }

  OpBuilder builder(context);
  Location location = operation ? operation->getLoc() : function.getLoc();
  auto targetEnv = parseAttribute(kTargetEnvironment, context);
  auto entryABI = parseAttribute(kEntryPointABI, context);
  if (!targetEnv || !entryABI) {
    return module.emitError(
      "failed to construct Vulkan SPIR-V target attributes");
  }
  module->setAttr("gpu.container_module", builder.getUnitAttr());
  module->setAttr("spirv.target_env", targetEnv);

  auto storageClass =
    spirv::StorageClassAttr::get(context, spirv::StorageClass::StorageBuffer);
  SmallVector<Type> argumentTypes;
  for (RankedTensorType inputType : inputTypes) {
    const std::int64_t inputCountValue =
      static_cast<std::int64_t>(*elementCount(inputType));
    const ArrayRef<std::int64_t> shape(&inputCountValue, 1);
    argumentTypes.push_back(
      MemRefType::get(shape, builder.getF32Type(), AffineMap(), storageClass));
  }
  const auto outputCountValue = static_cast<std::int64_t>(*count);
  argumentTypes.push_back(
    MemRefType::get(ArrayRef<std::int64_t>(&outputCountValue, 1),
                    builder.getF32Type(),
                    AffineMap(),
                    storageClass));
  FunctionType kernelType = FunctionType::get(context, argumentTypes, {});
  SmallVector<NamedAttribute> kernelAttrs{
    builder.getNamedAttr("gpu.kernel", builder.getUnitAttr()),
    builder.getNamedAttr("spirv.entry_point_abi", entryABI),
  };

  std::string moduleName = function.getSymName().str() + "_vulkan";
  builder.setInsertionPointToEnd(module.getBody());
  auto gpuModule = gpu::GPUModuleOp::create(
    builder, location, moduleName, ArrayRef<Attribute>{});
  Block* gpuModuleBody = gpuModule.getBody();
  if (!gpuModuleBody) {
    gpuModule->getRegion(0).push_back(new Block());
    gpuModuleBody = gpuModule.getBody();
  }
  builder.setInsertionPointToEnd(gpuModuleBody);
  auto gpuFunction = gpu::GPUFuncOp::create(
    builder, location, kind, kernelType, {}, {}, kernelAttrs);
  Block* entry = &gpuFunction.getBody().front();
  builder.setInsertionPointToStart(entry);

  if (kind == "add" || kind == "relu") {
    Value index = builder.create<gpu::GlobalIdOp>(location, gpu::Dimension::x);
    Value outputCount =
      constantIndex(builder, location, static_cast<std::int64_t>(*count));
    Value inBounds = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::ult, index, outputCount);
    builder.create<scf::IfOp>(
      location, inBounds, [&](OpBuilder& thenBuilder, Location thenLocation) {
        Value lhs = thenBuilder.create<memref::LoadOp>(
          thenLocation, entry->getArgument(0), ValueRange{index});
        Value value = lhs;
        if (kind == "add") {
          Value rhs = thenBuilder.create<memref::LoadOp>(
            thenLocation, entry->getArgument(1), ValueRange{index});
          value = thenBuilder.create<arith::AddFOp>(thenLocation, lhs, rhs);
        } else {
          Value zero = thenBuilder.create<arith::ConstantOp>(
            thenLocation, thenBuilder.getF32FloatAttr(0.0F));
          value =
            thenBuilder.create<arith::MaximumFOp>(thenLocation, lhs, zero);
        }
        thenBuilder.create<memref::StoreOp>(thenLocation,
                                            value,
                                            entry->getArgument(inputCount),
                                            ValueRange{index});
        thenBuilder.create<scf::YieldOp>(thenLocation);
      });
  } else {
    ArrayRef<std::int64_t> permutation;
    if (auto transpose = dyn_cast_or_null<linalg::TransposeOp>(operation)) {
      permutation = transpose.getPermutation();
    }
    emitCopyBody(
      builder, location, *entry, inputTypes.front(), outputType, permutation);
  }
  builder.setInsertionPointToEnd(entry);
  builder.create<gpu::ReturnOp>(location);

  std::vector<std::uint32_t> readIds;
  std::vector<DescriptorBinding> bindings;
  VulkanExecutionPlan plan{
    .schema_version = ncnn_mlir::vulkan::kExecutionPlanSchemaVersion,
    .model = function.getSymName().str(),
    .build_identity = std::format(
      "vulkan-v2-{:016x}", ncnn_mlir::stableHash64(function.getSymName())),
    .host_triple = llvm::sys::getDefaultTargetTriple(),
    .device_target = "vulkan1.1-spirv1.0",
    .resources = {},
    .kernels = {},
    .inputs = {},
    .outputs = {},
    .plan_hash = "",
  };
  const std::uint64_t bytes = *count * sizeof(float);
  for (unsigned index = 0; index < inputCount; ++index) {
    const std::uint32_t id = index + 1;
    plan.inputs.push_back(id);
    readIds.push_back(id);
    plan.resources.push_back(TensorResource{
      .id = id,
      .kind = ResourceKind::Input,
      .name = std::format("input_{}", index),
      .shape = planShape(inputTypes[index]),
      .logical_layout = "linear",
      .physical_layout = "linear",
      .host_layout = "linear",
      .layout_transform = std::nullopt,
      .weight_sha256 = "",
      .byte_size = bytes,
      .first_kernel = 0,
      .last_kernel = 0,
    });
    bindings.push_back(DescriptorBinding{
      .set = 0,
      .binding = index,
      .resource_id = id,
      .access = ResourceAccess::Read,
    });
  }
  const std::uint32_t outputId = inputCount + 1;
  plan.outputs.push_back(outputId);
  plan.resources.push_back(TensorResource{
    .id = outputId,
    .kind = ResourceKind::Output,
    .name = "output",
    .shape = planShape(outputType),
    .logical_layout = "linear",
    .physical_layout = "linear",
    .host_layout = "linear",
    .layout_transform = std::nullopt,
    .weight_sha256 = "",
    .byte_size = bytes,
    .first_kernel = 0,
    .last_kernel = 0,
  });
  bindings.push_back(DescriptorBinding{
    .set = 0,
    .binding = inputCount,
    .resource_id = outputId,
    .access = ResourceAccess::Write,
  });
  plan.kernels.push_back(KernelExecution{
    .id = 0,
    .operation = kind,
    .entry_point = kind,
    .local_size = {kLocalSize, 1, 1},
    .reads = std::move(readIds),
    .writes = {outputId},
    .bindings = std::move(bindings),
    .push_constants = {},
  });
  if (auto valid = ncnn_mlir::vulkan::verifyExecutionPlan(plan); !valid) {
    return module.emitError()
           << "invalid generated Vulkan execution plan: " << valid.error();
  }
  if (failed(writePlan(plan, planPath))) {
    return module.emitError()
           << "failed to write Vulkan execution plan to '" << planPath << "'";
  }
  return success();
}

class LowerNCNNVulkanModelPass final
  : public impl::LowerNCNNVulkanModelPassBase<LowerNCNNVulkanModelPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functions;
    module.walk([&](func::FuncOp function) {
      if (!function.isExternal()) {
        functions.push_back(function);
      }
    });
    if (functions.size() != 1) {
      module.emitError(
        "Vulkan V2 single-op lowering requires exactly one model function");
      signalPassFailure();
      return;
    }
    func::FuncOp function = functions.front();
    SmallVector<linalg::Conv2DNhwcHwcfOp> convolutions;
    function.walk([&](linalg::Conv2DNhwcHwcfOp convolution) {
      convolutions.push_back(convolution);
    });
    if (!convolutions.empty()) {
      if (convolutions.size() != 1) {
        function.emitError(
          "Vulkan V2.2 supports exactly one Conv2D operation per fixture");
        signalPassFailure();
        return;
      }
      auto returnOp =
        dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
      auto outputType =
        returnOp && returnOp.getNumOperands() == 1
          ? dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType())
          : RankedTensorType();
      if (!outputType ||
          failed(createConvKernel(
            module, function, convolutions.front(), outputType, planPath))) {
        signalPassFailure();
        return;
      }
      function.erase();
      return;
    }
    SmallVector<Operation*> computeOps;
    SmallVector<Operation*> layoutOps;
    function.walk([&](Operation* operation) {
      if (isa<linalg::GenericOp, linalg::CopyOp, linalg::TransposeOp>(
            operation)) {
        computeOps.push_back(operation);
      } else if (isa<tensor::ReshapeOp,
                     tensor::ExpandShapeOp,
                     tensor::CollapseShapeOp>(operation)) {
        layoutOps.push_back(operation);
      }
    });
    if (computeOps.size() > 1) {
      function.emitError(
        "Vulkan V2.1 currently lowers one compute operation per fixture");
      signalPassFailure();
      return;
    }
    Operation* operation =
      computeOps.empty() ? (layoutOps.size() == 1 ? layoutOps.front() : nullptr)
                         : computeOps.front();
    auto returnOp =
      dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1) {
      function.emitError(
        "Vulkan V2 single-op lowering requires one returned tensor");
      signalPassFailure();
      return;
    }
    auto outputType =
      dyn_cast<RankedTensorType>(returnOp.getOperand(0).getType());
    if (failed(verifySingleOperation(function, operation, outputType))) {
      signalPassFailure();
      return;
    }
    const bool isLayoutOnlyOp =
      operation &&
      isa<tensor::ReshapeOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
        operation);
    if (operation &&
        (isLayoutOnlyOp ? returnOp.getOperand(0) != operation->getResult(0)
                        : !isLayoutViewOf(returnOp.getOperand(0),
                                          operation->getResult(0)))) {
      operation->emitError(
        "Vulkan V2 single-op lowering requires the operation result with only "
        "layout views as output");
      signalPassFailure();
      return;
    }
    if (failed(
          createKernel(module, function, operation, outputType, planPath))) {
      signalPassFailure();
      return;
    }
    function.erase();
  }
};

}  // namespace

}  // namespace mlir::ncnn
