#include "VulkanGraphKernel.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/IR/Builders.h"

namespace mlir::ncnn::vulkan_graph {
namespace {

Value indexConstant(OpBuilder& builder, Location location, std::int64_t value) {
  return builder.create<arith::ConstantIndexOp>(location, value);
}

SmallVector<Value> coordinates(OpBuilder& builder,
                               Location location,
                               Value index,
                               RankedTensorType type) {
  SmallVector<Value> result(static_cast<std::size_t>(type.getRank()));
  for (std::int64_t dimension = type.getRank(); dimension-- > 0;) {
    Value extent = indexConstant(builder, location, type.getDimSize(dimension));
    result[dimension] = builder.create<arith::RemUIOp>(location, index, extent);
    index = builder.create<arith::DivUIOp>(location, index, extent);
  }
  return result;
}

Value linearOffset(OpBuilder& builder,
                   Location location,
                   ArrayRef<Value> coordinates,
                   RankedTensorType type) {
  Value offset = indexConstant(builder, location, 0);
  for (auto [dimension, coordinate] : llvm::enumerate(coordinates)) {
    offset = builder.create<arith::AddIOp>(
      location,
      builder.create<arith::MulIOp>(
        location,
        offset,
        indexConstant(builder, location, type.getDimSize(dimension))),
      coordinate);
  }
  return offset;
}

Value load(OpBuilder& builder, Location location, Value buffer, Value index) {
  return builder.create<memref::LoadOp>(location, buffer, ValueRange{index});
}

Value convValue(OpBuilder& builder,
                Location location,
                const GraphKernel& kernel,
                const ConvKernel& convolution,
                ArrayRef<Value> buffers,
                Value outputIndex) {
  const auto inputType = kernel.input_types[0];
  const auto filterType = kernel.input_types[1];
  auto outputCoordinates =
    coordinates(builder, location, outputIndex, kernel.output_type);
  Value outputY = outputCoordinates[1];
  Value outputX = outputCoordinates[2];
  Value outputChannel = outputCoordinates[3];
  Value accumulator = load(builder, location, buffers[2], outputChannel);
  const std::int64_t inputChannels = inputType.getDimSize(3);
  const std::int64_t kernelHeight = filterType.getDimSize(1);
  const std::int64_t kernelWidth = filterType.getDimSize(2);
  Value padTop = indexConstant(builder, location, convolution.padding[0]);
  Value padLeft = indexConstant(builder, location, convolution.padding[2]);
  Value paddedHeight = indexConstant(
    builder, location, inputType.getDimSize(1) + convolution.padding[0]);
  Value paddedWidth = indexConstant(
    builder, location, inputType.getDimSize(2) + convolution.padding[2]);
  Value strideY = indexConstant(builder, location, convolution.strides[0]);
  Value strideX = indexConstant(builder, location, convolution.strides[1]);
  Value zero = indexConstant(builder, location, 0);
  Value step = indexConstant(builder, location, 1);
  Value channels = indexConstant(builder, location, inputChannels);

  // The serial KH, KW, IC order is the shared TOSA/Linalg convolution order.
  // Keep IC as a loop rather than expanding every channel into shader code.
  for (std::int64_t kernelY = 0; kernelY < kernelHeight; ++kernelY) {
    for (std::int64_t kernelX = 0; kernelX < kernelWidth; ++kernelX) {
      Value paddedY = builder.create<arith::AddIOp>(
        location,
        builder.create<arith::MulIOp>(location, outputY, strideY),
        indexConstant(builder, location, kernelY));
      Value paddedX = builder.create<arith::AddIOp>(
        location,
        builder.create<arith::MulIOp>(location, outputX, strideX),
        indexConstant(builder, location, kernelX));
      Value yAfter = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::uge, paddedY, padTop);
      Value yBefore = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, paddedY, paddedHeight);
      Value xAfter = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::uge, paddedX, padLeft);
      Value xBefore = builder.create<arith::CmpIOp>(
        location, arith::CmpIPredicate::ult, paddedX, paddedWidth);
      Value inRange = builder.create<arith::AndIOp>(
        location,
        builder.create<arith::AndIOp>(location, yAfter, yBefore),
        builder.create<arith::AndIOp>(location, xAfter, xBefore));
      auto conditional = builder.create<scf::IfOp>(
        location, TypeRange{builder.getF32Type()}, inRange, true, true);
      OpBuilder thenBuilder = conditional.getThenBodyBuilder();
      Value inputY =
        thenBuilder.create<arith::SubIOp>(location, paddedY, padTop);
      Value inputX =
        thenBuilder.create<arith::SubIOp>(location, paddedX, padLeft);
      auto reduction = thenBuilder.create<scf::ForOp>(
        location,
        zero,
        channels,
        step,
        ValueRange{accumulator},
        [&](OpBuilder& body,
            Location bodyLocation,
            Value channel,
            ValueRange iterates) {
          SmallVector<Value> inputCoordinates{zero, inputY, inputX, channel};
          SmallVector<Value> weightCoordinates{
            outputChannel,
            indexConstant(body, bodyLocation, kernelY),
            indexConstant(body, bodyLocation, kernelX),
            channel};
          Value input =
            load(body,
                 bodyLocation,
                 buffers[0],
                 linearOffset(body, bodyLocation, inputCoordinates, inputType));
          Value weight = load(
            body,
            bodyLocation,
            buffers[1],
            linearOffset(body, bodyLocation, weightCoordinates, filterType));
          Value product =
            body.create<arith::MulFOp>(bodyLocation, input, weight);
          Value sum =
            body.create<arith::AddFOp>(bodyLocation, iterates[0], product);
          body.create<scf::YieldOp>(bodyLocation, sum);
        });
      thenBuilder.create<scf::YieldOp>(location, reduction.getResult(0));
      OpBuilder elseBuilder = conditional.getElseBodyBuilder();
      elseBuilder.create<scf::YieldOp>(location, accumulator);
      accumulator = conditional.getResult(0);
    }
  }
  return accumulator;
}

Value concatValue(OpBuilder& builder,
                  Location location,
                  const GraphKernel& kernel,
                  const ConcatKernel& concatenation,
                  ArrayRef<Value> buffers,
                  Value outputIndex) {
  auto outputCoordinates =
    coordinates(builder, location, outputIndex, kernel.output_type);
  Value axisCoordinate = outputCoordinates[concatenation.axis];
  Value value =
    builder.create<arith::ConstantOp>(location, builder.getF32FloatAttr(0.0F));
  std::int64_t begin = 0;
  for (auto [operand, type] : llvm::enumerate(kernel.input_types)) {
    const std::int64_t end = begin + type.getDimSize(concatenation.axis);
    Value low = indexConstant(builder, location, begin);
    Value high = indexConstant(builder, location, end);
    Value afterLow = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::uge, axisCoordinate, low);
    Value beforeHigh = builder.create<arith::CmpIOp>(
      location, arith::CmpIPredicate::ult, axisCoordinate, high);
    Value inRange =
      builder.create<arith::AndIOp>(location, afterLow, beforeHigh);
    auto conditional = builder.create<scf::IfOp>(
      location, TypeRange{builder.getF32Type()}, inRange, true, true);
    OpBuilder thenBuilder = conditional.getThenBodyBuilder();
    SmallVector<Value> inputCoordinates(outputCoordinates);
    inputCoordinates[concatenation.axis] =
      thenBuilder.create<arith::SubIOp>(location, axisCoordinate, low);
    Value input =
      load(thenBuilder,
           location,
           buffers[operand],
           linearOffset(thenBuilder, location, inputCoordinates, type));
    thenBuilder.create<scf::YieldOp>(location, input);
    OpBuilder elseBuilder = conditional.getElseBodyBuilder();
    elseBuilder.create<scf::YieldOp>(location, value);
    value = conditional.getResult(0);
    begin = end;
  }
  return value;
}

Value broadcastLoad(OpBuilder& builder,
                    Location location,
                    Value buffer,
                    ArrayRef<Value> outputCoordinates,
                    RankedTensorType inputType) {
  SmallVector<Value> inputCoordinates(outputCoordinates);
  for (auto [dimension, extent] : llvm::enumerate(inputType.getShape())) {
    if (extent == 1) {
      inputCoordinates[dimension] = indexConstant(builder, location, 0);
    }
  }
  return load(builder,
              location,
              buffer,
              linearOffset(builder, location, inputCoordinates, inputType));
}

Value maxPoolValue(OpBuilder& builder,
                   Location location,
                   const GraphKernel& kernel,
                   const MaxPoolKernel& pooling,
                   ArrayRef<Value> buffers,
                   Value outputIndex) {
  const auto inputType = kernel.input_types.front();
  const auto outputCoordinates =
    coordinates(builder, location, outputIndex, kernel.output_type);
  const Value outputY = outputCoordinates[1];
  const Value outputX = outputCoordinates[2];
  const Value channel = outputCoordinates[3];
  const Value zero = indexConstant(builder, location, 0);
  const Value strideY = indexConstant(builder, location, pooling.strides[0]);
  const Value strideX = indexConstant(builder, location, pooling.strides[1]);
  const Value top = indexConstant(builder, location, pooling.padding[0]);
  const Value left = indexConstant(builder, location, pooling.padding[2]);
  const Value paddedHeight = indexConstant(
    builder, location, inputType.getDimSize(1) + pooling.padding[0]);
  const Value paddedWidth = indexConstant(
    builder, location, inputType.getDimSize(2) + pooling.padding[2]);
  Value maximum = builder.create<arith::ConstantOp>(
    location, builder.getF32FloatAttr(-std::numeric_limits<float>::infinity()));

  for (std::int64_t kernelY = 0; kernelY < pooling.kernel[0]; ++kernelY) {
    for (std::int64_t kernelX = 0; kernelX < pooling.kernel[1]; ++kernelX) {
      const Value paddedY = builder.create<arith::AddIOp>(
        location,
        builder.create<arith::MulIOp>(location, outputY, strideY),
        indexConstant(builder, location, kernelY));
      const Value paddedX = builder.create<arith::AddIOp>(
        location,
        builder.create<arith::MulIOp>(location, outputX, strideX),
        indexConstant(builder, location, kernelX));
      const Value validY = builder.create<arith::AndIOp>(
        location,
        builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::uge, paddedY, top),
        builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::ult, paddedY, paddedHeight));
      const Value validX = builder.create<arith::AndIOp>(
        location,
        builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::uge, paddedX, left),
        builder.create<arith::CmpIOp>(
          location, arith::CmpIPredicate::ult, paddedX, paddedWidth));
      const Value valid =
        builder.create<arith::AndIOp>(location, validY, validX);
      auto conditional = builder.create<scf::IfOp>(
        location, TypeRange{builder.getF32Type()}, valid, true, true);
      OpBuilder thenBuilder = conditional.getThenBodyBuilder();
      const Value inputY =
        thenBuilder.create<arith::SubIOp>(location, paddedY, top);
      const Value inputX =
        thenBuilder.create<arith::SubIOp>(location, paddedX, left);
      const SmallVector<Value> inputCoordinates{zero, inputY, inputX, channel};
      const Value input =
        load(thenBuilder,
             location,
             buffers.front(),
             linearOffset(thenBuilder, location, inputCoordinates, inputType));
      const Value updated =
        thenBuilder.create<arith::MaximumFOp>(location, maximum, input);
      thenBuilder.create<scf::YieldOp>(location, updated);
      OpBuilder elseBuilder = conditional.getElseBodyBuilder();
      elseBuilder.create<scf::YieldOp>(location, maximum);
      maximum = conditional.getResult(0);
    }
  }
  return maximum;
}

Value globalAveragePoolValue(OpBuilder& builder,
                             Location location,
                             const GraphKernel& kernel,
                             ArrayRef<Value> buffers,
                             Value outputIndex) {
  const auto inputType = kernel.input_types.front();
  const Value channel = outputIndex;
  const Value zero = indexConstant(builder, location, 0);
  const Value one = indexConstant(builder, location, 1);
  const Value height =
    indexConstant(builder, location, inputType.getDimSize(1));
  const Value width = indexConstant(builder, location, inputType.getDimSize(2));
  Value sum =
    builder.create<arith::ConstantOp>(location, builder.getF32FloatAttr(0.0F));
  auto rows = builder.create<scf::ForOp>(
    location,
    zero,
    height,
    one,
    ValueRange{sum},
    [&](OpBuilder& rowBuilder,
        Location rowLocation,
        Value row,
        ValueRange rowSum) {
      auto columns = rowBuilder.create<scf::ForOp>(
        rowLocation,
        zero,
        width,
        one,
        rowSum,
        [&](OpBuilder& columnBuilder,
            Location columnLocation,
            Value column,
            ValueRange columnSum) {
          const SmallVector<Value> coordinates{zero, row, column, channel};
          const Value input =
            load(columnBuilder,
                 columnLocation,
                 buffers.front(),
                 linearOffset(
                   columnBuilder, columnLocation, coordinates, inputType));
          const Value accumulated = columnBuilder.create<arith::AddFOp>(
            columnLocation, columnSum.front(), input);
          columnBuilder.create<scf::YieldOp>(columnLocation, accumulated);
        });
      rowBuilder.create<scf::YieldOp>(rowLocation, columns.getResult(0));
    });
  const auto divisorValue = static_cast<float>(inputType.getDimSize(1)) *
                            static_cast<float>(inputType.getDimSize(2));
  const Value divisor = builder.create<arith::ConstantOp>(
    location, builder.getF32FloatAttr(divisorValue));
  return builder.create<arith::DivFOp>(location, rows.getResult(0), divisor);
}

Value reduceValue(OpBuilder& builder,
                  Location location,
                  const GraphKernel& kernel,
                  ArrayRef<Value> buffers,
                  Value outputIndex,
                  std::uint32_t axis,
                  bool maximumReduction) {
  const auto inputType = kernel.input_types.front();
  SmallVector<Value> inputCoordinates =
    coordinates(builder, location, outputIndex, kernel.output_type);
  const Value zero = indexConstant(builder, location, 0);
  const Value one = indexConstant(builder, location, 1);
  const Value extent =
    indexConstant(builder, location, inputType.getDimSize(axis));
  const float initialValue =
    maximumReduction ? -std::numeric_limits<float>::infinity() : 0.0F;
  Value initial = builder.create<arith::ConstantOp>(
    location, builder.getF32FloatAttr(initialValue));
  auto reduction = builder.create<scf::ForOp>(
    location,
    zero,
    extent,
    one,
    ValueRange{initial},
    [&](OpBuilder& body,
        Location bodyLocation,
        Value reducedIndex,
        ValueRange accumulated) {
      inputCoordinates[axis] = reducedIndex;
      const Value input =
        load(body,
             bodyLocation,
             buffers.front(),
             linearOffset(body, bodyLocation, inputCoordinates, inputType));
      const Value next = maximumReduction
                           ? Value(body.create<arith::MaximumFOp>(
                               bodyLocation, accumulated.front(), input))
                           : Value(body.create<arith::AddFOp>(
                               bodyLocation, accumulated.front(), input));
      body.create<scf::YieldOp>(bodyLocation, next);
    });
  return reduction.getResult(0);
}

Value outputValue(OpBuilder& builder,
                  Location location,
                  const GraphKernel& kernel,
                  ArrayRef<Value> buffers,
                  Value outputIndex) {
  if (std::holds_alternative<CopyKernel>(kernel.body)) {
    return load(builder, location, buffers[0], outputIndex);
  }
  if (std::holds_alternative<ReluKernel>(kernel.body)) {
    Value input = load(builder, location, buffers[0], outputIndex);
    Value zero = builder.create<arith::ConstantOp>(
      location, builder.getF32FloatAttr(0.0F));
    return builder.create<arith::MaximumFOp>(location, input, zero);
  }
  if (std::holds_alternative<AddKernel>(kernel.body)) {
    Value lhs = load(builder, location, buffers[0], outputIndex);
    Value rhs = load(builder, location, buffers[1], outputIndex);
    return builder.create<arith::AddFOp>(location, lhs, rhs);
  }
  if (const auto* transpose = std::get_if<TransposeKernel>(&kernel.body)) {
    auto outputCoordinates =
      coordinates(builder, location, outputIndex, kernel.output_type);
    SmallVector<Value> inputCoordinates(outputCoordinates.size());
    for (auto [dimension, source] : llvm::enumerate(transpose->permutation)) {
      inputCoordinates[source] = outputCoordinates[dimension];
    }
    return load(
      builder,
      location,
      buffers[0],
      linearOffset(builder, location, inputCoordinates, kernel.input_types[0]));
  }
  if (const auto* slice = std::get_if<SliceKernel>(&kernel.body)) {
    auto inputCoordinates =
      coordinates(builder, location, outputIndex, kernel.output_type);
    for (auto [dimension, offset] : llvm::enumerate(slice->offsets)) {
      inputCoordinates[dimension] =
        builder.create<arith::AddIOp>(location,
                                      inputCoordinates[dimension],
                                      indexConstant(builder, location, offset));
    }
    return load(
      builder,
      location,
      buffers[0],
      linearOffset(builder, location, inputCoordinates, kernel.input_types[0]));
  }
  if (const auto* convolution = std::get_if<ConvKernel>(&kernel.body)) {
    return convValue(
      builder, location, kernel, *convolution, buffers, outputIndex);
  }
  if (const auto* pooling = std::get_if<MaxPoolKernel>(&kernel.body)) {
    return maxPoolValue(
      builder, location, kernel, *pooling, buffers, outputIndex);
  }
  if (std::holds_alternative<GlobalAveragePoolKernel>(kernel.body)) {
    return globalAveragePoolValue(
      builder, location, kernel, buffers, outputIndex);
  }
  if (const auto* reduction = std::get_if<ReduceMaxKernel>(&kernel.body)) {
    return reduceValue(
      builder, location, kernel, buffers, outputIndex, reduction->axis, true);
  }
  if (const auto* reduction = std::get_if<ReduceSumKernel>(&kernel.body)) {
    return reduceValue(
      builder, location, kernel, buffers, outputIndex, reduction->axis, false);
  }
  if (std::holds_alternative<SubKernel>(kernel.body) ||
      std::holds_alternative<MulKernel>(kernel.body)) {
    const auto outputCoordinates =
      coordinates(builder, location, outputIndex, kernel.output_type);
    const Value lhs = broadcastLoad(
      builder, location, buffers[0], outputCoordinates, kernel.input_types[0]);
    const Value rhs = broadcastLoad(
      builder, location, buffers[1], outputCoordinates, kernel.input_types[1]);
    if (std::holds_alternative<SubKernel>(kernel.body)) {
      return builder.create<arith::SubFOp>(location, lhs, rhs);
    }
    return builder.create<arith::MulFOp>(location, lhs, rhs);
  }
  if (std::holds_alternative<ExpKernel>(kernel.body)) {
    return builder.create<math::ExpOp>(
      location, load(builder, location, buffers[0], outputIndex));
  }
  if (std::holds_alternative<ReciprocalKernel>(kernel.body)) {
    const Value one = builder.create<arith::ConstantOp>(
      location, builder.getF32FloatAttr(1.0F));
    return builder.create<arith::DivFOp>(
      location, one, load(builder, location, buffers[0], outputIndex));
  }
  if (const auto* concatenation = std::get_if<ConcatKernel>(&kernel.body)) {
    return concatValue(
      builder, location, kernel, *concatenation, buffers, outputIndex);
  }
  llvm_unreachable("all checked Vulkan graph kernel variants must be emitted");
}

}  // namespace

LogicalResult emitGraphKernel(
  ModuleOp module,
  const GraphKernel& kernel,
  const ncnn_mlir::vulkan::VulkanExecutionPlan& plan) {
  OpBuilder builder(module.getContext());
  auto targetEnv = parseAttribute(
    "#spirv.target_env<#spirv.vce<v1.0, [Shader], "
    "[SPV_KHR_storage_buffer_storage_class]>, api=Vulkan, "
    "#spirv.resource_limits<>>",
    module.getContext());
  auto entryABI = parseAttribute(
    "#spirv.entry_point_abi<workgroup_size = [64, 1, 1]>", module.getContext());
  if (!targetEnv || !entryABI) {
    return module.emitError(
      "cannot construct the Vulkan graph target environment");
  }
  module->setAttr("gpu.container_module", builder.getUnitAttr());
  module->setAttr("spirv.target_env", targetEnv);
  const auto storageClass = spirv::StorageClassAttr::get(
    module.getContext(), spirv::StorageClass::StorageBuffer);
  SmallVector<Type> argumentTypes;
  std::vector<std::uint32_t> argumentIds;
  for (const auto& binding : kernel.execution.bindings) {
    auto resource = std::ranges::find(plan.resources,
                                      binding.resource_id,
                                      &ncnn_mlir::vulkan::TensorResource::id);
    if (resource == plan.resources.end()) {
      return module.emitError("kernel binding has no planned resource");
    }
    const auto count =
      static_cast<std::int64_t>(resource->byte_size / sizeof(float));
    argumentTypes.push_back(MemRefType::get(
      {count}, builder.getF32Type(), AffineMap(), storageClass));
    argumentIds.push_back(binding.resource_id);
  }
  SmallVector<NamedAttribute> attributes{
    builder.getNamedAttr("gpu.kernel", builder.getUnitAttr()),
    builder.getNamedAttr("spirv.entry_point_abi", entryABI)};
  builder.setInsertionPointToEnd(module.getBody());
  const auto moduleName =
    std::format("{}_vulkan_k{}", plan.model, kernel.execution.id);
  auto deviceModule = gpu::GPUModuleOp::create(
    builder, kernel.location, moduleName, ArrayRef<Attribute>{});
  builder.setInsertionPointToEnd(deviceModule.getBody());
  auto function = gpu::GPUFuncOp::create(
    builder,
    kernel.location,
    kernel.execution.entry_point,
    FunctionType::get(module.getContext(), argumentTypes, {}),
    {},
    {},
    attributes);
  Block& entry = function.getBody().front();
  builder.setInsertionPointToStart(&entry);
  SmallVector<Value> operands;
  for (auto id : kernel.operands) {
    auto iterator = std::ranges::find(argumentIds, id);
    if (iterator == argumentIds.end()) {
      return module.emitError("kernel operand has no descriptor");
    }
    operands.push_back(
      entry.getArgument(static_cast<unsigned>(iterator - argumentIds.begin())));
  }
  const auto outputCount =
    static_cast<std::int64_t>(kernel.output_type.getNumElements());
  Value index =
    builder.create<gpu::GlobalIdOp>(kernel.location, gpu::Dimension::x);
  Value inBounds = builder.create<arith::CmpIOp>(
    kernel.location,
    arith::CmpIPredicate::ult,
    index,
    indexConstant(builder, kernel.location, outputCount));
  Value output = entry.getArguments().back();
  builder.create<scf::IfOp>(
    kernel.location, inBounds, [&](OpBuilder& body, Location location) {
      Value value = outputValue(body, location, kernel, operands, index);
      body.create<memref::StoreOp>(location, value, output, ValueRange{index});
      body.create<scf::YieldOp>(location);
    });
  builder.setInsertionPointToEnd(&entry);
  builder.create<gpu::ReturnOp>(kernel.location);
  return success();
}

}  // namespace mlir::ncnn::vulkan_graph
