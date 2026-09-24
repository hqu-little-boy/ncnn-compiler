#include "ncnn-mlir/Transforms/FuseLinalgEpilogue/FuseLinalgEpilogue.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_FUSELINALGEPILOGUEPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

bool isFusableElementwiseBody(linalg::GenericOp consumer,
                              bool allowCastChain,
                              unsigned maxBodyOperations) {
  Region& region = consumer->getRegion(0);
  if (!region.hasOneBlock()) {
    return false;
  }
  Block& block = region.front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1 ||
      block.getNumArguments() !=
        consumer.getNumDpsInputs() + consumer.getNumDpsInits()) {
    return false;
  }
  if (block.getArgument(0).use_empty()) {
    return false;
  }
  for (unsigned index = consumer.getNumDpsInputs();
       index < block.getNumArguments();
       ++index) {
    if (!block.getArgument(index).use_empty()) {
      return false;
    }
  }
  bool hasComputation = false;
  unsigned bodyOperations = 0;
  for (Operation& operation : block.without_terminator()) {
    if (++bodyOperations > maxBodyOperations) {
      return false;
    }
    const bool castOperation = isa<arith::ExtFOp,
                                   arith::TruncFOp,
                                   arith::SIToFPOp,
                                   arith::UIToFPOp,
                                   arith::FPToSIOp,
                                   arith::FPToUIOp,
                                   arith::ExtSIOp,
                                   arith::ExtUIOp,
                                   arith::TruncIOp>(operation);
    if (!isa<arith::MaximumFOp,
             arith::MinimumFOp,
             arith::SelectOp,
             arith::CmpFOp,
             arith::MulFOp,
             arith::AddFOp,
             arith::SubFOp,
             arith::DivFOp,
             arith::NegFOp,
             arith::ConstantOp,
             math::ExpOp,
             math::TanhOp,
             math::ErfOp>(operation) &&
        (!allowCastChain || !castOperation)) {
      return false;
    }
    hasComputation |= !isa<arith::ConstantOp>(operation);
  }
  return hasComputation;
}

bool isIdentityMap(AffineMap map, int64_t rank) {
  if (!map.isIdentity()) {
    return false;
  }
  const int64_t dimensions = map.getNumDims();
  const int64_t results = map.getNumResults();
  return dimensions == rank && results == rank;
}

// P22 广播映射只允许投影一个迭代维，或投影到静态 extent=1 的常量维。
// 这使 tile slice 的 offset/size 可在不猜测 alias 的情况下逐维推导。
bool isProjectedBroadcastMap(AffineMap map,
                             RankedTensorType sourceType,
                             int64_t loopRank) {
  if (!std::cmp_equal(map.getNumDims(), loopRank) ||
      !std::cmp_equal(map.getNumResults(), sourceType.getRank())) {
    return false;
  }
  SmallVector<bool> usedDimensions(loopRank, false);
  bool hasProjectedConstant = false;
  for (unsigned sourceDimension = 0; sourceDimension < sourceType.getRank();
       ++sourceDimension) {
    AffineExpr expression = map.getResult(sourceDimension);
    if (auto dimension = dyn_cast<AffineDimExpr>(expression)) {
      const unsigned position = dimension.getPosition();
      if (std::cmp_greater_equal(position, loopRank) ||
          usedDimensions[position]) {
        return false;
      }
      usedDimensions[position] = true;
      continue;
    }
    auto constant = dyn_cast<AffineConstantExpr>(expression);
    if (!constant || constant.getValue() != 0 ||
        sourceType.getShape()[sourceDimension] != 1) {
      return false;
    }
    hasProjectedConstant = true;
  }
  return sourceType.getRank() < loopRank || hasProjectedConstant;
}

bool isSupportedOperandMap(AffineMap map,
                           RankedTensorType sourceType,
                           int64_t loopRank,
                           bool allowBroadcast) {
  if (isIdentityMap(map, sourceType.getRank())) {
    return std::cmp_equal(map.getNumDims(), loopRank);
  }
  return allowBroadcast && isProjectedBroadcastMap(map, sourceType, loopRank);
}

StringRef fusionBroadcastKind(linalg::GenericOp consumer) {
  for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
    auto type =
      dyn_cast<RankedTensorType>(consumer.getDpsInputs()[index].getType());
    if (type && !isIdentityMap(consumer.getIndexingMapsArray()[index],
                               type.getRank())) {
      return "projected_broadcast";
    }
  }
  return "none";
}

StringRef getKnownLayout(Operation* operation) {
  for (StringRef attribute :
       {contract::kOutputLayout, contract::kLayout, contract::kInputLayout}) {
    if (auto value = operation->getAttrOfType<StringAttr>(attribute)) {
      return value.getValue();
    }
  }
  return {};
}

std::optional<int64_t> getKnownPackFactor(Operation* operation) {
  for (StringRef attribute :
       {contract::kPackFactor, contract::kLayoutPackFactor}) {
    if (auto value = operation->getAttrOfType<IntegerAttr>(attribute)) {
      return value.getInt();
    }
  }
  return std::nullopt;
}

bool isFusionLayoutCompatible(Operation* producer, Operation* consumer) {
  StringRef producerLayout = getKnownLayout(producer);
  StringRef consumerLayout = getKnownLayout(consumer);
  if (!producerLayout.empty() && !consumerLayout.empty() &&
      producerLayout != consumerLayout) {
    return false;
  }
  auto producerFactor = getKnownPackFactor(producer);
  auto consumerFactor = getKnownPackFactor(consumer);
  return !producerFactor || !consumerFactor ||
         *producerFactor == *consumerFactor;
}

StringRef fusionLayoutKind(Operation* producer, Operation* consumer) {
  StringRef producerLayout = getKnownLayout(producer);
  StringRef consumerLayout = getKnownLayout(consumer);
  if (!producerLayout.empty() || !consumerLayout.empty()) {
    return "metadata_compatible";
  }
  return "identity_contiguous";
}

Value sliceOperandForTile(RewriterBase& rewriter,
                          Location location,
                          Value operand,
                          AffineMap map,
                          ArrayRef<OpFoldResult> tileOffsets,
                          ArrayRef<OpFoldResult> tileSizes,
                          bool& isBroadcast) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type || type.getRank() == 0) {
    return operand;
  }
  isBroadcast = !isIdentityMap(map, type.getRank());
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  offsets.reserve(type.getRank());
  sizes.reserve(type.getRank());
  strides.assign(type.getRank(), rewriter.getIndexAttr(1));
  for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
    AffineExpr expression = map.getResult(dimension);
    if (auto projected = dyn_cast<AffineDimExpr>(expression)) {
      const unsigned position = projected.getPosition();
      offsets.push_back(tileOffsets[position]);
      sizes.push_back(tileSizes[position]);
    } else {
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(1));
    }
  }
  SmallVector<int64_t> resultShape;
  resultShape.reserve(type.getRank());
  for (OpFoldResult size : sizes) {
    auto attribute = dyn_cast_if_present<Attribute>(size);
    auto integer = attribute ? dyn_cast<IntegerAttr>(attribute) : nullptr;
    resultShape.push_back(integer ? integer.getInt() : ShapedType::kDynamic);
  }
  auto resultType = RankedTensorType::get(resultShape, type.getElementType());
  return {rewriter.create<tensor::ExtractSliceOp>(
    location, resultType, operand, offsets, sizes, strides)};
}

bool hasTensorViewProducer(Value value) {
  Operation* definition = value.getDefiningOp();
  if (!definition) {
    return false;
  }
  return isa<tensor::ExpandShapeOp,
             tensor::CollapseShapeOp,
             tensor::ExtractSliceOp,
             tensor::InsertSliceOp,
             tensor::CastOp>(definition);
}

bool hasOnlyDpsInputUses(Value result, linalg::GenericOp consumer) {
  for (OpOperand& use : result.getUses()) {
    if (use.getOwner() != consumer.getOperation() ||
        !llvm::is_contained(consumer.getDpsInputOperands(), &use)) {
      return false;
    }
  }
  return true;
}

int64_t countResidualInputs(linalg::GenericOp consumer, Value producerResult) {
  int64_t count = 0;
  for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
    count += consumer.getDpsInputs()[index] != producerResult;
  }
  return count;
}

bool hasRepeatedOperand(Operation* producer, linalg::GenericOp consumer) {
  SmallVector<Value> values;
  for (Value value : producer->getOperands()) {
    values.push_back(value);
  }
  for (Value value : consumer.getDpsInputs()) {
    if (value == consumer.getDpsInputs().front()) {
      continue;
    }
    values.push_back(value);
  }
  for (Value value : consumer.getDpsInits()) {
    values.push_back(value);
  }
  for (unsigned first = 0; first < values.size(); ++first) {
    for (unsigned second = first + 1; second < values.size(); ++second) {
      if (values[first] == values[second]) {
        return true;
      }
    }
  }
  return false;
}

const char* fusableConsumerFailure(linalg::GenericOp consumer,
                                   Operation* producer,
                                   int64_t producerRank,
                                   bool allowResidual,
                                   bool allowBroadcast,
                                   bool allowCastChain,
                                   unsigned maxChainLength) {
  if (consumer.getNumDpsInputs() < 1 || consumer.getNumDpsInits() != 1) {
    return "unsupported_consumer_arity";
  }
  if (!allowResidual && consumer.getNumDpsInputs() > 1) {
    return "residual_disabled";
  }
  if (consumer.getDpsInputOperand(0)->get() != producer->getResult(0)) {
    return "producer_not_flowing_input";
  }
  for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
    if (consumer.getDpsInputs()[index] == producer->getResult(0)) {
      return "repeated_producer_input";
    }
  }
  for (utils::IteratorType iteratorType : consumer.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return "non_parallel_consumer";
    }
  }
  auto maps = consumer.getIndexingMapsArray();
  const int64_t loopRank = consumer.getNumLoops();
  if (maps.size() != static_cast<size_t>(consumer.getNumDpsInputs() + 1)) {
    return "indexing_map_arity";
  }
  auto resultType = dyn_cast<RankedTensorType>(consumer.getResult(0).getType());
  auto producerType =
    dyn_cast<RankedTensorType>(producer->getResult(0).getType());
  auto initType =
    dyn_cast<RankedTensorType>(consumer.getDpsInits()[0].getType());
  if (!resultType || !producerType || !initType ||
      !resultType.hasStaticShape() || resultType != producerType ||
      initType != resultType ||
      !std::cmp_equal(maps.back().getNumDims(), loopRank) ||
      !isIdentityMap(maps.back(), producerRank)) {
    return "shape_mismatch";
  }
  if (!resultType.getElementType().isF32() ||
      !producerType.getElementType().isF32()) {
    return "unsupported_type";
  }
  auto producerMap = maps.front();
  if (!isIdentityMap(producerMap, producerRank) ||
      !std::cmp_equal(producerMap.getNumDims(), loopRank)) {
    return "producer_map";
  }
  for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
    auto residualType =
      dyn_cast<RankedTensorType>(consumer.getDpsInputs()[index].getType());
    if (!residualType || !residualType.hasStaticShape() ||
        !residualType.getElementType().isF32() ||
        !isSupportedOperandMap(
          maps[index], residualType, loopRank, allowBroadcast)) {
      if (residualType && std::cmp_equal(residualType.getRank(), loopRank) &&
          !isIdentityMap(maps[index], residualType.getRank())) {
        return "non_identity_map";
      }
      return allowBroadcast ? "unsupported_broadcast_map"
                            : "broadcast_disabled";
    }
  }
  if (!isFusableElementwiseBody(consumer, allowCastChain, maxChainLength)) {
    return "unsupported_body";
  }
  for (Value value : producer->getOperands()) {
    if (hasTensorViewProducer(value)) {
      return "tensor_view_or_alias";
    }
  }
  for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
    if (hasTensorViewProducer(consumer.getDpsInputs()[index])) {
      return "tensor_view_or_alias";
    }
  }
  for (Value value : consumer.getDpsInits()) {
    if (hasTensorViewProducer(value)) {
      return "tensor_view_or_alias";
    }
  }
  if (hasRepeatedOperand(producer, consumer)) {
    return "tensor_view_or_alias";
  }
  return nullptr;
}

bool isSupportedProducer(Operation* producer) {
  if (auto convolution = dyn_cast<linalg::Conv2DNhwcHwcfOp>(producer)) {
    auto inputType =
      dyn_cast<RankedTensorType>(convolution.getInputs()[0].getType());
    auto weightType =
      dyn_cast<RankedTensorType>(convolution.getInputs()[1].getType());
    auto initType = dyn_cast<RankedTensorType>(
      convolution.getDpsInitOperand(0)->get().getType());
    auto resultType =
      dyn_cast<RankedTensorType>(convolution.getResult(0).getType());
    if (!inputType || !weightType || !initType || !resultType ||
        !inputType.hasStaticShape() || !weightType.hasStaticShape() ||
        !initType.hasStaticShape() || !resultType.hasStaticShape() ||
        inputType.getRank() != 4 || weightType.getRank() != 4 ||
        initType != resultType || resultType.getRank() != 4 ||
        !inputType.getElementType().isF32() ||
        !weightType.getElementType().isF32() ||
        !resultType.getElementType().isF32()) {
      return false;
    }
    auto strides = convolution.getStrides().getValues<int64_t>();
    auto dilations = convolution.getDilations().getValues<int64_t>();
    return strides.size() == 2 && dilations.size() == 2 && strides[0] > 0 &&
           strides[1] > 0 && dilations[0] > 0 && dilations[1] > 0;
  }
  if (auto matmul = dyn_cast<linalg::MatmulOp>(producer)) {
    auto lhsType = dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
    auto initType =
      dyn_cast<RankedTensorType>(matmul.getDpsInitOperand(0)->get().getType());
    auto resultType = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!lhsType || !rhsType || !initType || !resultType ||
        !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !initType.hasStaticShape() || !resultType.hasStaticShape() ||
        lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        initType != resultType || resultType.getRank() != 2 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !resultType.getElementType().isF32()) {
      return false;
    }
    return lhsType.getShape()[1] == rhsType.getShape()[0] &&
           resultType.getShape()[0] == lhsType.getShape()[0] &&
           resultType.getShape()[1] == rhsType.getShape()[1];
  }
  return false;
}

Value indexConstant(RewriterBase& rewriter, Location location, int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(location, value);
}

int64_t fusionProfileId(StringRef functionName,
                        uint64_t ordinal,
                        StringRef suffix = {}) {
  const std::string key = functionName.str() + "/fusion-site#" +
                          std::to_string(ordinal) + suffix.str();
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char character : key) {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
  return static_cast<int64_t>(hash & 0x7fffffffffffffffULL);
}

linalg::GenericOp cloneEpilogue(RewriterBase& rewriter,
                                linalg::GenericOp consumer,
                                ValueRange tiledInputs,
                                Value emptyTile) {
  auto tileGeneric =
    rewriter.create<linalg::GenericOp>(consumer.getLoc(),
                                       TypeRange{emptyTile.getType()},
                                       tiledInputs,
                                       ValueRange{emptyTile},
                                       consumer.getIndexingMapsArray(),
                                       consumer.getIteratorTypesArray());
  IRMapping mapping;
  consumer->getRegion(0).cloneInto(&tileGeneric->getRegion(0), mapping);
  return tileGeneric;
}

using TileChunkBuilder =
  std::function<Value(RewriterBase&, Location, Value, int64_t, int64_t, Value)>;

Value buildStaticTiledSequence(RewriterBase& rewriter,
                               linalg::GenericOp consumer,
                               Operation* producer,
                               int64_t totalExtent,
                               int64_t tileSize,
                               int64_t profileId,
                               int64_t tailProfileId,
                               const TileChunkBuilder& buildChunk) {
  Location location = producer->getLoc();
  Value acc = consumer.getDpsInitOperand(0)->get();
  const int64_t mainTiles = totalExtent / tileSize;
  const int64_t tailWidth = totalExtent % tileSize;
  if (mainTiles > 0) {
    auto loop =
      rewriter.create<scf::ForOp>(location,
                                  indexConstant(rewriter, location, 0),
                                  indexConstant(rewriter, location, mainTiles),
                                  indexConstant(rewriter, location, 1),
                                  ValueRange{acc});
    contract::setInteger(loop, contract::kFusionSiteId, profileId);
    Block* body = loop.getBody();
    rewriter.setInsertionPointToStart(body);
    Value chunkResult = buildChunk(rewriter,
                                   location,
                                   body->getArgument(0),
                                   0,
                                   tileSize,
                                   body->getArgument(1));
    rewriter.create<scf::YieldOp>(location, chunkResult);
    acc = loop.getResult(0);
    rewriter.setInsertionPointAfter(loop);
  }
  if (tailWidth > 0) {
    acc = buildChunk(
      rewriter, location, Value(), mainTiles * tileSize, tailWidth, acc);
    if (auto insertSlice = acc.getDefiningOp<tensor::InsertSliceOp>()) {
      if (Operation* epilogue = insertSlice.getSource().getDefiningOp()) {
        contract::setInteger(epilogue,
                             contract::kFusionSiteId,
                             mainTiles > 0 ? tailProfileId : profileId);
      }
    }
  }
  return acc;
}

Value tileConvolution(RewriterBase& rewriter,
                      linalg::Conv2DNhwcHwcfOp convolution,
                      linalg::GenericOp consumer,
                      int64_t tileWidth,
                      int64_t profileId,
                      int64_t tailProfileId,
                      bool& fusionAnnotated) {
  Location location = convolution.getLoc();
  auto resultType = cast<RankedTensorType>(convolution.getResult(0).getType());
  const ArrayRef<int64_t> outputShape = resultType.getShape();
  const int64_t width = outputShape[2];
  const int64_t channels = outputShape[3];
  auto weightType =
    cast<RankedTensorType>(convolution.getInputs()[1].getType());
  SmallVector<int64_t> strides;
  SmallVector<int64_t> dilations;
  llvm::append_range(strides, convolution.getStrides().getValues<int64_t>());
  llvm::append_range(dilations,
                     convolution.getDilations().getValues<int64_t>());
  const int64_t effectiveWidth =
    ((weightType.getShape()[1] - 1) * dilations[1]) + 1;
  const int64_t strideWidth = strides[1];
  const int64_t inputChannels = weightType.getShape()[2];
  const int64_t inputHeight =
    cast<RankedTensorType>(convolution.getInputs()[0].getType()).getShape()[1];
  const int64_t outputHeight = outputShape[1];
  Value zero = indexConstant(rewriter, location, 0);
  Value weight = convolution.getInputs()[1];
  Value image = convolution.getInputs()[0];
  Value convolutionInit = convolution.getDpsInitOperand(0)->get();

  TileChunkBuilder buildChunk = [&](RewriterBase& rewriter,
                                    Location location,
                                    Value inductionVariable,
                                    int64_t staticOffset,
                                    int64_t tileWidth,
                                    Value acc) -> Value {
    const int64_t windowWidth =
      ((tileWidth - 1) * strideWidth) + effectiveWidth;
    OpFoldResult inputOffsetWidth;
    OpFoldResult outputOffsetWidth;
    if (inductionVariable) {
      inputOffsetWidth = OpFoldResult(rewriter.create<arith::MulIOp>(
        location,
        inductionVariable,
        indexConstant(rewriter, location, tileWidth * strideWidth)));
      outputOffsetWidth = OpFoldResult(rewriter.create<arith::MulIOp>(
        location,
        inductionVariable,
        indexConstant(rewriter, location, tileWidth)));
    } else {
      inputOffsetWidth = rewriter.getIndexAttr(staticOffset * strideWidth);
      outputOffsetWidth = rewriter.getIndexAttr(staticOffset);
    }
    auto imageTileType = RankedTensorType::get(
      {1, inputHeight, windowWidth, inputChannels}, rewriter.getF32Type());
    Value imageTile = rewriter.create<tensor::ExtractSliceOp>(
      location,
      imageTileType,
      image,
      SmallVector<OpFoldResult>{zero, zero, inputOffsetWidth, zero},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(inputHeight),
                                rewriter.getIndexAttr(windowWidth),
                                rewriter.getIndexAttr(inputChannels)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1)});
    auto initTileType = RankedTensorType::get(
      {1, outputHeight, tileWidth, channels}, rewriter.getF32Type());
    Value initTile = rewriter.create<tensor::ExtractSliceOp>(
      location,
      initTileType,
      convolutionInit,
      SmallVector<OpFoldResult>{zero, zero, outputOffsetWidth, zero},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(outputHeight),
                                rewriter.getIndexAttr(tileWidth),
                                rewriter.getIndexAttr(channels)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1)});
    auto tileConvolution =
      rewriter.create<linalg::Conv2DNhwcHwcfOp>(location,
                                                TypeRange{initTileType},
                                                ValueRange{imageTile, weight},
                                                ValueRange{initTile},
                                                convolution.getStrides(),
                                                convolution.getDilations());
    SmallVector<OpFoldResult> tileOffsets{zero, zero, outputOffsetWidth, zero};
    SmallVector<OpFoldResult> tileSizes{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(outputHeight),
                                        rewriter.getIndexAttr(tileWidth),
                                        rewriter.getIndexAttr(channels)};
    SmallVector<Value> tiledInputs{tileConvolution.getResult(0)};
    for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
      Value residual = consumer.getDpsInputs()[index];
      if (residual == convolution.getResult(0)) {
        tiledInputs.push_back(tileConvolution.getResult(0));
        continue;
      }
      bool broadcasted = false;
      Value residualTile =
        sliceOperandForTile(rewriter,
                            location,
                            residual,
                            consumer.getIndexingMapsArray()[index],
                            tileOffsets,
                            tileSizes,
                            broadcasted);
      tiledInputs.push_back(residualTile);
    }
    Value emptyTile = rewriter.create<tensor::EmptyOp>(
      location,
      ArrayRef<int64_t>{1, outputHeight, tileWidth, channels},
      rewriter.getF32Type());
    auto tileEpilogue =
      cloneEpilogue(rewriter, consumer, tiledInputs, emptyTile);
    if (!fusionAnnotated) {
      const int64_t bytes =
        cast<RankedTensorType>(convolution.getResult(0).getType())
          .getNumElements() *
        4;
      contract::annotateFusion(
        tileEpilogue.getOperation(),
        "conv2d_epilogue",
        "linalg.conv_2d_nhwc_hwcf",
        countResidualInputs(consumer, convolution.getResult(0)),
        tileWidth,
        bytes,
        bytes);
      contract::annotateFusionPlan(
        tileEpilogue.getOperation(),
        "ordered_elementwise",
        fusionBroadcastKind(consumer),
        fusionLayoutKind(convolution.getOperation(), consumer));
      fusionAnnotated = true;
    }
    return rewriter.create<tensor::InsertSliceOp>(
      location,
      tileEpilogue.getResult(0),
      acc,
      SmallVector<OpFoldResult>{zero, zero, outputOffsetWidth, zero},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(outputHeight),
                                rewriter.getIndexAttr(tileWidth),
                                rewriter.getIndexAttr(channels)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1)});
  };

  return buildStaticTiledSequence(rewriter,
                                  consumer,
                                  convolution.getOperation(),
                                  width,
                                  tileWidth,
                                  profileId,
                                  tailProfileId,
                                  buildChunk);
}

Value tileMatmul(RewriterBase& rewriter,
                 linalg::MatmulOp matmul,
                 linalg::GenericOp consumer,
                 int64_t tileWidth,
                 int64_t profileId,
                 int64_t tailProfileId,
                 bool& fusionAnnotated) {
  Location location = matmul.getLoc();
  auto resultType = cast<RankedTensorType>(matmul.getResult(0).getType());
  const int64_t rows = resultType.getShape()[0];
  const int64_t columns = resultType.getShape()[1];
  auto lhsType = cast<RankedTensorType>(matmul.getInputs()[0].getType());
  const int64_t contraction = lhsType.getShape()[1];
  Value zero = indexConstant(rewriter, location, 0);
  Value lhs = matmul.getInputs()[0];
  Value rhs = matmul.getInputs()[1];
  Value matmulInit = matmul.getDpsInitOperand(0)->get();

  TileChunkBuilder buildChunk = [&](RewriterBase& rewriter,
                                    Location location,
                                    Value inductionVariable,
                                    int64_t staticOffset,
                                    int64_t tileColumns,
                                    Value acc) -> Value {
    OpFoldResult columnOffset;
    if (inductionVariable) {
      columnOffset = OpFoldResult(rewriter.create<arith::MulIOp>(
        location,
        inductionVariable,
        indexConstant(rewriter, location, tileColumns)));
    } else {
      columnOffset = rewriter.getIndexAttr(staticOffset);
    }
    auto rhsTileType =
      RankedTensorType::get({contraction, tileColumns}, rewriter.getF32Type());
    Value rhsTile = rewriter.create<tensor::ExtractSliceOp>(
      location,
      rhsTileType,
      rhs,
      SmallVector<OpFoldResult>{zero, columnOffset},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(contraction),
                                rewriter.getIndexAttr(tileColumns)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1)});
    auto initTileType =
      RankedTensorType::get({rows, tileColumns}, rewriter.getF32Type());
    Value initTile = rewriter.create<tensor::ExtractSliceOp>(
      location,
      initTileType,
      matmulInit,
      SmallVector<OpFoldResult>{zero, columnOffset},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(rows),
                                rewriter.getIndexAttr(tileColumns)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1)});
    auto tileMatmul =
      rewriter.create<linalg::MatmulOp>(location,
                                        TypeRange{initTileType},
                                        ValueRange{lhs, rhsTile},
                                        ValueRange{initTile});
    SmallVector<OpFoldResult> tileOffsets{zero, columnOffset};
    SmallVector<OpFoldResult> tileSizes{rewriter.getIndexAttr(rows),
                                        rewriter.getIndexAttr(tileColumns)};
    SmallVector<Value> tiledInputs{tileMatmul.getResult(0)};
    for (unsigned index = 1; index < consumer.getNumDpsInputs(); ++index) {
      Value residual = consumer.getDpsInputs()[index];
      if (residual == matmul.getResult(0)) {
        tiledInputs.push_back(tileMatmul.getResult(0));
        continue;
      }
      bool broadcasted = false;
      Value residualTile =
        sliceOperandForTile(rewriter,
                            location,
                            residual,
                            consumer.getIndexingMapsArray()[index],
                            tileOffsets,
                            tileSizes,
                            broadcasted);
      tiledInputs.push_back(residualTile);
    }
    Value emptyTile = rewriter.create<tensor::EmptyOp>(
      location, ArrayRef<int64_t>{rows, tileColumns}, rewriter.getF32Type());
    auto tileEpilogue =
      cloneEpilogue(rewriter, consumer, tiledInputs, emptyTile);
    if (!fusionAnnotated) {
      const int64_t bytes =
        cast<RankedTensorType>(matmul.getResult(0).getType()).getNumElements() *
        4;
      contract::annotateFusion(
        tileEpilogue.getOperation(),
        "matmul_epilogue",
        "linalg.matmul",
        countResidualInputs(consumer, matmul.getResult(0)),
        tileWidth,
        bytes,
        bytes);
      contract::annotateFusionPlan(
        tileEpilogue.getOperation(),
        "ordered_elementwise",
        fusionBroadcastKind(consumer),
        fusionLayoutKind(matmul.getOperation(), consumer));
      fusionAnnotated = true;
    }
    return rewriter.create<tensor::InsertSliceOp>(
      location,
      tileEpilogue.getResult(0),
      acc,
      SmallVector<OpFoldResult>{zero, columnOffset},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(rows),
                                rewriter.getIndexAttr(tileColumns)},
      SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                                rewriter.getIndexAttr(1)});
  };

  return buildStaticTiledSequence(rewriter,
                                  consumer,
                                  matmul.getOperation(),
                                  columns,
                                  tileWidth,
                                  profileId,
                                  tailProfileId,
                                  buildChunk);
}

class FuseLinalgEpiloguePass final
  : public impl::FuseLinalgEpiloguePassBase<FuseLinalgEpiloguePass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    const int64_t tileWidthValue = this->tileWidth.getValue();
    contract::setBool(module, contract::kFusionEnabled, this->enable);
    contract::setBool(
      module, contract::kFusionAllowBroadcast, this->allowBroadcast);
    contract::setBool(
      module, contract::kFusionAllowCastChain, this->allowCastChain);
    contract::setBool(
      module, contract::kFusionLayoutAware, this->layoutAwareFusion);
    contract::setInteger(
      module, contract::kFusionMaxChain, this->maxChainLength);
    contract::setString(module, contract::kFusionRevision, "fusion-v2");
    int64_t selectedCount = 0;
    int64_t residualCount = 0;
    int64_t rejectedCount = 0;
    std::map<std::string, int64_t> rejectionReasons;
    SmallVector<std::pair<Operation*, linalg::GenericOp>> candidates;
    std::map<Operation*, std::pair<int64_t, int64_t>> fusionProfileIds;

    auto reject = [&](const char* reason, Operation* operation = nullptr) {
      ++rejectedCount;
      ++rejectionReasons[reason];
      if (operation) {
        contract::annotateFusionFallback(operation, reason);
      }
    };
    if (tileWidthValue <= 0) {
      reject("invalid_tile_width");
    }

    module.walk([&](func::FuncOp function) {
      uint64_t producerOrdinal = 0;
      function.walk([&](Operation* operation) {
        Value result;
        if (isa<linalg::Conv2DNhwcHwcfOp, linalg::MatmulOp>(operation)) {
          result = operation->getResult(0);
          const uint64_t ordinal = producerOrdinal++;
          fusionProfileIds[operation] = {
            fusionProfileId(function.getName(), ordinal),
            fusionProfileId(function.getName(), ordinal, "/tail")};
        } else {
          return;
        }
        if (!isSupportedProducer(operation)) {
          reject("unsupported_producer", operation);
          return;
        }
        if (result.use_empty()) {
          return;
        }
        OpOperand& firstUse = *result.getUses().begin();
        auto consumer = dyn_cast<linalg::GenericOp>(firstUse.getOwner());
        if (!consumer) {
          reject(result.hasOneUse() ? "unsupported_consumer" : "multi_consumer",
                 operation);
          return;
        }
        if (!hasOnlyDpsInputUses(result, consumer)) {
          reject("multi_consumer", operation);
          return;
        }
        if (operation->getBlock() != consumer->getBlock()) {
          reject("different_block", operation);
          return;
        }
        if (!operation->isBeforeInBlock(consumer)) {
          reject("non_dominating_producer", operation);
          return;
        }
        if (this->layoutAwareFusion &&
            !isFusionLayoutCompatible(operation, consumer)) {
          reject("layout_mismatch", operation);
          return;
        }
        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType) {
          reject("unsupported_type", operation);
          return;
        }
        if (const char* reason = fusableConsumerFailure(consumer,
                                                        operation,
                                                        resultType.getRank(),
                                                        this->allowResidual,
                                                        this->allowBroadcast,
                                                        this->allowCastChain,
                                                        this->maxChainLength)) {
          reject(reason, operation);
          return;
        }
        candidates.emplace_back(operation, consumer);
      });
    });

    if (this->enable && tileWidthValue > 0) {
      IRRewriter rewriter(&getContext());
      for (auto& [producer, consumer] : candidates) {
        if (!producer->getBlock() || !consumer->getBlock() ||
            producer->getBlock() != consumer->getBlock() ||
            !producer->isBeforeInBlock(consumer)) {
          reject("candidate_invalidated");
          continue;
        }
        rewriter.setInsertionPoint(consumer);
        const auto [profileId, tailProfileId] = fusionProfileIds.at(producer);
        bool fusionAnnotated = false;
        Value fused;
        if (auto convolution = dyn_cast<linalg::Conv2DNhwcHwcfOp>(producer)) {
          fused = tileConvolution(rewriter,
                                  convolution,
                                  consumer,
                                  tileWidthValue,
                                  profileId,
                                  tailProfileId,
                                  fusionAnnotated);
        } else if (auto matmul = dyn_cast<linalg::MatmulOp>(producer)) {
          fused = tileMatmul(rewriter,
                             matmul,
                             consumer,
                             tileWidthValue,
                             profileId,
                             tailProfileId,
                             fusionAnnotated);
        } else {
          reject("unsupported_producer");
          continue;
        }
        const int64_t candidateResidualCount =
          countResidualInputs(consumer, producer->getResult(0));
        const int64_t bytes =
          cast<RankedTensorType>(producer->getResult(0).getType())
            .getNumElements() *
          4;
        const bool isConvolution = isa<linalg::Conv2DNhwcHwcfOp>(producer);
        auto producerResultType =
          cast<RankedTensorType>(producer->getResult(0).getType());
        const int64_t tiledExtent =
          producerResultType.getShape()[isConvolution ? 2 : 1];
        std::optional<int64_t> tailProfileIdForRecord;
        if (tiledExtent / tileWidthValue > 0 &&
            tiledExtent % tileWidthValue > 0) {
          tailProfileIdForRecord = tailProfileId;
        }
        auto function = producer->getParentOfType<func::FuncOp>();
        contract::appendFusionRecord(
          module,
          function ? function.getName() : "",
          isConvolution ? "linalg.conv_2d_nhwc_hwcf" : "linalg.matmul",
          isConvolution ? "conv2d_epilogue" : "matmul_epilogue",
          isConvolution ? "linalg.conv_2d_nhwc_hwcf" : "linalg.matmul",
          candidateResidualCount,
          tileWidthValue,
          bytes,
          bytes,
          profileId,
          tailProfileIdForRecord);
        rewriter.replaceOp(consumer, fused);
        rewriter.eraseOp(producer);
        ++selectedCount;
        residualCount += candidateResidualCount;
      }
    }

    contract::setInteger(module, contract::kFusionSelectedCount, selectedCount);
    contract::setInteger(module, contract::kFusionResidualCount, residualCount);
    contract::setInteger(module, contract::kFusionRejectedCount, rejectedCount);
    std::string reasonText;
    for (auto [reason, count] : rejectionReasons) {
      if (!reasonText.empty()) {
        reasonText += ",";
      }
      reasonText += reason + "=" + std::to_string(count);
    }
    contract::setString(module, contract::kFusionRejectionReasons, reasonText);
  }
};

}  // namespace

}  // namespace mlir::ncnn
