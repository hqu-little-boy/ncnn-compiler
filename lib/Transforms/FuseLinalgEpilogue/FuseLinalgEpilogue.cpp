#include "ncnn-mlir/Transforms/FuseLinalgEpilogue/FuseLinalgEpilogue.hpp"

#include <cstdint>
#include <functional>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_FUSELINALGEPILOGUEPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

constexpr int64_t kConvTileWidth = 16;
constexpr int64_t kMatmulTileColumns = 16;

bool isFusableElementwiseBody(linalg::GenericOp consumer) {
  Region& region = consumer->getRegion(0);
  if (!region.hasOneBlock()) {
    return false;
  }
  Block& block = region.front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1) {
    return false;
  }
  if (block.getNumArguments() != 2) {
    return false;
  }
  Value input = block.getArgument(0);
  Value output = block.getArgument(1);
  if (input.use_empty() || !output.use_empty()) {
    return false;
  }
  bool hasComputation = false;
  for (Operation& operation : block.without_terminator()) {
    if (!isa<arith::MaximumFOp,
             arith::MinimumFOp,
             arith::SelectOp,
             arith::CmpFOp,
             arith::MulFOp,
             arith::AddFOp,
             arith::ConstantOp>(operation)) {
      return false;
    }
    hasComputation = !isa<arith::ConstantOp>(operation);
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

bool isFusableConsumer(linalg::GenericOp consumer,
                       Operation* producer,
                       int64_t producerRank) {
  if (consumer.getNumDpsInputs() != 1 || consumer.getNumDpsInits() != 1) {
    return false;
  }
  for (utils::IteratorType iteratorType : consumer.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  auto maps = consumer.getIndexingMapsArray();
  if (!isIdentityMap(maps[0], producerRank) ||
      !isIdentityMap(maps[1], producerRank)) {
    return false;
  }
  auto resultType = dyn_cast<RankedTensorType>(consumer.getResult(0).getType());
  auto producerType =
    dyn_cast<RankedTensorType>(producer->getResult(0).getType());
  if (!resultType || !producerType || !resultType.hasStaticShape() ||
      resultType != producerType) {
    return false;
  }
  return isFusableElementwiseBody(consumer);
}

Value indexConstant(RewriterBase& rewriter, Location location, int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(location, value);
}

linalg::GenericOp cloneEpilogue(RewriterBase& rewriter,
                                linalg::GenericOp consumer,
                                Value tiledProducer,
                                Value emptyTile) {
  auto tileGeneric =
    rewriter.create<linalg::GenericOp>(consumer.getLoc(),
                                       TypeRange{emptyTile.getType()},
                                       ValueRange{tiledProducer},
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
  }
  return acc;
}

Value tileConvolution(RewriterBase& rewriter,
                      linalg::Conv2DNhwcHwcfOp convolution,
                      linalg::GenericOp consumer) {
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
  const int64_t effectiveHeight =
    ((weightType.getShape()[0] - 1) * dilations[0]) + 1;
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
    Value emptyTile = rewriter.create<tensor::EmptyOp>(
      location,
      ArrayRef<int64_t>{1, outputHeight, tileWidth, channels},
      rewriter.getF32Type());
    auto tileEpilogue = cloneEpilogue(
      rewriter, consumer, tileConvolution.getResult(0), emptyTile);
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
                                  kConvTileWidth,
                                  buildChunk);
}

Value tileMatmul(RewriterBase& rewriter,
                 linalg::MatmulOp matmul,
                 linalg::GenericOp consumer) {
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
    Value emptyTile = rewriter.create<tensor::EmptyOp>(
      location, ArrayRef<int64_t>{rows, tileColumns}, rewriter.getF32Type());
    auto tileEpilogue =
      cloneEpilogue(rewriter, consumer, tileMatmul.getResult(0), emptyTile);
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
                                  kMatmulTileColumns,
                                  buildChunk);
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

class FuseLinalgEpiloguePass final
  : public impl::FuseLinalgEpiloguePassBase<FuseLinalgEpiloguePass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<std::pair<Operation*, linalg::GenericOp>> candidates;
    module.walk([&](func::FuncOp function) {
      function.walk([&](Operation* operation) {
        Value result;
        if (auto convolution = dyn_cast<linalg::Conv2DNhwcHwcfOp>(operation)) {
          result = convolution.getResult(0);
        } else if (auto matmul = dyn_cast<linalg::MatmulOp>(operation)) {
          result = matmul.getResult(0);
        } else {
          return;
        }
        if (!result.hasOneUse()) {
          return;
        }
        auto consumer = dyn_cast<linalg::GenericOp>(*result.user_begin());
        if (!consumer || consumer->getBlock() != operation->getBlock() ||
            !isFusableConsumer(
              consumer,
              operation,
              cast<RankedTensorType>(result.getType()).getRank())) {
          return;
        }
        candidates.emplace_back(operation, consumer);
      });
    });
    if (candidates.empty()) {
      return;
    }
    IRRewriter rewriter(&getContext());
    for (auto& [producer, consumer] : candidates) {
      if (!producer->getBlock() ||
          producer->getBlock() != consumer->getBlock()) {
        continue;
      }
      rewriter.setInsertionPoint(consumer);
      bool hasViewInput = false;
      for (Value producerOperand : producer->getOperands()) {
        if (hasTensorViewProducer(producerOperand)) {
          hasViewInput = true;
          break;
        }
      }
      if (hasViewInput) {
        continue;
      }
      Value fused;
      if (auto convolution = dyn_cast<linalg::Conv2DNhwcHwcfOp>(producer)) {
        fused = tileConvolution(rewriter, convolution, consumer);
      } else if (auto matmul = dyn_cast<linalg::MatmulOp>(producer)) {
        fused = tileMatmul(rewriter, matmul, consumer);
      } else {
        continue;
      }
      rewriter.replaceOp(consumer, fused);
      rewriter.eraseOp(producer);
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
