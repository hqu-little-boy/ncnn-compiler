#include "ncnn-mlir/Transforms/PackStaticMatmulNCNN/PackStaticMatmulNCNN.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_PACKSTATICMATMULNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class PackStaticMatmulNCNNPass final
  : public impl::PackStaticMatmulNCNNPassBase<PackStaticMatmulNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> candidates;
    SmallVector<linalg::MatmulTransposeBOp> int8Candidates;
    module.walk([&](linalg::MatmulOp matmul) {
      if (matmul.hasPureTensorSemantics()) {
        candidates.push_back(matmul);
      }
    });
    if (int8Enabled) {
      packedBytesUsed = getExistingF32PackedBytes(module);
      module.walk([&](linalg::MatmulTransposeBOp matmul) {
        if (!matmul.hasPureTensorSemantics()) {
          int8Candidates.push_back(matmul);
        }
      });
    }

    for (linalg::MatmulOp matmul : candidates) {
      process(matmul);
    }
    for (linalg::MatmulTransposeBOp matmul : int8Candidates) {
      processInt8MemRef(matmul, module);
    }
  }

 private:
  struct PackedInt8GlobalInfo {
    std::string symbol;
    int64_t rows = 0;
    int64_t depth = 0;
    int64_t paddedDepth = 0;
    int64_t rawBytes = 0;
    int64_t packedBytes = 0;
  };

  struct Int8RhsSource {
    memref::GlobalOp global;
    memref::GetGlobalOp access;
    SmallVector<Operation*> views;
    SmallVector<memref::SubViewOp> subviews;
  };

  // Keep object/rodata growth bounded for large models. A zero budget is an
  // explicit escape hatch used only by focused transform tests.
  mutable std::int64_t packedBytesUsed = 0;
  llvm::DenseMap<Operation*, PackedInt8GlobalInfo> packedInt8Globals;

  static int64_t getExistingF32PackedBytes(ModuleOp module) {
    int64_t total = 0;
    module.walk([&](Operation* operation) {
      auto schema = operation->getAttrOfType<StringAttr>(contract::kPackSchema);
      auto bytes = operation->getAttrOfType<IntegerAttr>(contract::kPackBytes);
      if (!schema || schema.getValue() != "p20-panel-nk-v1" || !bytes ||
          bytes.getInt() <= 0) {
        return;
      }
      const int64_t value = bytes.getInt();
      if (total > std::numeric_limits<int64_t>::max() - value) {
        total = std::numeric_limits<int64_t>::max();
      } else {
        total += value;
      }
    });
    return total;
  }

  // Match TileMatmulForall's static-factor rule. Packing before tiling is only
  // safe when the later pass will actually create an scf.forall consumer;
  // otherwise generic linalg lowering still interprets the RHS as row-major.
  static int64_t pickTileDivisor(int64_t extent, int64_t requested) {
    if (requested <= 0 || requested >= extent) {
      return extent;
    }
    for (int64_t candidate = requested; candidate > 1; --candidate) {
      if (extent % candidate == 0) {
        return candidate;
      }
    }
    return extent;
  }

  static arith::ConstantOp findConstant(Value value) {
    while (Operation* defining = value.getDefiningOp()) {
      if (auto constant = dyn_cast<arith::ConstantOp>(defining)) {
        return constant;
      }
      if (auto cast = dyn_cast<tensor::CastOp>(defining)) {
        value = cast.getSource();
        continue;
      }
      if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(defining)) {
        value = collapse.getSrc();
        continue;
      }
      if (auto expand = dyn_cast<tensor::ExpandShapeOp>(defining)) {
        value = expand.getSrc();
        continue;
      }
      return {};
    }
    return {};
  }

  static void annotateRejected(linalg::MatmulOp matmul, StringRef reason) {
    contract::annotatePacking(matmul.getOperation(),
                              reason,
                              /*packFactor=*/1,
                              /*packBytes=*/0,
                              /*unpackBytes=*/0);
    contract::setString(
      matmul.getOperation(), contract::kWeightLayout, "row_major_kxn");
    contract::setString(
      matmul.getOperation(), contract::kPackSchema, "p20-panel-nk-v1");
    contract::annotateFallback(matmul.getOperation(), reason);
  }

  static void annotateInt8Rejected(linalg::MatmulTransposeBOp matmul,
                                   StringRef reason) {
    contract::annotatePacking(matmul.getOperation(),
                              reason,
                              /*packFactor=*/16,
                              /*packBytes=*/0,
                              /*unpackBytes=*/0);
    contract::setString(
      matmul.getOperation(), contract::kWeightLayout, "row_major_nk");
    contract::annotateFallback(matmul.getOperation(), reason);
  }

  static std::optional<Int8RhsSource> findInt8RhsSource(ModuleOp module,
                                                        Value rhs) {
    Int8RhsSource source;
    Value current = rhs;
    while (Operation* defining = current.getDefiningOp()) {
      if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(defining)) {
        auto global = dyn_cast_or_null<memref::GlobalOp>(
          SymbolTable::lookupSymbolIn(module, getGlobal.getName()));
        if (!global) {
          return std::nullopt;
        }
        source.global = global;
        source.access = getGlobal;
        return source;
      }
      if (auto subview = dyn_cast<memref::SubViewOp>(defining)) {
        if (subview.getType().getRank() != 2 ||
            cast<MemRefType>(subview.getSource().getType()).getRank() != 2) {
          return std::nullopt;
        }
        source.views.push_back(subview.getOperation());
        source.subviews.push_back(subview);
        current = subview.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(defining)) {
        auto sourceType = dyn_cast<MemRefType>(cast.getSource().getType());
        auto resultType = dyn_cast<MemRefType>(cast.getType());
        if (!sourceType || !resultType || sourceType.getRank() != 2 ||
            resultType.getRank() != 2 ||
            sourceType.getElementType() != resultType.getElementType()) {
          return std::nullopt;
        }
        for (unsigned dimension = 0; dimension < 2; ++dimension) {
          const int64_t sourceSize = sourceType.getShape()[dimension];
          const int64_t resultSize = resultType.getShape()[dimension];
          if (sourceSize != resultSize && !ShapedType::isDynamic(sourceSize) &&
              !ShapedType::isDynamic(resultSize)) {
            return std::nullopt;
          }
        }
        source.views.push_back(cast.getOperation());
        current = cast.getSource();
        continue;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  static std::optional<int64_t> getStaticIndex(OpFoldResult value) {
    if (auto attribute = value.dyn_cast<Attribute>()) {
      if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
        return integer.getInt();
      }
    } else if (auto operand = value.dyn_cast<Value>()) {
      if (auto constant = operand.getDefiningOp<arith::ConstantIndexOp>()) {
        return constant.value();
      }
      if (auto constant = operand.getDefiningOp<arith::ConstantOp>()) {
        if (auto integer = dyn_cast<IntegerAttr>(constant.getValue())) {
          return integer.getInt();
        }
      }
    }
    return std::nullopt;
  }

  std::optional<Value> createPackedInt8View(linalg::MatmulTransposeBOp matmul,
                                            ModuleOp module,
                                            Int8RhsSource& rhsSource,
                                            int64_t rows,
                                            int64_t depth,
                                            int64_t paddedDepth,
                                            MemRefType packedType,
                                            DenseIntElementsAttr packedValues,
                                            int64_t rawBytes,
                                            int64_t packedBytes) {
    PackedInt8GlobalInfo* packedInfo = nullptr;
    auto found = packedInt8Globals.find(rhsSource.global.getOperation());
    if (found != packedInt8Globals.end()) {
      packedInfo = &found->second;
      if (packedInfo->rows != rows || packedInfo->depth != depth ||
          packedInfo->paddedDepth != paddedDepth) {
        return std::nullopt;
      }
    } else {
      const std::string baseName =
        rhsSource.global.getSymName().str() + "_p23_int8_kpad64";
      std::string symbolName = baseName;
      unsigned suffix = 0;
      while (SymbolTable::lookupSymbolIn(module, symbolName)) {
        symbolName = baseName + "_" + std::to_string(++suffix);
      }

      OpBuilder globalBuilder(module.getContext());
      globalBuilder.setInsertionPointToStart(module.getBody());
      auto global = globalBuilder.create<memref::GlobalOp>(
        rhsSource.global.getLoc(),
        symbolName,
        globalBuilder.getStringAttr("private"),
        packedType,
        packedValues,
        /*constant=*/true,
        globalBuilder.getI64IntegerAttr(64));
      PackedInt8GlobalInfo info{.symbol = symbolName,
                                .rows = rows,
                                .depth = depth,
                                .paddedDepth = paddedDepth,
                                .rawBytes = rawBytes,
                                .packedBytes = packedBytes};
      auto [iterator, inserted] = packedInt8Globals.try_emplace(
        rhsSource.global.getOperation(), std::move(info));
      (void)inserted;
      packedInfo = &iterator->second;
      packedBytesUsed += packedBytes;
    }

    OpBuilder builder(matmul);
    auto flatType =
      MemRefType::get({packedInfo->rows * packedInfo->paddedDepth},
                      cast<MemRefType>(packedType).getElementType());
    auto packedGlobal = builder.create<memref::GetGlobalOp>(
      matmul.getLoc(),
      flatType,
      FlatSymbolRefAttr::get(builder.getContext(), packedInfo->symbol));
    const SmallVector<int64_t> logicalShape{rows, depth};
    const SmallVector<int64_t> logicalStrides{paddedDepth, 1};
    auto layout = StridedLayoutAttr::get(
      builder.getContext(), /*offset=*/0, logicalStrides);
    auto logicalType = MemRefType::get(
      logicalShape, cast<MemRefType>(packedType).getElementType(), layout);
    Value replacement =
      builder.create<memref::ReinterpretCastOp>(matmul.getLoc(),
                                                logicalType,
                                                packedGlobal.getResult(),
                                                /*offset=*/0,
                                                logicalShape,
                                                logicalStrides);
    for (memref::SubViewOp subview : llvm::reverse(rhsSource.subviews)) {
      auto offsets = subview.getMixedOffsets();
      auto sizes = subview.getMixedSizes();
      auto strides = subview.getMixedStrides();
      MemRefType resultType = memref::SubViewOp::inferResultType(
        cast<MemRefType>(replacement.getType()), offsets, sizes, strides);
      replacement = builder.create<memref::SubViewOp>(
        matmul.getLoc(), resultType, replacement, offsets, sizes, strides);
    }
    auto replacementType = dyn_cast<MemRefType>(replacement.getType());
    auto originalType = dyn_cast<MemRefType>(matmul.getInputs()[1].getType());
    if (!replacementType || !originalType ||
        replacementType.getShape() != originalType.getShape() ||
        replacementType.getElementType() != originalType.getElementType()) {
      return std::nullopt;
    }
    matmul->setOperand(1, replacement);
    for (Operation* view : rhsSource.views) {
      if (view->use_empty()) {
        view->erase();
      }
    }
    if (rhsSource.access && rhsSource.access->use_empty()) {
      rhsSource.access.erase();
    }

    // Mark the exact global access so the execution-plan pass accounts for the
    // padded compile-time storage once, rather than once per tiled consumer.
    contract::setBool(
      packedGlobal.getOperation(), contract::kPackedWeight, true);
    contract::setInteger(packedGlobal.getOperation(),
                         contract::kPackBytes,
                         packedInfo->packedBytes);
    contract::setInteger(packedGlobal.getOperation(),
                         contract::kPackRawBytes,
                         packedInfo->rawBytes);
    contract::setInteger(
      packedGlobal.getOperation(), contract::kPackFactor, 16);
    contract::setInteger(packedGlobal.getOperation(), contract::kPackTileK, 64);
    contract::setString(packedGlobal.getOperation(),
                        contract::kPackSchema,
                        contract::kInt8PanelPackSchema);
    contract::setString(
      packedGlobal.getOperation(), contract::kWeightLayout, "panel_nk_kpad64");
    contract::setString(
      packedGlobal.getOperation(), contract::kAlignment, "64");
    contract::setString(
      packedGlobal.getOperation(), contract::kPackRuntime, "compile_time_B");
    return replacement;
  }

  void processInt8MemRef(linalg::MatmulTransposeBOp matmul, ModuleOp module) {
    auto lhsType = dyn_cast<MemRefType>(matmul.getInputs()[0].getType());
    auto rhsType = dyn_cast<MemRefType>(matmul.getInputs()[1].getType());
    auto outputType =
      dyn_cast<MemRefType>(matmul.getOutputs().front().getType());
    if (!lhsType || !rhsType || !outputType ||
        !lhsType.getElementType().isInteger(8) ||
        !rhsType.getElementType().isInteger(8) ||
        !outputType.getElementType().isInteger(32)) {
      return;
    }
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outputType.hasStaticShape() || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outputType.getRank() != 2) {
      annotateInt8Rejected(matmul, "packing_rejected_dynamic");
      return;
    }
    if (auto selected = matmul->getAttrOfType<StringAttr>(contract::kPacking);
        selected && selected.getValue() == "prepacked_B") {
      return;
    }
    if (auto fallback = matmul->getAttrOfType<StringAttr>(contract::kFallback);
        fallback && fallback.getValue() == "unsupported_consumer") {
      annotateInt8Rejected(matmul, "packing_rejected_layout");
      return;
    }

    auto rhsSource = findInt8RhsSource(module, matmul.getInputs()[1]);
    if (!rhsSource) {
      annotateInt8Rejected(matmul, "packing_rejected_nonconstant_rhs");
      return;
    }
    MemRefType sourceType = rhsSource->global.getType();
    if (sourceType.getRank() != 2 || !sourceType.hasStaticShape() ||
        !sourceType.getElementType().isInteger(8)) {
      annotateInt8Rejected(matmul, "packing_rejected_layout");
      return;
    }
    const int64_t rows = sourceType.getShape()[0];
    const int64_t depth = sourceType.getShape()[1];
    const int64_t lhsRows = lhsType.getShape()[0];
    const int64_t lhsDepth = lhsType.getShape()[1];
    const int64_t outputRows = outputType.getShape()[0];
    const int64_t outputColumns = outputType.getShape()[1];
    llvm::SmallVector<int64_t, 2> rhsStrides;
    int64_t rhsOffset = 0;
    if (rows <= 0 || depth <= 0 || lhsRows != outputRows || lhsDepth != depth ||
        rhsType.getShape()[1] != depth ||
        rhsType.getShape()[0] != outputColumns ||
        failed(rhsType.getStridesAndOffset(rhsStrides, rhsOffset)) ||
        rhsStrides.size() != 2 || rhsStrides[1] != 1) {
      annotateInt8Rejected(matmul, "packing_rejected_layout");
      return;
    }
    for (memref::SubViewOp subview : rhsSource->subviews) {
      auto offsets = subview.getMixedOffsets();
      auto sizes = subview.getMixedSizes();
      auto strides = subview.getMixedStrides();
      if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2 ||
          getStaticIndex(offsets[1]) != 0 ||
          getStaticIndex(sizes[1]) != depth ||
          getStaticIndex(strides[1]) != 1) {
        annotateInt8Rejected(matmul, "packing_rejected_layout");
        return;
      }
    }
    if (rows < 8 || depth < 32) {
      annotateInt8Rejected(matmul, "packing_skipped_small_shape");
      return;
    }
    if (packN != 16) {
      annotateInt8Rejected(matmul, "packing_rejected_layout");
      return;
    }
    if (rows > std::numeric_limits<int64_t>::max() / depth) {
      annotateInt8Rejected(matmul, "packing_rejected_budget");
      return;
    }
    const int64_t rawBytes = rows * depth;
    auto elements =
      dyn_cast<DenseIntElementsAttr>(rhsSource->global.getConstantInitValue());
    if (!rhsSource->global.getConstant() || !elements ||
        !std::cmp_equal(elements.getNumElements(), rawBytes)) {
      annotateInt8Rejected(matmul, "packing_rejected_nonconstant_rhs");
      return;
    }
    if (depth > std::numeric_limits<int64_t>::max() - 63) {
      annotateInt8Rejected(matmul, "packing_rejected_budget");
      return;
    }
    const int64_t paddedDepth = ((depth + 63) / 64) * 64;
    if (rows > std::numeric_limits<int64_t>::max() / paddedDepth) {
      annotateInt8Rejected(matmul, "packing_rejected_budget");
      return;
    }
    const int64_t packedBytes = rows * paddedDepth;
    const bool needsNewGlobal =
      !packedInt8Globals.contains(rhsSource->global.getOperation());
    if (needsNewGlobal &&
        (packedBytesUsed > std::numeric_limits<int64_t>::max() - packedBytes ||
         (maxTotalBytes > 0 &&
          packedBytesUsed > maxTotalBytes - packedBytes))) {
      annotateInt8Rejected(matmul, "packing_rejected_budget");
      return;
    }

    auto flatType = MemRefType::get({packedBytes}, sourceType.getElementType());
    DenseIntElementsAttr initializer;
    if (needsNewGlobal) {
      SmallVector<APInt> source;
      source.reserve(static_cast<size_t>(rawBytes));
      for (APInt value : elements.getValues<APInt>()) {
        source.push_back(value);
      }
      SmallVector<APInt> packed;
      packed.reserve(static_cast<size_t>(packedBytes));
      for (int64_t panel = 0; panel < rows; panel += 16) {
        const int64_t width = std::min<int64_t>(16, rows - panel);
        for (int64_t lane = 0; lane < width; ++lane) {
          const int64_t row = panel + lane;
          for (int64_t k = 0; k < paddedDepth; ++k) {
            packed.push_back(k < depth
                               ? source[static_cast<size_t>((row * depth) + k)]
                               : APInt(8, 0));
          }
        }
      }
      auto initializerType =
        RankedTensorType::get({packedBytes}, sourceType.getElementType());
      initializer = DenseIntElementsAttr::get(initializerType, packed);
    }

    auto packedView = createPackedInt8View(matmul,
                                           module,
                                           *rhsSource,
                                           rows,
                                           depth,
                                           paddedDepth,
                                           flatType,
                                           initializer,
                                           rawBytes,
                                           packedBytes);
    if (!packedView) {
      annotateInt8Rejected(matmul, "packing_rejected_layout");
      return;
    }
    contract::annotatePacking(
      matmul.getOperation(), "prepacked_B", 16, packedBytes, 0);
    contract::setString(
      matmul.getOperation(), contract::kWeightLayout, "panel_nk_kpad64");
    contract::setString(matmul.getOperation(),
                        contract::kPackSchema,
                        contract::kInt8PanelPackSchema);
    contract::setInteger(
      matmul.getOperation(), contract::kPackRawBytes, rawBytes);
    contract::setInteger(matmul.getOperation(), contract::kPackTileK, 64);
    contract::setString(
      matmul.getOperation(), contract::kPackRuntime, "compile_time_B");
    contract::setString(matmul.getOperation(), contract::kAlignment, "64");
    contract::setString(matmul.getOperation(), contract::kContract, "selected");
  }

  void process(linalg::MatmulOp matmul) const {
    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    Value output = matmul.getOutputs().front();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    if (!lhsType || !rhsType || !outputType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || outputType.getRank() != 2) {
      annotateRejected(matmul, "packing_rejected_layout");
      return;
    }
    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
        !outputType.hasStaticShape()) {
      annotateRejected(matmul, "packing_rejected_dynamic");
      return;
    }
    if (!rhsType.getElementType().isF32() ||
        !lhsType.getElementType().isF32() ||
        !outputType.getElementType().isF32()) {
      annotateRejected(matmul, "packing_rejected_layout");
      return;
    }

    const int64_t rows = lhsType.getShape()[0];
    const int64_t depth = lhsType.getShape()[1];
    const int64_t columns = rhsType.getShape()[1];
    if (rows <= 0 || depth <= 0 || columns <= 0 ||
        rhsType.getShape()[0] != depth || outputType.getShape()[0] != rows ||
        outputType.getShape()[1] != columns) {
      annotateRejected(matmul, "packing_rejected_layout");
      return;
    }
    if (!enabled) {
      annotateRejected(matmul, "unpacked_direct");
      return;
    }
    // Epilogue fusion records unsupported consumers explicitly. Such matmuls
    // remain in the generic/direct path and have no packed kernel consumer;
    // physically reordering their RHS here would make that fallback consume
    // the wrong layout.
    if (auto fallback = matmul->getAttrOfType<StringAttr>(contract::kFallback);
        fallback && fallback.getValue() == "unsupported_consumer") {
      annotateRejected(matmul, "packing_rejected_layout");
      return;
    }
    // A one-row GEMM is not tileable by the downstream forall pass and stays
    // as a generic linalg.matmul. Keep its RHS row-major so that an
    // unkernelized fallback never consumes a physical panel layout.
    if (rows < 2 || columns < minN || depth < minK) {
      annotateRejected(matmul, "packing_skipped_small_shape");
      return;
    }
    const int64_t rowChunk = pickTileDivisor(rows, tileRows);
    const int64_t columnChunk = pickTileDivisor(columns, tileColumns);
    if (rowChunk >= rows && columnChunk >= columns) {
      // The downstream forall tiler leaves this matmul at top level. A
      // physically packed RHS would then be consumed by generic row-major
      // lowering, so keep the original storage layout.
      annotateRejected(matmul, "packing_skipped_small_shape");
      return;
    }
    if (packN != 16 || tileK <= 0 || columns < packN || columns % 32 != 0) {
      // P20-v1 uses a fixed 16-lane physical panel. Do not accept a custom
      // panel width that the bufferized kernel may not be able to address
      // without crossing a panel boundary.
      annotateRejected(matmul, "packing_rejected_budget");
      return;
    }

    arith::ConstantOp constant = findConstant(rhs);
    if (!constant) {
      annotateRejected(matmul, "packing_rejected_dynamic");
      return;
    }
    auto elements = dyn_cast<DenseFPElementsAttr>(constant.getValueAttr());
    if (!elements || elements.getElementType() != rhsType.getElementType() ||
        elements.getNumElements() != depth * columns) {
      annotateRejected(matmul, "packing_rejected_layout");
      return;
    }
    if (matmul->hasAttr(contract::kPackedWeight)) {
      return;
    }

    const int64_t elementCount = depth * columns;
    if (depth != 0 && columns > std::numeric_limits<int64_t>::max() / depth) {
      annotateRejected(matmul, "packing_rejected_budget");
      return;
    }
    if (elementCount > std::numeric_limits<int64_t>::max() /
                         static_cast<int64_t>(sizeof(float))) {
      annotateRejected(matmul, "packing_rejected_budget");
      return;
    }
    const int64_t rawBytes = elementCount * static_cast<int64_t>(sizeof(float));
    if (maxTotalBytes > 0 && (packedBytesUsed > maxTotalBytes - rawBytes)) {
      annotateRejected(matmul, "packing_rejected_budget");
      return;
    }
    SmallVector<APFloat> source;
    source.reserve(static_cast<size_t>(elementCount));
    for (APFloat value : elements.getValues<APFloat>()) {
      source.push_back(value);
    }

    SmallVector<APFloat> packed;
    packed.reserve(source.size());
    for (int64_t panel = 0; panel < columns; panel += packN) {
      const int64_t width = std::min<int64_t>(packN, columns - panel);
      for (int64_t k = 0; k < depth; ++k) {
        for (int64_t lane = 0; lane < width; ++lane) {
          packed.push_back(
            source[static_cast<size_t>((k * columns) + panel + lane)]);
        }
      }
    }

    OpBuilder builder(matmul);
    auto packedConstant = builder.create<arith::ConstantOp>(
      matmul.getLoc(), rhsType, DenseFPElementsAttr::get(rhsType, packed));
    contract::setBool(
      packedConstant.getOperation(), contract::kPackedWeight, true);
    contract::setInteger(
      packedConstant.getOperation(), contract::kPackFactor, packN);
    contract::setInteger(
      packedConstant.getOperation(), contract::kPackTileK, tileK);
    contract::setString(
      packedConstant.getOperation(), contract::kPackSchema, "p20-panel-nk-v1");
    contract::setString(
      packedConstant.getOperation(), contract::kWeightLayout, "panel_nk");
    contract::setString(
      packedConstant.getOperation(), contract::kAlignment, "64");

    matmul->setOperand(1, packedConstant.getResult());
    packedBytesUsed += rawBytes;
    contract::annotatePacking(
      matmul.getOperation(), "prepacked_B", packN, rawBytes, 0);
    contract::setString(
      matmul.getOperation(), contract::kWeightLayout, "panel_nk");
    contract::setString(
      matmul.getOperation(), contract::kPackSchema, "p20-panel-nk-v1");
    contract::setInteger(
      matmul.getOperation(), contract::kPackRawBytes, rawBytes);
    contract::setInteger(matmul.getOperation(), contract::kPackTileK, tileK);
    contract::setString(
      matmul.getOperation(), contract::kPackRuntime, "compile_time_B");
    contract::setString(matmul.getOperation(), contract::kAlignment, "64");
    contract::setString(matmul.getOperation(), contract::kContract, "selected");

    // A StrategyNCNN-produced convolution matmul is the Conv side of P21.
    // The packed constant remains consumed by MatmulKernelNCNN; these fields
    // describe that physical contract rather than a metadata-only selection.
    auto family = matmul->getAttrOfType<StringAttr>(contract::kOperationFamily);
    auto packedLayout =
      matmul->getParentOfType<ModuleOp>()->getAttrOfType<BoolAttr>(
        "ncnn.packed_conv_depthwise");
    if (family && family.getValue() == "conv" && packedLayout &&
        packedLayout.getValue()) {
      contract::annotateOperationFamily(
        matmul.getOperation(), "conv", "conv_packed_gemm");
      contract::annotateLayoutIsland(matmul.getOperation(),
                                     "conv-packed-gemm",
                                     "selected",
                                     "abi_to_panel_nk",
                                     "panel_nk_to_abi",
                                     "packed_rhs_only",
                                     "static_conv_gemm",
                                     packN,
                                     columns / packN);
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
