#include "ncnn-mlir/Transforms/PackStaticMatmulNCNN/PackStaticMatmulNCNN.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
    module.walk([&](linalg::MatmulOp matmul) {
      if (matmul.hasPureTensorSemantics()) {
        candidates.push_back(matmul);
      }
    });

    for (linalg::MatmulOp matmul : candidates) {
      process(matmul);
    }
  }

 private:
  // Keep object/rodata growth bounded for large models. A zero budget is an
  // explicit escape hatch used only by focused transform tests.
  mutable std::int64_t packedBytesUsed = 0;

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
  }
};

}  // namespace

}  // namespace mlir::ncnn
