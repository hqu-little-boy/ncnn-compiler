#include "ncnn-mlir/Transforms/RewriteLinalgCopies/RewriteLinalgCopies.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_REWRITELINALGCOPIESPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

struct CopyRecord {
  std::string function;
  std::string kind;
  std::optional<int64_t> bytes;
  int64_t vectorLanes = 0;
};

struct CopyStats {
  int64_t eliminated = 0;
  int64_t vectorized = 0;
  int64_t fallback = 0;
  SmallVector<CopyRecord> records;

  void record(Operation* anchor,
              StringRef kind,
              std::optional<int64_t> bytes,
              int64_t vectorLanes = 0) {
    auto function = anchor->getParentOfType<func::FuncOp>();
    records.push_back(
      CopyRecord{.function = function ? function.getName().str() : "",
                 .kind = kind.str(),
                 .bytes = bytes,
                 .vectorLanes = vectorLanes});
  }
};

std::optional<int64_t> getElementBytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    const unsigned bits = integer.getWidth();
    if (bits != 0) {
      return static_cast<int64_t>((bits / 8) + (bits % 8 != 0));
    }
  }
  if (auto floating = dyn_cast<FloatType>(type)) {
    const unsigned bits = floating.getWidth();
    if (bits != 0) {
      return static_cast<int64_t>((bits / 8) + (bits % 8 != 0));
    }
  }
  return std::nullopt;
}

std::optional<int64_t> getStaticCopyBytes(MemRefType type) {
  if (!type.hasStaticShape()) {
    return std::nullopt;
  }
  auto elementBytes = getElementBytes(type.getElementType());
  if (!elementBytes) {
    return std::nullopt;
  }
  int64_t elements = 1;
  for (int64_t extent : type.getShape()) {
    if (extent < 0) {
      return std::nullopt;
    }
    if (extent == 0) {
      return int64_t{0};
    }
    if (elements > std::numeric_limits<int64_t>::max() / extent) {
      return std::nullopt;
    }
    elements *= extent;
  }
  if (elements > std::numeric_limits<int64_t>::max() / *elementBytes) {
    return std::nullopt;
  }
  return elements * *elementBytes;
}

bool hasContiguousIdentityLayout(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      strides.size() != static_cast<size_t>(type.getRank()) ||
      ShapedType::isDynamic(offset)) {
    return false;
  }

  int64_t expected = 1;
  for (int64_t index = type.getRank() - 1; index >= 0; --index) {
    if (strides[index] != expected) {
      return false;
    }
    const int64_t extent = type.getDimSize(index);
    if (extent < 0) {
      return false;
    }
    if (extent == 0) {
      return true;
    }
    if (expected > std::numeric_limits<int64_t>::max() / extent) {
      return false;
    }
    expected *= extent;
  }
  return true;
}

Operation* getStorageRoot(Value value) {
  Value current = value;
  for (unsigned depth = 0; depth != 8; ++depth) {
    Operation* definition = current.getDefiningOp();
    if (!definition) {
      return nullptr;
    }
    if (isa<memref::AllocOp, memref::GetGlobalOp>(definition)) {
      return definition;
    }
    if (!isa<memref::SubViewOp,
             memref::CastOp,
             memref::ReinterpretCastOp,
             memref::MemorySpaceCastOp>(definition) ||
        definition->getNumOperands() == 0) {
      return nullptr;
    }
    current = definition->getOperand(0);
  }
  return nullptr;
}

bool hasProvenDisjointStorage(Value source, Value target) {
  if (source == target) {
    return false;
  }
  Operation* sourceRoot = getStorageRoot(source);
  Operation* targetRoot = getStorageRoot(target);
  if (!sourceRoot || !targetRoot || sourceRoot == targetRoot) {
    return false;
  }
  auto sourceGlobal = dyn_cast<memref::GetGlobalOp>(sourceRoot);
  auto targetGlobal = dyn_cast<memref::GetGlobalOp>(targetRoot);
  if (sourceGlobal && targetGlobal) {
    Attribute sourceName = sourceGlobal->getAttr("name");
    Attribute targetName = targetGlobal->getAttr("name");
    return sourceName && targetName && sourceName != targetName;
  }
  return true;
}

bool isIdentityCopyBody(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1) {
    return false;
  }
  ArrayRef<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[0].isIdentity() || !maps[1].isIdentity()) {
    return false;
  }
  for (utils::IteratorType iteratorType : generic.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  Block& block = generic.getRegion().front();
  if (block.getOperations().size() != 1) {
    return false;
  }
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         !block.getArguments().empty() &&
         yield.getOperand(0) == block.getArgument(0);
}

void buildSequentialCopy(PatternRewriter& rewriter,
                         Location location,
                         Value source,
                         Value target,
                         Operation** root = nullptr) {
  const unsigned rank = cast<ShapedType>(source.getType()).getRank();
  SmallVector<Value> indices(rank);
  std::function<void(unsigned)> buildLevel = [&](unsigned level) {
    if (level == rank) {
      auto load = rewriter.create<memref::LoadOp>(location, source, indices);
      auto store = rewriter.create<memref::StoreOp>(
        location, load.getResult(), target, indices);
      if (root && !*root) {
        *root = store.getOperation();
      }
      return;
    }
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(location, 0);
    Value extent = rewriter.create<memref::DimOp>(location, source, level);
    Value step = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto loop = rewriter.create<scf::ForOp>(location, lowerBound, extent, step);
    if (root && !*root) {
      *root = loop.getOperation();
    }
    rewriter.setInsertionPointToStart(loop.getBody());
    indices[level] = loop.getInductionVar();
    buildLevel(level + 1);
    rewriter.setInsertionPointAfter(loop);
  };
  buildLevel(0);
}

class CopyEmitter {
 public:
  CopyEmitter(PatternRewriter& rewriter,
              Location location,
              Value source,
              Value target,
              MemRefType type,
              unsigned lanes,
              Operation** root)
    : rewriter(rewriter),
      location(location),
      source(source),
      target(target),
      type(type),
      lanes(lanes),
      root(root),
      indices(type.getRank()) {}

  void emit() {
    if (type.getRank() != 1) {
      buildOuter(0);
      return;
    }
    // One enclosing loop keeps the vector body and scalar tail under the same
    // copy contract and runtime profiling event.
    auto lower = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto upper = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto step = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto loop = rewriter.create<scf::ForOp>(location, lower, upper, step);
    if (root) {
      *root = loop.getOperation();
    }
    rewriter.setInsertionPointToStart(loop.getBody());
    emitInnermost();
    rewriter.setInsertionPointAfter(loop);
  }

 private:
  void buildOuter(unsigned level) {
    const unsigned rank = type.getRank();
    if (level + 1 == rank) {
      emitInnermost();
      return;
    }

    auto lower = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto upper =
      rewriter.create<arith::ConstantIndexOp>(location, type.getDimSize(level));
    auto step = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto loop = rewriter.create<scf::ForOp>(location, lower, upper, step);
    if (root && !*root) {
      *root = loop.getOperation();
    }
    rewriter.setInsertionPointToStart(loop.getBody());
    indices[level] = loop.getInductionVar();
    buildOuter(level + 1);
    rewriter.setInsertionPointAfter(loop);
  }

  void emitInnermost() {
    const int64_t extent = type.getDimSize(type.getRank() - 1);
    const int64_t vectorEnd = (extent / lanes) * lanes;
    auto lower = rewriter.create<arith::ConstantIndexOp>(location, 0);
    auto upper = rewriter.create<arith::ConstantIndexOp>(location, vectorEnd);
    auto step = rewriter.create<arith::ConstantIndexOp>(location, lanes);
    auto vectorLoop = rewriter.create<scf::ForOp>(location, lower, upper, step);
    if (root && !*root && type.getRank() > 1) {
      *root = vectorLoop.getOperation();
    }
    rewriter.setInsertionPointToStart(vectorLoop.getBody());
    indices.back() = vectorLoop.getInductionVar();

    auto elementType = type.getElementType();
    auto vectorType =
      VectorType::get({static_cast<int64_t>(lanes)}, elementType);
    Value padding = rewriter.create<arith::ConstantOp>(
      location, elementType, rewriter.getZeroAttr(elementType));
    SmallVector<bool> inBounds(1, true);
    Value loaded = rewriter.create<vector::TransferReadOp>(
      location, vectorType, source, indices, padding, inBounds);
    rewriter.create<vector::TransferWriteOp>(
      location, loaded, target, indices, inBounds);

    rewriter.setInsertionPointAfter(vectorLoop);
    auto tailLower =
      rewriter.create<arith::ConstantIndexOp>(location, vectorEnd);
    auto tailUpper = rewriter.create<arith::ConstantIndexOp>(location, extent);
    auto tailStep = rewriter.create<arith::ConstantIndexOp>(location, 1);
    auto tailLoop =
      rewriter.create<scf::ForOp>(location, tailLower, tailUpper, tailStep);
    if (root && !*root) {
      *root = tailLoop.getOperation();
    }
    rewriter.setInsertionPointToStart(tailLoop.getBody());
    indices.back() = tailLoop.getInductionVar();
    Value value = rewriter.create<memref::LoadOp>(location, source, indices);
    rewriter.create<memref::StoreOp>(location, value, target, indices);
    rewriter.setInsertionPointAfter(tailLoop);
  }

  PatternRewriter& rewriter;
  Location location;
  Value source;
  Value target;
  MemRefType type;
  unsigned lanes;
  Operation** root;
  SmallVector<Value> indices;
};

bool canVectorize(Value source,
                  Value target,
                  MemRefType sourceType,
                  MemRefType targetType,
                  unsigned lanes) {
  if (lanes == 0 || sourceType.getRank() == 0 ||
      sourceType.getElementType() != targetType.getElementType() ||
      sourceType.getShape() != targetType.getShape() ||
      !sourceType.hasStaticShape() || !targetType.hasStaticShape() ||
      !isa<IntegerType, FloatType, IndexType>(sourceType.getElementType()) ||
      !hasContiguousIdentityLayout(sourceType) ||
      !hasContiguousIdentityLayout(targetType) ||
      sourceType.getDimSize(sourceType.getRank() - 1) <
        static_cast<int64_t>(lanes) ||
      !hasProvenDisjointStorage(source, target)) {
    return false;
  }
  return true;
}

void annotateAndCount(Operation* operation,
                      CopyStats& stats,
                      StringRef kind,
                      std::optional<int64_t> bytes,
                      int64_t lanes = 0) {
  contract::annotateCopy(operation, "p22_identity_copy", kind, bytes, lanes);
  stats.record(operation, kind, bytes, lanes);
  if (kind == "vectorized") {
    ++stats.vectorized;
  } else {
    ++stats.fallback;
  }
}

LogicalResult lowerConcreteCopy(PatternRewriter& rewriter,
                                Operation* copy,
                                Value source,
                                Value target,
                                CopyStats& stats,
                                bool vectorize,
                                unsigned lanes) {
  if (source == target) {
    ++stats.eliminated;
    stats.record(copy, "eliminated", std::nullopt);
    rewriter.eraseOp(copy);
    return success();
  }
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto targetType = dyn_cast<MemRefType>(target.getType());
  if (!sourceType || !targetType ||
      failed(verifyCompatibleShape(sourceType, targetType)) ||
      sourceType.getElementType() != targetType.getElementType()) {
    return failure();
  }

  const auto bytes = getStaticCopyBytes(sourceType);
  Operation* root = nullptr;
  if (vectorize &&
      canVectorize(source, target, sourceType, targetType, lanes)) {
    CopyEmitter emitter(
      rewriter, copy->getLoc(), source, target, sourceType, lanes, &root);
    emitter.emit();
    if (root) {
      annotateAndCount(root, stats, "vectorized", bytes, lanes);
    }
  } else {
    buildSequentialCopy(rewriter, copy->getLoc(), source, target, &root);
    if (root) {
      annotateAndCount(root, stats, "fallback", bytes);
    }
  }
  rewriter.eraseOp(copy);
  return success();
}

struct LowerLinalgCopy : public OpRewritePattern<linalg::CopyOp> {
  LowerLinalgCopy(MLIRContext* context,
                  CopyStats& stats,
                  bool vectorize,
                  unsigned lanes)
    : OpRewritePattern<linalg::CopyOp>(context),
      stats(stats),
      vectorize(vectorize),
      lanes(lanes) {}

  LogicalResult matchAndRewrite(linalg::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    auto inputs = copy.getInputs();
    auto outputs = copy.getOutputs();
    if (inputs.size() != 1 || outputs.size() != 1) {
      return failure();
    }
    return lowerConcreteCopy(rewriter,
                             copy.getOperation(),
                             inputs.front(),
                             outputs.front(),
                             stats,
                             vectorize,
                             lanes);
  }

 private:
  CopyStats& stats;
  bool vectorize;
  unsigned lanes;
};

struct LowerMemRefCopy : public OpRewritePattern<memref::CopyOp> {
  LowerMemRefCopy(MLIRContext* context,
                  CopyStats& stats,
                  bool vectorize,
                  unsigned lanes)
    : OpRewritePattern<memref::CopyOp>(context),
      stats(stats),
      vectorize(vectorize),
      lanes(lanes) {}

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    return lowerConcreteCopy(rewriter,
                             copy.getOperation(),
                             copy.getSource(),
                             copy.getTarget(),
                             stats,
                             vectorize,
                             lanes);
  }

 private:
  CopyStats& stats;
  bool vectorize;
  unsigned lanes;
};

struct EliminateLinalgCopy : public OpRewritePattern<linalg::CopyOp> {
  EliminateLinalgCopy(MLIRContext* context, CopyStats& stats)
    : OpRewritePattern<linalg::CopyOp>(context), stats(stats) {}

  LogicalResult matchAndRewrite(linalg::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    if (copy->getNumOperands() != 2 ||
        copy->getOperand(0) != copy->getOperand(1)) {
      return failure();
    }
    ++stats.eliminated;
    stats.record(copy, "eliminated", std::nullopt);
    rewriter.eraseOp(copy);
    return success();
  }

 private:
  CopyStats& stats;
};

struct EliminateMemRefCopy : public OpRewritePattern<memref::CopyOp> {
  EliminateMemRefCopy(MLIRContext* context, CopyStats& stats)
    : OpRewritePattern<memref::CopyOp>(context), stats(stats) {}

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter& rewriter) const override {
    if (copy.getSource() != copy.getTarget()) {
      return failure();
    }
    ++stats.eliminated;
    stats.record(copy, "eliminated", std::nullopt);
    rewriter.eraseOp(copy);
    return success();
  }

 private:
  CopyStats& stats;
};

struct LinalgCopyToLoops : public OpRewritePattern<linalg::GenericOp> {
  LinalgCopyToLoops(MLIRContext* context,
                    CopyStats& stats,
                    bool vectorize,
                    unsigned lanes)
    : OpRewritePattern<linalg::GenericOp>(context),
      stats(stats),
      vectorize(vectorize),
      lanes(lanes) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter& rewriter) const override {
    if (!isIdentityCopyBody(generic)) {
      return failure();
    }
    Value source = generic.getInputs()[0];
    Value target = generic.getDpsInitOperand(0)->get();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    auto targetType = dyn_cast<MemRefType>(target.getType());
    if (!sourceType || !targetType ||
        failed(verifyCompatibleShape(sourceType, targetType)) ||
        sourceType.getElementType() != targetType.getElementType()) {
      return failure();
    }
    if (source == target) {
      ++stats.eliminated;
      stats.record(generic, "eliminated", std::nullopt);
      rewriter.eraseOp(generic);
      return success();
    }

    const auto bytes = getStaticCopyBytes(sourceType);
    Operation* root = nullptr;
    if (vectorize &&
        canVectorize(source, target, sourceType, targetType, lanes)) {
      CopyEmitter emitter(
        rewriter, generic.getLoc(), source, target, sourceType, lanes, &root);
      emitter.emit();
      if (root) {
        annotateAndCount(root, stats, "vectorized", bytes, lanes);
      }
    } else {
      buildSequentialCopy(rewriter, generic.getLoc(), source, target, &root);
      if (root) {
        annotateAndCount(root, stats, "fallback", bytes);
      }
    }
    rewriter.eraseOp(generic);
    return success();
  }

 private:
  CopyStats& stats;
  bool vectorize;
  unsigned lanes;
};

class RewriteLinalgCopiesPass final
  : public impl::RewriteLinalgCopiesPassBase<RewriteLinalgCopiesPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    CopyStats stats;
    RewritePatternSet patterns(&getContext());
    patterns.add<EliminateLinalgCopy>(&getContext(), stats);
    patterns.add<EliminateMemRefCopy>(&getContext(), stats);
    patterns.add<LowerLinalgCopy>(&getContext(), stats, vectorize, vectorLanes);
    patterns.add<LowerMemRefCopy>(&getContext(), stats, vectorize, vectorLanes);
    patterns.add<LinalgCopyToLoops>(
      &getContext(), stats, vectorize, vectorLanes);
    if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();
    const bool hadLedger = module->hasAttr(contract::kCopyRecords);
    // A repeated invocation may leave unsupported copies in place. Keep the
    // durable ledger instead of replacing it with an empty observation.
    if (stats.records.empty() && hadLedger) {
      module->removeAttr(contract::kCopyFusedCount);
      return;
    }

    module->removeAttr(contract::kCopyEliminatedCount);
    module->removeAttr(contract::kCopyVectorizedCount);
    module->removeAttr(contract::kCopyFallbackCount);
    module->removeAttr(contract::kCopyFusedCount);
    module->removeAttr(contract::kCopyRecords);
    SmallVector<Attribute> records;
    records.reserve(stats.records.size());
    for (const CopyRecord& copy : stats.records) {
      NamedAttrList record;
      record.set("function",
                 StringAttr::get(module.getContext(), copy.function));
      record.set("copy_contract",
                 StringAttr::get(module.getContext(), "p22_identity_copy"));
      record.set("copy_kind", StringAttr::get(module.getContext(), copy.kind));
      if (copy.bytes) {
        record.set("copy_bytes",
                   IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                    *copy.bytes));
      }
      record.set("copy_vector_lanes",
                 IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                  copy.vectorLanes));
      records.push_back(DictionaryAttr::get(module.getContext(), record));
    }
    module->setAttr(contract::kCopyRecords,
                    ArrayAttr::get(module.getContext(), records));
    contract::setInteger(
      module, contract::kCopyEliminatedCount, stats.eliminated);
    contract::setInteger(
      module, contract::kCopyVectorizedCount, stats.vectorized);
    contract::setInteger(module, contract::kCopyFallbackCount, stats.fallback);
  }
};

}  // namespace

}  // namespace mlir::ncnn
