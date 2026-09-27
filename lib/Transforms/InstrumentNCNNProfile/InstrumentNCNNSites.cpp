// InstrumentNCNNSites.cpp：插桩器共享积木。
//
// 这里只放「所有插桩器都要用」的东西：运行期入口声明、i64 常量、调用、
// 静态字节尺寸、以及站点候选判定。任何单一类别专用的逻辑放到对应 TU。
// 契约与插入点纪律见 InstrumentNCNNSites.hpp 顶部。

#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

#include <limits>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn::sites {

func::FuncOp declareRuntime(IRRewriter& rewriter,
                            ModuleOp module,
                            StringRef name,
                            ArrayRef<Type> argumentTypes) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
    return existing;
  }
  rewriter.setInsertionPointToStart(module.getBody());
  auto type = rewriter.getFunctionType(argumentTypes, {});
  auto function = rewriter.create<func::FuncOp>(module.getLoc(), name, type);
  function.setPrivate();
  function->setAttr(contract::kProfileRuntime,
                    UnitAttr::get(module.getContext()));
  return function;
}

Value emitConstant(IRRewriter& rewriter,
                   Location location,
                   std::int64_t value) {
  return rewriter.create<arith::ConstantOp>(
    location,
    rewriter.getI64Type(),
    rewriter.getIntegerAttr(rewriter.getI64Type(), value));
}

void emitCall(IRRewriter& rewriter,
              Location location,
              func::FuncOp callee,
              ArrayRef<Value> arguments) {
  rewriter.create<func::CallOp>(location, callee, arguments);
}

bool isViewLike(Operation& operation) {
  return isa<memref::CastOp,
             memref::SubViewOp,
             memref::ReinterpretCastOp,
             memref::CollapseShapeOp,
             memref::ExpandShapeOp,
             memref::MemorySpaceCastOp,
             memref::ReshapeOp,
             memref::TransposeOp,
             memref::ViewOp,
             memref::ExtractStridedMetadataOp,
             memref::AssumeAlignmentOp>(operation);
}

std::int64_t staticByteSize(ShapedType type) {
  if (!type || !type.hasStaticShape() ||
      (isa<MemRefType>(type) &&
       !cast<MemRefType>(type).getLayout().isIdentity()) ||
      !type.getElementType().isIntOrFloat()) {
    return -1;
  }
  std::uint64_t count = 1;
  for (std::int64_t extent : type.getShape()) {
    if (extent < 0) {
      return -1;
    }
    if (extent == 0) {
      count = 0;
      continue;
    }
    if (count > std::numeric_limits<std::uint64_t>::max() /
                  static_cast<std::uint64_t>(extent)) {
      return -1;
    }
    count *= static_cast<std::uint64_t>(extent);
  }
  const unsigned width =
    type.getElementType().isInteger()
      ? (type.getElementType().getIntOrFloatBitWidth() + 7) / 8
      : type.getElementType().getIntOrFloatBitWidth() / 8;
  if (width == 0 || count > static_cast<std::uint64_t>(
                              std::numeric_limits<std::int64_t>::max()) /
                              width) {
    return -1;
  }
  return static_cast<std::int64_t>(count * width);
}

bool isStaticIdentityMemRef(MemRefType type) {
  return type && type.hasStaticShape() && type.getLayout().isIdentity() &&
         staticByteSize(type) > 0;
}

bool isWholeBufferView(Operation& operation) {
  Value source;
  Value result;
  if (auto cast = dyn_cast<memref::CastOp>(operation)) {
    source = cast.getSource();
    result = cast.getResult();
  } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(operation)) {
    source = collapse.getSrc();
    result = collapse.getResult();
  } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(operation)) {
    source = expand.getSrc();
    result = expand.getResult();
  } else if (auto subview = dyn_cast<memref::SubViewOp>(operation)) {
    source = subview.getSource();
    result = subview.getResult();
  } else {
    return false;
  }

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto resultType = dyn_cast<MemRefType>(result.getType());
  if (!isStaticIdentityMemRef(sourceType) ||
      !isStaticIdentityMemRef(resultType) ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getMemorySpace() != resultType.getMemorySpace() ||
      staticByteSize(sourceType) != staticByteSize(resultType)) {
    return false;
  }

  if (isa<memref::CastOp>(operation)) {
    return sourceType.getShape() == resultType.getShape();
  }
  if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(operation)) {
    return true;
  }

  auto subview = cast<memref::SubViewOp>(operation);
  if (sourceType.getRank() != resultType.getRank()) {
    return false;
  }
  ArrayRef<int64_t> offsets = subview.getStaticOffsets();
  ArrayRef<int64_t> sizes = subview.getStaticSizes();
  ArrayRef<int64_t> strides = subview.getStaticStrides();
  if (static_cast<std::int64_t>(offsets.size()) != sourceType.getRank() ||
      static_cast<std::int64_t>(sizes.size()) != sourceType.getRank() ||
      static_cast<std::int64_t>(strides.size()) != sourceType.getRank()) {
    return false;
  }
  for (int64_t dimension = 0; dimension < sourceType.getRank(); ++dimension) {
    if (offsets[dimension] != 0 ||
        sizes[dimension] != sourceType.getDimSize(dimension) ||
        sizes[dimension] != resultType.getDimSize(dimension) ||
        strides[dimension] != 1) {
      return false;
    }
  }
  return true;
}

std::optional<Operation*> traceWholeAllocation(
  Value value, SmallVectorImpl<Value>& aliases) {
  while (value) {
    aliases.push_back(value);
    if (auto allocation = value.getDefiningOp<memref::AllocOp>()) {
      return allocation.getOperation();
    }
    Operation* defining = value.getDefiningOp();
    if (!defining || !isWholeBufferView(*defining) ||
        defining->getNumOperands() == 0) {
      return std::nullopt;
    }
    value = defining->getOperand(0);
  }
  return std::nullopt;
}

bool hasFullAccessMap(linalg::LinalgOp operation, OpOperand* operand) {
  auto type = dyn_cast<ShapedType>(operand->get().getType());
  if (!type || !type.hasStaticShape() ||
      !isa<MemRefType, RankedTensorType>(type)) {
    return false;
  }
  AffineMap map = operation.getMatchingIndexingMap(operand);
  if (!map || !map.isProjectedPermutation() ||
      map.getNumResults() != type.getRank()) {
    return false;
  }

  SmallVector<int64_t> loopRanges = operation.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t range) { return range <= 0; })) {
    return false;
  }
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
    auto loopDimension = dyn_cast<AffineDimExpr>(map.getResult(dimension));
    if (!loopDimension || loopDimension.getPosition() >= loopRanges.size() ||
        loopRanges[loopDimension.getPosition()] != type.getDimSize(dimension)) {
      return false;
    }
  }
  return true;
}

bool isProfileCandidate(Operation& operation) {
  if (isa<func::FuncOp, func::ReturnOp, memref::DeallocOp>(operation) ||
      operation.hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  if (isa<memref::AllocOp,
          memref::CopyOp,
          memref::TransposeOp,
          scf::ForallOp,
          scf::ParallelOp>(operation)) {
    return true;
  }
  StringRef name = operation.getName().getStringRef();
  return name.starts_with("linalg.") || name.starts_with("vector.") ||
         name.starts_with("scf.") || name.starts_with("omp.");
}

bool isDirectParallelWorkerCandidate(Operation& operation) {
  if (!isa<scf::ForallOp, scf::ParallelOp, omp::ParallelOp>(
        operation.getParentOp()) ||
      !isProfileCandidate(operation) ||
      isa<memref::AllocOp, memref::CopyOp, memref::TransposeOp>(operation)) {
    return false;
  }
  StringRef name = operation.getName().getStringRef();
  // Time coarse per-worker loop/kernel scopes only. Instrumenting each vector
  // lane or scalar operation would distort the profile and dominate tiny ops.
  return isa<scf::ForOp>(operation) || name.starts_with("linalg.");
}

}  // namespace mlir::ncnn::sites
