#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNProfile.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_INSTRUMENTNCNNPROFILEPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

constexpr StringLiteral kBegin = "__ncnn_profile_event_begin";
constexpr StringLiteral kEnd = "__ncnn_profile_event_end";
constexpr StringLiteral kAlloc = "__ncnn_profile_alloc";
constexpr StringLiteral kDealloc = "__ncnn_profile_dealloc";
constexpr StringLiteral kCopy = "__ncnn_profile_copy";
constexpr StringLiteral kMovement = "__ncnn_profile_movement";
constexpr StringLiteral kFlush = "__ncnn_profile_flush";

// Categories are part of the private profile ABI.  They intentionally remain
// numeric so the generated module does not depend on a compiler-side enum.
enum class EventCategory : std::int64_t {
  Operation = 0,
  Allocation = 1,
  Deallocation = 2,
  Copy = 3,
  Parallel = 4,
  Transpose = 5,
  Pack = 6,
  Unpack = 7,
};

std::uint64_t fnv1a(StringRef value) {
  std::uint64_t result = 14695981039346656037ULL;
  for (unsigned char character : value.bytes()) {
    result ^= character;
    result *= 1099511628211ULL;
  }
  return result;
}

func::FuncOp declare(IRRewriter& rewriter,
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
  function->setAttr("ncnn.profile_runtime", UnitAttr::get(module.getContext()));
  return function;
}

Value constant(IRRewriter& rewriter, Location location, std::int64_t value) {
  return rewriter.create<arith::ConstantOp>(
    location,
    rewriter.getI64Type(),
    rewriter.getIntegerAttr(rewriter.getI64Type(), value));
}

void call(IRRewriter& rewriter,
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

std::int64_t staticByteSize(MemRefType type) {
  if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat()) {
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

class InstrumentNCNNProfilePass final
  : public impl::InstrumentNCNNProfilePassBase<InstrumentNCNNProfilePass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    omp::OpenMPDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr("ncnn.profile_instrumented")) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    const SmallVector<Type> oneId{i64};
    const SmallVector<Type> twoIds{i64, i64};
    auto begin = declare(rewriter, module, kBegin, {i64, i64});  // id, category
    auto end = declare(rewriter, module, kEnd, oneId);
    auto alloc = declare(rewriter, module, kAlloc, twoIds);
    auto dealloc = declare(rewriter, module, kDealloc, oneId);
    auto copy = declare(rewriter, module, kCopy, twoIds);
    auto movement = declare(rewriter, module, kMovement, {i64, i64, i64});
    auto flush = declare(rewriter, module, kFlush, {});

    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr("ncnn.profile_runtime")) {
        functions.push_back(function);
      }
    }

    for (func::FuncOp function : functions) {
      std::map<Operation*, std::uint64_t> operation_ids;
      std::map<std::string, unsigned> operation_ordinals;
      function.walk([&](Operation* operation) {
        const std::string kind = operation->getName().getStringRef().str();
        const std::string ordinal_key = function.getName().str() + "/" + kind;
        const unsigned ordinal = operation_ordinals[ordinal_key]++;
        operation_ids.emplace(
          operation, fnv1a(ordinal_key + "#" + std::to_string(ordinal)));
      });

      SmallVector<Operation*> candidates;
      function.walk([&](Operation* operation) {
        // Nested memory events need no region wrapping. Keep duration timers
        // at function scope: worker TLS spans cannot be subtracted from the
        // parent parallel span, and timing individual lanes is too expensive.
        if (isa<memref::AllocOp, memref::CopyOp>(operation) ||
            (operation->getParentOp() == function.getOperation() &&
             isProfileCandidate(*operation))) {
          candidates.push_back(operation);
        }
      });

      std::optional<std::uint64_t> root_id;
      if (function->hasAttr("ncnn.entry_point")) {
        if (auto root = operation_ids.find(function.getOperation());
            root != operation_ids.end()) {
          root_id = root->second;
        }
      }
      if (root_id && !function.getBody().empty()) {
        rewriter.setInsertionPointToStart(&function.getBody().front());
        Value idValue = constant(
          rewriter, function.getLoc(), static_cast<std::int64_t>(*root_id));
        Value categoryValue =
          constant(rewriter,
                   function.getLoc(),
                   static_cast<std::int64_t>(EventCategory::Operation));
        call(rewriter, function.getLoc(), begin, {idValue, categoryValue});
      }

      for (Operation* operation : candidates) {
        const std::uint64_t id = operation_ids.at(operation);
        const auto category =
          isa<memref::AllocOp>(*operation)       ? EventCategory::Allocation
          : isa<memref::CopyOp>(*operation)      ? EventCategory::Copy
          : isa<memref::TransposeOp>(*operation) ? EventCategory::Transpose
          : isa<scf::ForallOp, scf::ParallelOp, omp::ParallelOp>(*operation)
            ? EventCategory::Parallel
            : EventCategory::Operation;
        rewriter.setInsertionPoint(operation);
        Value idValue = constant(
          rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
        const bool timed = operation->getParentOp() == function.getOperation();
        if (timed) {
          Value categoryValue = constant(
            rewriter, operation->getLoc(), static_cast<std::int64_t>(category));
          call(rewriter, operation->getLoc(), begin, {idValue, categoryValue});
        }

        if (auto allocOp = dyn_cast<memref::AllocOp>(operation)) {
          const std::int64_t bytes = staticByteSize(allocOp.getType());
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          call(rewriter, operation->getLoc(), alloc, {idValue, byteValue});
        }
        if (auto copyOp = dyn_cast<memref::CopyOp>(operation)) {
          const auto sourceType =
            dyn_cast<MemRefType>(copyOp.getSource().getType());
          const std::int64_t bytes =
            sourceType ? staticByteSize(sourceType) : -1;
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          call(rewriter, operation->getLoc(), copy, {idValue, byteValue});
        }
        if (auto transposeOp = dyn_cast<memref::TransposeOp>(operation)) {
          const auto resultType =
            dyn_cast<MemRefType>(transposeOp.getResult().getType());
          const std::int64_t bytes =
            resultType ? staticByteSize(resultType) : -1;
          Value kind = constant(rewriter, operation->getLoc(), 0);
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          call(rewriter,
               operation->getLoc(),
               movement,
               {idValue, kind, byteValue});
        }

        if (timed) {
          rewriter.setInsertionPointAfter(operation);
          Value endId = constant(
            rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
          call(rewriter, operation->getLoc(), end, {endId});
        }
      }

      // Deallocation has no useful duration, but it is still an explicit
      // event.  Keep it separate from the operation timer so allocation bytes
      // cannot be mistaken for arithmetic time.
      SmallVector<Operation*> deallocations;
      function.walk([&](memref::DeallocOp operation) {
        deallocations.push_back(operation.getOperation());
      });
      auto resolveAllocation = [&](Value value) -> Operation* {
        // Follow the SSA forwarding performed by scf.for in addition to
        // view-like definitions. A loop result (or an iter arg used by a
        // nested dealloc) still denotes the original allocation.
        while (true) {
          if (auto allocOp = value.getDefiningOp<memref::AllocOp>()) {
            return allocOp.getOperation();
          }
          if (Operation* defining = value.getDefiningOp()) {
            if (isViewLike(*defining) && defining->getNumOperands() != 0) {
              value = defining->getOperand(0);
              continue;
            }
            if (auto forOp = dyn_cast<scf::ForOp>(defining)) {
              auto result = dyn_cast<OpResult>(value);
              if (!result ||
                  result.getResultNumber() >= forOp.getInitArgs().size()) {
                return nullptr;
              }
              value = forOp.getInitArgs()[result.getResultNumber()];
              continue;
            }
            return nullptr;
          }
          auto blockArgument = dyn_cast<BlockArgument>(value);
          if (!blockArgument) {
            return nullptr;
          }
          auto forOp =
            dyn_cast<scf::ForOp>(blockArgument.getOwner()->getParentOp());
          const unsigned argumentNumber = blockArgument.getArgNumber();
          if (!forOp || argumentNumber == 0 ||
              argumentNumber - 1 >= forOp.getInitArgs().size()) {
            return nullptr;
          }
          value = forOp.getInitArgs()[argumentNumber - 1];
        }
      };
      for (Operation* operation : deallocations) {
        std::uint64_t id = operation_ids.at(operation);
        if (operation->getNumOperands() == 1) {
          if (Operation* allocation =
                resolveAllocation(operation->getOperand(0))) {
            id = operation_ids.at(allocation);
          }
        }
        rewriter.setInsertionPoint(operation);
        Value idValue = constant(
          rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
        call(rewriter, operation->getLoc(), dealloc, {idValue});
      }

      if (root_id) {
        SmallVector<func::ReturnOp> returns;
        function.walk(
          [&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
        for (func::ReturnOp returnOp : returns) {
          rewriter.setInsertionPoint(returnOp);
          Value idValue = constant(
            rewriter, returnOp.getLoc(), static_cast<std::int64_t>(*root_id));
          call(rewriter, returnOp.getLoc(), end, {idValue});
          call(rewriter, returnOp.getLoc(), flush, {});
        }
      }
    }
    module->setAttr("ncnn.profile_instrumented",
                    UnitAttr::get(module.getContext()));
  }
};

}  // namespace

}  // namespace mlir::ncnn
