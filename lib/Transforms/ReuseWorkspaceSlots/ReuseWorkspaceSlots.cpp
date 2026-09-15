#include "ncnn-mlir/Transforms/ReuseWorkspaceSlots/ReuseWorkspaceSlots.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_REUSEWORKSPACESLOTSPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

constexpr StringLiteral kReuseStatus = "ncnn.workspace_reuse_status";
constexpr StringLiteral kFallbackReason = "ncnn.workspace_fallback_reason";
constexpr StringLiteral kSlot = "ncnn.workspace_slot";
constexpr StringLiteral kSlotBytes = "ncnn.workspace_slot_bytes";
constexpr StringLiteral kSlotAlignment = "ncnn.workspace_slot_alignment";
constexpr StringLiteral kSlotThreadVisibility =
  "ncnn.workspace_slot_thread_visibility";
constexpr StringLiteral kSlotOwner = "ncnn.workspace_slot_owner";
constexpr StringLiteral kSlotLifetimeBegin =
  "ncnn.workspace_slot_lifetime_begin";
constexpr StringLiteral kSlotLifetimeEnd = "ncnn.workspace_slot_lifetime_end";
constexpr StringLiteral kReuseCount = "ncnn.workspace_reuse_count";

void setStringAttr(Operation* operation, StringRef name, StringRef value) {
  operation->setAttr(name, StringAttr::get(operation->getContext(), value));
}

void setIntegerAttr(Operation* operation, StringRef name, int64_t value) {
  operation->setAttr(
    name,
    IntegerAttr::get(IntegerType::get(operation->getContext(), 64), value));
}

std::optional<int64_t> staticByteSize(MemRefType type) {
  if (!type.hasStaticShape()) {
    return std::nullopt;
  }

  int64_t elementCount = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension < 0) {
      return std::nullopt;
    }
    if (dimension == 0) {
      elementCount = 0;
      continue;
    }
    if (elementCount > std::numeric_limits<int64_t>::max() / dimension) {
      return std::nullopt;
    }
    elementCount *= dimension;
  }

  unsigned elementBits = 0;
  Type elementType = type.getElementType();
  if (auto integer = dyn_cast<IntegerType>(elementType)) {
    elementBits = integer.getWidth();
  } else if (auto floating = dyn_cast<FloatType>(elementType)) {
    elementBits = floating.getWidth();
  } else if (isa<IndexType>(elementType)) {
    elementBits = 64;
  } else {
    return std::nullopt;
  }
  if (elementBits == 0 || elementBits % 8 != 0) {
    return std::nullopt;
  }

  const auto elementBytes = static_cast<int64_t>(elementBits / 8);
  if (elementCount > std::numeric_limits<int64_t>::max() / elementBytes) {
    return std::nullopt;
  }
  return elementCount * elementBytes;
}

int64_t explicitAlignment(memref::AllocOp allocation) {
  if (auto alignment = allocation->getAttrOfType<IntegerAttr>("alignment")) {
    return alignment.getInt();
  }
  return 0;
}

bool isIdentityLayout(MemRefType type) {
  return type.getLayout().isIdentity();
}

struct AllocationInfo final {
  memref::AllocOp allocation;
  memref::DeallocOp deallocation;
  MemRefType type;
  int64_t allocationPosition = 0;
  int64_t deallocationPosition = 0;
  int64_t lastUsePosition = 0;
  int64_t alignment = 0;
  int64_t bytes = 0;
  std::string fallbackReason;
  bool hasUse = false;

  bool eligible() const { return fallbackReason.empty(); }
};

struct Slot final {
  AllocationInfo* representative = nullptr;
  memref::DeallocOp latestDeallocation;
  int64_t latestDeallocationPosition = 0;
  int64_t alignment = 0;
  int64_t bytes = 0;
  int64_t memberCount = 0;
};

void markFallback(memref::AllocOp allocation, StringRef reason) {
  setStringAttr(allocation, kReuseStatus, "fallback");
  setStringAttr(allocation, kFallbackReason, reason);
}

bool hasMemRefResult(Operation* operation) {
  return llvm::any_of(operation->getResults(), [](Value result) {
    return isa<BaseMemRefType>(result.getType());
  });
}

class ReuseWorkspaceSlotsPass final
  : public impl::ReuseWorkspaceSlotsPassBase<ReuseWorkspaceSlotsPass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
  }

  static void markFunctionFallback(func::FuncOp function, StringRef reason) {
    function.walk(
      [&](memref::AllocOp allocation) { markFallback(allocation, reason); });
  }

  static std::optional<AllocationInfo> analyzeAllocation(
    memref::AllocOp allocation, DenseMap<Operation*, int64_t>& positions) {
    MemRefType type = allocation.getType();
    AllocationInfo info;
    info.allocation = allocation;
    info.type = type;
    info.allocationPosition = positions.lookup(allocation.getOperation());
    info.alignment = explicitAlignment(allocation);

    if (!type.hasStaticShape()) {
      info.fallbackReason = "dynamic_shape";
      return info;
    }
    if (!isIdentityLayout(type)) {
      info.fallbackReason = "non_identity_layout";
      return info;
    }
    std::optional<int64_t> bytes = staticByteSize(type);
    if (!bytes) {
      info.fallbackReason = "workspace_bytes_unknown";
      return info;
    }
    info.bytes = *bytes;

    for (Operation* user : allocation.getResult().getUsers()) {
      auto deallocation = dyn_cast<memref::DeallocOp>(user);
      if (deallocation) {
        if (info.deallocation) {
          info.fallbackReason = "multiple_deallocations";
          return info;
        }
        info.deallocation = deallocation;
        continue;
      }
      if (isa<func::CallOp, func::ReturnOp>(user)) {
        info.fallbackReason = "call_boundary";
        return info;
      }
      if (user->getNumRegions() != 0 || hasMemRefResult(user)) {
        info.fallbackReason = "alias_not_proven";
        return info;
      }
      auto position = positions.find(user);
      if (position == positions.end()) {
        info.fallbackReason = "control_flow_or_region";
        return info;
      }
      if (!info.hasUse || position->second > info.lastUsePosition) {
        info.lastUsePosition = position->second;
      }
      info.hasUse = true;
    }

    if (!info.deallocation) {
      info.fallbackReason = "no_matching_deallocation";
      return info;
    }
    auto deallocationPosition =
      positions.find(info.deallocation.getOperation());
    if (deallocationPosition == positions.end()) {
      info.fallbackReason = "control_flow_or_region";
      return info;
    }
    info.deallocationPosition = deallocationPosition->second;
    if (!info.hasUse) {
      info.fallbackReason = "no_direct_use";
      return info;
    }
    if (info.deallocationPosition <= info.lastUsePosition) {
      info.fallbackReason = "invalid_lifetime";
      return info;
    }
    return info;
  }

  static bool compatible(const Slot& slot, const AllocationInfo& info) {
    return slot.representative->type == info.type &&
           slot.latestDeallocationPosition < info.allocationPosition;
  }

  static void annotateSlot(const Slot& slot,
                           int64_t slotNumber,
                           func::FuncOp function) {
    memref::AllocOp allocation = slot.representative->allocation;
    setStringAttr(
      allocation, kReuseStatus, slot.memberCount > 1 ? "reused" : "dedicated");
    setIntegerAttr(allocation, kSlot, slotNumber);
    setIntegerAttr(allocation, kSlotBytes, slot.bytes);
    setIntegerAttr(allocation, kSlotAlignment, slot.alignment);
    setStringAttr(allocation, kSlotThreadVisibility, "function_serial");
    setStringAttr(allocation, kSlotOwner, function.getName());
    setIntegerAttr(
      allocation, kSlotLifetimeBegin, slot.representative->allocationPosition);
    setIntegerAttr(
      allocation, kSlotLifetimeEnd, slot.latestDeallocationPosition);
    setIntegerAttr(allocation, kReuseCount, slot.memberCount);
    allocation->removeAttr(kFallbackReason);
  }

  static void runOnFunction(func::FuncOp function) {
    if (!function.getBody().hasOneBlock()) {
      markFunctionFallback(function, "control_flow_or_region");
      return;
    }
    Block& block = function.getBody().front();
    DenseMap<Operation*, int64_t> positions;
    int64_t position = 0;
    for (Operation& operation : block) {
      positions[&operation] = position++;
      if (operation.getNumRegions() != 0) {
        markFunctionFallback(function, "control_flow_or_region");
        return;
      }
    }

    SmallVector<AllocationInfo> allocations;
    for (Operation& operation : block) {
      auto allocation = dyn_cast<memref::AllocOp>(&operation);
      if (!allocation) {
        continue;
      }
      std::optional<AllocationInfo> info =
        analyzeAllocation(allocation, positions);
      if (!info) {
        markFallback(allocation, "analysis_failed");
        continue;
      }
      if (!info->eligible()) {
        markFallback(allocation, info->fallbackReason);
      }
      allocations.push_back(std::move(*info));
    }

    SmallVector<Slot> slots;
    SmallVector<memref::AllocOp> allocationsToErase;
    SmallVector<memref::DeallocOp> deallocationsToErase;
    for (AllocationInfo& info : allocations) {
      if (!info.eligible()) {
        continue;
      }
      Slot* reusable = nullptr;
      for (Slot& slot : slots) {
        if (compatible(slot, info)) {
          reusable = &slot;
          break;
        }
      }

      if (!reusable) {
        Slot slot;
        slot.representative = &info;
        slot.latestDeallocation = info.deallocation;
        slot.latestDeallocationPosition = info.deallocationPosition;
        slot.alignment = info.alignment;
        slot.bytes = info.bytes;
        slot.memberCount = 1;
        slots.push_back(std::move(slot));
        continue;
      }

      info.allocation.getResult().replaceAllUsesWith(
        reusable->representative->allocation.getResult());
      deallocationsToErase.push_back(reusable->latestDeallocation);
      allocationsToErase.push_back(info.allocation);
      reusable->latestDeallocation = info.deallocation;
      reusable->latestDeallocationPosition = info.deallocationPosition;
      reusable->alignment = std::max(reusable->alignment, info.alignment);
      ++reusable->memberCount;
    }

    for (Slot& slot : slots) {
      if (slot.alignment > 0) {
        slot.representative->allocation->setAttr(
          "alignment",
          IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                           slot.alignment));
      }
      annotateSlot(slot, static_cast<int64_t>(&slot - slots.data()), function);
    }

    for (memref::DeallocOp deallocation : deallocationsToErase) {
      deallocation.erase();
    }
    for (memref::AllocOp allocation : allocationsToErase) {
      allocation.erase();
    }
  }

  void runOnOperation() final {
    getOperation().walk([](func::FuncOp function) { runOnFunction(function); });
  }
};

}  // namespace

}  // namespace mlir::ncnn
