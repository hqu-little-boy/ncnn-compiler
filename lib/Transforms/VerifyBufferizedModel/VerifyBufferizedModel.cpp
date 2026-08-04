#include "ncnn-mlir/Transforms/VerifyBufferizedModel/VerifyBufferizedModel.hpp"

#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYBUFFERIZEDMODELPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class VerifyBufferizedModelPass final
  : public impl::VerifyBufferizedModelPassBase<VerifyBufferizedModelPass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<memref::MemRefDialect>();
    memref::registerAllocationOpInterfaceExternalModels(registry);
  }

  struct Allocation {
    Value value;
    Operation* operation;
  };

  struct Release {
    Value value;
    Operation* operation;
  };

  static bool mayAlias(AliasResult result) { return !result.isNo(); }

  static bool blockHasReleaseAfter(Block* block,
                                   Operation* allocation,
                                   ArrayRef<Operation*> releases,
                                   DominanceInfo& dominance) {
    return llvm::any_of(releases, [&](Operation* release) {
      return release->getBlock() == block &&
             dominance.dominates(allocation, release);
    });
  }

  static bool releasesOnEveryPath(Operation* allocation,
                                  ArrayRef<Operation*> releases,
                                  DominanceInfo& dominance) {
    SmallVector<Block*> worklist{allocation->getBlock()};
    DenseSet<Block*> visited;
    while (!worklist.empty()) {
      Block* block = worklist.pop_back_val();
      if (!visited.insert(block).second) {
        continue;
      }
      if (blockHasReleaseAfter(block, allocation, releases, dominance)) {
        continue;
      }
      if (block->getSuccessors().empty()) {
        return false;
      }
      worklist.append(block->getSuccessors().begin(),
                      block->getSuccessors().end());
    }
    return true;
  }

  static bool isReachableAfter(Operation* first, Operation* second) {
    if (first->getBlock() == second->getBlock()) {
      return first->isBeforeInBlock(second);
    }

    SmallVector<Block*> worklist;
    DenseSet<Block*> visited;
    worklist.append(first->getBlock()->getSuccessors().begin(),
                    first->getBlock()->getSuccessors().end());
    while (!worklist.empty()) {
      Block* block = worklist.pop_back_val();
      if (!visited.insert(block).second) {
        continue;
      }
      if (block == second->getBlock()) {
        return true;
      }
      worklist.append(block->getSuccessors().begin(),
                      block->getSuccessors().end());
    }
    return false;
  }

  static void verifyAllocationLifetimes(func::FuncOp function,
                                        bool& failedVerification) {
    SmallVector<Allocation> allocations;
    SmallVector<Release> releases;
    function.walk([&](Operation* operation) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
      if (!effects) {
        return;
      }
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      if (isa<bufferization::AllocationOpInterface>(operation)) {
        for (const MemoryEffects::EffectInstance& effect : instances) {
          Value value = effect.getValue();
          if (!isa<MemoryEffects::Allocate>(effect.getEffect()) || !value ||
              !isa<BaseMemRefType>(value.getType()) ||
              isa<SideEffects::AutomaticAllocationScopeResource>(
                effect.getResource())) {
            continue;
          }
          allocations.push_back({value, operation});
        }
      }
      for (const MemoryEffects::EffectInstance& effect : instances) {
        Value value = effect.getValue();
        if (isa<MemoryEffects::Free>(effect.getEffect()) && value &&
            isa<BaseMemRefType>(value.getType())) {
          releases.push_back({value, operation});
        }
      }
    });

    AliasAnalysis aliases(function);
    BufferOriginAnalysis origins(function);
    DominanceInfo dominance(function);
    SmallVector<SmallVector<Operation*>> matchingReleases(allocations.size());
    for (const Release& release : releases) {
      bool callerOwned = false;
      for (BlockArgument argument : function.getArguments()) {
        if (isa<BaseMemRefType>(argument.getType()) &&
            mayAlias(aliases.alias(release.value, argument))) {
          callerOwned = true;
          break;
        }
      }
      if (callerOwned) {
        release.operation->emitOpError(
          "must not release a caller-owned function argument or its alias");
        failedVerification = true;
        continue;
      }

      SmallVector<unsigned> roots;
      for (auto [index, allocation] : llvm::enumerate(allocations)) {
        if (origins.isSameAllocation(release.value, allocation.value) == true) {
          roots.push_back(index);
        }
      }
      if (roots.size() != 1) {
        release.operation->emitOpError(
          "does not resolve to a unique heap allocation root");
        failedVerification = true;
        continue;
      }
      matchingReleases[roots.front()].push_back(release.operation);
    }

    for (auto [index, allocation] : llvm::enumerate(allocations)) {
      ArrayRef<Operation*> allocationReleases = matchingReleases[index];
      const size_t count = allocationReleases.size();
      if (count == 0) {
        allocation.operation->emitOpError("has no matching deallocation");
        failedVerification = true;
        continue;
      }
      if (!releasesOnEveryPath(
            allocation.operation, allocationReleases, dominance)) {
        allocation.operation->emitOpError(
          "does not have a deallocation on every path after allocation");
        failedVerification = true;
        continue;
      }

      for (auto [index, first] : llvm::enumerate(allocationReleases)) {
        for (Operation* second :
             ArrayRef<Operation*>(allocationReleases).drop_front(index + 1)) {
          if (isReachableAfter(first, second)) {
            second->emitOpError(
              "has a matching deallocation that may execute after it; "
              "expected exactly one to avoid double-free");
            failedVerification = true;
          }
        }
      }

      SmallVector<Value> allocationAliases;
      function.walk([&](Block* block) {
        for (BlockArgument argument : block->getArguments()) {
          if (isa<BaseMemRefType>(argument.getType()) &&
              origins.isSameAllocation(allocation.value, argument) == true) {
            allocationAliases.push_back(argument);
          }
        }
      });
      function.walk([&](Operation* operation) {
        for (Value result : operation->getResults()) {
          if (isa<BaseMemRefType>(result.getType()) &&
              origins.isSameAllocation(allocation.value, result) == true) {
            allocationAliases.push_back(result);
          }
        }
      });
      for (Value alias : allocationAliases) {
        for (Operation* user : alias.getUsers()) {
          for (Operation* dealloc : allocationReleases) {
            if (user != dealloc && isReachableAfter(dealloc, user)) {
              user->emitOpError(
                "may execute after or without the allocation deallocation");
              failedVerification = true;
              break;
            }
          }
        }
      }
    }
  }

  void runOnOperation() final {
    bool failedVerification = false;
    getOperation().walk([&](Operation* operation) {
      StringRef dialect = operation->getName().getDialectNamespace();
      if (dialect == "tensor" || dialect == "bufferization" ||
          dialect == "ncnn" || dialect == "tosa") {
        operation->emitOpError()
          << "belongs to forbidden residual dialect '" << dialect << "'";
        failedVerification = true;
      }
      if (isa<UnrealizedConversionCastOp>(operation)) {
        operation->emitOpError(
          "must not remain after bufferization conversion");
        failedVerification = true;
      }
      for (Type type : operation->getOperandTypes()) {
        if (isa<TensorType>(type)) {
          operation->emitOpError(
            "has a tensor-typed operand after bufferization");
          failedVerification = true;
        }
      }
      for (Type type : operation->getResultTypes()) {
        if (isa<TensorType>(type)) {
          operation->emitOpError(
            "has a tensor-typed result after bufferization");
          failedVerification = true;
        }
      }
    });

    getOperation().walk([&](func::FuncOp function) {
      for (Type type : function.getArgumentTypes()) {
        if (isa<TensorType>(type)) {
          function.emitOpError(
            "has a tensor-typed argument after bufferization");
          failedVerification = true;
        }
      }
      for (Type type : function.getResultTypes()) {
        if (isa<TensorType>(type)) {
          function.emitOpError(
            "has a tensor-typed function result after "
            "bufferization");
          failedVerification = true;
        }
      }
      verifyAllocationLifetimes(function, failedVerification);
      if (!function->hasAttr("ncnn.entry_point")) {
        return;
      }
      if (function.getNumResults() != 0) {
        function.emitOpError("must not return values after bufferization");
        failedVerification = true;
      }

      bool foundOutput = false;
      for (unsigned index = 0; index < function.getNumArguments(); ++index) {
        if (function.getArgAttr(index, "bufferize.result")) {
          foundOutput = true;
          if (!isa<MemRefType>(function.getArgumentTypes()[index])) {
            function.emitOpError()
              << "output parameter " << index << " must have memref type";
            failedVerification = true;
          }
        }
      }
      if (!foundOutput) {
        function.emitOpError("has no bufferize.result output parameter");
        failedVerification = true;
      }

      function.walk([&](func::ReturnOp returnOp) {
        if (returnOp.getNumOperands() != 0) {
          returnOp.emitOpError("must not return values after bufferization");
          failedVerification = true;
        }
      });
    });

    if (failedVerification) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
