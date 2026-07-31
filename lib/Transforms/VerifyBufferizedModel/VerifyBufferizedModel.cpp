#include "ncnn-mlir/Transforms/VerifyBufferizedModel/VerifyBufferizedModel.hpp"

#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {
namespace {

class VerifyBufferizedModelPass final
  : public PassWrapper<VerifyBufferizedModelPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyBufferizedModelPass)

  StringRef getArgument() const final { return "verify-bufferized-model"; }
  StringRef getDescription() const final {
    return "Verify the bufferized ncnn entry point and its buffer ownership";
  }

  static Value getAllocationRoot(Value value) {
    auto memrefValue = dyn_cast<TypedValue<BaseMemRefType>>(value);
    if (!memrefValue) {
      return {};
    }
    return memref::skipViewLikeOps(memrefValue);
  }

  static bool isFunctionArgument(Value value, func::FuncOp function) {
    auto argument = dyn_cast<BlockArgument>(value);
    return argument && argument.getOwner() == &function.getBody().front();
  }

  static void verifyAllocationLifetimes(func::FuncOp function,
                                        bool& failedVerification) {
    DenseMap<Value, SmallVector<memref::DeallocOp>> deallocations;
    function.walk([&](memref::DeallocOp dealloc) {
      Value root = getAllocationRoot(dealloc.getMemref());
      if (isFunctionArgument(root, function)) {
        dealloc.emitOpError(
          "must not release a caller-owned function argument or its alias");
        failedVerification = true;
        return;
      }
      if (!root.getDefiningOp<memref::AllocOp>()) {
        dealloc.emitOpError(
          "does not resolve to a unique heap allocation root");
        failedVerification = true;
        return;
      }
      deallocations[root].push_back(dealloc);
    });

    PostDominanceInfo postDominance(function);
    function.walk([&](memref::AllocOp allocation) {
      Value root = allocation.getResult();
      auto iterator = deallocations.find(root);
      const size_t count =
        iterator == deallocations.end() ? 0 : iterator->second.size();
      if (count == 0) {
        allocation.emitOpError("has no matching deallocation");
        failedVerification = true;
        return;
      }
      if (count != 1) {
        allocation.emitOpError() << "has " << count
                                 << " matching deallocations; expected exactly "
                                    "one to avoid double-free";
        failedVerification = true;
        return;
      }

      Operation* dealloc = iterator->second.front().getOperation();
      if (!postDominance.properlyPostDominates(dealloc,
                                               allocation.getOperation())) {
        allocation.emitOpError(
          "deallocation does not execute on every path after allocation");
        failedVerification = true;
      }
      function.walk([&](Operation* user) {
        if (user == dealloc) {
          return;
        }
        for (Value operand : user->getOperands()) {
          if (getAllocationRoot(operand) != root) {
            continue;
          }
          if (!postDominance.postDominates(dealloc, user)) {
            user->emitOpError(
              "may execute after or without the allocation deallocation");
            failedVerification = true;
          }
          break;
        }
      });
    });
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

std::unique_ptr<Pass> createVerifyBufferizedModelPass() {
  return std::make_unique<VerifyBufferizedModelPass>();
}

void registerVerifyBufferizedModelPasses() {
  static PassRegistration<VerifyBufferizedModelPass> registration;
}

}  // namespace mlir::ncnn
