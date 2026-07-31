#include "ncnn-mlir/Transforms/VerifyBufferizedModel/VerifyBufferizedModel.hpp"

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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

  void runOnOperation() final {
    bool failedVerification = false;
    getOperation().walk([&](Operation* operation) {
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
      function.walk([&](memref::DeallocOp dealloc) {
        auto argument = dyn_cast<BlockArgument>(dealloc.getMemref());
        if (argument && argument.getOwner() == &function.getBody().front()) {
          dealloc.emitOpError(
            "must not release a caller-owned function argument");
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
