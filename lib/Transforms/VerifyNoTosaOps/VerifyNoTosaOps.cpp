#include "ncnn-mlir/Transforms/VerifyNoTosaOps/VerifyNoTosaOps.hpp"

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {
namespace {

class VerifyNoTosaOpsPass final
  : public PassWrapper<VerifyNoTosaOpsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyNoTosaOpsPass)

  StringRef getArgument() const final { return "verify-no-tosa-ops"; }
  StringRef getDescription() const final {
    return "Fail if any TOSA dialect operation remains";
  }

  void runOnOperation() final {
    bool foundResidual = false;
    getOperation().walk([&](Operation* operation) {
      if (operation->getName().getDialectNamespace() != "tosa") {
        return;
      }
      foundResidual = true;
      operation->emitOpError() << "remains after lowering to Linalg; op="
                               << operation->getName().getStringRef();
    });
    if (foundResidual) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createVerifyNoTosaOpsPass() {
  return std::make_unique<VerifyNoTosaOpsPass>();
}

void registerVerifyNoTosaOpsPasses() {
  static PassRegistration<VerifyNoTosaOpsPass> registration;
}

}  // namespace mlir::ncnn
