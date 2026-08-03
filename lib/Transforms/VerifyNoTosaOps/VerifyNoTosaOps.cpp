#include "ncnn-mlir/Transforms/VerifyNoTosaOps/VerifyNoTosaOps.hpp"

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYNOTOSAOPSPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class VerifyNoTosaOpsPass final
  : public impl::VerifyNoTosaOpsPassBase<VerifyNoTosaOpsPass> {
 public:
  using Base::Base;

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

}  // namespace mlir::ncnn
