#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {
namespace {

class VerifyNoNCNNOpsPass final
  : public PassWrapper<VerifyNoNCNNOpsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyNoNCNNOpsPass)

  StringRef getArgument() const final { return "verify-no-ncnn-ops"; }
  StringRef getDescription() const final {
    return "Fail if any ncnn dialect operation remains";
  }

  void runOnOperation() final {
    bool foundResidual = false;
    getOperation().walk([&](Operation* operation) {
      if (operation->getName().getDialectNamespace() != "ncnn") {
        return;
      }
      foundResidual = true;
      auto name = operation->getAttrOfType<StringAttr>("ncnn.name");
      auto source = operation->getAttrOfType<IntegerAttr>("ncnn.source_layer");
      InFlightDiagnostic diagnostic =
        operation->emitOpError()
        << "remains after lowering; op=" << operation->getName().getStringRef()
        << ", ncnn.name=";
      if (name) {
        diagnostic << '"' << name.getValue() << '"';
      } else {
        diagnostic << "<missing>";
      }
      diagnostic << ", ncnn.source_layer=";
      if (source) {
        diagnostic << source.getInt();
      } else {
        diagnostic << "<missing>";
      }
    });
    if (foundResidual) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createVerifyNoNCNNOpsPass() {
  return std::make_unique<VerifyNoNCNNOpsPass>();
}

void registerVerifyNoNCNNOpsPasses() {
  static PassRegistration<VerifyNoNCNNOpsPass> registration;
}

}  // namespace mlir::ncnn
