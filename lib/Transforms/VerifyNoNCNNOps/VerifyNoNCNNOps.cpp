#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYNONCNNOPSPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class VerifyNoNCNNOpsPass final
  : public impl::VerifyNoNCNNOpsPassBase<VerifyNoNCNNOpsPass> {
 public:
  using Base::Base;

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

}  // namespace mlir::ncnn
