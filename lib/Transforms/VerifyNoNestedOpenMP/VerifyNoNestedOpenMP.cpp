#include "ncnn-mlir/Transforms/VerifyNoNestedOpenMP/VerifyNoNestedOpenMP.hpp"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYNONESTEDOPENMPPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class VerifyNoNestedOpenMPPass final
  : public impl::VerifyNoNestedOpenMPPassBase<VerifyNoNestedOpenMPPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    bool foundNested = false;
    getOperation().walk([&](omp::ParallelOp parallel) {
      if (parallel->getParentOfType<omp::ParallelOp>() == nullptr) {
        return;
      }
      parallel.emitOpError()
        << "nested omp.parallel is not part of the P12 single-team contract";
      foundNested = true;
    });
    if (foundNested) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
