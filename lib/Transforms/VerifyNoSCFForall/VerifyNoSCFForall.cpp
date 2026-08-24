#include "ncnn-mlir/Transforms/VerifyNoSCFForall/VerifyNoSCFForall.hpp"

#include <memory>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_VERIFYNOSCFFORALLPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// 并行发射完整性闸门：scf.forall / scf.parallel 必须已被
// forall-to-parallel → openmp（多线程）或 forall-to-for（串行）消费。
// convert-scf-to-openmp 对 scf.forall 是静默跳过（不报错、不转换），
// 残留即意味着并行化缺失——要么静默退化成串行，要么到 LLVM 下降才失败。
class VerifyNoSCFForallPass final
  : public impl::VerifyNoSCFForallPassBase<VerifyNoSCFForallPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    bool foundResidual = false;
    getOperation().walk([&](Operation* operation) {
      if (!isa<scf::ForallOp, scf::ParallelOp>(operation)) {
        return;
      }
      foundResidual = true;
      operation->emitOpError() << "remains after parallel lowering; op="
                               << operation->getName().getStringRef();
    });
    if (foundResidual) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
