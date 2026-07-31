#include "ncnn-mlir/Transforms/BufferizeNCNN/BufferizeNCNN.hpp"

#include <memory>

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {
namespace {

class BufferizeNCNNPass final
  : public PassWrapper<BufferizeNCNNPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferizeNCNNPass)

  StringRef getArgument() const final { return "bufferize-ncnn"; }
  StringRef getDescription() const final {
    return "One-shot bufferize with copies represented as Linalg operations";
  }

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<bufferization::BufferizationDialect,
                    func::FuncDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = true;
    options.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
    options.memCpyFn =
      [](OpBuilder& builder, Location location, Value source, Value target) {
        linalg::makeMemRefCopyOp(builder, location, source, target);
        return success();
      };

    bufferization::BufferizationState state;
    if (failed(bufferization::runOneShotModuleBufferize(
          getOperation(), options, state))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createBufferizeNCNNPass() {
  return std::make_unique<BufferizeNCNNPass>();
}

void registerBufferizeNCNNPasses() {
  static PassRegistration<BufferizeNCNNPass> registration;
}

}  // namespace mlir::ncnn
