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

#define GEN_PASS_DEF_BUFFERIZENCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

class BufferizeNCNNPass final
  : public impl::BufferizeNCNNPassBase<BufferizeNCNNPass> {
 public:
  using Base::Base;

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

}  // namespace mlir::ncnn
