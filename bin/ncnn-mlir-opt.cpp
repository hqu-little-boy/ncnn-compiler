// ncnn-mlir-opt：基于 MlirOptMain 的 mlir-opt 克隆。
//
// 注册全部上游方言/扩展/pass，再叠加 ncnn 方言，因此支持与 mlir-opt 一致的
// 标准选项（-o、--mlir-print-op-generic、--verify-diagnostics、--allow-unregistered-dialect
// 等）以及所有 pass（--tosa-to-linalg、--convert-arith-to-llvm、--canonicalize
// …）。 主要用于 lit 测试与下降管线的调试（parse -> pass -> verify -> print）。

#include "RegisterNCNNDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "ncnn-mlir/Conversion/NCNNToFunc/NCNNToFunc.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"
#include "ncnn-mlir/Transforms/NormalizeNCNN/NormalizeNCNN.hpp"
#include "ncnn-mlir/Transforms/VerifyNoNCNNOps/VerifyNoNCNNOps.hpp"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  mlir::ncnn::registerNCNNToFuncPasses();
  mlir::ncnn::registerNCNNToTosaPasses();
  mlir::ncnn::registerNormalizeNCNNPasses();
  mlir::ncnn::registerVerifyNoNCNNOpsPasses();
  // 叠加 ncnn 方言（arith/func 等已含于 registerAllDialects，重复插入无害）。
  ncnn_mlir::register_all_dialects(registry);
  return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "ncnn-mlir-opt\n", registry));
}
