#pragma once

// 注册编译器用到的全部方言。opt 工具与驱动共用此入口，保证方言集合一致。
// 遵循 MLIR 惯例：注册头文件以内联函数提供（同 RegisterAllDialects.h 模式）。

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.hpp"

namespace ncnn_mlir {

inline void register_all_dialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::ncnn::NCNNDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();
}

}  // namespace ncnn_mlir
