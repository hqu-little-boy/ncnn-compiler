#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::ncnn {

#define GEN_PASS_DECL_PACKSTATICMATMULNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

}  // namespace mlir::ncnn
