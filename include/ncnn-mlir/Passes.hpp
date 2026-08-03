#pragma once

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace mlir::ncnn {

#define GEN_PASS_DECL
#include "ncnn-mlir/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ncnn-mlir/Passes.h.inc"

}  // namespace mlir::ncnn
