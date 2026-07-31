#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace ncnn {

std::unique_ptr<Pass> createBufferizeNCNNPass();
void registerBufferizeNCNNPasses();

}  // namespace ncnn
}  // namespace mlir
