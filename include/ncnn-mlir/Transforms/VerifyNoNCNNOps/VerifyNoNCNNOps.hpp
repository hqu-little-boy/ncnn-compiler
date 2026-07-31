#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace ncnn {

std::unique_ptr<Pass> createVerifyNoNCNNOpsPass();
void registerVerifyNoNCNNOpsPasses();

}  // namespace ncnn
}  // namespace mlir
