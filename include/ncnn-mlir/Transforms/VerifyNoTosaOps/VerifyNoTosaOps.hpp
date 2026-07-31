#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace ncnn {

std::unique_ptr<Pass> createVerifyNoTosaOpsPass();
void registerVerifyNoTosaOpsPasses();

}  // namespace ncnn
}  // namespace mlir
