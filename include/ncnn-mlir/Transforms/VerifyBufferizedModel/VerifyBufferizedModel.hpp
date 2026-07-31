#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace ncnn {

std::unique_ptr<Pass> createVerifyBufferizedModelPass();
void registerVerifyBufferizedModelPasses();

}  // namespace ncnn
}  // namespace mlir
