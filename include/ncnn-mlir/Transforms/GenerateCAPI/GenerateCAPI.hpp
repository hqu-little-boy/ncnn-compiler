#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace ncnn {

std::unique_ptr<Pass> createGenerateCAPIPass();
std::unique_ptr<Pass> createFinalizeCAPIPass();
void registerGenerateCAPIPasses();

}  // namespace ncnn
}  // namespace mlir
