// Detection-output 降低。
//
// 职责
//   把 ncnn.detection_output（NMS + decode）降低为 tosa 的逐元素/归约
//   组合；单类 ConvertDetectionOutput 见 DetectionOutputLowering.inc。
//
// 不变量
//   * decode 的 anchor 解码公式与 ncnn 一致；
//   * NMS 的 IoU 阈值与 top-k 来自操作属性，不得写死。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后（bbox 布局已定）。
//
// 明确不做
//   * 不做检测后处理的算子融合；
//   * 不改 decode 公式。

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosaPatterns.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/TosaLoweringUtils.hpp"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"
#include "ncnn-mlir/Support/ConstantFold.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/ModelLedger.hpp"
#include "ncnn-mlir/Support/Precision.hpp"

namespace mlir::ncnn {
namespace {

// Shared lowering helpers used by this family (clang-tidy forbids
// `using namespace`, so imports are per-name).
using tosa_lowering::applyLowPrecisionBoundary;
using tosa_lowering::reshapeValue;

#include "DetectionOutputLowering.inc"

}  // namespace

void populateDetectionToTosaPatterns(RewritePatternSet& patterns,
                                     const TypeConverter& typeConverter,
                                     MLIRContext* context) {
  patterns.add<ConvertDetectionOutput>(typeConverter, context);
}

}  // namespace mlir::ncnn
