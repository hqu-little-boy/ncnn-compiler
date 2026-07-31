// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s --check-prefix=PARTIAL
// RUN: not ncnn-mlir-opt --convert-ncnn-to-tosa --verify-no-ncnn-ops %s 2>&1 | FileCheck %s --check-prefix=STRICT

func.func @partial(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %before = ncnn.relu %arg0 {ncnn.name = "before", ncnn.source_layer = 1 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %pool = ncnn.pooling %before {include_pad = false, kernel_h = 3 : i64, kernel_w = 4 : i64, kind = 0 : i64, mode = 2 : i64, ncnn.name = "adaptive", ncnn.source_layer = 2 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  %after = ncnn.relu %pool {ncnn.name = "after", ncnn.source_layer = 3 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %after : tensor<2x3x4xf32>
}

// PARTIAL-LABEL: func.func @partial
// PARTIAL: %[[INPUT:.*]] = tosa.reshape
// PARTIAL: %[[BEFORE:.*]] = tosa.clamp %[[INPUT]]
// PARTIAL: %[[TO_CHW:.*]] = tosa.transpose
// PARTIAL: %[[POOL:.*]] = ncnn.pooling %[[TO_CHW]]
// PARTIAL-SAME: ncnn.name = "adaptive"
// PARTIAL-SAME: ncnn.source_layer = 2 : i64
// PARTIAL: %[[FROM_CHW:.*]] = tosa.transpose %[[POOL]]
// PARTIAL: %[[AFTER:.*]] = tosa.clamp
// PARTIAL-NOT: ncnn.relu

// STRICT: error: 'ncnn.pooling' op remains after lowering; op=ncnn.pooling, ncnn.name="adaptive", ncnn.source_layer=2
