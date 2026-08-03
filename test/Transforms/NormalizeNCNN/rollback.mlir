// RUN: not ncnn-mlir-opt --mlir-print-ir-after-failure --normalize-ncnn %s 2>&1 | FileCheck %s

func.func @rollback(%arg0: tensor<1x2x1xf32>) -> tensor<1x2x1xf32> {
  %softmax = ncnn.softmax %arg0 {axis = -1 : i64} : (tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  %pool = ncnn.pooling %softmax {include_pad = false, kernel_h = 9223372036854775807 : i64, kernel_w = 1 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 2 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  return %pool : tensor<1x2x1xf32>
}

// CHECK: error: 'ncnn.pooling' op cannot resolve static SAME padding
// CHECK: IR Dump After {{.*}}NormalizeNCNNPass Failed
// CHECK: ncnn.softmax
// CHECK-SAME: axis = -1 : i64
// CHECK: ncnn.pooling
// CHECK-SAME: pad_mode = 2 : i64
