// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa --ncnn-tosa-to-linalg-pipeline --ncnn-linalg-to-memref-pipeline --ncnn-memref-to-llvm-pipeline %s | FileCheck %s --check-prefix=LLVM

func.func @detection_output(%location: tensor<12xf32>, %confidence: tensor<9xf32>, %prior: tensor<2x12xf32>) -> (tensor<2x6xf32>, tensor<2xi64>) {
  %result, %shape = ncnn.detection_output %location, %confidence, %prior {confidence_threshold = 5.000000e-01 : f32, keep_top_k = 2 : i64, nms_threshold = 5.000000e-02 : f32, nms_top_k = 3 : i64, num_class = 3 : i64, variance_h = 2.000000e-01 : f32, variance_w = 2.000000e-01 : f32, variance_x = 1.000000e-01 : f32, variance_y = 1.000000e-01 : f32} : (tensor<12xf32>, tensor<9xf32>, tensor<2x12xf32>) -> (tensor<2x6xf32>, tensor<2xi64>)
  return %result, %shape : tensor<2x6xf32>, tensor<2xi64>
}

// CHECK-LABEL: func.func @detection_output
// CHECK-NOT: ncnn.detection_output
// CHECK: scf.for
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: linalg.generic
// CHECK: tensor.from_elements
// CHECK: return {{.*}} : tensor<2x6xf32>, tensor<2xi64>

// LLVM-LABEL: llvm.func @detection_output
// LLVM-NOT: ncnn.detection_output
