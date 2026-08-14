// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s

func.func @complex_fp16(%input: tensor<4x2x2xf32>) -> tensor<4x2x2xf32> attributes {ncnn.fp16_accumulator = "f32", ncnn.precision = "fp16"} {
  %slope = arith.constant dense<1.0> : tensor<4xf32>
  %mean = arith.constant dense<0.0> : tensor<4xf32>
  %variance = arith.constant dense<1.0> : tensor<4xf32>
  %bias = arith.constant dense<0.0> : tensor<4xf32>
  %sigmoid = ncnn.sigmoid %input : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  %hard_sigmoid = ncnn.hard_sigmoid %sigmoid {alpha = 2.0e-1 : f32, beta = 5.0e-1 : f32} : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  %hard_swish = ncnn.hard_swish %hard_sigmoid {alpha = 2.0e-1 : f32, beta = 5.0e-1 : f32} : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  %gelu = ncnn.gelu %hard_swish {fast = false} : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  %softmax = ncnn.softmax %gelu {axis = 0 : i64} : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  %result = ncnn.batch_norm %softmax, %slope, %mean, %variance, %bias {epsilon = 1.0e-5 : f32} : (tensor<4x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4x2x2xf32>
  return %result : tensor<4x2x2xf32>
}

// CHECK-LABEL: func.func @complex_fp16
// CHECK: linalg.map { arith.truncf } {{.*}}tensor<1x2x2x4xf32>{{.*}}tensor<1x2x2x4xf16>
// CHECK: linalg.map { arith.extf } {{.*}}tensor<1x2x2x4xf16>{{.*}}tensor<1x2x2x4xf32>
// CHECK: tosa.sigmoid {{.*}} : (tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32>
// CHECK: linalg.map { math.erfc } {{.*}}tensor<1x2x2x4xf32>
// CHECK: tosa.reduce_sum {{.*}} : (tensor<1x2x2x4xf32>) -> tensor<1x2x2x1xf32>
// CHECK: tosa.pow {{.*}} : (tensor<1x1x1x4xf32>, tensor<1x1x1x4xf32>) -> tensor<1x1x1x4xf32>
// CHECK-NOT: tosa.exp {{.*}}f16
// CHECK-NOT: math.erfc {{.*}}f16

func.func @complex_bf16(%input: tensor<4x2x2xf32>) -> tensor<4x2x2xf32> attributes {ncnn.fp16_accumulator = "f32", ncnn.precision = "bf16"} {
  %result = ncnn.sigmoid %input : (tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
  return %result : tensor<4x2x2xf32>
}

// CHECK-LABEL: func.func @complex_bf16
// CHECK: linalg.map { arith.truncf } {{.*}}tensor<1x2x2x4xf32>{{.*}}tensor<1x2x2x4xbf16>
// CHECK: linalg.map { arith.extf } {{.*}}tensor<1x2x2x4xbf16>{{.*}}tensor<1x2x2x4xf32>
// CHECK: tosa.sigmoid {{.*}} : (tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32>

func.func @detection_fp16(%location: tensor<12xf32>, %confidence: tensor<9xf32>, %prior: tensor<2x12xf32>) -> (tensor<2x6xf32>, tensor<2xi64>) attributes {ncnn.fp16_accumulator = "f32", ncnn.precision = "fp16"} {
  %result, %shape = ncnn.detection_output %location, %confidence, %prior {confidence_threshold = 5.0e-1 : f32, keep_top_k = 2 : i64, nms_threshold = 5.0e-2 : f32, nms_top_k = 3 : i64, num_class = 3 : i64, variance_h = 2.0e-1 : f32, variance_w = 2.0e-1 : f32, variance_x = 1.0e-1 : f32, variance_y = 1.0e-1 : f32} : (tensor<12xf32>, tensor<9xf32>, tensor<2x12xf32>) -> (tensor<2x6xf32>, tensor<2xi64>)
  return %result, %shape : tensor<2x6xf32>, tensor<2xi64>
}

// CHECK-LABEL: func.func @detection_fp16
// CHECK: linalg.map { arith.truncf } {{.*}}tensor<12xf32>{{.*}}tensor<12xf16>
// CHECK: linalg.map { arith.extf } {{.*}}tensor<12xf16>{{.*}}tensor<12xf32>
// CHECK: math.exp {{.*}} : f32
// CHECK: arith.divf {{.*}} : f32
// CHECK: tensor.from_elements
// CHECK: return {{.*}} : tensor<2x6xf32>, tensor<2xi64>
