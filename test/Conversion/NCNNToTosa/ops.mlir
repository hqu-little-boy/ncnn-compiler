// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | FileCheck %s
// RUN: ncnn-mlir-opt --convert-ncnn-to-tosa %s | mlir-opt-21 --tosa-validate

func.func @spatial(%arg0: tensor<2x5x5xf32>) -> tensor<6x2x2xf32> {
  %weight = arith.constant dense<0.000000e+00> : tensor<3x2x3x3xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<3xf32>
  %conv = ncnn.convolution %arg0, %weight, %bias {dilation_h = 1 : i64, dilation_w = 1 : i64, has_bias = true, kernel_h = 3 : i64, kernel_w = 3 : i64, pad_bottom = 1 : i64, pad_left = 1 : i64, pad_right = 1 : i64, pad_top = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<2x5x5xf32>, tensor<3x2x3x3xf32>, tensor<3xf32>) -> tensor<3x3x3xf32>
  %relu = ncnn.relu %conv : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
  %pool = ncnn.pooling %relu {include_pad = false, kernel_h = 2 : i64, kernel_w = 2 : i64, kind = 0 : i64, mode = 0 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 2 : i64, stride_w = 2 : i64} : (tensor<3x3x3xf32>) -> tensor<3x2x2xf32>
  %split0, %split1 = ncnn.split %pool : (tensor<3x2x2xf32>) -> (tensor<3x2x2xf32>, tensor<3x2x2xf32>)
  %concat = ncnn.concat %split0, %split1 {axis = 0 : i64} : (tensor<3x2x2xf32>, tensor<3x2x2xf32>) -> tensor<6x2x2xf32>
  %dropout = ncnn.dropout %concat : (tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
  return %dropout : tensor<6x2x2xf32>
}

// CHECK-LABEL: func.func @spatial
// CHECK: %[[INPUT_TRANSPOSE:.*]] = tosa.transpose %arg0 {perms = array<i32: 1, 2, 0>} : (tensor<2x5x5xf32>) -> tensor<5x5x2xf32>
// CHECK: %[[INPUT:.*]] = tosa.reshape %[[INPUT_TRANSPOSE]]
// CHECK-SAME: -> tensor<1x5x5x2xf32>
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<0.000000e+00> : tensor<3x3x3x2xf32>
// CHECK: %[[CONV:.*]] = tosa.conv2d %[[INPUT]], %[[WEIGHT]],
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
// CHECK-SAME: stride = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x3x3x3xf32>
// CHECK: %[[RELU:.*]] = tosa.clamp %[[CONV]]
// CHECK: %[[POOL:.*]] = tosa.max_pool2d %[[RELU]]
// CHECK-SAME: pad = array<i64: 0, 1, 0, 1>
// CHECK: %[[CONCAT:.*]] = tosa.concat %[[POOL]], %[[POOL]] {axis = 3 : i32}
// CHECK-NOT: ncnn.split
// CHECK-NOT: ncnn.dropout
// CHECK: %[[HWC:.*]] = tosa.reshape %[[CONCAT]]
// CHECK-SAME: -> tensor<2x2x6xf32>
// CHECK: %[[OUTPUT:.*]] = tosa.transpose %[[HWC]] {perms = array<i32: 2, 0, 1>}
// CHECK: return %[[OUTPUT]] : tensor<6x2x2xf32>

func.func @global_softmax(%arg0: tensor<4x2x2xf32>) -> tensor<4xf32> {
  %pool = ncnn.pooling %arg0 {include_pad = false, kernel_h = 0 : i64, kernel_w = 0 : i64, kind = 1 : i64, mode = 1 : i64, pad_bottom = 0 : i64, pad_left = 0 : i64, pad_mode = 0 : i64, pad_right = 0 : i64, pad_top = 0 : i64, stride_h = 1 : i64, stride_w = 1 : i64} : (tensor<4x2x2xf32>) -> tensor<4xf32>
  %softmax = ncnn.softmax %pool {axis = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
  %scaled = ncnn.dropout %softmax {scale = 5.000000e-01 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %scaled : tensor<4xf32>
}

// CHECK-LABEL: func.func @global_softmax
// CHECK: %[[AVG:.*]] = tosa.avg_pool2d
// CHECK-SAME: kernel = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x1x1x4xf32>
// CHECK: %[[FLAT:.*]] = tosa.reshape %[[AVG]]
// CHECK-SAME: -> tensor<4xf32>
// CHECK: %[[MAX:.*]] = tosa.reduce_max %[[FLAT]] {axis = 0 : i32}
// CHECK: %[[SUB:.*]] = tosa.sub %[[FLAT]], %[[MAX]]
// CHECK: %[[EXP:.*]] = tosa.exp %[[SUB]]
// CHECK: %[[SUM:.*]] = tosa.reduce_sum %[[EXP]] {axis = 0 : i32}
// CHECK: %[[RECIP:.*]] = tosa.reciprocal %[[SUM]]
// CHECK: %[[SOFTMAX:.*]] = tosa.mul %[[EXP]], %[[RECIP]],
// CHECK: %[[SCALE:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1xf32>}>
// CHECK: %[[SCALED:.*]] = tosa.mul %[[SOFTMAX]], %[[SCALE]],
// CHECK: return %[[SCALED]] : tensor<4xf32>

func.func @anglenet_ops(%arg0: tensor<4x3x4xf32>) -> tensor<4xf32> {
  %shuffle = ncnn.shuffle_channel %arg0 {group = 2 : i64, reverse = false} : (tensor<4x3x4xf32>) -> tensor<4x3x4xf32>
  %left, %right = ncnn.slice %shuffle {axis = 0 : i64, slices = array<i64: 2, 2>} : (tensor<4x3x4xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>)
  %joined = ncnn.concat %left, %right {axis = 0 : i64} : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<4x3x4xf32>
  %mean = ncnn.reduction %joined {axes = array<i64: 1, 2>, coeff = 1.000000e+00 : f32, keepdims = false, kind = 3 : i64, reduce_all = false} : (tensor<4x3x4xf32>) -> tensor<4xf32>
  return %mean : tensor<4xf32>
}

// CHECK-LABEL: func.func @anglenet_ops
// CHECK: tosa.reshape {{.*}} -> tensor<1x3x4x2x2xf32>
// CHECK: tosa.transpose {{.*}}perms = array<i32: 0, 1, 2, 4, 3>
// CHECK: tosa.slice {{.*}} -> tensor<1x3x4x2xf32>
// CHECK: tosa.slice {{.*}} -> tensor<1x3x4x2xf32>
// CHECK: tosa.reduce_sum {{.*}}axis = 1 : i32
// CHECK: tosa.reduce_sum {{.*}}axis = 2 : i32
// CHECK: tosa.mul
// CHECK: tosa.reshape {{.*}} -> tensor<4xf32>
// CHECK: return

func.func @leaky_relu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %relu = ncnn.relu %arg0 {negative_slope = 2.500000e-01 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %relu : tensor<4xf32>
}

// CHECK-LABEL: func.func @leaky_relu
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[SLOPE:.*]] = "tosa.const"() <{values = dense<2.500000e-01> : tensor<1xf32>}>
// CHECK: %[[NEGATIVE:.*]] = tosa.mul %arg0, %[[SLOPE]],
// CHECK: %[[CONDITION:.*]] = tosa.greater_equal %arg0, %[[ZERO]]
// CHECK: %[[LEAKY:.*]] = tosa.select %[[CONDITION]], %arg0, %[[NEGATIVE]]
// CHECK: return %[[LEAKY]] : tensor<4xf32>

func.func @ppocr_rec_ops(%arg0: tensor<2x1x3xf32>) -> tensor<3x4xf32> {
  %slope = arith.constant dense<1.000000e+00> : tensor<2xf32>
  %mean = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %variance = arith.constant dense<1.000000e+00> : tensor<2xf32>
  %bias = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %weight = arith.constant dense<0.000000e+00> : tensor<4x2xf32>
  %gemm_bias = arith.constant dense<0.000000e+00> : tensor<4xf32>
  %gelu = ncnn.gelu %arg0 {fast = false} : (tensor<2x1x3xf32>) -> tensor<2x1x3xf32>
  %squeezed = ncnn.squeeze %gelu {axes = array<i64: 1>} : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  %normalized = ncnn.batch_norm %squeezed, %slope, %mean, %variance, %bias {epsilon = 1.000000e-05 : f32} : (tensor<2x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x3xf32>
  %expanded = ncnn.expand_dims %normalized {axes = array<i64: 1>} : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  %again = ncnn.squeeze %expanded {axes = array<i64: 1>} : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  %transposed = ncnn.permute %again {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %output = ncnn.gemm %transposed, %weight, %gemm_bias {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32} : (tensor<3x2xf32>, tensor<4x2xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %output : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @ppocr_rec_ops
// CHECK: linalg.map
// CHECK: math.erfc
// CHECK: %[[SQUEEZED:.*]] = tosa.reshape {{.*}} -> tensor<2x3xf32>
// CHECK: %[[INVERSE_STD:.*]] = tosa.pow
// CHECK: %[[SAFE_INVERSE_STD:.*]] = tosa.select {{.*}}, {{.*}}, %[[INVERSE_STD]]
// CHECK: %[[NORMALIZED_SCALE:.*]] = tosa.mul {{.*}}, %[[SAFE_INVERSE_STD]],
// CHECK: %[[NORMALIZED:.*]] = tosa.add {{.*}} -> tensor<2x3xf32>
// CHECK: %[[EXPANDED:.*]] = tosa.reshape %[[NORMALIZED]], {{.*}} -> tensor<2x1x3xf32>
// CHECK: %[[SQUEEZED_AGAIN:.*]] = tosa.reshape {{.*}} -> tensor<2x3xf32>
// CHECK: tosa.transpose %[[SQUEEZED_AGAIN]] {perms = array<i32: 1, 0>}
// CHECK: tosa.matmul
// CHECK: tosa.add
// CHECK: return

// CHECK-NOT: ncnn.convolution
// CHECK-NOT: ncnn.relu
// CHECK-NOT: ncnn.pooling
// CHECK-NOT: ncnn.split
// CHECK-NOT: ncnn.concat
// CHECK-NOT: ncnn.dropout
// CHECK-NOT: ncnn.softmax
// CHECK-NOT: ncnn.shuffle_channel
// CHECK-NOT: ncnn.slice
// CHECK-NOT: ncnn.reduction
// CHECK-NOT: ncnn.gelu
// CHECK-NOT: ncnn.squeeze
// CHECK-NOT: ncnn.batch_norm
// CHECK-NOT: ncnn.expand_dims
// CHECK-NOT: ncnn.permute
// CHECK-NOT: ncnn.gemm
