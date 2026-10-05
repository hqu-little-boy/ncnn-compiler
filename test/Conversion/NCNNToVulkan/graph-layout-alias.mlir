// RUN: ncnn-mlir-opt --split-input-file --lower-ncnn-vulkan-graph %s | FileCheck %s

func.func @aliases(%input: tensor<2x3x5xf32>) -> (tensor<3x10xf32>, tensor<2x3x5xf32>) {
  %shape = "tosa.const_shape"() <{values = dense<[3, 10]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %view = "tosa.reshape"(%input, %shape) : (tensor<2x3x5xf32>, !tosa.shape<2>) -> tensor<3x10xf32>
  return %view, %input : tensor<3x10xf32>, tensor<2x3x5xf32>
}
// CHECK: gpu.module @aliases_vulkan_k0
// CHECK: gpu.func @kernel_0
// CHECK: memref.store
// CHECK: gpu.module @aliases_vulkan_k1
// CHECK: gpu.func @kernel_1
// CHECK: memref.store
// CHECK-NOT: func.func
// CHECK-NOT: tosa.

// -----

func.func @layout(%input: tensor<3x5x7xf32>) -> tensor<5x6x3xf32> {
  %transposed = "tosa.transpose"(%input) <{perms = array<i32: 1, 2, 0>}> : (tensor<3x5x7xf32>) -> tensor<5x7x3xf32>
  %start = "tosa.const_shape"() <{values = dense<[0, 1, 0]> : tensor<3xindex>}> : () -> !tosa.shape<3>
  %size = "tosa.const_shape"() <{values = dense<[5, 6, 3]> : tensor<3xindex>}> : () -> !tosa.shape<3>
  %slice = "tosa.slice"(%transposed, %start, %size) : (tensor<5x7x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x6x3xf32>
  return %slice : tensor<5x6x3xf32>
}
// CHECK: gpu.module @layout_vulkan_k0
// CHECK: arith.remui
// CHECK: memref.load
// CHECK: gpu.module @layout_vulkan_k1
// CHECK: arith.addi
// CHECK: gpu.module @layout_vulkan_k2
// CHECK-NOT: func.func
// CHECK-NOT: tosa.

// -----

func.func @duplicate_operand(%input: tensor<7xf32>) -> tensor<7xf32> {
  %sum = "tosa.add"(%input, %input) : (tensor<7xf32>, tensor<7xf32>) -> tensor<7xf32>
  return %sum : tensor<7xf32>
}
// CHECK: gpu.module @duplicate_operand_vulkan_k0
// CHECK: gpu.func @kernel_0(%{{[^:]+}}: memref<7xf32
// CHECK-SAME: %{{[^:]+}}: memref<7xf32
// CHECK: arith.addf
// CHECK: gpu.module @duplicate_operand_vulkan_k1
