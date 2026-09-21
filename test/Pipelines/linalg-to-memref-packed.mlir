// RUN: ncnn-mlir-opt '--ncnn-linalg-to-memref-pipeline=vector-tail=true matmul-packing=auto' %s | FileCheck %s

module {
  func.func @packed_matmul(%arg0: tensor<32x256xf32>) -> tensor<32x64xf32> {
    %rhs = arith.constant dense<1.0> : tensor<256x64xf32>
    %out = tensor.empty() : tensor<32x64xf32>
    %mm = linalg.matmul ins(%arg0, %rhs : tensor<32x256xf32>, tensor<256x64xf32>)
                     outs(%out : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %mm : tensor<32x64xf32>
  }
}

// CHECK: memref.global "private" constant @__constant_256x32xf32
// CHECK: memref.get_global @__constant_256x32xf32 : memref<256x32xf32> {ncnn.alignment = "64"{{.*}}ncnn.packed_weight = true{{.*}}ncnn.weight_layout = "panel_nk"}
// CHECK: scf.for {{.*}} to {{.*}} step %{{.*}}128
// CHECK: vector.fma
// CHECK: ncnn.kernel = "f32_packed_mxn_fma"
// CHECK: ncnn.pack_runtime = "compile_time_B"
