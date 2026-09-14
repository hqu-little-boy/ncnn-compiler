// RUN: ncnn-mlir-opt '--vectorize-ncnn=lanes=4 scalable=true' %s | FileCheck %s

// Scalable vectors are rejected explicitly until the vectorizer emits scalable
// VectorType values instead of fixed-width vectors.
module {
  func.func @relu(%arg0: tensor<6x8xf32>) -> tensor<6x8xf32> {
    %empty = tensor.empty() : tensor<6x8xf32>
    %zero = arith.constant 0.0 : f32
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<6x8xf32>) outs(%empty : tensor<6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %zero : f32
      linalg.yield %max : f32
    } -> tensor<6x8xf32>
    return %result : tensor<6x8xf32>
  }
}

// CHECK-DAG: linalg.generic
// CHECK-DAG: ncnn.contract = "fallback"
// CHECK-DAG: ncnn.fallback_reason = "unsupported_scalable_vector"
// CHECK-NOT: vector.transfer_read
