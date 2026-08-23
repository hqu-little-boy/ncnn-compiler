// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=4 %s | FileCheck %s --check-prefix=ON
// RUN: ncnn-mlir-opt --vectorize-ncnn=lanes=0 %s | FileCheck %s --check-prefix=OFF

// lanes=0 时 pass 为无操作，标量 Linalg 保持原样。

func.func @relu(%arg0: tensor<6x8xf32>) -> tensor<6x8xf32> {
  %empty = tensor.empty() : tensor<6x8xf32>
  %zero = arith.constant 0.0 : f32
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<6x8xf32>) outs(%empty : tensor<6x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %zero : f32
    linalg.yield %max : f32
  } -> tensor<6x8xf32>
  return %result : tensor<6x8xf32>
}

// ON-NOT: linalg.generic
// OFF-LABEL: func.func @relu
// OFF: linalg.generic
// OFF-NOT: vector.transfer_read
