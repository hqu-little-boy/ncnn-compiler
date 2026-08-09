// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  ncnn.model @relu_rank2 attributes {
    ncnn.dynamic_rank,
    ncnn.rank_variant = 2 : i32
  } {
    %input = ncnn.input {blob_name = "data", layer_name = "input"} : tensor<?x?xf32>
    %relu = ncnn.relu %input : (tensor<?x?xf32>) -> tensor<?x?xf32>
    ncnn.output %relu {blob_name = "output"} : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @relu_rank2(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>
// CHECK-SAME: ncnn.shape_program = [array<i64>, array<i64>]
// CHECK-SAME: ncnn.shape_source_input = 0 : i32
// CHECK-SAME: attributes {{.*}}ncnn.dynamic_rank{{.*}}ncnn.rank_variant = 2 : i32
// CHECK: %[[RELU:.*]] = ncnn.relu %arg0
// CHECK: return %[[RELU]] : tensor<?x?xf32>
