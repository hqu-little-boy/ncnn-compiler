// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %s | FileCheck %s

module {
  "ncnn.model"() <{sym_name = "reshape_from_reference"}> ({
    %data = "ncnn.input"() {blob_name = "data", layer_name = "data"}
      : () -> tensor<1x?x?xf32>
    %reference = "ncnn.input"() {blob_name = "reference", layer_name = "reference"}
      : () -> tensor<1x?x?xf32>
    %result = "ncnn.reshape"(%data, %reference) {
      shape = array<i64: 1, -9223372036854775808, -9223372036854775808>,
      shape_expression = "1w,1h,1c",
      shape_sources = array<i64: 1, 0, 1, 1, 1, 2>}
      : (tensor<1x?x?xf32>, tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
    "ncnn.output"(%result) {blob_name = "output"}
      : (tensor<1x?x?xf32>) -> ()
  }) : () -> ()
}

// CHECK: %[[REF:.*]] = tensor.expand_shape
// CHECK: %[[H:.*]] = tensor.dim %[[REF]], %c1
// CHECK: %[[W:.*]] = tensor.dim %[[REF]], %c2
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %c1, %[[H]], %[[W]]
// CHECK: tensor.reshape {{.*}}(%[[SHAPE]])
// CHECK-NOT: ncnn.
