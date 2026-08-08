// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %s | FileCheck %s
// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %s | ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline | ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline | FileCheck %s --check-prefix=MEMREF

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

// MEMREF-LABEL: func.func @reshape_from_reference(
// MEMREF-SAME: %[[DATA:[A-Za-z0-9_]+]]: memref<1x?x?xf32>,
// MEMREF-SAME: %[[REFERENCE:[A-Za-z0-9_]+]]: memref<1x?x?xf32>,
// MEMREF-SAME: %[[OUTPUT:[A-Za-z0-9_]+]]: memref<1x?x?xf32> {bufferize.result, ncnn.shape_program = [array<i64>, array<i64>, array<i64>], ncnn.shape_source_input = 1 : i32})
// MEMREF: %[[RESHAPE:.*]] = memref.reshape %[[DATA]]
// MEMREF: memref.copy %[[RESHAPE]], %[[OUTPUT]]
// MEMREF-NOT: memref<1x?x?xf32, strided
// MEMREF-NOT: tensor.
// MEMREF-NOT: bufferization.
