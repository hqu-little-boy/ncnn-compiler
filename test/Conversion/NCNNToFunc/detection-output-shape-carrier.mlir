// RUN: ncnn-mlir-opt --convert-ncnn-model-to-func %s | FileCheck %s

module {
  "ncnn.model"() <{sym_name = "detection"}> ({
    %location = "ncnn.input"() {blob_name = "location", layer_name = "location"} : () -> tensor<12xf32>
    %confidence = "ncnn.input"() {blob_name = "confidence", layer_name = "confidence"} : () -> tensor<9xf32>
    %prior = "ncnn.input"() {blob_name = "prior", layer_name = "prior"} : () -> tensor<2x12xf32>
    %result, %shape = "ncnn.detection_output"(%location, %confidence, %prior) {confidence_threshold = 5.000000e-01 : f32, keep_top_k = 2 : i64, nms_threshold = 5.000000e-02 : f32, nms_top_k = 3 : i64, num_class = 3 : i64, variance_h = 2.000000e-01 : f32, variance_w = 2.000000e-01 : f32, variance_x = 1.000000e-01 : f32, variance_y = 1.000000e-01 : f32} : (tensor<12xf32>, tensor<9xf32>, tensor<2x12xf32>) -> (tensor<2x6xf32>, tensor<2xi64>)
    "ncnn.output"(%result) {blob_name = "output"} : (tensor<2x6xf32>) -> ()
  }) : () -> ()
}

// CHECK-LABEL: func.func @detection
// CHECK-SAME: -> (tensor<2x6xf32> {ncnn.data_dependent_dim_mask = 1 : i32}, tensor<2xi64> {ncnn.shape_carrier})
// CHECK: %[[RESULT:.*]], %[[SHAPE:.*]] = ncnn.detection_output
// CHECK: return %[[RESULT]], %[[SHAPE]]

// -----

module {
  "ncnn.model"() <{sym_name = "detection_then_dynamic"}> ({
    %location = "ncnn.input"() {blob_name = "location", layer_name = "location"} : () -> tensor<12xf32>
    %confidence = "ncnn.input"() {blob_name = "confidence", layer_name = "confidence"} : () -> tensor<9xf32>
    %prior = "ncnn.input"() {blob_name = "prior", layer_name = "prior"} : () -> tensor<2x12xf32>
    %dynamic = "ncnn.input"() {blob_name = "dynamic", layer_name = "dynamic"} : () -> tensor<3x?x?xf32>
    %result, %shape = "ncnn.detection_output"(%location, %confidence, %prior) {confidence_threshold = 5.000000e-01 : f32, keep_top_k = 2 : i64, nms_threshold = 5.000000e-02 : f32, nms_top_k = 3 : i64, num_class = 3 : i64, variance_h = 2.000000e-01 : f32, variance_w = 2.000000e-01 : f32, variance_x = 1.000000e-01 : f32, variance_y = 1.000000e-01 : f32} : (tensor<12xf32>, tensor<9xf32>, tensor<2x12xf32>) -> (tensor<2x6xf32>, tensor<2xi64>)
    "ncnn.output"(%result) {blob_name = "detections"} : (tensor<2x6xf32>) -> ()
    "ncnn.output"(%dynamic) {blob_name = "dynamic_output"} : (tensor<3x?x?xf32>) -> ()
  }) : () -> ()
}

// CHECK-LABEL: func.func @detection_then_dynamic
// CHECK-SAME: -> (tensor<2x6xf32> {ncnn.data_dependent_dim_mask = 1 : i32}, tensor<2xi64> {ncnn.shape_carrier}, tensor<3x?x?xf32> {ncnn.shape_program = [array<i64>, array<i64>, array<i64>], ncnn.shape_source_input = 3 : i32})
