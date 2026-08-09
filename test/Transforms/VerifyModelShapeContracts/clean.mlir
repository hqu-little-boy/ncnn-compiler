// RUN: ncnn-mlir-opt --verify-ncnn-model-shape-contracts %s -o /dev/null

module {
  func.func @static(%input: memref<3x32x32xf32>,
                    %output: memref<1x32x32xf32> {bufferize.result})
      attributes {ncnn.entry_point} {
    return
  }

  func.func @mixed(
      %static_input: memref<4xf32>,
      %dynamic_input: memref<3x?x?xf32>,
      %static_output: memref<4xf32> {bufferize.result},
      %dynamic_output: memref<1x?x?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>, array<i64>, array<i64>],
        ncnn.shape_source_input = 1 : i32})
      attributes {ncnn.entry_point} {
    return
  }

  func.func @bounded(
      %input: memref<8xf32>,
      %output: memref<8xf32> {bufferize.result,
        ncnn.data_dependent_dim_mask = 1 : i32},
      %actual_shape: memref<1xi64> {bufferize.result, ncnn.shape_carrier})
      attributes {ncnn.entry_point} {
    return
  }
}
