// RUN: not ncnn-mlir-opt --generate-ncnn-c-api='export-name=model' --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @model(
      %input: memref<2xf32>,
      %carrier: memref<1xi64> {bufferize.result, ncnn.shape_carrier})
      attributes {ncnn.entry_point} {
    return
  }
}

// CHECK: error: 'func.func' op argument 1 is not paired with a data-dependent output

// -----

module {
  func.func @model(
      %input: memref<2xf32>,
      %output: memref<2xf32> {bufferize.result,
        ncnn.data_dependent_dim_mask = 1 : i32},
      %carrier: memref<i64> {bufferize.result, ncnn.shape_carrier})
      attributes {ncnn.entry_point} {
    return
  }
}

// CHECK: error: 'func.func' op output 1 must be followed by an i64 shape carrier of its rank
