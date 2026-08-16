// RUN: not ncnn-mlir-opt --generate-ncnn-c-api='export-name=invalid_relation' %s 2>&1 | FileCheck %s

module {
  func.func @model(
      %input: memref<?x4xf32>,
      %output: memref<?x4xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>, array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {ncnn.entry_point,
        ncnn.input_dim_relations = [array<i64: 0, 2, 0, 0, 0>]} {
    return
  }
}

// CHECK: error: 'func.func' op has invalid input dimension relations
