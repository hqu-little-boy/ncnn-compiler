// RUN: ncnn-mlir-opt --verify-diagnostics --verify-ncnn-model-shape-contracts --split-input-file %s

module {
  // expected-error@+1 {{static shape route has dynamic argument 1}}
  func.func @static_with_dynamic_output(
      %input: memref<4xf32>,
      %output: memref<?xf32> {bufferize.result})
      attributes {ncnn.entry_point} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{dynamic output 1 has no valid input shape source}}
  func.func @missing_source(
      %input: memref<?xf32>,
      %output: memref<?xf32> {bufferize.result})
      attributes {ncnn.entry_point} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{dynamic output 2 shape source must be a dynamic input of the same rank}}
  func.func @static_source(
      %static_input: memref<4xf32>,
      %dynamic_input: memref<?xf32>,
      %output: memref<?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {ncnn.entry_point} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{dynamic output 1 has no complete shape program}}
  func.func @incomplete_program(
      %input: memref<?x?xf32>,
      %output: memref<?x?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {ncnn.entry_point} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{data-dependent output 1 has no shape carrier}}
  func.func @missing_carrier(
      %input: memref<8xf32>,
      %output: memref<8xf32> {bufferize.result,
        ncnn.data_dependent_dim_mask = 1 : i32})
      attributes {ncnn.entry_point} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{output 1 has an invalid data-dependent shape contract}}
  func.func @mask_exceeds_rank(
      %input: memref<8xf32>,
      %output: memref<1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1x1xf32> {bufferize.result,
        ncnn.data_dependent_dim_mask = 1099511627776 : i64})
      attributes {ncnn.entry_point} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{dynamic_rank and rank_variant attributes must appear together}}
  func.func @missing_rank_variant(
      %input: memref<?xf32>,
      %output: memref<?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {ncnn.dynamic_rank, ncnn.entry_point} {
    return
  }
}

// -----

// expected-error@+1 {{dynamic rank route requires unique specializations for ranks 1 through 4}}
module {
  func.func @rank1(
      %input: memref<?xf32>,
      %output: memref<?xf32> {bufferize.result,
        ncnn.shape_program = [array<i64>],
        ncnn.shape_source_input = 0 : i32})
      attributes {ncnn.dynamic_rank, ncnn.entry_point,
                  ncnn.rank_variant = 1 : i32} {
    return
  }
}
