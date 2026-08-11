// RUN: not ncnn-mlir-opt --finalize-ncnn-c-api --split-input-file %s 2>&1 | FileCheck %s

module attributes {
  ncnn.c_api.export_name = "model",
  ncnn.c_api.internal_name = "__ncnn_internal_model",
  ncnn.c_api.argument_types = [memref<?xf32>, memref<?xf32>],
  ncnn.c_api.output_indices = array<i32: 1>,
  ncnn.c_api.output_shape_sources = array<i32>,
  ncnn.c_api.output_shape_program_versions = array<i32: 1>,
  ncnn.c_api.output_shape_programs = [[]],
  ncnn.c_api.shape_carrier_indices = array<i32>,
  ncnn.c_api.input_shape_constraints = []
} {
  llvm.func private @__ncnn_internal_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: error: has inconsistent prepared output metadata

// -----

module attributes {
  ncnn.c_api.export_name = "model",
  ncnn.c_api.internal_name = "__ncnn_internal_model",
  ncnn.c_api.argument_types = [memref<?xf32>, memref<?xf32>],
  ncnn.c_api.output_indices = array<i32: 2>,
  ncnn.c_api.output_shape_sources = array<i32: 0>,
  ncnn.c_api.output_shape_program_versions = array<i32: 1>,
  ncnn.c_api.output_shape_programs = [[array<i64>]],
  ncnn.c_api.shape_carrier_indices = array<i32>,
  ncnn.c_api.input_shape_constraints = []
} {
  llvm.func private @__ncnn_internal_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: error: has invalid prepared output index metadata

// -----

module attributes {
  ncnn.c_api.export_name = "model",
  ncnn.c_api.internal_name = "__ncnn_internal_model",
  ncnn.c_api.argument_types = [memref<?xf32>, memref<?xf32>],
  ncnn.c_api.output_indices = array<i32: 1>,
  ncnn.c_api.output_shape_sources = array<i32: -1>,
  ncnn.c_api.output_shape_program_versions = array<i32: 2>,
  ncnn.c_api.output_shape_programs = [[array<i64: 1, 1, 0>]],
  ncnn.c_api.shape_carrier_indices = array<i32>,
  ncnn.c_api.input_shape_constraints = []
} {
  llvm.func private @__ncnn_internal_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: error: has invalid serialized V2 shape program

// -----

module attributes {
  ncnn.c_api.export_name = "model",
  ncnn.c_api.internal_name = "__ncnn_internal_model",
  ncnn.c_api.argument_types = [memref<?xf32>, memref<?xf32>],
  ncnn.c_api.output_indices = array<i32: 1>,
  ncnn.c_api.output_shape_sources = array<i32: -1>,
  ncnn.c_api.output_shape_program_versions = array<i32: 2>,
  ncnn.c_api.output_shape_programs = [[array<i64: 1, 0, 1>]],
  ncnn.c_api.shape_carrier_indices = array<i32>,
  ncnn.c_api.input_shape_constraints = []
} {
  llvm.func private @__ncnn_internal_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: error: has invalid serialized V2 shape program

// -----

module attributes {
  ncnn.c_api.export_name = "model",
  ncnn.c_api.internal_name = "__ncnn_internal_model",
  ncnn.c_api.argument_types = [memref<2xf32>, memref<1xi64>],
  ncnn.c_api.output_indices = array<i32: 1>,
  ncnn.c_api.output_shape_sources = array<i32: -1>,
  ncnn.c_api.output_shape_program_versions = array<i32: 0>,
  ncnn.c_api.output_shape_programs = [[]],
  ncnn.c_api.shape_carrier_indices = array<i32: 1>,
  ncnn.c_api.input_shape_constraints = []
} {
  llvm.func private @__ncnn_internal_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: error: has invalid shape carrier metadata

// -----

module attributes {
  ncnn.c_api.export_name = "model",
  ncnn.c_api.internal_name = "__ncnn_internal_model",
  ncnn.c_api.argument_types = [memref<2xf32>, memref<2xf32>, memref<i64>],
  ncnn.c_api.output_indices = array<i32: 1, 2>,
  ncnn.c_api.output_shape_sources = array<i32: -1, -1>,
  ncnn.c_api.output_shape_program_versions = array<i32: 1, 0>,
  ncnn.c_api.output_shape_programs = [[], []],
  ncnn.c_api.shape_carrier_indices = array<i32: 2>,
  ncnn.c_api.input_shape_constraints = []
} {
  llvm.func private @__ncnn_internal_model(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.return
  }
}

// CHECK: error: has invalid shape carrier metadata
