// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=compact_positive manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @model(
      %input: memref<8xf32>,
      %output: memref<8xf32> {bufferize.result,
        ncnn.data_dependent_dim_mask = 1 : i32},
      %actual_shape: memref<1xi64> {bufferize.result, ncnn.shape_carrier})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %count = scf.for %i = %c0 to %c8 step %c1 iter_args(%written = %c0) -> index {
      %value = memref.load %input[%i] : memref<8xf32>
      %positive = arith.cmpf ogt, %value, %zero : f32
      %next = scf.if %positive -> index {
        memref.store %value, %output[%written] : memref<8xf32>
        %incremented = arith.addi %written, %c1 : index
        scf.yield %incremented : index
      } else {
        scf.yield %written : index
      }
      scf.yield %next : index
    }
    %count_i64 = arith.index_cast %count : index to i64
    memref.store %count_i64, %actual_shape[%c0] : memref<1xi64>
    return
  }
}

// MANIFEST: "maximum_shape": [
// MANIFEST-NEXT: 8
// MANIFEST: "shape": [
// MANIFEST-NEXT: -1
// MANIFEST: "shape_depends_on_data": true
// MANIFEST-NOT: "name": "output2"

// LLVM-LABEL: llvm.func @compact_positive(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32
// LLVM: llvm.call @__ncnn_internal_compact_positive(
// LLVM: llvm.store
// LLVM-NOT: llvm.func @compact_positive_infer_output_shapes
