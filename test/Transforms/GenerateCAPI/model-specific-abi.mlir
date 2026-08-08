// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=model_specific manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  // Deliberately interleave outputs and inputs. The public ABI must still be:
  // dynamic input group, static input, all output data, then shape metadata.
  func.func @model(
      %dependent: memref<8xf32> {bufferize.result,
        ncnn.data_dependent_dim_mask = 1 : i32},
      %actual_shape: memref<1xi64> {bufferize.result, ncnn.shape_carrier},
      %ordinary: memref<4xi64> {bufferize.result},
      %dynamic_input: memref<?x4xi32>,
      %static_input: memref<2xf32>)
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : i64
    memref.store %c8, %actual_shape[%c0] : memref<1xi64>
    return
  }
}

// MANIFEST: "element_type": "i32"
// MANIFEST: "name": "input1"
// MANIFEST: "element_type": "f32"
// MANIFEST: "name": "input2"
// MANIFEST: "maximum_shape": [
// MANIFEST-NEXT: 8
// MANIFEST: "name": "output1"
// MANIFEST: "shape_depends_on_data": true
// MANIFEST: "element_type": "i64"
// MANIFEST: "name": "output2"

// Public order:
//   input1_data, input1_shape, input2_data,
//   output1_data, output2_data,
//   output1_actual_shape, output1_shape_capacity, output1_rank.
// LLVM-LABEL: llvm.func @model_specific
// LLVM-SAME: (%[[INPUT1:.*]]: !llvm.ptr, %[[INPUT1_SHAPE:.*]]: !llvm.ptr, %[[INPUT2:.*]]: !llvm.ptr, %[[OUTPUT1:.*]]: !llvm.ptr, %[[OUTPUT2:.*]]: !llvm.ptr, %[[OUTPUT1_SHAPE:.*]]: !llvm.ptr, %[[OUTPUT1_CAPACITY:.*]]: i32, %[[OUTPUT1_RANK:.*]]: !llvm.ptr) -> i32
// LLVM: llvm.icmp "uge" %[[OUTPUT1_CAPACITY]]
// LLVM: llvm.call @__ncnn_internal_model_specific(
// LLVM: llvm.store %{{.*}}, %[[OUTPUT1_RANK]]
