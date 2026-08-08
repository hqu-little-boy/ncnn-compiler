// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=dynamic_rank manifest-path=%t.json' %s -o %t.mlir
// RUN: FileCheck %s --check-prefix=MANIFEST < %t.json
// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline %t.mlir | FileCheck %s --check-prefix=LLVM

module {
  func.func @rank1(%input: memref<?xf32>, %output: memref<?xf32> {bufferize.result, ncnn.shape_program = [array<i64>], ncnn.shape_source_input = 0 : i32}) attributes {llvm.emit_c_interface, ncnn.dynamic_rank, ncnn.entry_point, ncnn.rank_variant = 1 : i32} { memref.copy %input, %output : memref<?xf32> to memref<?xf32> return }
  func.func @rank2(%input: memref<?x?xf32>, %output: memref<?x?xf32> {bufferize.result, ncnn.shape_program = [array<i64>, array<i64>], ncnn.shape_source_input = 0 : i32}) attributes {llvm.emit_c_interface, ncnn.dynamic_rank, ncnn.entry_point, ncnn.rank_variant = 2 : i32} { memref.copy %input, %output : memref<?x?xf32> to memref<?x?xf32> return }
  func.func @rank3(%input: memref<?x?x?xf32>, %output: memref<?x?x?xf32> {bufferize.result, ncnn.shape_program = [array<i64>, array<i64>, array<i64>], ncnn.shape_source_input = 0 : i32}) attributes {llvm.emit_c_interface, ncnn.dynamic_rank, ncnn.entry_point, ncnn.rank_variant = 3 : i32} { memref.copy %input, %output : memref<?x?x?xf32> to memref<?x?x?xf32> return }
  func.func @rank4(%input: memref<?x?x?x?xf32>, %output: memref<?x?x?x?xf32> {bufferize.result, ncnn.shape_program = [array<i64>, array<i64>, array<i64>, array<i64>], ncnn.shape_source_input = 0 : i32}) attributes {llvm.emit_c_interface, ncnn.dynamic_rank, ncnn.entry_point, ncnn.rank_variant = 4 : i32} { memref.copy %input, %output : memref<?x?x?x?xf32> to memref<?x?x?x?xf32> return }
}

// MANIFEST: "dynamic_rank": true
// MANIFEST: "rank_max": 4
// MANIFEST: "rank_min": 1
// LLVM-LABEL: llvm.func @dynamic_rank(
// LLVM-SAME: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr, %{{.*}}: i64) -> i32
// LLVM: llvm.call @__ncnn_internal_dynamic_rank_rank1
// LLVM: llvm.call @__ncnn_internal_dynamic_rank_rank2
// LLVM: llvm.call @__ncnn_internal_dynamic_rank_rank3
// LLVM: llvm.call @__ncnn_internal_dynamic_rank_rank4
// LLVM-LABEL: llvm.func @dynamic_rank_infer_output_shapes(
// LLVM-SAME: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32
