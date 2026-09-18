// RUN: ncnn-mlir-opt --instrument-ncnn-profile %s | FileCheck %s
// RUN: ncnn-mlir-opt --instrument-ncnn-profile --ncnn-memref-to-llvm-pipeline='threads=4 vector-lowering=true' %s -o %t
// Memory callbacks inside nested control flow must survive OpenMP lowering.
module {
  func.func @model(%input: memref<16xf32>, %choose: i1) attributes {ncnn.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%i) = (%c0) to (%c4) step (%c1) {
      scf.if %choose {
        %buffer = memref.alloc() : memref<16xf32>
        memref.copy %input, %buffer : memref<16xf32> to memref<16xf32>
        memref.dealloc %buffer : memref<16xf32>
      }
      scf.reduce
    }
    return
  }

  func.func @forwarded_alloc() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %buffer = memref.alloc() : memref<16xf32>
    %forwarded = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %buffer) -> (memref<16xf32>) {
      scf.yield %iter : memref<16xf32>
    }
    memref.dealloc %forwarded : memref<16xf32>
    return
  }
}
// CHECK-LABEL: func.func @model
// CHECK: scf.parallel
// CHECK: scf.if
// CHECK-NOT: call @__ncnn_profile_event_begin
// CHECK: call @__ncnn_profile_alloc
// CHECK: memref.alloc
// CHECK: call @__ncnn_profile_copy
// CHECK: memref.copy
// CHECK: call @__ncnn_profile_dealloc
// CHECK: memref.dealloc
// CHECK-NOT: memref.alloca_scope
// CHECK: call @__ncnn_profile_flush
// CHECK-LABEL: func.func @forwarded_alloc
// CHECK: call @__ncnn_profile_alloc(%[[ALLOC_ID:[^, ]+]],
// CHECK: %[[ALLOC_ID]]_2 = arith.constant {{-?[0-9]+}} : i64
// CHECK-NEXT: call @__ncnn_profile_dealloc(%[[ALLOC_ID]]_2)
