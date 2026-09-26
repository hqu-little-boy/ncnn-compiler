// RUN: ncnn-mlir-opt --instrument-ncnn-profile %s -o %t
// RUN: FileCheck %s --input-file=%t

module {
  func.func @model(%input: memref<4x4xf32>) attributes {ncnn.entry_point} {
    %buffer = memref.alloc() : memref<4x4xf32>
    memref.copy %input, %buffer : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }
  func.func @parallel_work() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.forall (%iv) in (%c4) {
      scf.for %j = %c0 to %c4 step %c1 {
        %sum = arith.addi %iv, %j : index
      }
    }
    return
  }
}

// CHECK: func.func private @__ncnn_profile_flush
// CHECK: func.func private @__ncnn_profile_copy
// CHECK: func.func private @__ncnn_profile_dealloc
// CHECK: func.func private @__ncnn_profile_alloc
// CHECK: func.func private @__ncnn_profile_worker_event_end
// CHECK: func.func private @__ncnn_profile_worker_event_begin
// CHECK: func.func private @__ncnn_profile_event_end
// CHECK: func.func private @__ncnn_profile_event_begin
// CHECK: call @__ncnn_profile_alloc
// CHECK: call @__ncnn_profile_copy
// CHECK: call @__ncnn_profile_dealloc
// CHECK: call @__ncnn_profile_flush
// CHECK-LABEL: func.func @parallel_work
// CHECK: scf.forall
// CHECK: call @__ncnn_profile_worker_event_begin
// CHECK: scf.for
// CHECK: call @__ncnn_profile_worker_event_end
