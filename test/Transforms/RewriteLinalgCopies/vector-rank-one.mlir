// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" %s | FileCheck %s
// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" --instrument-ncnn-profile %s | FileCheck %s --check-prefix=PROFILE

module {
  func.func @full_vectors() {
    %source = memref.alloc() : memref<16xf32>
    %target = memref.alloc() : memref<16xf32>
    memref.copy %source, %target : memref<16xf32> to memref<16xf32>
    return
  }

  func.func @vectors_with_tail() {
    %source = memref.alloc() : memref<10xf32>
    %target = memref.alloc() : memref<10xf32>
    memref.copy %source, %target : memref<10xf32> to memref<10xf32>
    return
  }
}

// CHECK: module attributes {{.*}}ncnn.copy_vectorized_count = 2 : i64
// CHECK-LABEL: func.func @full_vectors
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: ncnn.copy_kind = "vectorized"
// CHECK: ncnn.copy_vector_lanes = 4 : i64
// CHECK-LABEL: func.func @vectors_with_tail
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: ncnn.copy_kind = "vectorized"
// CHECK: ncnn.copy_vector_lanes = 4 : i64

// PROFILE-LABEL: func.func @full_vectors
// PROFILE: call @__ncnn_profile_event_begin
// PROFILE: scf.for
// PROFILE: scf.for
// PROFILE: vector.transfer_write
// PROFILE: call @__ncnn_profile_event_end
// PROFILE: call @__ncnn_profile_copy
// PROFILE-LABEL: func.func @vectors_with_tail
// PROFILE: call @__ncnn_profile_event_begin
// PROFILE: scf.for
// PROFILE: vector.transfer_write
// PROFILE: scf.for
// PROFILE: memref.store
// PROFILE: call @__ncnn_profile_event_end
// PROFILE: call @__ncnn_profile_copy
