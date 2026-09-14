// RUN: ncnn-mlir-opt --instrument-ncnn-profile %s -o %t
// RUN: FileCheck %s --input-file=%t

module {
  func.func @model(%input: memref<4x4xf32>) attributes {ncnn.entry_point} {
    %buffer = memref.alloc() : memref<4x4xf32>
    memref.copy %input, %buffer : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }
}

// CHECK: func.func private @__ncnn_profile_flush
// CHECK: func.func private @__ncnn_profile_copy
// CHECK: func.func private @__ncnn_profile_dealloc
// CHECK: func.func private @__ncnn_profile_alloc
// CHECK: func.func private @__ncnn_profile_event_end
// CHECK: func.func private @__ncnn_profile_event_begin
// CHECK: call @__ncnn_profile_alloc
// CHECK: call @__ncnn_profile_copy
// CHECK: call @__ncnn_profile_dealloc
// CHECK: call @__ncnn_profile_flush
