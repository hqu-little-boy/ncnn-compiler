// RUN: ncnn-mlir-opt --instrument-ncnn-profile %s | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>
#identity_1d = affine_map<(d0) -> (d0)>
#empty_writer_input = affine_map<(d0, d1) -> (d0, d1)>
#full_1d = affine_map<(d0, d1) -> (d0)>
#empty_reader_output = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @model(%input: memref<4x4xf32>, %output: memref<4x4xf32>) attributes {ncnn.entry_point} {
    %buffer = memref.alloc() : memref<4x4xf32>
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<4x4xf32>) outs(%buffer : memref<4x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%buffer : memref<4x4xf32>) outs(%output : memref<4x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }

  func.func @full_view_chain(%input: memref<16xf32>,
                             %output: memref<16xf32>) {
    %buffer = memref.alloc() : memref<4x4xf32>
    %view = memref.collapse_shape %buffer [[0, 1]] : memref<4x4xf32> into memref<16xf32>
    linalg.generic {
      indexing_maps = [#identity_1d, #identity_1d],
      iterator_types = ["parallel"]
    } ins(%input : memref<16xf32>) outs(%view : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      indexing_maps = [#identity_1d, #identity_1d],
      iterator_types = ["parallel"]
    } ins(%view : memref<16xf32>) outs(%output : memref<16xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }

  func.func @partial_view(%input: memref<2x4xf32>,
                          %output: memref<2x4xf32>) {
    %buffer = memref.alloc() : memref<4x4xf32>
    %view = memref.subview %buffer[0, 0][2, 4][1, 1] : memref<4x4xf32> to memref<2x4xf32>
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<2x4xf32>) outs(%view : memref<2x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%view : memref<2x4xf32>) outs(%output : memref<2x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }

  func.func @dynamic_view(%input: memref<?x4xf32>,
                          %output: memref<?x4xf32>, %rows: index) {
    %buffer = memref.alloc() : memref<4x4xf32>
    %view = memref.subview %buffer[0, 0][%rows, 4][1, 1] : memref<4x4xf32> to memref<?x4xf32>
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<?x4xf32>) outs(%view : memref<?x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%view : memref<?x4xf32>) outs(%output : memref<?x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }

  func.func @unconsumed(%input: memref<4x4xf32>) {
    %buffer = memref.alloc() : memref<4x4xf32>
    linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input : memref<4x4xf32>) outs(%buffer : memref<4x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.dealloc %buffer : memref<4x4xf32>
    return
  }

  func.func @empty_domain(%empty: memref<4x0xf32>,
                          %empty_output: memref<4x0xf32>) {
    %buffer = memref.alloc() : memref<4xf32>
    linalg.generic {
      indexing_maps = [#empty_writer_input, #full_1d],
      iterator_types = ["parallel", "parallel"]
    } ins(%empty : memref<4x0xf32>) outs(%buffer : memref<4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      indexing_maps = [#full_1d, #empty_reader_output],
      iterator_types = ["parallel", "parallel"]
    } ins(%buffer : memref<4xf32>) outs(%empty_output : memref<4x0xf32>) {
    ^bb0(%value: f32, %unused: f32):
      linalg.yield %value : f32
    }
    memref.dealloc %buffer : memref<4xf32>
    return
  }

  func.func @copy_contract() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      scf.yield
    } {ncnn.copy_contract = "p22_identity_copy", ncnn.copy_kind = "fallback", ncnn.copy_bytes = 16 : i64}
    return
  }
}

// CHECK: func.func private @__ncnn_profile_materialized
// CHECK-LABEL: func.func @model
// CHECK: linalg.generic
// CHECK: call @__ncnn_profile_materialized({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: call @__ncnn_profile_materialized({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK-LABEL: func.func @full_view_chain
// CHECK: call @__ncnn_profile_materialized({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK: call @__ncnn_profile_materialized({{.*}}, {{.*}}, {{.*}}, {{.*}})
// CHECK-LABEL: func.func @partial_view
// CHECK-NOT: call @__ncnn_profile_materialized
// CHECK-LABEL: func.func @dynamic_view
// CHECK-NOT: call @__ncnn_profile_materialized
// CHECK-LABEL: func.func @unconsumed
// CHECK-NOT: call @__ncnn_profile_materialized
// CHECK-LABEL: func.func @empty_domain
// CHECK-NOT: call @__ncnn_profile_materialized
// CHECK-LABEL: func.func @copy_contract
// CHECK: scf.for
// CHECK: call @__ncnn_profile_copy
