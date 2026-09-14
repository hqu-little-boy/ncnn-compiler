// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=metadata_alias target-triple=x86_64-pc-linux-gnu threads=1" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

module {
  func.func @metadata_alias(%input: memref<4x4xf32>) {
    %buffer = memref.alloc() : memref<4x4xf32, strided<[4, 1], offset: 0>>
    %base, %offset, %size0, %size1, %stride0, %stride1 =
      memref.extract_strided_metadata %buffer :
        memref<4x4xf32, strided<[4, 1], offset: 0>> ->
        memref<f32>, index, index, index, index, index
    %value = memref.load %base[] : memref<f32>
    memref.dealloc %base : memref<f32>
    return
  }
}

// CHECK: "model": "metadata_alias"
// CHECK: "static_liveness": [
// CHECK: "status": "proven"
// CHECK: "peak_workspace_proven": true
