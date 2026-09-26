// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" %s | FileCheck %s

// 非连续布局：canVectorize 要求两侧都是连续恒等布局（最后一维步距 1）。
// 步距不为 1、或最内维长度小于 lane 数时，一律落 sequential fallback。

func.func @strided_source() {
  %source = memref.alloc() : memref<2x10xf32, strided<[20, 2]>>
  %target = memref.alloc() : memref<2x10xf32>
  linalg.copy ins(%source : memref<2x10xf32, strided<[20, 2]>>) outs(%target : memref<2x10xf32>)
  memref.dealloc %source : memref<2x10xf32, strided<[20, 2]>>
  memref.dealloc %target : memref<2x10xf32>
  return
}

// CHECK-LABEL: func.func @strided_source
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vector.transfer_read
// CHECK: ncnn.copy_kind = "fallback"

// -----

// 最内维 2 < lanes 4：装不下一个向量，不值得向量化。
func.func @narrow_inner_dim() {
  %source = memref.alloc() : memref<2x2xf32>
  %target = memref.alloc() : memref<2x2xf32>
  linalg.copy ins(%source : memref<2x2xf32>) outs(%target : memref<2x2xf32>)
  memref.dealloc %source : memref<2x2xf32>
  memref.dealloc %target : memref<2x2xf32>
  return
}

// CHECK-LABEL: func.func @narrow_inner_dim
// CHECK: scf.for
// CHECK-NOT: vector.transfer_read
// CHECK: ncnn.copy_kind = "fallback"
