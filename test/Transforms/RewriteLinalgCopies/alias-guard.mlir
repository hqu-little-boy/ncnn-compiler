// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" %s | FileCheck %s

// 别名守卫（负向）：向量化要求 hasProvenDisjointStorage——两侧必须能追到
// 不同的存储根。追不到根（函数块参数）或根相同（同一 alloc 的两个
// subview）时**不得**向量化，落 sequential fallback。

func.func @same_storage_root(%base: memref<8x10xf32>) {
  %source = memref.subview %base[0, 0][2, 10][1, 1] : memref<8x10xf32> to memref<2x10xf32, strided<[10, 1]>>
  %target = memref.subview %base[2, 0][2, 10][1, 1] : memref<8x10xf32> to memref<2x10xf32, strided<[10, 1], offset: 20>>
  linalg.copy ins(%source : memref<2x10xf32, strided<[10, 1]>>) outs(%target : memref<2x10xf32, strided<[10, 1], offset: 20>>)
  return
}

// CHECK-LABEL: func.func @same_storage_root
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vector.transfer_read
// CHECK: ncnn.copy_kind = "fallback"

// -----

// 块参数没有存储根，无法证明不相交。
func.func @block_arguments(%source: memref<2x10xf32>, %target: memref<2x10xf32>) {
  linalg.copy ins(%source : memref<2x10xf32>) outs(%target : memref<2x10xf32>)
  return
}

// CHECK-LABEL: func.func @block_arguments
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vector.transfer_read
// CHECK: ncnn.copy_kind = "fallback"

// -----

module {
  memref.global constant @shared : memref<2x10xf32> = dense<1.0>

  // 两次 get_global 指向同一个全局：根相同，不相交性不成立。
  func.func @same_global() {
    %source = memref.get_global @shared : memref<2x10xf32>
    %target = memref.get_global @shared : memref<2x10xf32>
    linalg.copy ins(%source : memref<2x10xf32>) outs(%target : memref<2x10xf32>)
    return
  }
}

// CHECK-LABEL: func.func @same_global
// CHECK-NOT: vector.transfer_read
// CHECK: ncnn.copy_kind = "fallback"
