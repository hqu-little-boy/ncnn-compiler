// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" %s | FileCheck %s

// 同值消除：source == target 的拷贝是空操作，直接擦除（copy_kind =
// "eliminated"），不生成任何循环。memref.copy 与 linalg.copy 两条入口都走
// lowerConcreteCopy 的同一分支；linalg.copy 另有 EliminateLinalgCopy 先行。

func.func @same_value() {
  %target = memref.alloc() : memref<2x10xf32>
  memref.copy %target, %target : memref<2x10xf32> to memref<2x10xf32>
  linalg.copy ins(%target : memref<2x10xf32>) outs(%target : memref<2x10xf32>)
  memref.dealloc %target : memref<2x10xf32>
  return
}

// CHECK-LABEL: func.func @same_value
// CHECK-NOT: memref.copy
// CHECK-NOT: linalg.copy
// CHECK-NOT: scf.for
// CHECK: return

// -----

// 唯一结果被用掉时同样消除，只是结果值换名。
func.func @same_value_used() -> memref<2x10xf32> {
  %target = memref.alloc() : memref<2x10xf32>
  linalg.copy ins(%target : memref<2x10xf32>) outs(%target : memref<2x10xf32>)
  return %target : memref<2x10xf32>
}

// CHECK-LABEL: func.func @same_value_used
// CHECK-NOT: linalg.copy
// CHECK: return
