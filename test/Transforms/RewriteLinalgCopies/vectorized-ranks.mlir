// RUN: ncnn-mlir-opt --rewrite-linalg-copies="vectorize=true vector-lanes=4" %s | FileCheck %s

// 向量化主块/尾块：最内维按 lane 数分块，主块走 vector.transfer_read/write，
// 尾块走逐元素 memref.load/store。外层维各起一层 scf.for，最内维的主块
// 与尾块在同一拷贝契约下（都标 ncnn.copy_kind = "vectorized"）。

func.func @rank1_tail() {
  %source = memref.alloc() : memref<10xf32>
  %target = memref.alloc() : memref<10xf32>
  linalg.copy ins(%source : memref<10xf32>) outs(%target : memref<10xf32>)
  memref.dealloc %source : memref<10xf32>
  memref.dealloc %target : memref<10xf32>
  return
}

// CHECK-LABEL: func.func @rank1_tail
// CHECK: scf.for
// CHECK: vector.transfer_read %{{.*}}: memref<10xf32>, vector<4xf32>
// CHECK: vector.transfer_write %{{.*}}: vector<4xf32>, memref<10xf32>
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: ncnn.copy_kind = "vectorized", ncnn.copy_vector_lanes = 4

// -----

func.func @rank2_exactly_one_vector() {
  %source = memref.alloc() : memref<2x4xf32>
  %target = memref.alloc() : memref<2x4xf32>
  linalg.copy ins(%source : memref<2x4xf32>) outs(%target : memref<2x4xf32>)
  memref.dealloc %source : memref<2x4xf32>
  memref.dealloc %target : memref<2x4xf32>
  return
}

// CHECK-LABEL: func.func @rank2_exactly_one_vector
// CHECK: vector.transfer_read %{{.*}}: memref<2x4xf32>, vector<4xf32>
// CHECK: ncnn.copy_kind = "vectorized", ncnn.copy_vector_lanes = 4

// -----

func.func @rank3() {
  %source = memref.alloc() : memref<2x3x4xf32>
  %target = memref.alloc() : memref<2x3x4xf32>
  linalg.copy ins(%source : memref<2x3x4xf32>) outs(%target : memref<2x3x4xf32>)
  memref.dealloc %source : memref<2x3x4xf32>
  memref.dealloc %target : memref<2x3x4xf32>
  return
}

// CHECK-LABEL: func.func @rank3
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read %{{.*}}: memref<2x3x4xf32>, vector<4xf32>
// CHECK: vector.transfer_write %{{.*}}: vector<4xf32>, memref<2x3x4xf32>
// CHECK: ncnn.copy_kind = "vectorized", ncnn.copy_vector_lanes = 4
