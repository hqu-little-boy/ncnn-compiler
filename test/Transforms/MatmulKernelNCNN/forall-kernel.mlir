// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// forall 内的静态 memref matmul 改写为显式向量 FMA 内核：K 循环读 B 行
// 向量，M 循环 broadcast(A) · B 行 vector.fma 累加进 C 行（单舍入，
// 替代会阻止 LLVM 合成 vfmadd 的 mulf+addf 分离形态）。

// CHECK-LABEL: scf.forall
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} : memref<9x32xf32, strided<[32, 1], offset: ?>>, vector<32xf32>
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args
// CHECK: memref.load
// CHECK: vector.broadcast
// CHECK: vector.transfer_read {{.*}} : memref<27x32xf32>, vector<32xf32>
// CHECK: vector.fma
// CHECK-NOT: arith.mulf
// CHECK: scf.yield
// CHECK: vector.transfer_write {{.*}} : vector<32xf32>, memref<9x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NOT: linalg.matmul
module {
  func.func @tiled(%im2col: memref<103041x27xf32>, %weight: memref<27x32xf32>,
                   %out: memref<103041x32xf32>) {
    scf.forall (%i) = (0) to (103041) step (9) {
      %a = memref.subview %im2col[%i, 0][9, 27][1, 1]
        : memref<103041x27xf32> to memref<9x27xf32, strided<[27, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][9, 32][1, 1]
        : memref<103041x32xf32> to memref<9x32xf32, strided<[32, 1], offset: ?>>
      linalg.matmul ins(%a, %weight : memref<9x27xf32, strided<[27, 1], offset: ?>>,
                                   memref<27x32xf32>)
                    outs(%c : memref<9x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}
