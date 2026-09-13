// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// P6-C 批量收缩内核：[B,M,K]×[B,K,N]→[B,M,N] 改写为 forall(b) 网格 +
// 逐批 A1b 形态内核（M 行段 × N 列块 × K 单链 vector.fma，[b] 面板经
// subview + collapse 视图取得，零拷贝）。批维互不依赖，forall 提供
// OpenMP 并行；K 归约保持在单批内，累加顺序与逐批标量串行一致。
// MHA 的 scores/context（heads 批维）原走通用下降——标量 mul+add 且
// 逐 k 读写回 C，clang 无法向量化（medium_rec 实测 29% 热点）。

// CHECK-LABEL: func.func @batch
// CHECK: scf.forall {{.*}} in (8)
// CHECK: memref.subview
// CHECK: memref.collapse_shape
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} : memref<40x40xf32, strided<[40, 1], offset: ?>>, vector<16xf32>
// CHECK: scf.for {{.*}} iter_args
// CHECK: vector.transfer_read {{.*}} : memref<24x40xf32, strided<[40, 1], offset: ?>>, vector<16xf32>
// CHECK: memref.load
// CHECK: vector.broadcast
// CHECK: vector.fma
// CHECK: scf.yield
// CHECK-NOT: linalg.batch_matmul
module {
  func.func @batch(%a: memref<8x40x24xf32>, %b: memref<8x24x40xf32>,
                   %c: memref<8x40x40xf32>) {
    linalg.batch_matmul ins(%a, %b : memref<8x40x24xf32>, memref<8x24x40xf32>)
                        outs(%c : memref<8x40x40xf32>)
    return
  }
}

// -----

// 非静态形态守护：动态批数（无静态形状）不满足内核形态，保持原样。
// CHECK-LABEL: func.func @not_kernelizable
// CHECK: linalg.batch_matmul
module {
  func.func @not_kernelizable(%a: memref<?x40x24xf32>, %b: memref<?x24x40xf32>,
                              %c: memref<?x40x40xf32>) {
    linalg.batch_matmul ins(%a, %b : memref<?x40x24xf32>, memref<?x24x40xf32>)
                        outs(%c : memref<?x40x40xf32>)
    return
  }
}

// -----

// Static MHA score/context shapes can have a wide sequence dimension.  The
// batch kernel must keep the same bounded N panels as ordinary matmul.
// CHECK-LABEL: func.func @batch_wide
// CHECK: scf.forall {{.*}} in (8)
// CHECK: vector.transfer_read
// CHECK: vector<16xf32>
// CHECK: vector<1xf32>
// CHECK-NOT: vector<2049xf32>
// CHECK-NOT: linalg.batch_matmul
module {
  func.func @batch_wide(%a: memref<8x40x24xf32>, %b: memref<8x24x2049xf32>,
                        %c: memref<8x40x2049xf32>) {
    linalg.batch_matmul ins(%a, %b : memref<8x40x24xf32>,
                                    memref<8x24x2049xf32>)
                        outs(%c : memref<8x40x2049xf32>)
    return
  }
}
