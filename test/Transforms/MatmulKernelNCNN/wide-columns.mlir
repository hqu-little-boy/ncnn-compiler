// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// Wide-N f32 matmul must use the existing bounded column panels instead of
// being rejected by the elementwise row-width limit.  The tail is emitted as
// a narrow vector and no vector grows with the complete output width.
// CHECK-LABEL: func.func @wide_f32
// CHECK: vector.transfer_read {{.*}} : memref<8x2049xf32, strided<[2049, 1], offset: ?>>, vector<16xf32>
// CHECK: vector.transfer_read {{.*}} : memref<27x2049xf32>, vector<16xf32>
// CHECK: vector.transfer_read {{.*}} : memref<8x2049xf32, strided<[2049, 1], offset: ?>>, vector<1xf32>
// CHECK: vector.fma
// CHECK-NOT: vector<2049xf32>
// CHECK-NOT: linalg.matmul
module {
  func.func @wide_f32(%lhs: memref<64x27xf32>, %rhs: memref<27x2049xf32>,
                       %out: memref<64x2049xf32>) {
    scf.forall (%i) = (0) to (64) step (8) {
      %a = memref.subview %lhs[%i, 0][8, 27][1, 1]
        : memref<64x27xf32> to memref<8x27xf32, strided<[27, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][8, 2049][1, 1]
        : memref<64x2049xf32> to memref<8x2049xf32, strided<[2049, 1], offset: ?>>
      linalg.matmul ins(%a, %rhs : memref<8x27xf32, strided<[27, 1], offset: ?>>,
                                  memref<27x2049xf32>)
                    outs(%c : memref<8x2049xf32, strided<[2049, 1], offset: ?>>)
    }
    return
  }
}

// The int8 row-dot kernel has the same bounded-N property.  Its scalar stores
// are intentional: LLVM performs the accepted i8 widening/vectorization later.
// CHECK-LABEL: func.func @wide_i8
// CHECK: arith.extsi
// CHECK: arith.muli
// CHECK: memref.store {{.*}} : memref<8x2051xi32, strided<[2051, 1], offset: ?>>
// CHECK-NOT: vector<2051xi32>
// CHECK-NOT: linalg.matmul_transpose_b
module {
  func.func @wide_i8(%lhs: memref<64x27xi8>, %rhs: memref<2051x27xi8>,
                      %out: memref<64x2051xi32>) {
    scf.forall (%i) = (0) to (64) step (8) {
      %a = memref.subview %lhs[%i, 0][8, 27][1, 1]
        : memref<64x27xi8> to memref<8x27xi8, strided<[27, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][8, 2051][1, 1]
        : memref<64x2051xi32> to memref<8x2051xi32, strided<[2051, 1], offset: ?>>
      linalg.matmul_transpose_b
        ins(%a, %rhs : memref<8x27xi8, strided<[27, 1], offset: ?>>,
                       memref<2051x27xi8>)
        outs(%c : memref<8x2051xi32, strided<[2051, 1], offset: ?>>)
    }
    return
  }
}
