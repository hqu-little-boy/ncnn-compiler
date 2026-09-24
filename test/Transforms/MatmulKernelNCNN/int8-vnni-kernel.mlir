// RUN: ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=vnni int8-target=avx-vnni' %s | FileCheck %s
//
// P16：目标 ISA 感知的 INT8 row-dot。avx-vnni 能力 + vnni policy 下，
// K 满块由 256-bit vpdpbusd（unsigned A xor 0x80 × signed B，i32 部分和
// 再减 128*Σb 修正）承接，K 标量尾与 portable 路径共用；kernel 契约
// 标记 int8_vnni_row_dot 与 int8_isa，portable 拒绝路径见
// int8-rowdot-kernel.mlir 与 int8-vnni-fallback.mlir。

// CHECK-LABEL: scf.forall
// CHECK: scf.for
// CHECK: vector.load {{.*}} : memref<8x72xi8, strided<[72, 1], offset: ?>>, vector<32xi8>
// CHECK: arith.xori {{.*}} : vector<32xi8>
// CHECK: llvm.call @llvm.x86.avx512.vpdpbusd.256
// CHECK-COUNT-8: memref.store {{.*}} : memref<8x8xi32, strided<[8, 1], offset: ?>>
// CHECK: ncnn.contract = "selected"
// CHECK: ncnn.fma = "vpdpbusd_signed_correction"
// CHECK: ncnn.int8_backend = "vpdpbusd_256_u8s8_signed_mac"{{.*}}ncnn.int8_intrinsic = "llvm.x86.avx512.vpdpbusd.256"{{.*}}ncnn.int8_isa = "avx-vnni"{{.*}}ncnn.int8_k_alignment = 32 : i64{{.*}}ncnn.int8_reduction_tail = 8 : i64{{.*}}ncnn.int8_required_isa = "avx-vnni"{{.*}}ncnn.int8_signedness_correction = "xor_a_signbit_then_subtract_128_sum_b"
// CHECK: ncnn.kernel = "int8_vnni_row_dot"
// CHECK-NOT: linalg.matmul
module {
  func.func @tiled_i8(%im2col: memref<4096x72xi8>, %weight: memref<8x72xi8>,
                      %out: memref<4096x8xi32>) {
    scf.forall (%i) = (0) to (4096) step (8) {
      %a = memref.subview %im2col[%i, 0][8, 72][1, 1]
        : memref<4096x72xi8> to memref<8x72xi8, strided<[72, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][8, 8][1, 1]
        : memref<4096x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%a, %weight : memref<8x72xi8, strided<[72, 1], offset: ?>>,
                                              memref<8x72xi8>)
                                outs(%c : memref<8x8xi32, strided<[8, 1], offset: ?>>)
    }
    return
  }
}

// -----

// K=6 < 32：VNNI 满块不适用，整体回退 portable 标量 MAC 形态。
// CHECK-LABEL: func.func @tiled_i8_shallow_vnni_request
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: ncnn.fallback_reason = "reduction_k_lt_32"
// CHECK-NOT: llvm.call @llvm.x86.avx512.vpdpbusd.256
module {
  func.func @tiled_i8_shallow_vnni_request(%im2col: memref<4096x6xi8>,
                                           %weight: memref<8x6xi8>,
                                           %out: memref<4096x8xi32>) {
    scf.forall (%i) = (0) to (4096) step (8) {
      %a = memref.subview %im2col[%i, 0][8, 6][1, 1]
        : memref<4096x6xi8> to memref<8x6xi8, strided<[6, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][8, 8][1, 1]
        : memref<4096x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%a, %weight : memref<8x6xi8, strided<[6, 1], offset: ?>>,
                                              memref<8x6xi8>)
                                outs(%c : memref<8x8xi32, strided<[8, 1], offset: ?>>)
    }
    return
  }
}

// ----

// A row-padded RHS still exposes contiguous K lanes to the VNNI vector load.
// CHECK-LABEL: func.func @tiled_i8_padded_rhs
// CHECK: vector.load {{.*}} : memref<8x72xi8, strided<[128, 1]
// CHECK: llvm.call @llvm.x86.avx512.vpdpbusd.256
// CHECK: ncnn.kernel = "int8_vnni_row_dot"
module {
  func.func @tiled_i8_padded_rhs(%im2col: memref<4096x72xi8>,
                                 %weight: memref<8x72xi8, strided<[128, 1]>>,
                                 %out: memref<4096x8xi32>) {
    scf.forall (%i) = (0) to (4096) step (8) {
      %a = memref.subview %im2col[%i, 0][8, 72][1, 1]
        : memref<4096x72xi8> to memref<8x72xi8, strided<[72, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][8, 8][1, 1]
        : memref<4096x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%a, %weight : memref<8x72xi8, strided<[72, 1], offset: ?>>,
                                              memref<8x72xi8, strided<[128, 1]>>)
                                outs(%c : memref<8x8xi32, strided<[8, 1], offset: ?>>)
    }
    return
  }
}
