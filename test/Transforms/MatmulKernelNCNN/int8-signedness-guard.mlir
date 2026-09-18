// RUN: ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=vnni int8-target=avx-vnni' --split-input-file %s | FileCheck %s
//
// P16 signedness 守卫：cast 属性或自定义 region 改变扩宽 signedness 时，
// int8 row-dot / vpdpbusd 内核绝不改写（保持 named op 通用下降）；
// canonical signed named-op（空 region）仍照常进入 VNNI 内核。

// cast 属性改变扩宽语义：0xff×1 = 8160 而非 signed 的 -32，必须拒绝内核化。
// CHECK-LABEL: func.func @unsigned_cast_keeps_named_op
// CHECK-NOT: llvm.call @llvm.x86.avx512.vpdpbusd.256
// CHECK: linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_unsigned>}
// CHECK-NOT: ncnn.kernel
// CHECK: return
module {
  func.func @unsigned_cast_keeps_named_op(%a: memref<2x33xi8>, %b: memref<2x33xi8>, %c: memref<2x2xi32>) {
    scf.forall (%tile) in (1) {
      linalg.matmul_transpose_b {cast = #linalg.type_fn<cast_unsigned>}
        ins(%a, %b : memref<2x33xi8>, memref<2x33xi8>) outs(%c : memref<2x2xi32>)
    }
    return
  }
}

// -----

// 自定义 region 用 extui（unsigned 扩宽）：同样不匹配 signed MAC 语义。
// named op 的自定义 region 只能用 generic 引号语法书写；printer 会把它
// 回显为 pretty 形式（不带引号）。
// CHECK-LABEL: func.func @extui_body_keeps_named_op
// CHECK-NOT: llvm.call @llvm.x86.avx512.vpdpbusd.256
// CHECK: linalg.matmul_transpose_b {indexing_maps
// CHECK-NOT: ncnn.kernel
// CHECK: return
module {
  func.func @extui_body_keeps_named_op(%a: memref<2x33xi8>, %b: memref<2x33xi8>, %c: memref<2x2xi32>) {
    scf.forall (%tile) in (1) {
      "linalg.matmul_transpose_b" (%a, %b, %c) ({
      ^bb0(%x: i8, %y: i8, %out: i32):
        %0 = arith.extui %x : i8 to i32
        %1 = arith.extui %y : i8 to i32
        %2 = arith.muli %0, %1 : i32
        %3 = arith.addi %out, %2 : i32
        linalg.yield %3 : i32
      }) {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], operandSegmentSizes = array<i32: 2, 1>} : (memref<2x33xi8>, memref<2x33xi8>, memref<2x2xi32>) -> ()
    }
    return
  }
}

// -----

// canonical signed named-op（空 region）：仍照常进入 VNNI 内核。
// CHECK-LABEL: func.func @canonical_signed_selected
// CHECK: llvm.call @llvm.x86.avx512.vpdpbusd.256
// CHECK: ncnn.kernel = "int8_vnni_row_dot"
module {
  func.func @canonical_signed_selected(%a: memref<2x33xi8>, %b: memref<2x33xi8>, %c: memref<2x2xi32>) {
    scf.forall (%tile) in (1) {
      linalg.matmul_transpose_b
        ins(%a, %b : memref<2x33xi8>, memref<2x33xi8>) outs(%c : memref<2x2xi32>)
    }
    return
  }
}
