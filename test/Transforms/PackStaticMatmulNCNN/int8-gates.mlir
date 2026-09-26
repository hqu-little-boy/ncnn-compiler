// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=enabled=false int8-enabled=true' %s | FileCheck %s
// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=enabled=false int8-enabled=true pack-n=8' %s | FileCheck %s --check-prefix=N8

// int8 路径的两道守卫，与 f32 路径的口径**不同**，必须各自锁定：
//   * 小形状（rows < 8 或 depth < 32）→ packing_skipped_small_shape
//     （VNNI 内核的寄存器分块与 kpad64 面板都吃不消小形状）。rows/depth
//     取自 memref.global 的全局形状，不是 subview 的形状；
//   * pack-n != 16 → packing_rejected_layout
//     （f32 路径同样情形报的是 packing_rejected_budget，见
//      panel-width-guard.mlir）。两者的 reason 字符串都进 plan，不得漂移。

module {
  memref.global constant @small_rows : memref<4x36xi8> = dense<1>
  memref.global constant @short_k : memref<16x16xi8> = dense<1>
  memref.global constant @ok : memref<16x36xi8> = dense<1>

  // 全局 rows=4 < 8：小形状。
  func.func @small_row_count(%lhs: memref<4x36xi8>, %out: memref<4x4xi32>) {
    %rhs = memref.get_global @small_rows : memref<4x36xi8>
    %rhs_view = memref.subview %rhs[0, 0][4, 36][1, 1]
      : memref<4x36xi8> to memref<4x36xi8, strided<[36, 1]>>
    %out_view = memref.subview %out[0, 0][4, 4][1, 1]
      : memref<4x4xi32> to memref<4x4xi32, strided<[4, 1]>>
    linalg.matmul_transpose_b ins(%lhs, %rhs_view : memref<4x36xi8>,
                                                     memref<4x36xi8, strided<[36, 1]>>)
                              outs(%out_view : memref<4x4xi32, strided<[4, 1]>>)
    return
  }

  // 全局 depth=16 < 32：小形状（VNNI 以 32 为归约分块）。
  func.func @short_depth(%lhs: memref<4x16xi8>, %out: memref<4x16xi32>) {
    %rhs = memref.get_global @short_k : memref<16x16xi8>
    %rhs_view = memref.subview %rhs[0, 0][8, 16][1, 1]
      : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1]>>
    %out_view = memref.subview %out[0, 0][4, 8][1, 1]
      : memref<4x16xi32> to memref<4x8xi32, strided<[16, 1]>>
    linalg.matmul_transpose_b ins(%lhs, %rhs_view : memref<4x16xi8>,
                                                     memref<8x16xi8, strided<[16, 1]>>)
                              outs(%out_view : memref<4x8xi32, strided<[16, 1]>>)
    return
  }

  // 形状合规，只因 pack-n 被改。
  func.func @panel_width_guard(%lhs: memref<4x36xi8>, %out: memref<4x16xi32>) {
    %rhs = memref.get_global @ok : memref<16x36xi8>
    %rhs_view = memref.subview %rhs[0, 0][8, 36][1, 1]
      : memref<16x36xi8> to memref<8x36xi8, strided<[36, 1]>>
    %out_view = memref.subview %out[0, 0][4, 8][1, 1]
      : memref<4x16xi32> to memref<4x8xi32, strided<[16, 1]>>
    linalg.matmul_transpose_b ins(%lhs, %rhs_view : memref<4x36xi8>,
                                                     memref<8x36xi8, strided<[36, 1]>>)
                              outs(%out_view : memref<4x8xi32, strided<[16, 1]>>)
    return
  }
}

// CHECK-LABEL: func.func @small_row_count
// CHECK: linalg.matmul_transpose_b {ncnn.contract = "fallback"
// CHECK-SAME: ncnn.fallback_reason = "packing_skipped_small_shape"
// CHECK-SAME: ncnn.packing = "packing_skipped_small_shape"

// CHECK-LABEL: func.func @short_depth
// CHECK: linalg.matmul_transpose_b {ncnn.contract = "fallback"
// CHECK-SAME: ncnn.fallback_reason = "packing_skipped_small_shape"
// CHECK-SAME: ncnn.packing = "packing_skipped_small_shape"

// CHECK-LABEL: func.func @panel_width_guard
// CHECK: linalg.matmul_transpose_b
// CHECK-SAME: ncnn.pack_schema = "p23-int8-panel-row-kpad64-v1"
// CHECK-SAME: ncnn.packing = "prepacked_B"

// N8-LABEL: func.func @panel_width_guard
// N8: linalg.matmul_transpose_b {ncnn.contract = "fallback"
// N8-SAME: ncnn.fallback_reason = "packing_rejected_layout"
// N8-SAME: ncnn.packing = "packing_rejected_layout"
