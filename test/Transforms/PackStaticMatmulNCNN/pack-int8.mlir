// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=enabled=false int8-enabled=true' %s | FileCheck %s

module {
  memref.global constant @int8_weights : memref<16x36xi8> = dense<1>

  func.func @packs_static_int8_rhs(%lhs: memref<4x36xi8>,
                                   %out: memref<4x16xi32>) {
    %rhs = memref.get_global @int8_weights : memref<16x36xi8>
    %panel_width = arith.constant 8 : index
    scf.forall (%tile) in (2) {
      %column = arith.muli %tile, %panel_width : index
      %rhs_view = memref.subview %rhs[%column, 0][8, 36][1, 1]
        : memref<16x36xi8> to memref<8x36xi8, strided<[36, 1], offset: ?>>
      %out_view = memref.subview %out[0, %column][4, 8][1, 1]
        : memref<4x16xi32> to memref<4x8xi32, strided<[16, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%lhs, %rhs_view : memref<4x36xi8>,
                                                        memref<8x36xi8, strided<[36, 1], offset: ?>>)
                                outs(%out_view : memref<4x8xi32, strided<[16, 1], offset: ?>>)
    }
    return
  }

  func.func @rejects_nonconstant_int8_rhs(%lhs: memref<4x36xi8>,
                                          %rhs: memref<16x36xi8>,
                                          %out: memref<4x16xi32>) {
    linalg.matmul_transpose_b ins(%lhs, %rhs : memref<4x36xi8>, memref<16x36xi8>)
                              outs(%out : memref<4x16xi32>)
    return
  }
}

// CHECK: memref.global "private" constant @int8_weights_p23_int8_kpad64 : memref<1024xi8> = dense<
// CHECK-LABEL: func.func @packs_static_int8_rhs
// CHECK: %[[PACKED:.*]] = memref.get_global {{.*}}@int8_weights_p23_int8_kpad64{{.*}}memref<1024xi8>
// CHECK: memref.reinterpret_cast %[[PACKED]] to offset: [0], sizes: [16, 36], strides: [64, 1]
// CHECK: linalg.matmul_transpose_b {{.*}}ncnn.pack_bytes = 1024 : i64{{.*}}ncnn.pack_raw_bytes = 576 : i64{{.*}}ncnn.pack_schema = "p23-int8-panel-row-kpad64-v1"{{.*}}ncnn.packing = "prepacked_B"{{.*}}ncnn.weight_layout = "panel_nk_kpad64"
// CHECK-LABEL: func.func @rejects_nonconstant_int8_rhs
// CHECK: linalg.matmul_transpose_b {ncnn.contract = "fallback"
// CHECK-SAME: ncnn.fallback_reason = "packing_rejected_nonconstant_rhs"
// CHECK-SAME: ncnn.packing = "packing_rejected_nonconstant_rhs"
