// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// im2col gather 识别与向量化：窗口映射 (0, sh*d0+dh*d2, sw*d1+dw*d3, d4)
// 的纯转发拷贝改写为 scf.parallel + 整 IC 行 transfer_read/write——
// 并行语义保留，行内为 SIMD 拷贝。

module {
  func.func @im2col(%padded: memref<1x7x7x8xf32>,
                    %out: memref<3x3x2x2x8xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0,d1,d2,d3,d4) -> (0, d0*2+d2, d1*2+d3, d4)>,
        affine_map<(d0,d1,d2,d3,d4) -> (d0,d1,d2,d3,d4)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%padded : memref<1x7x7x8xf32>)
      outs(%out : memref<3x3x2x2x8xf32>) {
    ^bb0(%in: f32, %out_: f32):
      linalg.yield %in : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @im2col
// CHECK: scf.parallel
// CHECK: vector.transfer_read {{.*}} : memref<1x7x7x8xf32>, vector<8xf32>
// CHECK: vector.transfer_write {{.*}} : vector<8xf32>, memref<3x3x2x2x8xf32>
// CHECK-NOT: linalg.generic
