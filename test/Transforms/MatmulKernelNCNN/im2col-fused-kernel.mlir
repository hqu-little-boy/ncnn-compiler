// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// P6 im2col 物化消除：forall 内 matmul 的 A 面板为
// collapse(gather-generic(alloc)) 生产链时，内核直取源图窗口——
// gather/collapse/alloc 三件套整体删除，K 循环拆为外层 kp（窗口位置
// kh·KW+kw）+ 内层 ic（通道）两层静态界循环，k = kp·IC + ic 与折叠
// [[0,1],[2,3,4]] 展平一致；窗口行/列 = sh·oh + dh·kh、sw·ow + dw·kw。
// 这里 OH=3、OW=3、KH=2、KW=2、IC=4、sh=2、dh=1（sw/dw 同理用
// d1*2+d3 与 0 系数区分），M=9、K=16。

// CHECK-LABEL: func.func @fused
// CHECK-NOT: linalg.generic
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.parallel
// 内核主体：M 行段循环 + 外层 kp（窗口位置 kh·KW+kw，divsi/remsi 拆解）
// + 内层 ic（通道）双层 K 循环；A 标量按窗口映射直取源图 %padded
// （1x7x7x4），B 行按 k = kp·IC + ic 读权重面板。
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} : memref<9x8xf32>, vector<8xf32>
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK: memref.load %{{[a-zA-Z0-9_]+}}[%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}] : memref<1x7x7x4xf32>
// CHECK: vector.fma
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: return
module {
  func.func @fused(%padded: memref<1x7x7x4xf32>, %weight: memref<16x8xf32>,
                   %out: memref<9x8xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3x2x2x4xf32>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0,d1,d2,d3,d4) -> (0, d0*2+d2, d1*2+d3, d4)>,
        affine_map<(d0,d1,d2,d3,d4) -> (d0,d1,d2,d3,d4)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%padded : memref<1x7x7x4xf32>)
      outs(%alloc : memref<3x3x2x2x4xf32>) {
    ^bb0(%in: f32, %out_: f32):
      linalg.yield %in : f32
    }
    %collapsed = memref.collapse_shape %alloc [[0, 1], [2, 3, 4]]
      : memref<3x3x2x2x4xf32> into memref<9x16xf32>
    %zero = arith.constant 0 : index
    %nine = arith.constant 9 : index
    %one = arith.constant 1 : index
    scf.forall (%i) = (0) to (9) step (9) {
      linalg.matmul ins(%collapsed, %weight : memref<9x16xf32>,
                                    memref<16x8xf32>)
                    outs(%out : memref<9x8xf32>)
    }
    return
  }
}

// -----

// 非融合守护：collapse 有第二个消费者（memref.copy）时保持常规形态
// ——im2col 物化照旧、gather 走向量化窗口拷贝、内核从物化缓冲读 A。
// CHECK-LABEL: func.func @not_fused
// CHECK: scf.parallel
// CHECK: memref.collapse_shape
// CHECK: memref.load %collapse_shape
// CHECK: vector.fma
module {
  func.func @not_fused(%padded: memref<1x7x7x4xf32>, %weight: memref<16x8xf32>,
                       %out: memref<9x8xf32>, %extra: memref<9x16xf32>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x3x2x2x4xf32>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0,d1,d2,d3,d4) -> (0, d0*2+d2, d1*2+d3, d4)>,
        affine_map<(d0,d1,d2,d3,d4) -> (d0,d1,d2,d3,d4)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%padded : memref<1x7x7x4xf32>)
      outs(%alloc : memref<3x3x2x2x4xf32>) {
    ^bb0(%in: f32, %out_: f32):
      linalg.yield %in : f32
    }
    %collapsed = memref.collapse_shape %alloc [[0, 1], [2, 3, 4]]
      : memref<3x3x2x2x4xf32> into memref<9x16xf32>
    memref.copy %collapsed, %extra : memref<9x16xf32> to memref<9x16xf32>
    scf.forall (%i) = (0) to (9) step (9) {
      linalg.matmul ins(%collapsed, %weight : memref<9x16xf32>,
                                    memref<16x8xf32>)
                    outs(%out : memref<9x8xf32>)
    }
    return
  }
}
