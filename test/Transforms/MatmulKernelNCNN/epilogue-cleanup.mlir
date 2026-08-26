// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// B4 清理：①恒等自拷贝嵌套删除；②顶层 relu 型 memref generic 行向量
// 化（rank-4、最内维 32）。

module {
  func.func @tile(%im2col: memref<64x16xf32>, %w: memref<16x8xf32>,
                  %acc: memref<64x8xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.forall (%i) = (0) to (64) step (16) {
      %a = memref.subview %im2col[%i, 0][16, 16][1, 1]
        : memref<64x16xf32> to memref<16x16xf32, strided<[16, 1], offset: ?>>
      %c = memref.subview %acc[%i, 0][16, 8][1, 1]
        : memref<64x8xf32> to memref<16x8xf32, strided<[8, 1], offset: ?>>
      linalg.matmul ins(%a, %w : memref<16x16xf32, strided<[16, 1], offset: ?>>,
                                   memref<16x8xf32>)
                    outs(%c : memref<16x8xf32, strided<[8, 1], offset: ?>>)
      scf.for %r = %c0 to %c16 step %c1 {
        scf.for %q = %c0 to %c8 step %c1 {
          %v = memref.load %c[%r, %q]
            : memref<16x8xf32, strided<[8, 1], offset: ?>>
          memref.store %v, %c[%r, %q]
            : memref<16x8xf32, strided<[8, 1], offset: ?>>
        }
      }
    }
    return
  }

  func.func @relu(%in: memref<1x320x320x32xf32>, %out: memref<1x320x320x32xf32>) {
    %zero = arith.constant 0.0 : f32
    linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                       affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%in : memref<1x320x320x32xf32>)
      outs(%out : memref<1x320x320x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = arith.maximumf %x, %zero : f32
      linalg.yield %v : f32
    }
    return
  }
}

// 恒等自拷贝被整体删除。
// CHECK: vector.transfer_write {{.*}} : vector<8xf32>, memref<16x8xf32, strided<[8, 1], offset: ?>>
// 自拷贝嵌套被整体删除：forall 内不再存在标量 load/store 对。
// CHECK-NOT: memref.store

// relu generic 行向量化为三层循环 + 整行 transfer。
// CHECK-LABEL: func.func @relu
// CHECK-COUNT-3: scf.for
// CHECK: vector.transfer_read {{.*}} : memref<1x320x320x32xf32>, vector<32xf32>
// CHECK: arith.maximumf
// CHECK: vector.transfer_write {{.*}} : vector<32xf32>, memref<1x320x320x32xf32>
// CHECK-NOT: linalg.generic
