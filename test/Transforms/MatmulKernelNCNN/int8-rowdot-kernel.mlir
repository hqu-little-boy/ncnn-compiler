// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s

// forall 内的静态 memref matmul_transpose_b（i8×i8→i32，P4）改写为
// row-dot 内核：每个 (m,n) 输出一条标量 i32-MAC 累加链（iter_args 贯穿
// K 循环，初值 = C 初始化，语义 C += A·B），链数 = tileRows×accColumns
// （默认 2×4，寄存器预算 8，见 MatmulKernelNCNN.cpp
// kInt8AccumulatorBudget）。发射刻意保持标量形态——clang -O3 的循环向
// 量化器对其稳定生成 vpmovsxbw + vpmaddwd（ncnn AVX2 int8 内核同款）。

// 8x8 tile：m 满块 4 组（步 2）×n 满块 2 块（步 4），K=72。
// CHECK-LABEL: scf.forall
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args({{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}})
// 每轮每行一个 A 字节、每列一个 B 字节，sext 到 i32 后 muli/addi。
// CHECK: memref.load {{.*}} : memref<8x72xi8, strided<[72, 1], offset: ?>>
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: memref.load {{.*}} : memref<8x72xi8>
// CHECK-COUNT-8: arith.muli {{.*}} : i32
// CHECK: scf.yield
// CHECK-COUNT-8: memref.store {{.*}} : memref<8x8xi32, strided<[8, 1], offset: ?>>
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

// K=6：同一形态，链长 6（LV 按静态 trip 自主决策向量化宽度）。
// CHECK-LABEL: func.func @tiled_i8_shallow
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args
// CHECK: memref.load {{.*}} : memref<8x6xi8, strided<[6, 1], offset: ?>>
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: arith.muli {{.*}} : i32
// CHECK: memref.store {{.*}} : memref<8x8xi32, strided<[8, 1], offset: ?>>
// CHECK-NOT: linalg.matmul
module {
  func.func @tiled_i8_shallow(%im2col: memref<4096x6xi8>, %weight: memref<8x6xi8>,
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
