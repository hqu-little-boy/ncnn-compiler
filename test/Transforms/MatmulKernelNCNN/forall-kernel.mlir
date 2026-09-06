// RUN: ncnn-mlir-opt --matmul-kernel-ncnn %s | FileCheck %s
// RUN: ncnn-mlir-opt --matmul-kernel-ncnn="matmul-m-rows=1" %s | FileCheck %s --check-prefix=SINGLE

// forall 内的静态 memref matmul 改写为显式向量 FMA 内核（P3 M×N 寄存器
// 分块）：M 方向 tileRows 行 accumulator 驻留寄存器，K 循环每轮只读一次
// B 行向量，与各行 A 标量的 broadcast 做 vector.fma 独立累加（单舍入，
// 替代会阻止 LLVM 合成 vfmadd 的 mulf+addf 分离形态）；n 块循环按
// accColumns 步距扫 N。默认 tileRows=4、accColumns=16，寄存器预算
// 4×16=64 浮点（见 MatmulKernelNCNN.cpp kAccumulatorFloatBudget）；
// matmul-m-rows=1 退回历史单行内核供 A/B 对照。

// 9x32 tile：m 满块 2 组（步 4）+ 余数 1 行；n 满块 2 块（步 16）无尾块。
// CHECK-LABEL: scf.forall
// CHECK: scf.for
// CHECK: scf.for
// CHECK-COUNT-4: vector.transfer_read {{.*}} : memref<9x32xf32, strided<[32, 1], offset: ?>>, vector<16xf32>
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args({{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}})
// CHECK: vector.transfer_read {{.*}} : memref<27x32xf32>, vector<16xf32>
// CHECK: memref.load
// CHECK: vector.broadcast
// CHECK-COUNT-4: vector.fma
// CHECK-NOT: arith.mulf
// CHECK: scf.yield
// CHECK-COUNT-4: vector.transfer_write {{.*}} : vector<16xf32>, memref<9x32xf32, strided<[32, 1], offset: ?>>
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} : memref<9x32xf32, strided<[32, 1], offset: ?>>, vector<16xf32>
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args
// CHECK: memref.load
// CHECK: vector.fma
// CHECK: vector.transfer_write {{.*}} : vector<16xf32>, memref<9x32xf32, strided<[32, 1], offset: ?>>
// CHECK-NOT: vector<32xf32>
// CHECK-NOT: linalg.matmul
module {
  func.func @tiled(%im2col: memref<103041x27xf32>, %weight: memref<27x32xf32>,
                   %out: memref<103041x32xf32>) {
    scf.forall (%i) = (0) to (103041) step (9) {
      %a = memref.subview %im2col[%i, 0][9, 27][1, 1]
        : memref<103041x27xf32> to memref<9x27xf32, strided<[27, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][9, 32][1, 1]
        : memref<103041x32xf32> to memref<9x32xf32, strided<[32, 1], offset: ?>>
      linalg.matmul ins(%a, %weight : memref<9x27xf32, strided<[27, 1], offset: ?>>,
                                   memref<27x32xf32>)
                    outs(%c : memref<9x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}

// -----

// 9x24 tile：n 满块 1 块（16）+ 列尾块 8——尾块仍以窄向量走满 tileRows
// 行 accumulator（4×8=32 浮点，预算内）。
// CHECK-LABEL: scf.forall
// CHECK: scf.for
// CHECK: vector.transfer_read {{.*}} : memref<9x24xf32, strided<[24, 1], offset: ?>>, vector<16xf32>
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args({{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}})
// CHECK: vector.fma
// CHECK-COUNT-4: vector.transfer_write {{.*}} : vector<16xf32>, memref<9x24xf32, strided<[24, 1], offset: ?>>
// CHECK: vector.transfer_read {{.*}} : memref<9x24xf32, strided<[24, 1], offset: ?>>, vector<8xf32>
// CHECK: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args({{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}}, {{%[^)]*}})
// CHECK: vector.transfer_read {{.*}} : memref<27x24xf32>, vector<8xf32>
// CHECK-COUNT-4: vector.fma
// CHECK-COUNT-4: vector.transfer_write {{.*}} : vector<8xf32>, memref<9x24xf32, strided<[24, 1], offset: ?>>
// CHECK-NOT: linalg.matmul
module {
  func.func @tiled_tail(%im2col: memref<103041x27xf32>, %weight: memref<27x24xf32>,
                        %out: memref<103041x24xf32>) {
    scf.forall (%i) = (0) to (103041) step (9) {
      %a = memref.subview %im2col[%i, 0][9, 27][1, 1]
        : memref<103041x27xf32> to memref<9x27xf32, strided<[27, 1], offset: ?>>
      %c = memref.subview %out[%i, 0][9, 24][1, 1]
        : memref<103041x24xf32> to memref<9x24xf32, strided<[24, 1], offset: ?>>
      linalg.matmul ins(%a, %weight : memref<9x27xf32, strided<[27, 1], offset: ?>>,
                                   memref<27x24xf32>)
                    outs(%c : memref<9x24xf32, strided<[24, 1], offset: ?>>)
    }
    return
  }
}

// SINGLE-LABEL: scf.forall
// SINGLE: scf.for
// SINGLE: scf.for
// SINGLE: vector.transfer_read {{.*}} : memref<9x32xf32, strided<[32, 1], offset: ?>>, vector<16xf32>
// SINGLE: scf.for {{%[^ ]*}} = {{%[^ ]*}} to {{%[^ ]*}} step {{%[^ ]*}} iter_args
// SINGLE: memref.load
// SINGLE: vector.broadcast
// SINGLE: vector.fma
// SINGLE: vector.transfer_write {{.*}} : vector<16xf32>, memref<9x32xf32, strided<[32, 1], offset: ?>>
// SINGLE-NOT: linalg.matmul
