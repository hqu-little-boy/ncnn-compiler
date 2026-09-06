// P3 隔离 matmul bench（A1b 内核微基准，docs/ncnn-performance-parity-plan.md
// §3-P3）：两个 forall 分块 matmul——浅 K（M=1024, K=576, N=64，
// resnet18 layer1 卷积 GEMM 的代表形状）与深 K（M=256, K=2304, N=256，
// resnet18 layer3 同型）——不含 im2col 与 epilogue，度量纯内核质量。
// 驱动见 support/matmul_bench_main.c；tile 尺寸 32x32 与
// tile-matmul-forall 的默认分块一致。
module {
  func.func @bench_shallow(
    %im2col: memref<1024x576xf32>, %weight: memref<576x64xf32>,
    %out: memref<1024x64xf32>) attributes {llvm.emit_c_interface} {
    scf.forall (%i, %j) = (0, 0) to (1024, 64) step (32, 32) {
      %a = memref.subview %im2col[%i, 0][32, 576][1, 1]
        : memref<1024x576xf32> to memref<32x576xf32, strided<[576, 1], offset: ?>>
      %w = memref.subview %weight[0, %j][576, 32][1, 1]
        : memref<576x64xf32> to memref<576x32xf32, strided<[64, 1], offset: ?>>
      %c = memref.subview %out[%i, %j][32, 32][1, 1]
        : memref<1024x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
      linalg.matmul ins(%a, %w : memref<32x576xf32, strided<[576, 1], offset: ?>>,
                                   memref<576x32xf32, strided<[64, 1], offset: ?>>)
                    outs(%c : memref<32x32xf32, strided<[64, 1], offset: ?>>)
    }
    return
  }

  func.func @bench_deep(
    %im2col: memref<256x2304xf32>, %weight: memref<2304x256xf32>,
    %out: memref<256x256xf32>) attributes {llvm.emit_c_interface} {
    scf.forall (%i, %j) = (0, 0) to (256, 256) step (32, 32) {
      %a = memref.subview %im2col[%i, 0][32, 2304][1, 1]
        : memref<256x2304xf32> to memref<32x2304xf32, strided<[2304, 1], offset: ?>>
      %w = memref.subview %weight[0, %j][2304, 32][1, 1]
        : memref<2304x256xf32> to memref<2304x32xf32, strided<[256, 1], offset: ?>>
      %c = memref.subview %out[%i, %j][32, 32][1, 1]
        : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
      linalg.matmul ins(%a, %w : memref<32x2304xf32, strided<[2304, 1], offset: ?>>,
                                   memref<2304x32xf32, strided<[256, 1], offset: ?>>)
                    outs(%c : memref<32x32xf32, strided<[256, 1], offset: ?>>)
    }
    return
  }
}
