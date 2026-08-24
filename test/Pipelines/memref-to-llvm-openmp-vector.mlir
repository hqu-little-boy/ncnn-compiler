// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4 vector-lowering=true' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4 vector-lowering=true' %s | FileCheck %s --check-prefix=LOWERED

// A3 目标形态：bufferized scf.forall（外层线程并行）× 内层 rank-1 向量行。
// forall 先经 scf-forall-to-parallel 归一为 scf.parallel，再统一进 OpenMP
// 转换；向量下降在其后处理 omp 区域内的 vector op。

module {
  func.func @rows_vec(%in: memref<64x128xf32>, %out: memref<64x128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    scf.forall (%i) in (%c64) {
      %lb = arith.muli %i, %c8 : index
      %sin = memref.subview %in[%lb, 0][8, 128][1, 1] : memref<64x128xf32> to memref<8x128xf32, strided<[128, 1], offset: ?>>
      %sout = memref.subview %out[%lb, 0][8, 128][1, 1] : memref<64x128xf32> to memref<8x128xf32, strided<[128, 1], offset: ?>>
      scf.for %j = %c0 to %c8 step %c1 {
        %v = vector.transfer_read %sin[%j, %c0], %pad {in_bounds = [true]} : memref<8x128xf32, strided<[128, 1], offset: ?>>, vector<128xf32>
        %w = arith.negf %v : vector<128xf32>
        vector.transfer_write %w, %sout[%j, %c0] {in_bounds = [true]} : vector<128xf32>, memref<8x128xf32, strided<[128, 1], offset: ?>>
      }
    }
    return
  }
}

// PIPELINE: convert-scf-to-openmp{{.*}}num-threads=4
// PIPELINE: verify-no-scf-forall
// PIPELINE: convert-vector-to-llvm
// PIPELINE: convert-openmp-to-llvm

// LOWERED: llvm.func @rows_vec
// LOWERED: omp.parallel
// LOWERED: omp.wsloop
// LOWERED-NOT: scf.
// LOWERED-NOT: linalg.
// LOWERED-NOT: vector.
