// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=1 vector-lowering=true' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=1 vector-lowering=true' %s | FileCheck %s --check-prefix=LOWERED

// 串行回退：threads=1 时 bufferized scf.forall 经 scf-forall-to-for 顺序
// 执行后走既有下降；不产生 OpenMP 结构，内层 SIMD 保持。

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

// PIPELINE: scf-forall-to-for
// PIPELINE: verify-no-scf-forall
// PIPELINE-NOT: convert-scf-to-openmp
// PIPELINE-NOT: convert-openmp-to-llvm

// LOWERED: llvm.func @rows_vec
// LOWERED-NOT: omp.
// LOWERED-NOT: scf.forall
// LOWERED-NOT: linalg.
// LOWERED-NOT: vector.
