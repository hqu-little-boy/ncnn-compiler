// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' %s | FileCheck %s --check-prefix=LOWERED

// 残差 linalg 的多线程发射：全域 convert-linalg-to-parallel-loops 保留
// （运行时边界容忍任意 extent），与张量级分块 forall 共用同一条 OpenMP
// 转换；forall 区域内的嵌套团队由 libomp 默认非嵌套语义串行化。

module {
  func.func @parallel_copy(%input: memref<64xf32>, %output: memref<64xf32>) {
    linalg.copy ins(%input : memref<64xf32>) outs(%output : memref<64xf32>)
    return
  }
}

// PIPELINE: convert-linalg-to-parallel-loops
// PIPELINE: scf-forall-to-parallel
// PIPELINE: convert-scf-to-openmp{{.*}}num-threads=4
// PIPELINE: verify-no-scf-forall
// PIPELINE: convert-openmp-to-llvm

// LOWERED: llvm.func @parallel_copy
// LOWERED: omp.parallel
// LOWERED: omp.wsloop
// LOWERED-NOT: linalg.
// LOWERED-NOT: scf.
