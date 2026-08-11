// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' %s | FileCheck %s --check-prefix=LOWERED

module {
  func.func @parallel_copy(%input: memref<64xf32>, %output: memref<64xf32>) {
    linalg.copy ins(%input : memref<64xf32>) outs(%output : memref<64xf32>)
    return
  }
}

// PIPELINE: convert-linalg-to-parallel-loops
// PIPELINE: scf-parallel-loop-fusion
// PIPELINE: convert-scf-to-openmp{{.*}}num-threads=4
// PIPELINE: loop-invariant-code-motion
// PIPELINE: convert-openmp-to-llvm

// LOWERED: llvm.func @parallel_copy
// LOWERED: omp.parallel
// LOWERED: omp.wsloop
// LOWERED-NOT: linalg.
// LOWERED-NOT: scf.
