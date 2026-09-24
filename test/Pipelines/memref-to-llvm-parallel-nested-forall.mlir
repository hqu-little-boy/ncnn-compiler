// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' %s | FileCheck %s --check-prefix=LOWERED

// Nested forall dimensions are not separate OpenMP teams. Keep the outer
// forall parallel and lower the inner forall to serial loops.
module {
  func.func @nested_forall(%output: memref<4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    scf.forall (%i) in (%c4) {
      scf.forall (%j) in (%c4) {
        memref.store %zero, %output[%i, %j] : memref<4x4xf32>
      }
    }
    return
  }
}

// PIPELINE: convert-nested-forall-to-for
// PIPELINE: convert-nested-linalg-to-loops
// PIPELINE: convert-linalg-to-parallel-loops
// PIPELINE: scf-forall-to-parallel
// PIPELINE: convert-scf-to-openmp
// PIPELINE: verify-no-nested-openmp

// LOWERED-LABEL: llvm.func @nested_forall
// LOWERED-COUNT-1: omp.parallel
// LOWERED: omp.wsloop
// LOWERED-NOT: omp.parallel
// LOWERED-NOT: scf.forall
// LOWERED-NOT: scf.parallel
