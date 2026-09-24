// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' %s | FileCheck %s --check-prefix=LOWERED

// Scalar operands are valid for bufferized Linalg operations nested in a tile
// forall; only shaped operands must be memrefs before OpenMP lowering.
module {
  func.func @parallel_fill(%output: memref<4x4xf32>) {
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    scf.forall (%i) in (%c4) {
      %slice = memref.subview %output[%i, 0] [1, 4] [1, 1]
        : memref<4x4xf32> to memref<1x4xf32, strided<[4, 1], offset: ?>>
      linalg.fill ins(%zero : f32)
        outs(%slice : memref<1x4xf32, strided<[4, 1], offset: ?>>)
    }
    return
  }
}

// LOWERED: llvm.func @parallel_fill
// LOWERED: omp.parallel
// LOWERED: omp.wsloop
// LOWERED-NOT: linalg.fill
// LOWERED-NOT: scf.forall
