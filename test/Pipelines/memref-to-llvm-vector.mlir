// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=1 vector-size=8' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=1 vector-size=8' %s | FileCheck %s --check-prefix=LOWERED

module {
  func.func @vector_copy(%input: memref<64xf32>, %output: memref<64xf32>) {
    linalg.copy ins(%input : memref<64xf32>) outs(%output : memref<64xf32>)
    return
  }
}

// PIPELINE: convert-linalg-to-affine-loops
// PIPELINE: affine-super-vectorize{{.*}}vectorize-reductions=true{{.*}}virtual-vector-size={{.*}}8
// PIPELINE: convert-vector-to-llvm

// LOWERED: llvm.func @vector_copy
// LOWERED: vector<8xf32>
// LOWERED: llvm.intr.masked.load
// LOWERED: llvm.intr.masked.store
// LOWERED-NOT: linalg.
// LOWERED-NOT: affine.
// LOWERED-NOT: vector.
