// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline --dump-pass-pipeline %s 2>&1 | FileCheck %s

module {}

// CHECK: convert-linalg-to-loops
// CHECK: lower-affine
// CHECK: convert-scf-to-cf
// CHECK: convert-math-to-libm
// CHECK: expand-strided-metadata
// CHECK: convert-arith-to-llvm
// CHECK: finalize-memref-to-llvm
// CHECK: convert-func-to-llvm
// CHECK: convert-cf-to-llvm
// CHECK: reconcile-unrealized-casts
