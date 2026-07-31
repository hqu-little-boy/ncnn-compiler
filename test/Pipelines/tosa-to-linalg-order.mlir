// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline --dump-pass-pipeline %s 2>&1 | FileCheck %s

module {}

// CHECK: tosa-optional-decompositions
// CHECK: canonicalize
// CHECK: tosa-infer-shapes
// CHECK: tosa-make-broadcastable
// CHECK: tosa-to-linalg-named{{.*}}prefer-conv2d-kernel-layout-hwcf=true
// CHECK: tosa-layerwise-constant-fold
// CHECK: tosa-to-linalg
// CHECK: tosa-to-tensor
// CHECK: tosa-to-arith
// CHECK: cse
// CHECK: verify-no-tosa-ops
