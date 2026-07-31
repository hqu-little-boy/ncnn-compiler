// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline --dump-pass-pipeline %s 2>&1 | FileCheck %s

module {}

// CHECK: Pass Manager with 6 passes:
// CHECK-NEXT: builtin.module(
// CHECK-NEXT: convert-ncnn-model-to-func,
// CHECK-NEXT: normalize-ncnn,
// CHECK-NEXT: convert-ncnn-to-tosa,
// CHECK-NEXT: canonicalize{{.*}},
// CHECK-NEXT: cse,
// CHECK-NEXT: verify-no-ncnn-ops
// CHECK-NEXT: )
