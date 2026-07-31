// RUN: ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline --dump-pass-pipeline %s 2>&1 | FileCheck %s

module {}

// CHECK: one-shot-bufferize{{.*}}bufferize-function-boundaries=true{{.*}}function-boundary-type-conversion=identity-layout-map
// CHECK: buffer-results-to-out-params{{.*}}add-result-attr=true{{.*}}hoist-static-allocs=true
// CHECK: ownership-based-buffer-deallocation{{.*}}private-function-dynamic-ownership=false
// CHECK: bufferization-lower-deallocations
// CHECK: verify-bufferized-model
