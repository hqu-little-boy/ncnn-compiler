// RUN: not ncnn-mlir-opt --verify-no-nested-openmp %s 2>&1 | FileCheck %s

module {
  func.func @nested() {
    omp.parallel {
      omp.parallel {
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}

// CHECK: error: 'omp.parallel' op nested omp.parallel is not part of the P12 single-team contract
