// RUN: ncnn-mlir-opt --verify-no-nested-openmp %s

module {
  func.func @single_team() {
    omp.parallel {
      omp.terminator
    }
    return
  }
}
