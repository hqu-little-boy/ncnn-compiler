// RUN: ncnn-mlir-opt --strategy-ncnn --canonicalize '--fuse-linalg-epilogue=tile-width=16' '--ncnn-linalg-to-memref-pipeline=vector-lanes=8 vector-tail=true' %s -o %t.mem
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4 vector-lowering=true' --dump-pass-pipeline %t.mem 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4 vector-lowering=true' %t.mem | FileCheck %s --check-prefix=LOWERED
// RUN: ncnn-mlir-opt %t.mem | FileCheck %s --check-prefix=CONTRACT

// A3 终态（matmul 形态）：conv→matmul 的 M/N 分块 forall 与行向量化激活
// 同体，经单轨 forall→OpenMP 发射；残差 linalg 由串行兜底消化。

module {
  func.func @conv_relu(%arg0: tensor<1x8x8x3xf32>) -> tensor<1x8x8x16xf32> {
    %cst = arith.constant dense<0.5> : tensor<1x1x3x16xf32>
    %init = tensor.empty() : tensor<1x8x8x16xf32>
    %conv = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %cst : tensor<1x8x8x3xf32>, tensor<1x1x3x16xf32>) outs(%init : tensor<1x8x8x16xf32>) -> tensor<1x8x8x16xf32>
    %zero = arith.constant 0.0 : f32
    %out = tensor.empty() : tensor<1x8x8x16xf32>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%conv : tensor<1x8x8x16xf32>) outs(%out : tensor<1x8x8x16xf32>) {
    ^bb0(%in: f32, %o: f32):
      %max = arith.maximumf %in, %zero : f32
      linalg.yield %max : f32
    } -> tensor<1x8x8x16xf32>
    return %relu : tensor<1x8x8x16xf32>
  }
}

// PIPELINE: convert-linalg-to-parallel-loops
// PIPELINE: scf-forall-to-parallel
// PIPELINE: convert-scf-to-openmp{{.*}}num-threads=4
// PIPELINE: verify-no-scf-forall
// PIPELINE: convert-openmp-to-llvm

// LOWERED: llvm.func @conv_relu
// LOWERED: omp.parallel
// LOWERED: omp.wsloop
// LOWERED-NOT: scf.
// LOWERED-NOT: linalg.
// LOWERED-NOT: vector.

// CONTRACT-DAG: ncnn.contract = "selected"
// CONTRACT-DAG: ncnn.kernel = "f32_mxn_fma"
// CONTRACT-DAG: ncnn.parallel = "outer_tile+inner_simd"
// CONTRACT-DAG: ncnn.fma = "vector.fma"
