// RUN: %ncnn-mlir-driver %squeezenet-param --bin %squeezenet-bin -o %t.ncnn.mlir
// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %t.ncnn.mlir -o %t.tosa.mlir
// RUN: FileCheck %s --check-prefix=TOSA < %t.tosa.mlir
// RUN: %mlir-opt --tosa-validate %t.tosa.mlir -o /dev/null
// RUN: %mlir-opt --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named{prefer-conv2d-kernel-layout-hwcf=true},tosa-to-tensor,tosa-to-arith))' %t.tosa.mlir -o %t.linalg.mlir
// RUN: FileCheck %s --check-prefix=LINALG < %t.linalg.mlir

// TOSA-LABEL: func.func @model(%arg0: tensor<3x227x227xf32>) -> tensor<1000xf32>
// TOSA: tosa.conv2d
// TOSA: tosa.max_pool2d
// TOSA: tosa.concat
// TOSA: tosa.avg_pool2d
// TOSA: tosa.reduce_max
// TOSA: tosa.reduce_sum
// TOSA-NOT: ncnn.model
// TOSA-NOT: = ncnn.

// LINALG-LABEL: func.func @model(%arg0: tensor<3x227x227xf32>) -> tensor<1000xf32>
// LINALG: linalg.conv_2d_nhwc_hwcf
// LINALG-NOT: tosa.conv2d
