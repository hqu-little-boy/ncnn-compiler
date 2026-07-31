// RUN: %ncnn-mlir-driver %squeezenet-param --bin %squeezenet-bin -o %t.ncnn.mlir
// RUN: ncnn-mlir-opt --ncnn-to-tosa-pipeline %t.ncnn.mlir -o %t.tosa.mlir
// RUN: FileCheck %s --check-prefix=TOSA < %t.tosa.mlir
// RUN: %mlir-opt --tosa-validate %t.tosa.mlir -o /dev/null
// RUN: ncnn-mlir-opt --ncnn-tosa-to-linalg-pipeline %t.tosa.mlir -o %t.linalg.mlir
// RUN: FileCheck %s --check-prefix=LINALG < %t.linalg.mlir
// RUN: ncnn-mlir-opt --ncnn-linalg-to-memref-pipeline %t.linalg.mlir -o %t.memref.mlir
// RUN: FileCheck %s --check-prefix=MEMREF < %t.memref.mlir

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
// LINALG: math.exp
// LINALG-NOT: tosa.

// MEMREF-LABEL: func.func @model(
// MEMREF-SAME: %[[INPUT:.*]]: memref<3x227x227xf32>,
// MEMREF-SAME: %[[OUTPUT:.*]]: memref<1000xf32> {bufferize.result})
// MEMREF-NOT: ->
// MEMREF: linalg.generic {{.*}}outs(%[[OUTPUT]] : memref<1000xf32>)
// MEMREF: math.exp
// MEMREF: linalg.generic {{.*}}outs(%[[OUTPUT]] : memref<1000xf32>)
// MEMREF-NOT: memref.copy {{.*}}, %[[OUTPUT]]
// MEMREF-NOT: memref.dealloc %[[INPUT]]
// MEMREF-NOT: memref.dealloc %[[OUTPUT]]
// MEMREF: return
