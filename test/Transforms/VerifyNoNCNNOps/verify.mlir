// RUN: not ncnn-mlir-opt --split-input-file --verify-no-ncnn-ops %s 2>&1 | FileCheck %s

module {
  func.func @residual(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ncnn.relu %arg0 {ncnn.name = "relu_17", ncnn.source_layer = 17 : i64} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = ncnn.dropout %0 {ncnn.name = "dropout_18", ncnn.source_layer = 18 : i64} : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// CHECK-DAG: error: 'ncnn.relu' op remains after lowering; op=ncnn.relu, ncnn.name="relu_17", ncnn.source_layer=17
// CHECK-DAG: error: 'ncnn.dropout' op remains after lowering; op=ncnn.dropout, ncnn.name="dropout_18", ncnn.source_layer=18

// -----

module {
  func.func @missing_trace(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = ncnn.relu %arg0 : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: error: 'ncnn.relu' op remains after lowering; op=ncnn.relu, ncnn.name=<missing>, ncnn.source_layer=<missing>
