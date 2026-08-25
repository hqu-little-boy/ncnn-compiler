// RUN: ncnn-mlir-opt --lower-vector-math-ncnn='backend=sleef lanes=8 abi=8_u10' %s | FileCheck %s

// SLEEF 后端：dispatch 入口符号为 Sleef_<基名><标签>；一元与二元共用同一
// 签名形态（整向量进出）。

// CHECK-DAG: func.func private @Sleef_expf8_u10(
// CHECK-DAG: func.func private @Sleef_powf8_u10(
// CHECK-LABEL: func.func @sample
// CHECK-NOT: math.
// CHECK: call @Sleef_expf8_u10(
func.func @sample(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %0 = math.exp %arg0 : vector<8xf32>
  %1 = math.powf %0, %arg1 : vector<8xf32>
  return %1 : vector<8xf32>
}
