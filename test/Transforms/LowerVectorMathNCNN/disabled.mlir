// RUN: ncnn-mlir-opt %s | FileCheck %s
// RUN: ncnn-mlir-opt --lower-vector-math-ncnn='backend=none lanes=8 abi=_ZGVdN8' %s | FileCheck %s

// 默认（backend=none）与本测试的显式 none 均为严格无操作。

// CHECK-LABEL: func.func @sample
// CHECK: math.exp
// CHECK-NOT: call @_ZGVdN8v_expf
func.func @sample(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.exp %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}
