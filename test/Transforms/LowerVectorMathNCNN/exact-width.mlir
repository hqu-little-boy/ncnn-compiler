// RUN: ncnn-mlir-opt --lower-vector-math-ncnn='backend=libmvec lanes=8 abi=_ZGVdN8' %s | FileCheck %s

// ABI 等宽向量整体替换为 libmvec 调用；声明为私有纯函数。二元 pow 的
// libmvec mangle 是双 v（_ZGVdN8vv_powf）。

// CHECK-DAG: func.func private @_ZGVdN8v_expf({{[^)]*}}) -> vector<8xf32> attributes {llvm.readnone}
// CHECK-DAG: func.func private @_ZGVdN8v_tanhf(
// CHECK-DAG: func.func private @_ZGVdN8v_erff(
// CHECK-DAG: func.func private @_ZGVdN8v_logf(
// CHECK-DAG: func.func private @_ZGVdN8vv_powf(

// CHECK-LABEL: func.func @unary
// CHECK-NOT: math.
// CHECK: call @_ZGVdN8v_expf({{[^)]*}}) : (vector<8xf32>) -> vector<8xf32>
// CHECK: call @_ZGVdN8v_tanhf(
// CHECK: call @_ZGVdN8v_erff(
// CHECK: call @_ZGVdN8v_logf(
// CHECK: return
func.func @unary(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.exp %arg0 : vector<8xf32>
  %1 = math.tanh %0 : vector<8xf32>
  %2 = math.erf %1 : vector<8xf32>
  %3 = math.log %2 : vector<8xf32>
  return %3 : vector<8xf32>
}

// CHECK-LABEL: func.func @binary
// CHECK-NOT: math.
// CHECK: call @_ZGVdN8vv_powf({{[^)]*}}) : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
// CHECK: return
func.func @binary(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %0 = math.powf %arg0, %arg1 : vector<8xf32>
  return %0 : vector<8xf32>
}
