// RUN: ncnn-mlir-opt --lower-vector-math-ncnn='backend=libmvec lanes=8 abi=_ZGVdN8' %s | FileCheck %s

// lanes 整数倍且不超过展开上限的整行切成 slice/call/insert 链。

// CHECK-LABEL: func.func @wide_row
// CHECK-COUNT-4: = vector.extract_strided_slice
// CHECK-COUNT-4: call @_ZGVdN8v_expf(
// CHECK-COUNT-4: = vector.insert_strided_slice
// CHECK-NOT: math.exp
// CHECK: return
func.func @wide_row(%arg0: vector<32xf32>) -> vector<32xf32> {
  %0 = math.exp %arg0 : vector<32xf32>
  return %0 : vector<32xf32>
}

// 超过展开上限（4×lanes）的整行保持不动，交由标量 MathToLibm 兜底。

// CHECK-LABEL: func.func @too_wide_row
// CHECK: math.exp
// CHECK-NOT: call @_ZGV
func.func @too_wide_row(%arg0: vector<64xf32>) -> vector<64xf32> {
  %0 = math.exp %arg0 : vector<64xf32>
  return %0 : vector<64xf32>
}
