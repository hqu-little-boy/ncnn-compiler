// RUN: ncnn-mlir-opt --lower-vector-math-ncnn='backend=libmvec lanes=8 abi=_ZGVdN8' %s | FileCheck %s

// 无向量变体或不在覆盖范围的形态保持不动，交由标量 MathToLibm 兜底：
// 奇数尾宽、非 f32、scalable、无向量版函数（erfc）。

// CHECK-LABEL: func.func @odd_tail
// CHECK: math.exp
func.func @odd_tail(%arg0: vector<7xf32>) -> vector<7xf32> {
  %0 = math.exp %arg0 : vector<7xf32>
  return %0 : vector<7xf32>
}

// CHECK-LABEL: func.func @narrower_than_lanes
// CHECK: math.exp
func.func @narrower_than_lanes(%arg0: vector<4xf32>) -> vector<4xf32> {
  %0 = math.exp %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func.func @double_precision
// CHECK: math.exp
func.func @double_precision(%arg0: vector<8xf64>) -> vector<8xf64> {
  %0 = math.exp %arg0 : vector<8xf64>
  return %0 : vector<8xf64>
}

// CHECK-LABEL: func.func @scalable
// CHECK: math.exp
func.func @scalable(%arg0: vector<[8]xf32>) -> vector<[8]xf32> {
  %0 = math.exp %arg0 : vector<[8]xf32>
  return %0 : vector<[8]xf32>
}

// CHECK-LABEL: func.func @no_vector_variant
// CHECK: math.erfc
func.func @no_vector_variant(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.erfc %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}
