// RUN: ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=1 vector-lowering=true vector-math=libmvec vector-math-abi=_ZGVdN8 vector-math-lanes=8' %s | FileCheck %s

// 向量数学后端贯通端到端：管线串解析 → lower-vector-math-ncnn 整体替换 →
// 声明与调用存活至 LLVM 方言；标量 MathToLibm 不再产生逐 lane 调用。

module {
  func.func @row_exp(%input: memref<64x8xf32>, %output: memref<64x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c64 step %c1 {
      %v = vector.transfer_read %input[%i, %c0], %pad {in_bounds = [true]} : memref<64x8xf32>, vector<8xf32>
      %w = math.exp %v : vector<8xf32>
      vector.transfer_write %w, %output[%i, %c0] {in_bounds = [true]} : vector<8xf32>, memref<64x8xf32>
    }
    return
  }
}

// CHECK-LABEL: llvm.func @_ZGVdN8v_expf(vector<8xf32>) -> vector<8xf32>
// CHECK-SAME: sym_visibility = "private"
// CHECK-LABEL: llvm.func @row_exp
// CHECK-NOT: math.
// CHECK: llvm.call @_ZGVdN8v_expf({{[^)]*}}) : (vector<8xf32>) -> vector<8xf32>
// CHECK-NOT: llvm.call @expf
