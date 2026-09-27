// RUN: ncnn-mlir-opt '--pack-static-matmul-ncnn=min-n=128 min-k=128' %s 2>&1 | FileCheck %s
// RUN: env NCNN_WARNINGS_AS_ERRORS=1 not ncnn-mlir-opt '--pack-static-matmul-ncnn=min-n=128 min-k=128' %s 2>&1 | FileCheck %s --check-prefix=ERR

// T-M5 诊断分级。三条纪律各锁一处：
//   1. 只对「想走但没走成」的少数 reason 报警（packing_rejected_layout 这类
//      常态拒绝不报），避免 stderr 被冲垮；
//   2. 同一 reason 一次编译只报一次（去重键 = reason 本身）；
//   3. 默认是 warning 且编译**不失败**；NCNN_WARNINGS_AS_ERRORS=1 时升为
//      error 并让 pass 失败（CI 收紧用）。

func.func @small_shape(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %rhs = arith.constant dense<1.0> : tensor<64x64xf32>
  %out = tensor.empty() : tensor<64x64xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

// 形状要盖过 min-n/min-k 门控，否则会先撞 packing_skipped_small_shape
// 并被去重，测不到非恒定 RHS 这条 reason。
func.func @non_constant_rhs(%arg0: tensor<256x256xf32>, %rhs: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %out = tensor.empty() : tensor<256x256xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<256x256xf32>, tensor<256x256xf32>)
                   outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %mm : tensor<256x256xf32>
}

// 同一 reason 的第二次出现被抑制：本函数也触发 packing_skipped_small_shape，
// 但 stderr 上只应出现一次。
func.func @small_shape_again(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %rhs = arith.constant dense<1.0> : tensor<32x32xf32>
  %out = tensor.empty() : tensor<32x32xf32>
  %mm = linalg.matmul ins(%arg0, %rhs : tensor<32x32xf32>, tensor<32x32xf32>)
                   outs(%out : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %mm : tensor<32x32xf32>
}

// 默认：warning 形态，两个 reason 各一次。
// CHECK: warning: ncnn: packing_skipped_small_shape: linalg.matmul
// CHECK: warning: ncnn: packing_rejected_dynamic: linalg.matmul
// CHECK-NOT: warning: ncnn: packing_skipped_small_shape

// 升级：error 形态，reason 与去重口径不变。
// ERR: error: ncnn: packing_skipped_small_shape: linalg.matmul
// ERR: error: ncnn: packing_rejected_dynamic: linalg.matmul
// ERR-NOT: error: ncnn: packing_skipped_small_shape
