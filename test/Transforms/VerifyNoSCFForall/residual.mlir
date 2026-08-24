// RUN: not ncnn-mlir-opt --verify-no-scf-forall %s 2>&1 | FileCheck %s

// 并行发射完整性闸门：残留 scf.parallel / scf.forall 即失败——
// convert-scf-to-openmp 对 scf.forall 静默跳过，必须显式拦截。

func.func @residual_parallel(%out: memref<64xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.parallel (%i) = (%c0) to (%c64) step (%c1) {
    memref.store %cst, %out[%i] : memref<64xf32>
  }
  return
}

func.func @residual_forall(%in: memref<64x128xf32>, %out: memref<64x128xf32>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  scf.forall (%i) in (%c64) {
    %lb = arith.muli %i, %c8 : index
    %sin = memref.subview %in[%lb, 0][8, 128][1, 1] : memref<64x128xf32> to memref<8x128xf32, strided<[128, 1], offset: ?>>
    %sout = memref.subview %out[%lb, 0][8, 128][1, 1] : memref<64x128xf32> to memref<8x128xf32, strided<[128, 1], offset: ?>>
    memref.copy %sin, %sout : memref<8x128xf32, strided<[128, 1], offset: ?>> to memref<8x128xf32, strided<[128, 1], offset: ?>>
  }
  return
}

// CHECK: error: 'scf.parallel' op remains after parallel lowering; op=scf.parallel
// CHECK: error: 'scf.forall' op remains after parallel lowering; op=scf.forall
