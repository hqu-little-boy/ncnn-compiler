// RUN: ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=portable int8-target=avx-vnni' %s | FileCheck %s --check-prefix=PORTABLE
// RUN: ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=auto int8-target=portable' %s | FileCheck %s --check-prefix=PORTABLE
// RUN: ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=vnni int8-target=avx-vnni' %s | FileCheck %s --check-prefix=STRIDED
// RUN: not ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=vnni int8-target=portable' %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=invalid' %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not ncnn-mlir-opt --matmul-kernel-ncnn='int8-target=invalid' %s 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: invalid INT8 policy or unavailable VNNI target
// PORTABLE-NOT: llvm.call
// PORTABLE-LABEL: func.func @strided_k
// PORTABLE: arith.extsi
// PORTABLE: arith.muli
// PORTABLE: ncnn.kernel = "int8_row_dot"
// PORTABLE-NOT: llvm.call
// STRIDED-NOT: llvm.call
// STRIDED-LABEL: func.func @strided_k
// STRIDED: arith.extsi
// STRIDED: arith.muli
// STRIDED: ncnn.kernel = "int8_row_dot"
// STRIDED-NOT: llvm.call
// PORTABLE-LABEL: func.func @zero_columns
// PORTABLE: linalg.matmul_transpose_b
module {
  func.func @strided_k(%a: memref<3x33xi8, strided<[66, 2]>>,
                      %b: memref<5x33xi8>, %c: memref<3x5xi32>) {
    scf.forall (%tile) in (1) {
      linalg.matmul_transpose_b
        ins(%a, %b : memref<3x33xi8, strided<[66, 2]>>, memref<5x33xi8>)
        outs(%c : memref<3x5xi32>)
    }
    return
  }

  func.func @zero_columns(%a: memref<2x4xi8>, %b: memref<0x4xi8>,
                          %c: memref<2x0xi32>) {
    scf.forall (%tile) in (1) {
      linalg.matmul_transpose_b
        ins(%a, %b : memref<2x4xi8>, memref<0x4xi8>)
        outs(%c : memref<2x0xi32>)
    }
    return
  }
}
