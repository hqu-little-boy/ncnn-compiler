// RUN: not ncnn-mlir-opt --matmul-kernel-ncnn='int8-kernel=vnni int8-target=avx-vnni' %s 2>&1 | FileCheck %s
// CHECK: incompatible VNNI intrinsic declaration
module {
  llvm.func @llvm.x86.avx512.vpdpbusd.256(i32) -> i32
}
