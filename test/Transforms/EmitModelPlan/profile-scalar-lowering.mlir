// RUN: ncnn-mlir-opt --ncnn-memref-to-llvm-pipeline='threads=1 vector-size=0 vector-lowering=false' %s | FileCheck %s
// SCF introduces arithmetic after the initial memref/arith conversion.
// Scalar mode must lower that arithmetic without depending on a vector tail.
func.func @scalar_loop(%out: memref<?xi32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %value = arith.constant 7 : i32
  scf.for %i = %c0 to %n step %c1 {
    memref.store %value, %out[%i] : memref<?xi32>
  }
  return
}
// CHECK-LABEL: llvm.func @scalar_loop
// CHECK-NOT: arith.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.icmp
// CHECK-NOT: arith.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.add
// CHECK-NOT: arith.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: llvm.return
