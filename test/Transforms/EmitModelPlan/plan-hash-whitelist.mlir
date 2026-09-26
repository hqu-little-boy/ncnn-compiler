// Plan-hash field whitelist (T-C1).  The hash input is assembled from a fixed
// set of fields; everything else in the plan manifest must leave `plan_hash`
// (and therefore `build_identity`) untouched.
//
// On the whitelist: model, target triple, threads, vector lanes/scalable/tail,
// codegen identity, the fusion/copy/low-precision/layout-island/tuning/
// attention ledger fragments, and every walked operation's id, kind,
// operand/result types and attribute dictionary.
//
// Deliberately NOT on the whitelist: the output path.  Two runs that differ
// only in where the manifest is written must produce the same plan_hash,
// otherwise the hash would not be a property of the model.

// RUN: rm -f %t.a.json %t.b.json %t.c.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.a.json model=hashwhitelist target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.a.out
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.b.json model=hashwhitelist target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.b.out
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.c.json model=hashwhitelist target-triple=x86_64-pc-linux-gnu threads=8 vector-lanes=8 vector-tail=true" %s -o %t.c.out
// RUN: python3 %S/plan_hash_whitelist_check.py %t.a.json %t.b.json %t.c.json
// RUN: FileCheck %s --input-file=%t.a.json

module {
  func.func @model(%arg0: memref<4x8xf32>, %arg1: memref<8x8xf32>) -> memref<4x8xf32> {
    %out = memref.alloc() : memref<4x8xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<4x8xf32>, memref<8x8xf32>) outs(%out : memref<4x8xf32>)
    return %out : memref<4x8xf32>
  }
}

// Keys serialise alphabetically, so build_identity precedes plan_hash.
// CHECK-DAG: "build_identity":
// CHECK-DAG: "plan_hash":
