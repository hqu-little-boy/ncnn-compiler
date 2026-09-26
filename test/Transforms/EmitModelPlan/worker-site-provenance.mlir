// RUN: rm -f %t.plan.json
// RUN: ncnn-mlir-opt --emit-ncnn-execution-plan="path=%t.plan.json model=worker_site target-triple=x86_64-pc-linux-gnu threads=4 vector-lanes=8 vector-tail=true" %s -o %t.out
// RUN: FileCheck %s --input-file=%t.plan.json

// Parallel worker sites (direct children of scf.forall) are generated loops
// without provenance of their own.  They inherit (source_layer, source_name)
// from the operations inside their region only when those agree; disagreement
// is surfaced in diagnostics.unknown_fields instead of a guessed value.
// The location carrier loc("conv7"("ncnn-layer":7:0)) also takes precedence
// over the stale ncnn.name / ncnn.source_layer attributes on the same op.
// The scf.forall wrappers themselves are direct children of func.func, so
// they are not worker sites and stay unattributed.

module {
  func.func @model() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    // Positive: the whole region agrees on conv7 via the location carrier.
    scf.forall (%iv) in (%c4) {
      scf.for %j = %c0 to %c4 step %c1 {
        %sum = arith.addi %iv, %j {ncnn.name = "stale_attr", ncnn.source_layer = 42 : i64} : index loc("conv7"("ncnn-layer":7:0))
      }
    }
    // Negative: conv9 and relu10 disagree, so the worker site does not guess.
    scf.forall (%kv) in (%c4) {
      scf.for %k = %c0 to %c4 step %c1 {
        %left = arith.addi %kv, %k : index loc("conv9"("ncnn-layer":9:0))
        %right = arith.addi %left, %k : index loc("relu10"("ncnn-layer":10:0))
      }
    }
    return
  }
}

// The ambiguity is reported, never valued.
// CHECK: "unknown_fields": [
// CHECK: "worker_site_source_ambiguous"
// The location carrier wins over the stale attributes on the same operation.
// CHECK: "id": "model/arith.addi#0"
// CHECK-NOT: "source_layer": 42
// CHECK: "source_layer": 7
// CHECK: "source_name": "conv7"
// The worker site inherits the provenance its region agrees on.
// CHECK: "id": "model/scf.for#0"
// CHECK: "source_layer": 7
// CHECK: "source_name": "conv7"
// The scf.forall wrapper is not a worker site and stays unattributed.
// CHECK: "id": "model/scf.forall#0"
// Directly carried provenance still reports each conflicting layer.
// CHECK: "id": "model/arith.addi#1"
// CHECK: "source_layer": 9
// CHECK: "source_name": "conv9"
// CHECK: "id": "model/arith.addi#2"
// CHECK: "source_layer": 10
// CHECK: "source_name": "relu10"
// The ambiguous worker site reports explicit nulls, not a guessed layer.
// CHECK: "id": "model/scf.for#1"
// CHECK: "source_layer": null
// CHECK: "source_name": null
// provenance[] records the inherited worker site and the carried ones...
// CHECK: "provenance": [
// CHECK: "layer": 7
// CHECK: "name": "conv7"
// CHECK: "operation": "model/arith.addi#0"
// CHECK: "operation": "model/scf.for#0"
// ...but never the ambiguous worker site.
// CHECK-NOT: "operation": "model/scf.for#1"
// CHECK: "summary": {
