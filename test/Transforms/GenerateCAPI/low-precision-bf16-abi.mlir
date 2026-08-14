// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=low_precision_bf16 manifest-path=%t.json' %s -o /dev/null
// RUN: FileCheck %s < %t.json

module {
  func.func @model(%input: memref<4xbf16>,
                   %output: memref<4xbf16> {bufferize.result})
      attributes {llvm.emit_c_interface, ncnn.entry_point,
                  ncnn.fp16_accumulator = "f32", ncnn.precision = "bf16"} {
    return
  }
}

// CHECK: "element_type": "bf16"
// CHECK: "name": "input1"
// CHECK: "element_type": "bf16"
// CHECK: "name": "output1"
// CHECK: "precision_policy": {
// CHECK: "complex_accumulator": "f32"
// CHECK: "complex_math": "f32"
// CHECK: "storage": "bf16"
