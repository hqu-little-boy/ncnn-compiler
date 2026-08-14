// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=low_precision_fp16 manifest-path=%t.json' %s -o /dev/null
// RUN: FileCheck %s < %t.json

module {
  func.func @model(%input: memref<4xf16>,
                   %output: memref<4xf16> {bufferize.result})
      attributes {llvm.emit_c_interface, ncnn.entry_point,
                  ncnn.fp16_accumulator = "f32", ncnn.precision = "fp16"} {
    return
  }
}

// CHECK: "element_type": "f16"
// CHECK: "name": "input1"
// CHECK: "element_type": "f16"
// CHECK: "name": "output1"
// CHECK: "precision_policy": {
// CHECK: "complex_accumulator": "f32"
// CHECK: "complex_math": "f32"
// CHECK: "fallback": false
// CHECK: "fp16_accumulator": "f32"
// CHECK: "storage": "fp16"
