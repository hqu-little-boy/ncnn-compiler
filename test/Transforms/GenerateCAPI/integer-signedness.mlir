// RUN: ncnn-mlir-opt --generate-ncnn-c-api='export-name=integer_signedness manifest-path=%t.json' %s -o /dev/null
// RUN: FileCheck %s < %t.json

module {
  func.func @model(%unsigned_input: memref<4xui8>,
                   %signed_output: memref<4xi8> {bufferize.result})
      attributes {llvm.emit_c_interface, ncnn.entry_point} {
    return
  }
}

// CHECK: "element_type": "ui8"
// CHECK: "name": "input1"
// CHECK: "element_type": "i8"
// CHECK: "name": "output1"
