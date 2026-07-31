module {
  func.func private @model(%input: memref<1xf32>,
                           %output: memref<1xf32> {bufferize.result}) {
    %temporary = memref.alloc() : memref<1048576xf32>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%temporary : memref<1048576xf32>)
    %index = arith.constant 0 : index
    %value = memref.load %input[%index] : memref<1xf32>
    memref.store %value, %output[%index] : memref<1xf32>
    memref.dealloc %temporary : memref<1048576xf32>
    return
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %input = memref.alloc() : memref<1xf32>
    %output = memref.alloc() : memref<1xf32>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %iterations = arith.constant 500 : index
    %value = arith.constant 1.0 : f32
    memref.store %value, %input[%zero] : memref<1xf32>
    scf.for %iteration = %zero to %iterations step %one {
      func.call @model(%input, %output) : (memref<1xf32>, memref<1xf32>) -> ()
    }
    memref.dealloc %input : memref<1xf32>
    memref.dealloc %output : memref<1xf32>
    return
  }
}
