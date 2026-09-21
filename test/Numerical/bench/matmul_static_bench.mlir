// P20 static-B packed GEMM benchmark. The RHS is a compile-time tensor
// constant; --matmul-packing=off is the direct baseline and auto is the
// physical panel-NK candidate. The generated public functions keep the same
// A/output descriptor ABI in both variants.
module {
  func.func @bench_shallow(%arg0: tensor<1024x576xf32>) -> tensor<1024x64xf32> attributes {llvm.emit_c_interface} {
    %rhs = arith.constant dense<1.0> : tensor<576x64xf32>
    %out = tensor.empty() : tensor<1024x64xf32>
    %mm = linalg.matmul ins(%arg0, %rhs : tensor<1024x576xf32>, tensor<576x64xf32>)
                     outs(%out : tensor<1024x64xf32>) -> tensor<1024x64xf32>
    return %mm : tensor<1024x64xf32>
  }

  func.func @bench_deep(%arg0: tensor<256x2304xf32>) -> tensor<256x256xf32> attributes {llvm.emit_c_interface} {
    %rhs = arith.constant dense<1.0> : tensor<2304x256xf32>
    %out = tensor.empty() : tensor<256x256xf32>
    %mm = linalg.matmul ins(%arg0, %rhs : tensor<256x2304xf32>, tensor<2304x256xf32>)
                     outs(%out : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %mm : tensor<256x256xf32>
  }
}
