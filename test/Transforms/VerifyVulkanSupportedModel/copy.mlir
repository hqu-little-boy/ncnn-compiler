// RUN: ncnn-mlir-opt --verify-vulkan-supported-model %s -o /dev/null

func.func @copy(%input: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %copied = linalg.copy ins(%input : tensor<4xf32>) outs(%empty : tensor<4xf32>) -> tensor<4xf32>
  return %copied : tensor<4xf32>
}
