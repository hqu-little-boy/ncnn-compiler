// RUN: ncnn-mlir-opt '--ncnn-linalg-to-memref-pipeline=vector-lanes=4 vector-scalable=false' %s -o /dev/null

// 回归守护：最内维宽度 1 的逐元素 generic（如 matmul N=1 输出的 clamp）
// 必须被跳过——vector<1xf32> 退化行会触发上游 TransferWriteOp::build 的
// 越界段错误，且没有向量化收益。本测试以崩溃为失败信号。

module {
  func.func @clamp_width1(%in: tensor<409600x1xf32>) -> tensor<409600x1xf32> {
    %lo = arith.constant -1.0 : f32
    %hi = arith.constant 1.0 : f32
    %e = tensor.empty() : tensor<409600x1xf32>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%in : tensor<409600x1xf32>) outs(%e : tensor<409600x1xf32>) {
    ^bb0(%a: f32, %b: f32):
      %1 = arith.minimumf %a, %hi : f32
      %2 = arith.maximumf %1, %lo : f32
      linalg.yield %2 : f32
    } -> tensor<409600x1xf32>
    return %0 : tensor<409600x1xf32>
  }
}
