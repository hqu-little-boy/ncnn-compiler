#pragma once

namespace ncnn_frontend {

enum class OperationKind {
  Constant,
  Convolution,
  Relu,
  Pooling,
  Split,
  Concat,
  Dropout,
  Softmax,
  // 哨兵：始终排在最后，等于枚举中真实算子的个数。
  // 用于在测试中断言 OperationAttributes 的 alternative 数量与本枚举一致，
  // 防止「只加枚举、忘了加 variant alternative」的错位。
  Count,
};

}  // namespace ncnn_frontend
