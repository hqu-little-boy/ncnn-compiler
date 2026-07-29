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
};

}  // namespace ncnn_frontend
