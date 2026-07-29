#pragma once

#include <variant>

#include "ncnn_frontend/Ops/Concat.hpp"
#include "ncnn_frontend/Ops/Const.hpp"
#include "ncnn_frontend/Ops/Conv2D.hpp"
#include "ncnn_frontend/Ops/Dropout.hpp"
#include "ncnn_frontend/Ops/Pool2D.hpp"
#include "ncnn_frontend/Ops/Relu.hpp"
#include "ncnn_frontend/Ops/Softmax.hpp"
#include "ncnn_frontend/Ops/Split.hpp"

namespace ncnn_frontend {

// 算子属性的封闭集合。新增算子：在 Ops/ 下加头文件、在此 include 并追加一个
// alternative、在 OperationKind.hpp 加枚举值即可，分发器无需改动。
// 注意：alternative 顺序与 OperationKind 枚举顺序一一对应（见 ir.cpp
// get_kind）。
using OperationAttributes = std::variant<ConstOp,
                                         Conv2DOp,
                                         ReluOp,
                                         Pool2DOp,
                                         SplitOp,
                                         ConcatOp,
                                         DropoutOp,
                                         SoftmaxOp>;

}  // namespace ncnn_frontend
