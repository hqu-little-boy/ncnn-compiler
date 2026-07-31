#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "ncnn-mlir/Graph/graph.hpp"

namespace ncnn_importer {

// 导入错误的结构化上下文，对应原 ncnn_frontend::ImportError。
class ImportError {
 public:
  ImportError(std::size_t layer_index,
              std::string layer_type,
              std::string layer_name,
              std::string message);

  std::size_t get_layer_index() const noexcept;
  std::string_view get_layer_type() const noexcept;
  std::string_view get_layer_name() const noexcept;
  std::string_view get_message() const noexcept;
  std::string to_string() const;

 private:
  std::size_t layer_index_;
  std::string layer_type_;
  std::string layer_name_;
  std::string message_;
};

// 把解析好的 ncnn 计算图提升为 MLIR 模块：一个 ncnn.model，输入、权重和输出
// 分别由 ncnn.input、ncnn.const 和 ncnn.output 表示，计算层是 ncnn 方言算子。
// 后续先运行 convert-ncnn-model-to-func 建立 func.func，再 normalize 并由
// TOSA、 Linalg/SCF 或 Host conversion 消费计算算子。 失败时返回带 layer
// 上下文的 ImportError；形状/类型推断的诊断信息会被捕获进 message。
[[nodiscard]] std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
import_graph(const ncnn_graph::Graph& graph, mlir::MLIRContext* context);

}  // namespace ncnn_importer
