#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "ncnn-mlir/Graph/graph.hpp"

namespace ncnn_importer {

using InputShape = std::vector<std::int64_t>;
inline constexpr std::int64_t kDynamicExtent = -1;

struct InputDimConstraint {
  std::uint32_t input;
  std::uint32_t dimension;
  std::int64_t minimum;
  std::int64_t multiple_of;

  bool operator==(const InputDimConstraint&) const = default;
};

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

bool has_layer_importer(std::string_view layer_type) noexcept;
std::size_t get_layer_importer_count() noexcept;

struct ImportOptions {
  // Used only when the model has one Input layer whose w/h/c are all omitted.
  std::optional<InputShape> input_shape;

  // One shape per Input layer, in source-layer order.
  std::vector<InputShape> input_shapes;

  // Runtime constraints for dynamic input dimensions.
  std::vector<InputDimConstraint> input_dim_constraints;

  // Import one specialization for each logical ncnn rank in [1, 4].
  bool dynamic_rank = false;

  // Internal rank selected while materializing a dynamic-rank specialization.
  std::optional<std::uint32_t> rank_specialization;
};

// 把解析好的 ncnn 计算图提升为 MLIR 模块：一个 ncnn.model，输入、权重和输出
// 分别由 ncnn.input、ncnn.const 和 ncnn.output 表示，计算层是 ncnn 方言算子。
// 后续先运行 convert-ncnn-model-to-func 建立 func.func，再 normalize 并由
// TOSA、 Linalg/SCF 或 Host conversion 消费计算算子。 失败时返回带 layer
// 上下文的 ImportError；形状/类型推断的诊断信息会被捕获进 message。
[[nodiscard]] std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
import_graph(const ncnn_graph::Graph& graph, mlir::MLIRContext& context);

[[nodiscard]] std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
import_graph(const ncnn_graph::Graph& graph,
             mlir::MLIRContext& context,
             const ImportOptions& options);

}  // namespace ncnn_importer
