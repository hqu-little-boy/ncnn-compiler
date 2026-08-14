#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"
#include "ncnn-mlir/Graph/graph.hpp"
#include "ncnn-mlir/Importer/NCNNImporter.hpp"
#include "ncnn-mlir/Support/ShapeMode.hpp"

namespace ncnn_importer::detail {

struct LayerContext {
  std::size_t index;
  const ncnn_graph::Layer& layer;
};

using ImportResult = std::expected<void, ImportError>;

ImportError make_error(const LayerContext& context, std::string message);

const ncnn_graph::ParamValue* find_param(const ncnn_graph::ParamDict& params,
                                         int id);
std::expected<void, std::string> validate_param_ids(
  const ncnn_graph::ParamDict& params, std::span<const int> allowed);
std::expected<std::int64_t, std::string> get_int(
  const ncnn_graph::ParamDict& params,
  int id,
  std::int64_t default_value,
  std::string_view name);
std::expected<float, std::string> get_float(const ncnn_graph::ParamDict& params,
                                            int id,
                                            float default_value,
                                            std::string_view name);
std::expected<void, std::string> expect_boolean(std::int64_t value,
                                                std::string_view name);
std::expected<void, std::string> expect_source_arity(
  const ncnn_graph::Layer& layer, std::size_t inputs, std::size_t outputs);
std::expected<void, std::string> validate_feature_mask(
  const ncnn_graph::ParamDict& params);

class ImportContext {
 public:
  ImportContext(mlir::MLIRContext& context, ImportOptions options);

  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> run(
    const ncnn_graph::Graph& source);

  mlir::OpBuilder& builder() noexcept;
  const std::optional<ncnn_importer::InputShape>& input_shape() const noexcept;
  const std::vector<ncnn_importer::InputShape>& input_shapes() const noexcept;
  mlir::ncnn::ShapeMode shape_mode(bool has_dynamic_input) const noexcept;
  std::optional<ncnn_importer::InputShape> next_input_shape(
    bool dimensions_omitted) noexcept;
  std::expected<mlir::Value, ImportError> find_blob(const LayerContext& context,
                                                    std::string_view name);
  ImportResult bind_blob(const LayerContext& context,
                         std::string name,
                         mlir::Value value);
  std::expected<mlir::Value, ImportError> make_constant(
    const LayerContext& context,
    const ncnn_graph::Tensor& tensor,
    std::size_t weight_index,
    bool precision_storage = false);
  void tag_source(mlir::Operation* operation, const LayerContext& context);

  template <typename Op>
  mlir::FailureOr<mlir::RankedTensorType> infer_single_tensor_result(
    mlir::Location location,
    mlir::ValueRange operands,
    typename Op::Properties& properties) {
    captured_diag_.clear();
    mlir::ScopedDiagnosticHandler handler(context_,
                                          [this](mlir::Diagnostic& diagnostic) {
                                            captured_diag_ = diagnostic.str();
                                            return mlir::success();
                                          });
    llvm::SmallVector<mlir::Type, 1> inferred_types;
    if (mlir::failed(Op::inferReturnTypes(context_,
                                          location,
                                          operands,
                                          mlir::DictionaryAttr{},
                                          mlir::OpaqueProperties(&properties),
                                          mlir::RegionRange{},
                                          inferred_types)) ||
        inferred_types.size() != 1) {
      return mlir::failure();
    }
    auto result =
      llvm::dyn_cast<mlir::RankedTensorType>(inferred_types.front());
    if (!result) {
      return mlir::failure();
    }
    return result;
  }

  template <typename Op>
  mlir::LogicalResult infer_tensor_results(
    mlir::Location location,
    mlir::ValueRange operands,
    typename Op::Properties& properties,
    llvm::SmallVectorImpl<mlir::Type>& inferred_types) {
    captured_diag_.clear();
    mlir::ScopedDiagnosticHandler handler(context_,
                                          [this](mlir::Diagnostic& diagnostic) {
                                            captured_diag_ = diagnostic.str();
                                            return mlir::success();
                                          });
    return Op::inferReturnTypes(context_,
                                location,
                                operands,
                                mlir::DictionaryAttr{},
                                mlir::OpaqueProperties(&properties),
                                mlir::RegionRange{},
                                inferred_types);
  }

  const std::string& captured_diagnostic() const noexcept;

 private:
  ImportResult prepare_model();
  ImportResult import_layer(const LayerContext& context);
  std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> finish(
    const ncnn_graph::Graph& source);

  mlir::MLIRContext* context_;
  ImportOptions options_;
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  mlir::ncnn::ModelOp model_;
  llvm::StringMap<mlir::Value> blobs_;
  std::string captured_diag_;
  std::size_t imported_input_count_ = 0;
  bool sparse_input_shapes_ = false;
};

ImportResult import_input(ImportContext& importer, const LayerContext& context);
ImportResult import_convolution(ImportContext& importer,
                                const LayerContext& context);
ImportResult import_convolution_depthwise(ImportContext& importer,
                                          const LayerContext& context);
ImportResult import_deconvolution(ImportContext& importer,
                                  const LayerContext& context);
ImportResult import_padding(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_interp(ImportContext& importer,
                           const LayerContext& context);
ImportResult import_sigmoid(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_detection_output(ImportContext& importer,
                                     const LayerContext& context);
ImportResult import_hard_sigmoid(ImportContext& importer,
                                 const LayerContext& context);
ImportResult import_hard_swish(ImportContext& importer,
                               const LayerContext& context);
ImportResult import_reshape(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_binary_op(ImportContext& importer,
                              const LayerContext& context);
ImportResult import_inner_product(ImportContext& importer,
                                  const LayerContext& context);
ImportResult import_shuffle_channel(ImportContext& importer,
                                    const LayerContext& context);
ImportResult import_slice(ImportContext& importer, const LayerContext& context);
ImportResult import_reduction(ImportContext& importer,
                              const LayerContext& context);
ImportResult import_pooling(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_relu(ImportContext& importer, const LayerContext& context);
ImportResult import_dropout(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_softmax(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_split(ImportContext& importer, const LayerContext& context);
ImportResult import_concat(ImportContext& importer,
                           const LayerContext& context);
ImportResult import_gelu(ImportContext& importer, const LayerContext& context);
ImportResult import_squeeze(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_batch_norm(ImportContext& importer,
                               const LayerContext& context);
ImportResult import_expand_dims(ImportContext& importer,
                                const LayerContext& context);
ImportResult import_permute(ImportContext& importer,
                            const LayerContext& context);
ImportResult import_gemm(ImportContext& importer, const LayerContext& context);
ImportResult import_quantize(ImportContext& importer,
                             const LayerContext& context);
ImportResult import_dequantize(ImportContext& importer,
                               const LayerContext& context);
ImportResult import_requantize(ImportContext& importer,
                               const LayerContext& context);
ImportResult import_cast(ImportContext& importer, const LayerContext& context);

}  // namespace ncnn_importer::detail
