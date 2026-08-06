#include "ncnn-mlir/Importer/NCNNImporter.hpp"

#include "ImporterInternal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Verifier.h"

namespace ncnn_importer {

ImportError::ImportError(std::size_t layer_index,
                         std::string layer_type,
                         std::string layer_name,
                         std::string message)
  : layer_index_(layer_index),
    layer_type_(std::move(layer_type)),
    layer_name_(std::move(layer_name)),
    message_(std::move(message)) {}

std::size_t ImportError::get_layer_index() const noexcept {
  return layer_index_;
}

std::string_view ImportError::get_layer_type() const noexcept {
  return layer_type_;
}

std::string_view ImportError::get_layer_name() const noexcept {
  return layer_name_;
}

std::string_view ImportError::get_message() const noexcept {
  return message_;
}

std::string ImportError::to_string() const {
  return std::format(
    "layer {} ({}, {}): {}", layer_index_, layer_type_, layer_name_, message_);
}

namespace detail {
namespace {

using ImportHandler = ImportResult (*)(ImportContext&, const LayerContext&);

struct ImportEntry {
  std::string_view type;
  ImportHandler handler;
};

std::span<const ImportEntry> importers() noexcept {
  static constexpr std::array kImporters{
    ImportEntry{.type = "Input", .handler = import_input},
    ImportEntry{.type = "Convolution", .handler = import_convolution},
    ImportEntry{.type = "ConvolutionDepthWise",
                .handler = import_convolution_depthwise},
    ImportEntry{.type = "HardSigmoid", .handler = import_hard_sigmoid},
    ImportEntry{.type = "HardSwish", .handler = import_hard_swish},
    ImportEntry{.type = "Reshape", .handler = import_reshape},
    ImportEntry{.type = "BinaryOp", .handler = import_binary_op},
    ImportEntry{.type = "InnerProduct", .handler = import_inner_product},
    ImportEntry{.type = "ReLU", .handler = import_relu},
    ImportEntry{.type = "Pooling", .handler = import_pooling},
    ImportEntry{.type = "Split", .handler = import_split},
    ImportEntry{.type = "Concat", .handler = import_concat},
    ImportEntry{.type = "Dropout", .handler = import_dropout},
    ImportEntry{.type = "Softmax", .handler = import_softmax},
  };
  return kImporters;
}

std::expected<mlir::DenseElementsAttr, std::string> make_dense_attr(
  mlir::MLIRContext* context, const ncnn_graph::Tensor& tensor) {
  mlir::Type element;
  std::size_t width = 0;
  switch (tensor.get_dtype()) {
    case ncnn_graph::DataType::Float32:
      element = mlir::Float32Type::get(context);
      width = 4;
      break;
    case ncnn_graph::DataType::Float16:
      element = mlir::Float16Type::get(context);
      width = 2;
      break;
    case ncnn_graph::DataType::Int8:
      element = mlir::IntegerType::get(context, 8);
      width = 1;
      break;
    case ncnn_graph::DataType::Unknown:
      return std::unexpected("constant has unknown element type");
  }
  llvm::SmallVector<std::int64_t> shape(tensor.get_shape().begin(),
                                        tensor.get_shape().end());
  auto type = mlir::RankedTensorType::get(shape, element);
  const std::span<const std::byte> raw = tensor.get_data();
  if (raw.size() != tensor.byte_size()) {
    return std::unexpected("constant payload size does not match its type");
  }
  std::vector<char> native(raw.size());
  std::memcpy(native.data(), raw.data(), raw.size());
  if constexpr (std::endian::native == std::endian::big) {
    for (std::size_t offset = 0; offset + width <= native.size();
         offset += width) {
      std::reverse(
        native.begin() + static_cast<std::ptrdiff_t>(offset),
        native.begin() + static_cast<std::ptrdiff_t>(offset + width));
    }
  }
  auto attr = mlir::DenseElementsAttr::getFromRawBuffer(
    type, llvm::ArrayRef<char>(native));
  if (!attr) {
    return std::unexpected("constant payload is not a valid dense buffer");
  }
  return attr;
}

}  // namespace

ImportError make_error(const LayerContext& context, std::string message) {
  return {context.index,
          std::string(context.layer.get_type()),
          std::string(context.layer.get_name()),
          std::move(message)};
}

const ncnn_graph::ParamValue* find_param(const ncnn_graph::ParamDict& params,
                                         int id) {
  for (const auto& entry : params.get_entries()) {
    if (entry.first == id) {
      return &entry.second;
    }
  }
  return nullptr;
}

std::expected<void, std::string> validate_param_ids(
  const ncnn_graph::ParamDict& params, std::span<const int> allowed) {
  for (const auto& entry : params.get_entries()) {
    if ((entry.first == 30 || entry.first == 31) ||
        std::ranges::find(allowed, entry.first) != allowed.end()) {
      continue;
    }
    return std::unexpected(
      std::format("unsupported parameter id {}", entry.first));
  }
  return {};
}

std::expected<std::int64_t, std::string> get_int(
  const ncnn_graph::ParamDict& params,
  int id,
  std::int64_t default_value,
  std::string_view name) {
  const auto* value = find_param(params, id);
  if (value == nullptr) {
    return default_value;
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Int) {
    return std::unexpected(
      std::format("parameter {} ({}) must be integer", id, name));
  }
  return *value->get_int();
}

std::expected<float, std::string> get_float(const ncnn_graph::ParamDict& params,
                                            int id,
                                            float default_value,
                                            std::string_view name) {
  const auto* value = find_param(params, id);
  if (value == nullptr) {
    return default_value;
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Float) {
    return std::unexpected(
      std::format("parameter {} ({}) must be float", id, name));
  }
  if (!std::isfinite(*value->get_float())) {
    return std::unexpected(
      std::format("parameter {} ({}) must be finite", id, name));
  }
  return *value->get_float();
}

std::expected<void, std::string> expect_boolean(std::int64_t value,
                                                std::string_view name) {
  if (value != 0 && value != 1) {
    return std::unexpected(std::format("{} must be 0 or 1", name));
  }
  return {};
}

std::expected<void, std::string> expect_source_arity(
  const ncnn_graph::Layer& layer, std::size_t inputs, std::size_t outputs) {
  if (layer.get_inputs().size() != inputs ||
      layer.get_outputs().size() != outputs) {
    return std::unexpected(
      std::format("requires {} inputs and {} outputs, got {} and {}",
                  inputs,
                  outputs,
                  layer.get_inputs().size(),
                  layer.get_outputs().size()));
  }
  return {};
}

std::expected<void, std::string> validate_feature_mask(
  const ncnn_graph::ParamDict& params) {
  const auto* value = find_param(params, 31);
  if (value == nullptr) {
    return {};
  }
  if (value->get_kind() != ncnn_graph::ParamValue::Kind::Int ||
      !std::in_range<int>(*value->get_int())) {
    return std::unexpected(
      "parameter 31 (feature mask) must fit the ncnn int type");
  }
  return {};
}

ImportContext::ImportContext(mlir::MLIRContext& context,
                             const ImportOptions& options)
  : context_(&context),
    options_(options),
    builder_(&context),
    module_(mlir::ModuleOp::create(builder_.getUnknownLoc())),
    model_(),
    blobs_(),
    captured_diag_() {
  builder_.setInsertionPointToEnd(module_->getBody());
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
ImportContext::run(const ncnn_graph::Graph& source) {
  if (options_.input_shape && source.layer_count_of("Input") != 1) {
    return std::unexpected(
      ImportError(0,
                  "Input",
                  "input-shape",
                  "input shape override requires exactly one Input layer"));
  }
  auto prepared = prepare_model();
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  for (std::size_t index = 0; index < source.get_layers().size(); ++index) {
    auto imported =
      import_layer({.index = index, .layer = source.get_layers()[index]});
    if (!imported) {
      return std::unexpected(imported.error());
    }
  }
  return finish(source);
}

mlir::OpBuilder& ImportContext::builder() noexcept {
  return builder_;
}

const std::optional<std::vector<std::int64_t>>& ImportContext::input_shape()
  const noexcept {
  return options_.input_shape;
}

std::expected<mlir::Value, ImportError> ImportContext::find_blob(
  const LayerContext& context, std::string_view name) {
  auto iterator = blobs_.find(name);
  if (iterator == blobs_.end()) {
    return std::unexpected(
      make_error(context, std::format("unresolved blob {}", name)));
  }
  return iterator->second;
}

ImportResult ImportContext::bind_blob(const LayerContext& context,
                                      std::string name,
                                      mlir::Value value) {
  auto inserted = blobs_.insert({name, value});
  if (!inserted.second) {
    return std::unexpected(make_error(
      context, std::format("blob {} has multiple definitions", name)));
  }
  return {};
}

std::expected<mlir::Value, ImportError> ImportContext::make_constant(
  const LayerContext& context,
  const ncnn_graph::Tensor& tensor,
  std::size_t weight_index) {
  auto attr = make_dense_attr(context_, tensor);
  if (!attr) {
    return std::unexpected(make_error(context, attr.error()));
  }
  auto constant = builder_.create<mlir::ncnn::ConstOp>(
    builder_.getUnknownLoc(),
    attr->getType(),
    builder_.getStringAttr(
      std::format("{}.weight.{}", context.layer.get_name(), weight_index)),
    *attr);
  tag_source(constant.getOperation(), context);
  return constant.getOutput();
}

void ImportContext::tag_source(mlir::Operation* operation,
                               const LayerContext& context) {
  operation->setAttr("ncnn.name",
                     builder_.getStringAttr(context.layer.get_name()));
  operation->setAttr(
    "ncnn.source_layer",
    builder_.getI64IntegerAttr(static_cast<std::int64_t>(context.index)));
}

const std::string& ImportContext::captured_diagnostic() const noexcept {
  return captured_diag_;
}

ImportResult ImportContext::prepare_model() {
  model_ = builder_.create<mlir::ncnn::ModelOp>(
    builder_.getUnknownLoc(), builder_.getStringAttr("model"));
  mlir::Block* block = &model_.getBody().emplaceBlock();
  builder_.setInsertionPointToStart(block);
  return {};
}

ImportResult ImportContext::import_layer(const LayerContext& context) {
  const auto entries = importers();
  const auto importer =
    std::ranges::find(entries, context.layer.get_type(), &ImportEntry::type);
  if (importer == entries.end()) {
    return std::unexpected(make_error(
      context,
      std::format("unsupported layer type {}", context.layer.get_type())));
  }
  return importer->handler(*this, context);
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError>
ImportContext::finish(const ncnn_graph::Graph& source) {
  llvm::SmallVector<mlir::Value> outputs;
  for (const auto& name : source.get_output_blob_names()) {
    auto value = blobs_.find(name);
    if (value == blobs_.end()) {
      return std::unexpected(ImportError(source.get_layers().size(),
                                         "GraphOutput",
                                         name,
                                         "unresolved output blob"));
    }
    outputs.push_back(value->second);
  }
  builder_.setInsertionPointToEnd(&model_.getBody().front());
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    builder_.create<mlir::ncnn::OutputOp>(
      builder_.getUnknownLoc(),
      outputs[index],
      builder_.getStringAttr(source.get_output_blob_names()[index]));
  }
  captured_diag_.clear();
  mlir::ScopedDiagnosticHandler handler(context_,
                                        [this](mlir::Diagnostic& diagnostic) {
                                          captured_diag_ = diagnostic.str();
                                          return mlir::success();
                                        });
  if (mlir::failed(mlir::verify(module_.get().getOperation()))) {
    return std::unexpected(ImportError(
      source.get_layers().size(),
      "Verifier",
      "module",
      captured_diag_.empty() ? "module verification failed" : captured_diag_));
  }
  return std::move(module_);
}

}  // namespace detail

bool has_layer_importer(std::string_view layer_type) noexcept {
  const auto entries = detail::importers();
  return std::ranges::find(entries, layer_type, &detail::ImportEntry::type) !=
         entries.end();
}

std::size_t get_layer_importer_count() noexcept {
  return detail::importers().size();
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> import_graph(
  const ncnn_graph::Graph& graph, mlir::MLIRContext& context) {
  return import_graph(graph, context, ImportOptions{});
}

std::expected<mlir::OwningOpRef<mlir::ModuleOp>, ImportError> import_graph(
  const ncnn_graph::Graph& graph,
  mlir::MLIRContext& context,
  const ImportOptions& options) {
  detail::ImportContext importer(context, options);
  return importer.run(graph);
}

}  // namespace ncnn_importer
