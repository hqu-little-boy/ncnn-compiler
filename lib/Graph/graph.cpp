#include "ncnn-mlir/Graph/graph.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include "ncnn-mlir/Graph/parser.hpp"

namespace ncnn_graph {

// ───────────────────────── ParamValue ─────────────────────────
ParamValue ParamValue::make_int(std::int64_t value) {
  return ParamValue(Storage(value));
}

ParamValue ParamValue::make_float(float value) {
  return ParamValue(Storage(value));
}

ParamValue ParamValue::make_int_array(std::vector<std::int64_t> value) {
  return ParamValue(Storage(std::move(value)));
}

ParamValue ParamValue::make_float_array(std::vector<float> value) {
  return ParamValue(Storage(std::move(value)));
}

ParamValue ParamValue::make_string(std::string value) {
  return ParamValue(Storage(std::move(value)));
}

ParamValue::Kind ParamValue::get_kind() const noexcept {
  switch (storage_.index()) {
    case 0:
      return Kind::Int;
    case 1:
      return Kind::Float;
    case 2:
      return Kind::IntArray;
    case 3:
      return Kind::FloatArray;
    case 4:
      return Kind::String;
    default:
      return Kind::Int;
  }
}

std::optional<std::int64_t> ParamValue::get_int() const noexcept {
  const auto* value = std::get_if<std::int64_t>(&storage_);
  return value ? std::optional(*value) : std::nullopt;
}

std::optional<float> ParamValue::get_float() const noexcept {
  const auto* value = std::get_if<float>(&storage_);
  return value ? std::optional(*value) : std::nullopt;
}

std::optional<std::span<const std::int64_t>> ParamValue::get_int_array()
  const noexcept {
  const auto* value = std::get_if<std::vector<std::int64_t>>(&storage_);
  return value ? std::optional(std::span<const std::int64_t>(*value))
               : std::nullopt;
}

std::optional<std::span<const float>> ParamValue::get_float_array()
  const noexcept {
  const auto* value = std::get_if<std::vector<float>>(&storage_);
  return value ? std::optional(std::span<const float>(*value)) : std::nullopt;
}

std::optional<std::string_view> ParamValue::get_string() const noexcept {
  const auto* value = std::get_if<std::string>(&storage_);
  return value ? std::optional(std::string_view(*value)) : std::nullopt;
}

ParamValue::ParamValue(Storage storage) : storage_(std::move(storage)) {}

// ───────────────────────── ParamDict ─────────────────────────
ParamDict::ParamDict() : entries_() {}

std::optional<std::reference_wrapper<const ParamValue>> ParamDict::find_value(
  int id) const noexcept {
  auto iterator = std::ranges::find(
    entries_, id, [](const auto& entry) { return entry.first; });
  if (iterator == entries_.end()) {
    return std::nullopt;
  }
  return std::cref(iterator->second);
}

bool ParamDict::has(int id) const noexcept {
  return find_value(id).has_value();
}

std::span<const std::pair<int, ParamValue>> ParamDict::get_entries()
  const noexcept {
  return entries_;
}

void ParamDict::set_value(int id, ParamValue value) {
  auto iterator = std::ranges::find(
    entries_, id, [](const auto& entry) { return entry.first; });
  if (iterator != entries_.end()) {
    iterator->second = std::move(value);
    return;
  }
  entries_.emplace_back(id, std::move(value));
}

std::int64_t ParamDict::get_int(int id, std::int64_t default_value) const {
  auto value = find_value(id);
  if (!value || value->get().get_kind() != ParamValue::Kind::Int) {
    return default_value;
  }
  return value->get().get_int().value_or(default_value);
}

float ParamDict::get_float(int id, float default_value) const {
  auto value = find_value(id);
  if (!value || value->get().get_kind() != ParamValue::Kind::Float) {
    return default_value;
  }
  return value->get().get_float().value_or(default_value);
}

std::optional<std::reference_wrapper<const std::string>> ParamDict::get_string(
  int id) const noexcept {
  auto value = find_value(id);
  if (!value || value->get().get_kind() != ParamValue::Kind::String) {
    return std::nullopt;
  }
  const auto* string_value = std::get_if<std::string>(&value->get().storage_);
  if (!string_value) {
    return std::nullopt;
  }
  return std::cref(*string_value);
}

std::span<const std::int64_t> ParamDict::get_int_array(int id) const {
  auto value = find_value(id);
  if (!value) {
    return {};
  }
  return value->get().get_int_array().value_or(std::span<const std::int64_t>());
}

std::span<const float> ParamDict::get_float_array(int id) const {
  auto value = find_value(id);
  if (!value) {
    return {};
  }
  return value->get().get_float_array().value_or(std::span<const float>());
}

namespace {

std::expected<std::int64_t, std::string> decode_int_param(
  const ParamDict& params,
  int id,
  std::int64_t default_value,
  std::string_view name) {
  for (const auto& [entry_id, value] : params.get_entries()) {
    if (entry_id != id) {
      continue;
    }
    auto result = value.get_int();
    if (!result) {
      return std::unexpected(
        std::format("parameter {} ({}) must be integer", id, name));
    }
    return *result;
  }
  return default_value;
}

std::expected<float, std::string> decode_float_param(const ParamDict& params,
                                                     int id,
                                                     float default_value,
                                                     std::string_view name) {
  for (const auto& [entry_id, value] : params.get_entries()) {
    if (entry_id != id) {
      continue;
    }
    auto result = value.get_float();
    if (!result) {
      return std::unexpected(
        std::format("parameter {} ({}) must be float", id, name));
    }
    if (!std::isfinite(*result)) {
      return std::unexpected(
        std::format("parameter {} ({}) must be finite", id, name));
    }
    return *result;
  }
  return default_value;
}

}  // namespace

std::size_t ConvolutionParams::expected_weight_tensors() const noexcept {
  if (dynamic_weight) {
    return 0;
  }
  return 1 + static_cast<std::size_t>(has_bias) +
         (int8_scale_term == 0 ? 0 : 2) + (int8_scale_term > 100 ? 1 : 0);
}

std::expected<ConvolutionParams, std::string> decode_convolution_params(
  const ParamDict& params) {
  ConvolutionParams result;
  auto dynamic_weight = decode_int_param(params, 19, 0, "dynamic_weight");
  if (!dynamic_weight) {
    return std::unexpected(dynamic_weight.error());
  }
  if (*dynamic_weight != 0 && *dynamic_weight != 1) {
    return std::unexpected("convolution dynamic_weight must be 0 or 1");
  }
  result.dynamic_weight = *dynamic_weight == 1;
  if (result.dynamic_weight) {
    return result;
  }

  auto output_channels = decode_int_param(params, 0, 0, "num_output");
  auto kernel_w = decode_int_param(params, 1, 0, "kernel_w");
  if (!output_channels || !kernel_w) {
    return std::unexpected(!output_channels ? output_channels.error()
                                            : kernel_w.error());
  }
  auto kernel_h = decode_int_param(params, 11, *kernel_w, "kernel_h");
  auto dilation_w = decode_int_param(params, 2, 1, "dilation_w");
  auto dilation_h = decode_int_param(params, 12, *dilation_w, "dilation_h");
  auto stride_w = decode_int_param(params, 3, 1, "stride_w");
  auto stride_h = decode_int_param(params, 13, *stride_w, "stride_h");
  auto pad_left = decode_int_param(params, 4, 0, "pad_left");
  auto pad_right = decode_int_param(params, 15, *pad_left, "pad_right");
  auto pad_top = decode_int_param(params, 14, *pad_left, "pad_top");
  auto pad_bottom = decode_int_param(params, 16, *pad_top, "pad_bottom");
  auto bias_term = decode_int_param(params, 5, 0, "bias_term");
  auto weight_count = decode_int_param(params, 6, 0, "weight_data_size");
  auto int8_scale_term = decode_int_param(params, 8, 0, "int8_scale_term");
  auto activation_type = decode_int_param(params, 9, 0, "activation_type");
  auto pad_value = decode_float_param(params, 18, 0.0F, "pad_value");
  if (!kernel_h || !dilation_w || !dilation_h || !stride_w || !stride_h ||
      !pad_left || !pad_right || !pad_top || !pad_bottom || !bias_term ||
      !weight_count || !int8_scale_term || !activation_type || !pad_value) {
    return std::unexpected("invalid convolution parameter type");
  }
  if (*output_channels <= 0) {
    return std::unexpected("convolution num_output must be positive");
  }
  if (*weight_count <= 0) {
    return std::unexpected("convolution weight count must be positive");
  }
  if (*kernel_w <= 0 || *kernel_h <= 0) {
    return std::unexpected("convolution kernel dimensions must be positive");
  }
  if (*dilation_w <= 0 || *dilation_h <= 0) {
    return std::unexpected("convolution dilation must be positive");
  }
  if (*stride_w <= 0 || *stride_h <= 0) {
    return std::unexpected("convolution stride must be positive");
  }
  if (*bias_term != 0 && *bias_term != 1) {
    return std::unexpected("convolution bias_term must be 0 or 1");
  }
  if (*int8_scale_term != 0 && *int8_scale_term != 1 && *int8_scale_term != 2 &&
      *int8_scale_term != 101 && *int8_scale_term != 102) {
    return std::unexpected(
      "convolution int8_scale_term must be 0, 1, 2, 101, or 102");
  }

  result.output_channels = *output_channels;
  result.kernel_w = *kernel_w;
  result.kernel_h = *kernel_h;
  result.dilation_w = *dilation_w;
  result.dilation_h = *dilation_h;
  result.stride_w = *stride_w;
  result.stride_h = *stride_h;
  result.pad_left = *pad_left;
  result.pad_right = *pad_right;
  result.pad_top = *pad_top;
  result.pad_bottom = *pad_bottom;
  result.has_bias = *bias_term == 1;
  result.weight_count = *weight_count;
  result.int8_scale_term = *int8_scale_term;
  result.activation_type = *activation_type;
  result.has_activation_params = params.has(10);
  result.pad_value = *pad_value;
  return result;
}

std::size_t ConvolutionDepthWiseParams::expected_weight_tensors()
  const noexcept {
  if (dynamic_weight) {
    return 0;
  }
  return 1 + static_cast<std::size_t>(has_bias) +
         (int8_scale_term == 0 ? 0 : 2) + (int8_scale_term > 100 ? 1 : 0);
}

std::expected<ConvolutionDepthWiseParams, std::string>
decode_convolution_depthwise_params(const ParamDict& params) {
  ConvolutionDepthWiseParams result;
  auto dynamic_weight = decode_int_param(params, 19, 0, "dynamic_weight");
  if (!dynamic_weight) {
    return std::unexpected(dynamic_weight.error());
  }
  if (*dynamic_weight != 0 && *dynamic_weight != 1) {
    return std::unexpected(
      "convolution depthwise dynamic_weight must be 0 or 1");
  }
  result.dynamic_weight = *dynamic_weight == 1;
  if (result.dynamic_weight) {
    return result;
  }

  auto output_channels = decode_int_param(params, 0, 0, "num_output");
  auto kernel_w = decode_int_param(params, 1, 0, "kernel_w");
  if (!output_channels || !kernel_w) {
    return std::unexpected(!output_channels ? output_channels.error()
                                            : kernel_w.error());
  }
  auto kernel_h = decode_int_param(params, 11, *kernel_w, "kernel_h");
  auto dilation_w = decode_int_param(params, 2, 1, "dilation_w");
  auto dilation_h = decode_int_param(params, 12, *dilation_w, "dilation_h");
  auto stride_w = decode_int_param(params, 3, 1, "stride_w");
  auto stride_h = decode_int_param(params, 13, *stride_w, "stride_h");
  auto pad_left = decode_int_param(params, 4, 0, "pad_left");
  auto pad_right = decode_int_param(params, 15, *pad_left, "pad_right");
  auto pad_top = decode_int_param(params, 14, *pad_left, "pad_top");
  auto pad_bottom = decode_int_param(params, 16, *pad_top, "pad_bottom");
  auto group = decode_int_param(params, 7, 1, "group");
  auto bias_term = decode_int_param(params, 5, 0, "bias_term");
  auto weight_count = decode_int_param(params, 6, 0, "weight_data_size");
  auto int8_scale_term = decode_int_param(params, 8, 0, "int8_scale_term");
  if (!kernel_h || !dilation_w || !dilation_h || !stride_w || !stride_h ||
      !pad_left || !pad_right || !pad_top || !pad_bottom || !group ||
      !bias_term || !weight_count || !int8_scale_term) {
    return std::unexpected("invalid convolution depthwise parameter type");
  }
  if (*output_channels <= 0) {
    return std::unexpected("convolution depthwise num_output must be positive");
  }
  if (*kernel_w <= 0 || *kernel_h <= 0) {
    return std::unexpected(
      "convolution depthwise kernel dimensions must be positive");
  }
  if (*dilation_w <= 0 || *dilation_h <= 0) {
    return std::unexpected("convolution depthwise dilation must be positive");
  }
  if (*stride_w <= 0 || *stride_h <= 0) {
    return std::unexpected("convolution depthwise stride must be positive");
  }
  if (*weight_count <= 0) {
    return std::unexpected(
      "convolution depthwise weight count must be positive");
  }
  if (*group <= 0 || *output_channels % *group != 0) {
    return std::unexpected(
      "convolution depthwise group must be positive and divide num_output");
  }
  if (*bias_term != 0 && *bias_term != 1) {
    return std::unexpected("convolution depthwise bias_term must be 0 or 1");
  }
  if (*int8_scale_term != 0 && *int8_scale_term != 1 && *int8_scale_term != 2 &&
      *int8_scale_term != 101 && *int8_scale_term != 102) {
    return std::unexpected(
      "convolution depthwise int8_scale_term must be 0, 1, 2, 101, or 102");
  }
  result.output_channels = *output_channels;
  result.kernel_w = *kernel_w;
  result.kernel_h = *kernel_h;
  result.dilation_w = *dilation_w;
  result.dilation_h = *dilation_h;
  result.stride_w = *stride_w;
  result.stride_h = *stride_h;
  result.pad_left = *pad_left;
  result.pad_right = *pad_right;
  result.pad_top = *pad_top;
  result.pad_bottom = *pad_bottom;
  result.group = *group;
  result.has_bias = *bias_term == 1;
  result.weight_count = *weight_count;
  result.int8_scale_term = *int8_scale_term;
  return result;
}

std::size_t DeconvolutionParams::expected_weight_tensors() const noexcept {
  return dynamic_weight ? 0 : 1 + static_cast<std::size_t>(has_bias);
}

std::expected<DeconvolutionParams, std::string> decode_deconvolution_params(
  const ParamDict& params) {
  DeconvolutionParams result;
  auto outputChannels = decode_int_param(params, 0, 0, "num_output");
  auto kernelW = decode_int_param(params, 1, 0, "kernel_w");
  if (!outputChannels || !kernelW) {
    return std::unexpected(!outputChannels ? outputChannels.error()
                                           : kernelW.error());
  }
  auto kernelH = decode_int_param(params, 11, *kernelW, "kernel_h");
  auto dilationW = decode_int_param(params, 2, 1, "dilation_w");
  auto dilationH = decode_int_param(params, 12, *dilationW, "dilation_h");
  auto strideW = decode_int_param(params, 3, 1, "stride_w");
  auto strideH = decode_int_param(params, 13, *strideW, "stride_h");
  auto padLeft = decode_int_param(params, 4, 0, "pad_left");
  auto padRight = decode_int_param(params, 15, *padLeft, "pad_right");
  auto padTop = decode_int_param(params, 14, *padLeft, "pad_top");
  auto padBottom = decode_int_param(params, 16, *padTop, "pad_bottom");
  auto outputPadRight = decode_int_param(params, 18, 0, "output_pad_right");
  auto outputPadBottom =
    decode_int_param(params, 19, *outputPadRight, "output_pad_bottom");
  auto outputW = decode_int_param(params, 20, 0, "output_w");
  auto outputH = decode_int_param(params, 21, *outputW, "output_h");
  auto biasTerm = decode_int_param(params, 5, 0, "bias_term");
  auto weightCount = decode_int_param(params, 6, 0, "weight_data_size");
  auto activationType = decode_int_param(params, 9, 0, "activation_type");
  auto dynamicWeight = decode_int_param(params, 28, 0, "dynamic_weight");
  if (!kernelH || !dilationW || !dilationH || !strideW || !strideH ||
      !padLeft || !padRight || !padTop || !padBottom || !outputPadRight ||
      !outputPadBottom || !outputW || !outputH || !biasTerm || !weightCount ||
      !activationType || !dynamicWeight) {
    return std::unexpected("invalid deconvolution parameter type");
  }
  if (*dynamicWeight != 0 && *dynamicWeight != 1) {
    return std::unexpected("deconvolution dynamic_weight must be 0 or 1");
  }
  result.dynamic_weight = *dynamicWeight == 1;
  if (result.dynamic_weight) {
    return result;
  }
  if (*outputChannels <= 0 || *kernelW <= 0 || *kernelH <= 0 ||
      *dilationW <= 0 || *dilationH <= 0 || *strideW <= 0 || *strideH <= 0 ||
      *weightCount <= 0) {
    return std::unexpected(
      "deconvolution channels, kernel, stride, dilation, and weight count "
      "must be positive");
  }
  if (*padLeft < 0 || *padRight < 0 || *padTop < 0 || *padBottom < 0 ||
      *outputPadRight < 0 || *outputPadBottom < 0 || *outputW < 0 ||
      *outputH < 0) {
    return std::unexpected(
      "deconvolution padding and output dimensions must be non-negative");
  }
  if (*biasTerm != 0 && *biasTerm != 1) {
    return std::unexpected("deconvolution bias_term must be 0 or 1");
  }
  result.output_channels = *outputChannels;
  result.kernel_w = *kernelW;
  result.kernel_h = *kernelH;
  result.dilation_w = *dilationW;
  result.dilation_h = *dilationH;
  result.stride_w = *strideW;
  result.stride_h = *strideH;
  result.pad_left = *padLeft;
  result.pad_right = *padRight;
  result.pad_top = *padTop;
  result.pad_bottom = *padBottom;
  result.output_pad_right = *outputPadRight;
  result.output_pad_bottom = *outputPadBottom;
  result.output_w = *outputW;
  result.output_h = *outputH;
  result.has_bias = *biasTerm == 1;
  result.weight_count = *weightCount;
  result.activation_type = *activationType;
  result.has_activation_params = params.has(10);
  return result;
}

std::size_t InnerProductParams::expected_weight_tensors() const noexcept {
  return 1 + static_cast<std::size_t>(has_bias) +
         (int8_scale_term == 0 ? 0 : 2);
}

std::expected<InnerProductParams, std::string> decode_inner_product_params(
  const ParamDict& params) {
  InnerProductParams result;
  auto output_channels = decode_int_param(params, 0, 0, "num_output");
  auto bias_term = decode_int_param(params, 1, 0, "bias_term");
  auto weight_count = decode_int_param(params, 2, 0, "weight_data_size");
  auto int8_scale_term = decode_int_param(params, 8, 0, "int8_scale_term");
  if (!output_channels || !bias_term || !weight_count || !int8_scale_term) {
    return std::unexpected("invalid inner product parameter type");
  }
  if (*output_channels <= 0) {
    return std::unexpected("inner product num_output must be positive");
  }
  if (*weight_count <= 0 || *weight_count % *output_channels != 0) {
    return std::unexpected(
      "inner product weight count must be positive and divisible by "
      "num_output");
  }
  if (*bias_term != 0 && *bias_term != 1) {
    return std::unexpected("inner product bias_term must be 0 or 1");
  }
  if (*int8_scale_term != 0 && *int8_scale_term != 1 && *int8_scale_term != 2) {
    return std::unexpected("inner product int8_scale_term must be 0, 1, or 2");
  }
  result.output_channels = *output_channels;
  result.has_bias = *bias_term == 1;
  result.weight_count = *weight_count;
  result.int8_scale_term = *int8_scale_term;
  return result;
}

std::expected<BatchNormParams, std::string> decode_batch_norm_params(
  const ParamDict& params) {
  auto channels = decode_int_param(params, 0, 0, "channels");
  auto epsilon = decode_float_param(params, 1, 0.0F, "eps");
  if (!channels || !epsilon) {
    return std::unexpected(!channels ? channels.error() : epsilon.error());
  }
  if (*channels <= 0 || *epsilon < 0.0F) {
    return std::unexpected(
      "BatchNorm channels must be positive and eps must be non-negative");
  }
  return BatchNormParams{.channels = *channels, .epsilon = *epsilon};
}

std::expected<LayerNormParams, std::string> decode_layer_norm_params(
  const ParamDict& params) {
  auto affine_size = decode_int_param(params, 0, 0, "affine_size");
  auto epsilon = decode_float_param(params, 1, 0.001F, "eps");
  auto affine = decode_int_param(params, 2, 1, "affine");
  if (!affine_size || !epsilon || !affine) {
    return std::unexpected(!affine_size ? affine_size.error()
                           : !epsilon   ? epsilon.error()
                                        : affine.error());
  }
  if (*affine != 0 && *affine != 1) {
    return std::unexpected("LayerNorm affine must be 0 or 1");
  }
  if (*epsilon < 0.0F) {
    return std::unexpected("LayerNorm eps must be non-negative");
  }
  if (*affine == 1 && *affine_size <= 0) {
    return std::unexpected(
      "LayerNorm affine_size must be positive when affine is enabled");
  }
  if (*affine == 0 && *affine_size < 0) {
    return std::unexpected("LayerNorm affine_size must be non-negative");
  }
  return LayerNormParams{
    .affine_size = *affine_size, .epsilon = *epsilon, .affine = *affine == 1};
}

std::expected<EmbedParams, std::string> decode_embed_params(
  const ParamDict& params) {
  auto outputChannels = decode_int_param(params, 0, 0, "num_output");
  auto inputDim = decode_int_param(params, 1, 0, "input_dim");
  auto bias = decode_int_param(params, 2, 0, "bias_term");
  auto weightCount = decode_int_param(params, 3, 0, "weight_data_size");
  auto int8Scale = decode_int_param(params, 18, 0, "int8_scale_term");
  if (!outputChannels || !inputDim || !bias || !weightCount || !int8Scale) {
    return std::unexpected("invalid Embed parameter type");
  }
  if (*outputChannels <= 0 || *inputDim <= 0 || (*bias != 0 && *bias != 1) ||
      *weightCount <= 0 ||
      *inputDim > std::numeric_limits<std::int64_t>::max() / *outputChannels ||
      *inputDim * *outputChannels != *weightCount) {
    return std::unexpected(
      "Embed requires positive dimensions and input_dim * num_output weights");
  }
  return EmbedParams{.output_channels = *outputChannels,
                     .input_dim = *inputDim,
                     .has_bias = *bias == 1,
                     .weight_count = *weightCount,
                     .int8_scale_term = *int8Scale};
}

std::expected<MultiHeadAttentionParams, std::string>
decode_multi_head_attention_params(const ParamDict& params) {
  auto embed_dim = decode_int_param(params, 0, 0, "embed_dim");
  auto num_heads = decode_int_param(params, 1, 1, "num_heads");
  auto weight_count = decode_int_param(params, 2, 0, "weight_data_size");
  if (!embed_dim || !num_heads || !weight_count) {
    return std::unexpected(!embed_dim   ? embed_dim.error()
                           : !num_heads ? num_heads.error()
                                        : weight_count.error());
  }
  if (*embed_dim <= 0 || *num_heads <= 0 || *embed_dim % *num_heads != 0) {
    return std::unexpected(
      "MultiHeadAttention embed_dim and num_heads must be positive and "
      "embed_dim must be divisible by num_heads");
  }
  if (*weight_count <= 0 || *weight_count % *embed_dim != 0) {
    return std::unexpected(
      "MultiHeadAttention weight_data_size must be positive and divisible by "
      "embed_dim");
  }

  auto key_dim = decode_int_param(params, 3, *embed_dim, "kdim");
  auto value_dim = decode_int_param(params, 4, *embed_dim, "vdim");
  auto attention_mask = decode_int_param(params, 5, 0, "attn_mask");
  const float default_scale =
    1.0F / std::sqrt(static_cast<float>(*embed_dim / *num_heads));
  auto scale = decode_float_param(params, 6, default_scale, "scale");
  auto kv_cache = decode_int_param(params, 7, 0, "kv_cache");
  auto int8_scale_term = decode_int_param(params, 18, 0, "int8_scale_term");
  if (!key_dim || !value_dim || !attention_mask || !scale || !kv_cache ||
      !int8_scale_term) {
    return std::unexpected(!key_dim          ? key_dim.error()
                           : !value_dim      ? value_dim.error()
                           : !attention_mask ? attention_mask.error()
                           : !scale          ? scale.error()
                           : !kv_cache       ? kv_cache.error()
                                             : int8_scale_term.error());
  }
  if (*key_dim <= 0 || *value_dim <= 0) {
    return std::unexpected("MultiHeadAttention kdim and vdim must be positive");
  }
  if ((*attention_mask != 0 && *attention_mask != 1) ||
      (*kv_cache != 0 && *kv_cache != 1)) {
    return std::unexpected(
      "MultiHeadAttention attn_mask and kv_cache must be 0 or 1");
  }

  return MultiHeadAttentionParams{.embed_dim = *embed_dim,
                                  .num_heads = *num_heads,
                                  .weight_count = *weight_count,
                                  .query_dim = *weight_count / *embed_dim,
                                  .key_dim = *key_dim,
                                  .value_dim = *value_dim,
                                  .has_attention_mask = *attention_mask == 1,
                                  .scale = *scale,
                                  .kv_cache = *kv_cache == 1,
                                  .int8_scale_term = *int8_scale_term};
}

std::expected<SDPAParams, std::string> decode_sdpa_params(
  const ParamDict& params) {
  auto attentionMask = decode_int_param(params, 5, 0, "attn_mask");
  auto scale = decode_float_param(params, 6, 0.0F, "scale");
  auto kvCache = decode_int_param(params, 7, 0, "kv_cache");
  auto int8ScaleTerm = decode_int_param(params, 18, 0, "int8_scale_term");
  if (!attentionMask || !scale || !kvCache || !int8ScaleTerm) {
    return std::unexpected(!attentionMask ? attentionMask.error()
                           : !scale       ? scale.error()
                           : !kvCache     ? kvCache.error()
                                          : int8ScaleTerm.error());
  }
  if ((*attentionMask != 0 && *attentionMask != 1) ||
      (*kvCache != 0 && *kvCache != 1)) {
    return std::unexpected("SDPA attn_mask and kv_cache must be 0 or 1");
  }
  if (*int8ScaleTerm != 0) {
    return std::unexpected("quantized SDPA is unsupported");
  }
  return SDPAParams{.has_attention_mask = *attentionMask == 1,
                    .scale = *scale,
                    .kv_cache = *kvCache == 1};
}

std::expected<GemmParams, std::string> decode_gemm_params(
  const ParamDict& params) {
  GemmParams result;
  auto alpha = decode_float_param(params, 0, 1.0F, "alpha");
  auto beta = decode_float_param(params, 1, 1.0F, "beta");
  auto trans_a = decode_int_param(params, 2, 0, "transA");
  auto trans_b = decode_int_param(params, 3, 0, "transB");
  auto constant_a = decode_int_param(params, 4, 0, "constantA");
  auto constant_b = decode_int_param(params, 5, 0, "constantB");
  auto constant_c = decode_int_param(params, 6, 0, "constantC");
  auto m = decode_int_param(params, 7, 0, "constantM");
  auto n = decode_int_param(params, 8, 0, "constantN");
  auto k = decode_int_param(params, 9, 0, "constantK");
  auto broadcast = decode_int_param(params, 10, 0, "broadcast_type_C");
  auto n1m = decode_int_param(params, 11, 0, "output_N1M");
  auto elempack = decode_int_param(params, 12, 0, "output_elempack");
  auto elemtype = decode_int_param(params, 13, 0, "output_elemtype");
  auto transpose = decode_int_param(params, 14, 0, "output_transpose");
  auto quantize = decode_int_param(params, 18, 0, "int8_scale_term");
  if (!alpha || !beta || !trans_a || !trans_b || !constant_a || !constant_b ||
      !constant_c || !m || !n || !k || !broadcast || !n1m || !elempack ||
      !elemtype || !transpose || !quantize) {
    return std::unexpected("invalid Gemm parameter type");
  }
  for (const auto value :
       {*trans_a, *trans_b, *constant_a, *constant_b, *constant_c}) {
    if (value != 0 && value != 1) {
      return std::unexpected("Gemm boolean parameter must be 0 or 1");
    }
  }
  if (*constant_b == 1 && (*n <= 0 || *k <= 0)) {
    return std::unexpected(
      "Gemm constantN and constantK must be positive for constant B");
  }
  if (*constant_c == 1 && (*broadcast < -1 || *broadcast > 4)) {
    return std::unexpected("Gemm C broadcast type must be -1 through 4");
  }
  result = GemmParams{.alpha = *alpha,
                      .beta = *beta,
                      .transpose_a = *trans_a == 1,
                      .transpose_b = *trans_b == 1,
                      .constant_a = *constant_a == 1,
                      .constant_b = *constant_b == 1,
                      .constant_c = *constant_c == 1,
                      .constant_m = *m,
                      .constant_n = *n,
                      .constant_k = *k,
                      .broadcast_c = *broadcast,
                      .output_n1m = *n1m,
                      .output_elempack = *elempack,
                      .output_elemtype = *elemtype,
                      .output_transpose = *transpose,
                      .int8_scale_term = *quantize};
  return result;
}

std::expected<MemoryDataParams, std::string> decode_memory_data_params(
  const ParamDict& params) {
  auto width = decode_int_param(params, 0, 0, "width");
  auto height = decode_int_param(params, 1, 0, "height");
  auto channels = decode_int_param(params, 2, 0, "channels");
  auto depth = decode_int_param(params, 11, 0, "depth");
  auto loadType = decode_int_param(params, 21, 1, "load_type");
  if (!width || !height || !channels || !depth || !loadType) {
    return std::unexpected("invalid MemoryData parameter type");
  }
  if (*width <= 0 || *height < 0 || *depth < 0 || *channels < 0 ||
      (*loadType != 0 && *loadType != 1)) {
    return std::unexpected(
      "MemoryData requires positive width, non-negative dimensions, and "
      "load_type 0 or 1");
  }
  if ((*depth != 0 && (*height == 0 || *channels == 0)) ||
      (*channels != 0 && *height == 0)) {
    return std::unexpected("MemoryData dimensions must be densely specified");
  }
  return MemoryDataParams{.width = *width,
                          .height = *height,
                          .depth = *depth,
                          .channels = *channels,
                          .load_type = *loadType};
}

std::expected<QuantizeParams, std::string> decode_quantize_params(
  const ParamDict& params) {
  auto count = decode_int_param(params, 0, 1, "scale_data_size");
  if (!count || *count <= 0) {
    return std::unexpected(count ? "Quantize scale count must be positive"
                                 : count.error());
  }
  return QuantizeParams{.scale_count = *count};
}

std::expected<DequantizeParams, std::string> decode_dequantize_params(
  const ParamDict& params) {
  auto scale = decode_int_param(params, 0, 1, "scale_data_size");
  auto bias = decode_int_param(params, 1, 0, "bias_data_size");
  if (!scale || !bias) {
    return std::unexpected(!scale ? scale.error() : bias.error());
  }
  if (*scale <= 0 || *bias < 0) {
    return std::unexpected(
      "Dequantize scale count must be positive and bias count non-negative");
  }
  return DequantizeParams{.scale_count = *scale, .bias_count = *bias};
}

std::expected<RequantizeParams, std::string> decode_requantize_params(
  const ParamDict& params) {
  auto input = decode_int_param(params, 0, 1, "scale_in_data_size");
  auto output = decode_int_param(params, 1, 1, "scale_out_data_size");
  auto bias = decode_int_param(params, 2, 0, "bias_data_size");
  auto activation = decode_int_param(params, 3, 0, "activation_type");
  if (!input || !output || !bias || !activation) {
    return std::unexpected("invalid Requantize parameter type");
  }
  if (*input <= 0 || *output <= 0 || *bias < 0 || *activation < 0 ||
      *activation > 6) {
    return std::unexpected("invalid Requantize scale, bias, or activation");
  }
  return RequantizeParams{.input_scale_count = *input,
                          .output_scale_count = *output,
                          .bias_count = *bias,
                          .activation_type = *activation};
}

std::expected<CastParams, std::string> decode_cast_params(
  const ParamDict& params) {
  auto from = decode_int_param(params, 0, 0, "type_from");
  auto to = decode_int_param(params, 1, 0, "type_to");
  if (!from || !to) {
    return std::unexpected(!from ? from.error() : to.error());
  }
  if (*from < 0 || *from > 4 || *to < 0 || *to > 4) {
    return std::unexpected("Cast type must be auto, fp32, fp16, int8, or bf16");
  }
  return CastParams{.type_from = *from, .type_to = *to};
}

// ───────────────────────── Tensor ─────────────────────────
Tensor::Tensor() : shape_(), dtype_(DataType::Unknown), data_() {}

std::span<const std::int64_t> Tensor::get_shape() const noexcept {
  return shape_;
}

std::expected<void, std::string> Tensor::set_shape(
  std::vector<std::int64_t> shape) {
  std::size_t count = 1;
  for (std::int64_t dimension : shape) {
    if (dimension < 0) {
      return std::unexpected("tensor shape contains a negative dimension");
    }
    if (!std::in_range<std::size_t>(dimension)) {
      return std::unexpected("tensor shape dimension does not fit size_t");
    }
    auto unsigned_dimension = static_cast<std::size_t>(dimension);
    if (count != 0 &&
        unsigned_dimension > std::numeric_limits<std::size_t>::max() / count) {
      return std::unexpected("tensor element count overflows size_t");
    }
    count *= unsigned_dimension;
  }
  constexpr std::size_t kMaximumElementWidth = sizeof(float);
  if (count > std::numeric_limits<std::size_t>::max() / kMaximumElementWidth) {
    return std::unexpected("tensor byte size overflows size_t");
  }
  std::size_t element_width = 0;
  switch (dtype_) {
    case DataType::Unknown:
      break;
    case DataType::Float32:
      element_width = sizeof(float);
      break;
    case DataType::Float16:
    case DataType::BFloat16:
      element_width = 2;
      break;
    case DataType::Int8:
      element_width = 1;
      break;
  }
  if (dtype_ != DataType::Unknown && count * element_width != data_.size()) {
    return std::unexpected(
      std::format("tensor shape requires {} data bytes, tensor has {}",
                  count * element_width,
                  data_.size()));
  }
  shape_ = std::move(shape);
  return {};
}

DataType Tensor::get_dtype() const noexcept {
  return dtype_;
}

std::span<const std::byte> Tensor::get_data() const noexcept {
  return data_;
}

std::expected<void, std::string> Tensor::set_contents(
  std::vector<std::int64_t> shape,
  DataType dtype,
  std::vector<std::byte> data) {
  Tensor candidate;
  auto shape_result = candidate.set_shape(std::move(shape));
  if (!shape_result) {
    return std::unexpected(shape_result.error());
  }
  candidate.dtype_ = dtype;
  if (candidate.byte_size() != data.size()) {
    return std::unexpected(
      std::format("tensor byte size mismatch: shape and dtype require {}, "
                  "data has {}",
                  candidate.byte_size(),
                  data.size()));
  }
  shape_ = std::move(candidate.shape_);
  dtype_ = dtype;
  data_ = std::move(data);
  return {};
}

std::size_t Tensor::element_count() const noexcept {
  std::size_t count = 1;
  for (std::int64_t dimension : shape_) {
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

std::size_t Tensor::byte_size() const noexcept {
  switch (dtype_) {
    case DataType::Unknown:
      return 0;
    case DataType::Float32:
      return element_count() * sizeof(float);
    case DataType::Float16:
    case DataType::BFloat16:
      return element_count() * 2;
    case DataType::Int8:
      return element_count();
  }
  return 0;
}

// ───────────────────────── Blob ─────────────────────────
Blob::Blob(std::string name)
  : name_(std::move(name)), producer_(-1), consumer_(-1) {}

std::string_view Blob::get_name() const noexcept {
  return name_;
}

void Blob::set_name(std::string name) {
  name_ = std::move(name);
}

int Blob::get_producer() const noexcept {
  return producer_;
}

void Blob::set_producer(int producer) noexcept {
  producer_ = producer;
}

int Blob::get_consumer() const noexcept {
  return consumer_;
}

void Blob::set_consumer(int consumer) noexcept {
  consumer_ = consumer;
}

// ───────────────────────── Layer ─────────────────────────
Layer::Layer()
  : type_(), name_(), inputs_(), outputs_(), params_(), weights_() {}

std::string_view Layer::get_type() const noexcept {
  return type_;
}

void Layer::set_type(std::string type) {
  type_ = std::move(type);
}

std::string_view Layer::get_name() const noexcept {
  return name_;
}

void Layer::set_name(std::string name) {
  name_ = std::move(name);
}

std::span<const std::string> Layer::get_inputs() const noexcept {
  return inputs_;
}

void Layer::set_inputs(std::vector<std::string> inputs) {
  inputs_ = std::move(inputs);
}

void Layer::add_input(std::string input) {
  inputs_.push_back(std::move(input));
}

std::span<const std::string> Layer::get_outputs() const noexcept {
  return outputs_;
}

void Layer::set_outputs(std::vector<std::string> outputs) {
  outputs_ = std::move(outputs);
}

void Layer::add_output(std::string output) {
  outputs_.push_back(std::move(output));
}

const ParamDict& Layer::get_params() const noexcept {
  return params_;
}

void Layer::set_params(ParamDict params) {
  params_ = std::move(params);
}

std::span<const Tensor> Layer::get_weights() const noexcept {
  return weights_;
}

void Layer::set_weights(std::vector<Tensor> weights) {
  weights_ = std::move(weights);
}

void Layer::add_weight(Tensor weight) {
  weights_.push_back(std::move(weight));
}

std::int64_t Layer::get_param_int(int id, std::int64_t default_value) const {
  return params_.get_int(id, default_value);
}

float Layer::get_param_float(int id, float default_value) const {
  return params_.get_float(id, default_value);
}

// ───────────────────────── Graph ─────────────────────────
Graph::Graph()
  : layers_(),
    blobs_(),
    input_blob_names_(),
    output_blob_names_(),
    weights_loaded_(false) {}

std::span<const Layer> Graph::get_layers() const noexcept {
  return layers_;
}

void Graph::set_layers(std::vector<Layer> layers) {
  layers_ = std::move(layers);
}

void Graph::add_layer(Layer layer) {
  layers_.push_back(std::move(layer));
}

std::span<const Blob> Graph::get_blobs() const noexcept {
  return blobs_;
}

void Graph::set_blobs(std::vector<Blob> blobs) {
  blobs_ = std::move(blobs);
}

void Graph::add_blob(Blob blob) {
  blobs_.push_back(std::move(blob));
}

std::span<const std::string> Graph::get_input_blob_names() const noexcept {
  return input_blob_names_;
}

void Graph::set_input_blob_names(std::vector<std::string> names) {
  input_blob_names_ = std::move(names);
}

void Graph::add_input_blob_name(std::string name) {
  input_blob_names_.push_back(std::move(name));
}

std::span<const std::string> Graph::get_output_blob_names() const noexcept {
  return output_blob_names_;
}

void Graph::set_output_blob_names(std::vector<std::string> names) {
  output_blob_names_ = std::move(names);
}

void Graph::add_output_blob_name(std::string name) {
  output_blob_names_.push_back(std::move(name));
}

bool Graph::get_weights_loaded() const noexcept {
  return weights_loaded_;
}

void Graph::set_weights_loaded(bool weights_loaded) noexcept {
  weights_loaded_ = weights_loaded;
}

namespace {

constexpr std::uint32_t kFloat16Flag = 0x01306B47;
constexpr std::uint32_t kInt8Flag = 0x000D4B38;
constexpr std::uint32_t kFloat32Flag = 0x0002C056;
constexpr std::size_t kWeightAlignment = 4;

std::expected<std::size_t, std::string> checked_multiply(
  std::size_t left, std::size_t right, std::string_view description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return std::unexpected(std::format("{} overflows size_t", description));
  }
  return left * right;
}

std::expected<std::size_t, std::string> positive_size(
  std::int64_t value, std::string_view description) {
  if (value <= 0) {
    return std::unexpected(std::format("{} must be positive", description));
  }
  if (!std::in_range<std::size_t>(value)) {
    return std::unexpected(std::format("{} does not fit size_t", description));
  }
  return static_cast<std::size_t>(value);
}

template <typename Value>
std::expected<Value, std::string> parse_integer(std::string_view token,
                                                std::string_view description) {
  if (token.empty()) {
    return std::unexpected(std::format("empty {}", description));
  }
  Value value{};
  auto [end, error] =
    std::from_chars(token.data(), token.data() + token.size(), value);
  if (error == std::errc::result_out_of_range) {
    return std::unexpected(
      std::format("{} out of range: {}", description, token));
  }
  if (error != std::errc{} || end != token.data() + token.size()) {
    return std::unexpected(std::format("bad {}: {}", description, token));
  }
  return value;
}

std::expected<int, std::string> parse_nonnegative_int(
  std::string_view token, std::string_view description) {
  auto parsed = parse_integer<std::int64_t>(token, description);
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  if (*parsed < 0) {
    return std::unexpected(std::format("{} must be non-negative", description));
  }
  if (!std::in_range<int>(*parsed)) {
    return std::unexpected(std::format("{} does not fit int", description));
  }
  return static_cast<int>(*parsed);
}

std::expected<std::vector<std::byte>, std::string> read_file_bytes(
  const std::filesystem::path& path) {
  std::error_code error;
  std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error) {
    return std::unexpected(std::format(
      "cannot get file size for {}: {}", path.string(), error.message()));
  }

  std::vector<std::byte> buffer;
  if (file_size > buffer.max_size() ||
      file_size > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::streamsize>::max())) {
    return std::unexpected(std::format("file too large: {}", path.string()));
  }
  buffer.resize(static_cast<std::size_t>(file_size));

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(std::format("cannot open file: {}", path.string()));
  }
  if (!buffer.empty()) {
    auto expected_size = static_cast<std::streamsize>(buffer.size());
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
    if (file.gcount() != expected_size) {
      return std::unexpected(
        std::format("short read from file: {}", path.string()));
    }
  }
  return buffer;
}

std::expected<std::vector<std::string>, std::string> read_lines(
  const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::unexpected(
      std::format("cannot open param file: {}", path.string()));
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    auto first = line.find_first_not_of(" \t\r\n");
    auto last = line.find_last_not_of(" \t\r\n");
    if (first != std::string::npos) {
      lines.push_back(line.substr(first, last - first + 1));
    }
  }
  if (file.bad()) {
    return std::unexpected(std::format("read error: {}", path.string()));
  }
  return lines;
}

class BinCursor {
 public:
  explicit BinCursor(std::span<const std::byte> data);

  std::expected<std::uint32_t, std::string> read_u32_le();
  std::expected<std::span<const std::byte>, std::string> read_bytes(
    std::size_t count);
  std::expected<void, std::string> align4();

  std::size_t get_size() const noexcept;
  std::size_t get_position() const noexcept;
  std::size_t get_remaining() const noexcept;

 private:
  std::span<const std::byte> data_;
  std::size_t position_;
};

BinCursor::BinCursor(std::span<const std::byte> data)
  : data_(data), position_(0) {}

std::expected<std::span<const std::byte>, std::string> BinCursor::read_bytes(
  std::size_t count) {
  if (count > get_remaining()) {
    return std::unexpected(
      std::format("unexpected EOF at offset {}: requested {} bytes, {} remain",
                  position_,
                  count,
                  get_remaining()));
  }
  auto result = data_.subspan(position_, count);
  position_ += count;
  return result;
}

std::expected<std::uint32_t, std::string> BinCursor::read_u32_le() {
  auto bytes = read_bytes(sizeof(std::uint32_t));
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  std::array<std::byte, sizeof(std::uint32_t)> value_bytes;
  std::ranges::copy(*bytes, value_bytes.begin());
  auto value = std::bit_cast<std::uint32_t>(value_bytes);
  if constexpr (std::endian::native == std::endian::big) {
    value = std::byteswap(value);
  }
  return value;
}

std::expected<void, std::string> BinCursor::align4() {
  std::size_t remainder = position_ % kWeightAlignment;
  std::size_t padding = remainder == 0 ? 0 : kWeightAlignment - remainder;
  auto skipped = read_bytes(padding);
  if (!skipped) {
    return std::unexpected(skipped.error());
  }
  return {};
}

std::size_t BinCursor::get_size() const noexcept {
  return data_.size();
}

std::size_t BinCursor::get_position() const noexcept {
  return position_;
}

std::size_t BinCursor::get_remaining() const noexcept {
  return data_.size() - position_;
}

std::expected<Tensor, std::string> load_weight(
  BinCursor& cursor,
  std::int64_t element_count,
  int type,
  std::vector<std::int64_t> shape) {
  auto count = positive_size(element_count, "weight element count");
  if (!count) {
    return std::unexpected(count.error());
  }

  DataType dtype = DataType::Unknown;
  std::size_t element_width = 0;
  bool needs_alignment = false;
  if (type == 0) {
    const std::size_t flag_offset = cursor.get_position();
    auto flag = cursor.read_u32_le();
    if (!flag) {
      return std::unexpected(std::format("weight flag: {}", flag.error()));
    }
    if (*flag == kFloat16Flag) {
      dtype = DataType::Float16;
      element_width = 2;
      needs_alignment = true;
    } else if (*flag == kInt8Flag) {
      dtype = DataType::Int8;
      element_width = 1;
      needs_alignment = true;
    } else if (*flag == kFloat32Flag || *flag == 0) {
      dtype = DataType::Float32;
      element_width = sizeof(float);
    } else {
      return std::unexpected(std::format(
        "quantized lookup-table weight flag 0x{:08x} at offset {} is "
        "unsupported",
        *flag,
        flag_offset));
    }
  } else if (type == 1) {
    dtype = DataType::Float32;
    element_width = sizeof(float);
  } else {
    return std::unexpected(
      std::format("unsupported weight load type: {}", type));
  }

  auto byte_count = checked_multiply(*count, element_width, "weight byte size");
  if (!byte_count) {
    return std::unexpected(byte_count.error());
  }
  auto bytes = cursor.read_bytes(*byte_count);
  if (!bytes) {
    return std::unexpected(std::format("weight data: {}", bytes.error()));
  }
  std::vector<std::byte> data(bytes->begin(), bytes->end());
  if (needs_alignment) {
    auto aligned = cursor.align4();
    if (!aligned) {
      return std::unexpected(
        std::format("weight alignment: {}", aligned.error()));
    }
  }
  Tensor tensor;
  auto contents = tensor.set_contents(std::move(shape), dtype, std::move(data));
  if (!contents) {
    return std::unexpected(contents.error());
  }
  return tensor;
}

std::expected<Tensor, std::string> promote_float16_to_float32(
  const Tensor& source) {
  if (source.get_dtype() != DataType::Float16) {
    return source;
  }
  std::vector<std::byte> data(source.element_count() * sizeof(float));
  std::span<const std::byte> input = source.get_data();
  for (std::size_t index = 0; index < source.element_count(); ++index) {
    const std::uint16_t half =
      static_cast<std::uint16_t>(input[index * 2]) |
      (static_cast<std::uint16_t>(input[(index * 2) + 1]) << 8);
    const std::uint32_t sign =
      static_cast<std::uint32_t>(half & UINT16_C(0x8000)) << 16;
    std::uint32_t exponent = (half >> 10) & UINT16_C(0x1f);
    std::uint32_t significand = half & UINT16_C(0x03ff);
    std::uint32_t bits;
    if (exponent == 0) {
      if (significand == 0) {
        bits = sign;
      } else {
        int unbiasedExponent = -14;
        while ((significand & UINT32_C(0x0400)) == 0) {
          significand <<= 1;
          --unbiasedExponent;
        }
        significand &= UINT32_C(0x03ff);
        bits = sign |
               (static_cast<std::uint32_t>(unbiasedExponent + 127) << 23) |
               (significand << 13);
      }
    } else if (exponent == UINT32_C(0x1f)) {
      bits = sign | UINT32_C(0x7f800000) | (significand << 13);
    } else {
      bits = sign | ((exponent + UINT32_C(112)) << 23) | (significand << 13);
    }
    for (unsigned byte = 0; byte < sizeof(float); ++byte) {
      data[(index * sizeof(float)) + byte] =
        static_cast<std::byte>(bits >> (byte * 8));
    }
  }
  Tensor result;
  auto contents =
    result.set_contents(std::vector<std::int64_t>(source.get_shape().begin(),
                                                  source.get_shape().end()),
                        DataType::Float32,
                        std::move(data));
  if (!contents) {
    return std::unexpected(contents.error());
  }
  return result;
}

std::expected<void, std::string> load_convolution_weights(Layer& layer,
                                                          BinCursor& cursor) {
  auto decoded = decode_convolution_params(layer.get_params());
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  const ConvolutionParams& params = *decoded;
  if (params.dynamic_weight) {
    return {};
  }

  auto num_output =
    positive_size(params.output_channels, "convolution num_output");
  auto weight_count =
    positive_size(params.weight_count, "convolution weight count");
  auto kernel_w = positive_size(params.kernel_w, "convolution kernel width");
  auto kernel_h = positive_size(params.kernel_h, "convolution kernel height");
  if (!num_output) {
    return std::unexpected(num_output.error());
  }
  if (!weight_count) {
    return std::unexpected(weight_count.error());
  }
  if (!kernel_w) {
    return std::unexpected(kernel_w.error());
  }
  if (!kernel_h) {
    return std::unexpected(kernel_h.error());
  }

  auto output_kernel =
    checked_multiply(*num_output, *kernel_h, "convolution output/kernel size");
  if (!output_kernel) {
    return std::unexpected(output_kernel.error());
  }
  output_kernel = checked_multiply(
    *output_kernel, *kernel_w, "convolution output/kernel size");
  if (!output_kernel) {
    return std::unexpected(output_kernel.error());
  }
  if (*weight_count % *output_kernel != 0) {
    return std::unexpected(
      "convolution weight count is not divisible by output/kernel size");
  }
  std::size_t num_input = *weight_count / *output_kernel;
  if (num_input == 0 ||
      num_input >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::unexpected("convolution input channel count is invalid");
  }

  auto weight = load_weight(cursor,
                            params.weight_count,
                            0,
                            {params.output_channels,
                             static_cast<std::int64_t>(num_input),
                             params.kernel_h,
                             params.kernel_w});
  if (!weight) {
    return std::unexpected(std::format("conv weight: {}", weight.error()));
  }
  layer.add_weight(std::move(*weight));

  if (params.has_bias) {
    auto bias =
      load_weight(cursor, params.output_channels, 1, {params.output_channels});
    if (!bias) {
      return std::unexpected(std::format("conv bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }

  if (params.int8_scale_term != 0) {
    auto weight_scales =
      load_weight(cursor, params.output_channels, 1, {params.output_channels});
    if (!weight_scales) {
      return std::unexpected(
        std::format("conv weight int8 scales: {}", weight_scales.error()));
    }
    layer.add_weight(std::move(*weight_scales));

    auto bottom_scale = load_weight(cursor, 1, 1, {1});
    if (!bottom_scale) {
      return std::unexpected(
        std::format("conv bottom int8 scale: {}", bottom_scale.error()));
    }
    layer.add_weight(std::move(*bottom_scale));
  }
  if (params.int8_scale_term > 100) {
    auto top_scale = load_weight(cursor, 1, 1, {1});
    if (!top_scale) {
      return std::unexpected(
        std::format("conv top int8 scale: {}", top_scale.error()));
    }
    layer.add_weight(std::move(*top_scale));
  }
  return {};
}

std::expected<void, std::string> load_convolution_depthwise_weights(
  Layer& layer, BinCursor& cursor) {
  auto decoded = decode_convolution_depthwise_params(layer.get_params());
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  const auto& params = *decoded;
  if (params.dynamic_weight) {
    return {};
  }
  auto num_output =
    positive_size(params.output_channels, "depthwise num_output");
  auto kernel_h = positive_size(params.kernel_h, "depthwise kernel height");
  auto kernel_w = positive_size(params.kernel_w, "depthwise kernel width");
  if (!num_output || !kernel_h || !kernel_w) {
    return std::unexpected(!num_output ? num_output.error()
                           : !kernel_h ? kernel_h.error()
                                       : kernel_w.error());
  }
  auto output_kernel =
    checked_multiply(*num_output, *kernel_h, "depthwise output/kernel size");
  if (!output_kernel) {
    return std::unexpected(output_kernel.error());
  }
  output_kernel =
    checked_multiply(*output_kernel, *kernel_w, "depthwise output/kernel size");
  if (!output_kernel) {
    return std::unexpected(output_kernel.error());
  }
  auto weight_count =
    positive_size(params.weight_count, "depthwise weight count");
  if (!weight_count) {
    return std::unexpected(weight_count.error());
  }
  if (*weight_count % *output_kernel != 0) {
    return std::unexpected(
      "depthwise weight count is not divisible by output/kernel size");
  }
  const auto input_channels_per_group = *weight_count / *output_kernel;
  if (input_channels_per_group == 0 ||
      input_channels_per_group >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::unexpected("depthwise input channels per group is invalid");
  }
  auto weight =
    load_weight(cursor,
                params.weight_count,
                0,
                {params.output_channels,
                 static_cast<std::int64_t>(input_channels_per_group),
                 params.kernel_h,
                 params.kernel_w});
  if (!weight) {
    return std::unexpected(std::format("depthwise weight: {}", weight.error()));
  }
  layer.add_weight(std::move(*weight));
  if (params.has_bias) {
    auto bias =
      load_weight(cursor, params.output_channels, 1, {params.output_channels});
    if (!bias) {
      return std::unexpected(std::format("depthwise bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  if (params.int8_scale_term != 0) {
    const std::int64_t scale_count =
      params.int8_scale_term == 1 || params.int8_scale_term == 101
        ? params.group
        : 1;
    auto scales = load_weight(cursor, scale_count, 1, {scale_count});
    if (!scales) {
      return std::unexpected(
        std::format("depthwise weight int8 scales: {}", scales.error()));
    }
    layer.add_weight(std::move(*scales));
    auto bottom = load_weight(cursor, 1, 1, {1});
    if (!bottom) {
      return std::unexpected(
        std::format("depthwise bottom int8 scale: {}", bottom.error()));
    }
    layer.add_weight(std::move(*bottom));
  }
  if (params.int8_scale_term > 100) {
    auto top = load_weight(cursor, 1, 1, {1});
    if (!top) {
      return std::unexpected(
        std::format("depthwise top int8 scale: {}", top.error()));
    }
    layer.add_weight(std::move(*top));
  }
  return {};
}

std::expected<void, std::string> load_deconvolution_weights(Layer& layer,
                                                            BinCursor& cursor) {
  auto decoded = decode_deconvolution_params(layer.get_params());
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  const auto& params = *decoded;
  if (params.dynamic_weight) {
    return {};
  }
  auto outputChannels =
    positive_size(params.output_channels, "deconvolution num_output");
  auto kernelH = positive_size(params.kernel_h, "deconvolution kernel height");
  auto kernelW = positive_size(params.kernel_w, "deconvolution kernel width");
  auto weightCount =
    positive_size(params.weight_count, "deconvolution weight count");
  if (!outputChannels || !kernelH || !kernelW || !weightCount) {
    return std::unexpected(!outputChannels ? outputChannels.error()
                           : !kernelH      ? kernelH.error()
                           : !kernelW      ? kernelW.error()
                                           : weightCount.error());
  }
  auto outputKernel = checked_multiply(
    *outputChannels, *kernelH, "deconvolution output/kernel size");
  if (outputKernel) {
    outputKernel = checked_multiply(
      *outputKernel, *kernelW, "deconvolution output/kernel size");
  }
  if (!outputKernel) {
    return std::unexpected(outputKernel.error());
  }
  if (*weightCount % *outputKernel != 0) {
    return std::unexpected(
      "deconvolution weight count is not divisible by output/kernel size");
  }
  const std::size_t inputChannels = *weightCount / *outputKernel;
  if (inputChannels == 0 ||
      inputChannels >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::unexpected("deconvolution input channel count is invalid");
  }
  auto weight = load_weight(cursor,
                            params.weight_count,
                            0,
                            {params.output_channels,
                             static_cast<std::int64_t>(inputChannels),
                             params.kernel_h,
                             params.kernel_w});
  if (!weight) {
    return std::unexpected(
      std::format("deconvolution weight: {}", weight.error()));
  }
  layer.add_weight(std::move(*weight));
  if (params.has_bias) {
    auto bias =
      load_weight(cursor, params.output_channels, 1, {params.output_channels});
    if (!bias) {
      return std::unexpected(
        std::format("deconvolution bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  return {};
}

std::expected<void, std::string> load_inner_product_weights(Layer& layer,
                                                            BinCursor& cursor) {
  auto decoded = decode_inner_product_params(layer.get_params());
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  const auto& params = *decoded;
  const auto num_input = params.weight_count / params.output_channels;
  auto weight = load_weight(
    cursor, params.weight_count, 0, {params.output_channels, num_input});
  if (!weight) {
    return std::unexpected(
      std::format("inner product weight: {}", weight.error()));
  }
  layer.add_weight(std::move(*weight));
  if (params.has_bias) {
    auto bias =
      load_weight(cursor, params.output_channels, 1, {params.output_channels});
    if (!bias) {
      return std::unexpected(
        std::format("inner product bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  if (params.int8_scale_term != 0) {
    auto scales =
      load_weight(cursor, params.output_channels, 1, {params.output_channels});
    if (!scales) {
      return std::unexpected(
        std::format("inner product weight int8 scales: {}", scales.error()));
    }
    layer.add_weight(std::move(*scales));
    auto bottom = load_weight(cursor, 1, 1, {1});
    if (!bottom) {
      return std::unexpected(
        std::format("inner product bottom int8 scale: {}", bottom.error()));
    }
    layer.add_weight(std::move(*bottom));
  }
  return {};
}

std::expected<void, std::string> load_batch_norm_weights(Layer& layer,
                                                         BinCursor& cursor) {
  auto params = decode_batch_norm_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  for (std::string_view name : {"slope", "mean", "variance", "bias"}) {
    auto weight = load_weight(cursor, params->channels, 1, {params->channels});
    if (!weight) {
      return std::unexpected(
        std::format("BatchNorm {}: {}", name, weight.error()));
    }
    layer.add_weight(std::move(*weight));
  }
  return {};
}

std::expected<void, std::string> load_prelu_weights(Layer& layer,
                                                    BinCursor& cursor) {
  const std::int64_t count = layer.get_param_int(0);
  if (count <= 0) {
    return std::unexpected("PReLU num_slope must be positive");
  }
  auto slope = load_weight(cursor, count, 1, {count});
  if (!slope) {
    return std::unexpected(std::format("PReLU slope: {}", slope.error()));
  }
  layer.add_weight(std::move(*slope));
  return {};
}

std::expected<void, std::string> load_layer_norm_weights(Layer& layer,
                                                         BinCursor& cursor) {
  auto params = decode_layer_norm_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  if (!params->affine) {
    return {};
  }
  for (std::string_view name : {"gamma", "beta"}) {
    auto weight =
      load_weight(cursor, params->affine_size, 1, {params->affine_size});
    if (!weight) {
      return std::unexpected(
        std::format("LayerNorm {}: {}", name, weight.error()));
    }
    layer.add_weight(std::move(*weight));
  }
  return {};
}

std::expected<void, std::string> load_embed_weights(Layer& layer,
                                                    BinCursor& cursor) {
  auto params = decode_embed_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  if (params->int8_scale_term != 0) {
    return std::unexpected("quantized Embed weights are unsupported");
  }
  auto weight = load_weight(cursor,
                            params->weight_count,
                            0,
                            {params->input_dim, params->output_channels});
  if (!weight) {
    return std::unexpected(std::format("Embed weight: {}", weight.error()));
  }
  layer.add_weight(std::move(*weight));
  if (params->has_bias) {
    auto bias = load_weight(
      cursor, params->output_channels, 1, {params->output_channels});
    if (!bias) {
      return std::unexpected(std::format("Embed bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  return {};
}

std::expected<void, std::string> load_multi_head_attention_weights(
  Layer& layer, BinCursor& cursor) {
  auto params = decode_multi_head_attention_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  if (params->int8_scale_term != 0) {
    return std::unexpected(
      "quantized MultiHeadAttention weights are unsupported");
  }

  auto embed_dim = positive_size(params->embed_dim, "MHA embed_dim");
  auto query_dim = positive_size(params->query_dim, "MHA query_dim");
  auto key_dim = positive_size(params->key_dim, "MHA key_dim");
  auto value_dim = positive_size(params->value_dim, "MHA value_dim");
  if (!embed_dim || !query_dim || !key_dim || !value_dim) {
    return std::unexpected(!embed_dim   ? embed_dim.error()
                           : !query_dim ? query_dim.error()
                           : !key_dim   ? key_dim.error()
                                        : value_dim.error());
  }
  auto key_count =
    checked_multiply(*embed_dim, *key_dim, "MHA key weight element count");
  auto value_count =
    checked_multiply(*embed_dim, *value_dim, "MHA value weight element count");
  auto out_count =
    checked_multiply(*query_dim, *embed_dim, "MHA out weight element count");
  if (!key_count || !value_count || !out_count) {
    return std::unexpected(!key_count     ? key_count.error()
                           : !value_count ? value_count.error()
                                          : out_count.error());
  }
  constexpr auto kInt64Max =
    static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
  if (*key_count > kInt64Max || *value_count > kInt64Max ||
      *out_count > kInt64Max) {
    return std::unexpected("MHA weight element count does not fit int64_t");
  }

  struct ProjectionWeights {
    std::string_view name;
    std::int64_t weight_count;
    std::vector<std::int64_t> weight_shape;
    std::int64_t bias_count;
  };
  const std::array projections{
    ProjectionWeights{.name = "q",
                      .weight_count = params->weight_count,
                      .weight_shape = {params->embed_dim, params->query_dim},
                      .bias_count = params->embed_dim},
    ProjectionWeights{.name = "k",
                      .weight_count = static_cast<std::int64_t>(*key_count),
                      .weight_shape = {params->embed_dim, params->key_dim},
                      .bias_count = params->embed_dim},
    ProjectionWeights{.name = "v",
                      .weight_count = static_cast<std::int64_t>(*value_count),
                      .weight_shape = {params->embed_dim, params->value_dim},
                      .bias_count = params->embed_dim},
    ProjectionWeights{.name = "out",
                      .weight_count = static_cast<std::int64_t>(*out_count),
                      .weight_shape = {params->query_dim, params->embed_dim},
                      .bias_count = params->query_dim},
  };
  for (const auto& projection : projections) {
    auto weight =
      load_weight(cursor, projection.weight_count, 0, projection.weight_shape);
    if (!weight) {
      return std::unexpected(std::format(
        "MultiHeadAttention {} weight: {}", projection.name, weight.error()));
    }
    layer.add_weight(std::move(*weight));
    auto bias =
      load_weight(cursor, projection.bias_count, 1, {projection.bias_count});
    if (!bias) {
      return std::unexpected(std::format(
        "MultiHeadAttention {} bias: {}", projection.name, bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  return {};
}

std::expected<void, std::string> load_gemm_weights(Layer& layer,
                                                   BinCursor& cursor) {
  auto params = decode_gemm_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  const bool dynamicMatrices =
    !params->constant_a && !params->constant_b && !params->constant_c;
  if (dynamicMatrices && !params->transpose_a && !params->transpose_b &&
      params->output_n1m == 0 && params->output_elempack == 0 &&
      params->output_elemtype == 0 && params->output_transpose == 0 &&
      params->int8_scale_term == 0) {
    return {};
  }
  if (params->constant_a || !params->constant_b || !params->constant_c ||
      params->transpose_a || !params->transpose_b ||
      (params->broadcast_c != -1 && params->broadcast_c != 4) ||
      params->output_n1m != 0 || params->output_elempack != 0 ||
      params->output_elemtype != 0 || params->output_transpose != 0 ||
      (params->int8_scale_term != 0 && params->int8_scale_term != 1 &&
       params->int8_scale_term != 2)) {
    return std::unexpected(
      "only dynamic-A, transposed constant-B Gemm with optional row bias and "
      "optional int8 B is supported");
  }
  auto outputSize = positive_size(params->constant_n, "Gemm constantN");
  auto inputSize = positive_size(params->constant_k, "Gemm constantK");
  if (!outputSize || !inputSize) {
    return std::unexpected(!outputSize ? outputSize.error()
                                       : inputSize.error());
  }
  auto weightCount =
    checked_multiply(*outputSize, *inputSize, "Gemm B element count");
  if (!weightCount ||
      *weightCount >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::unexpected(weightCount ? "Gemm B element count is too large"
                                       : weightCount.error());
  }
  auto weight = load_weight(cursor,
                            static_cast<std::int64_t>(*weightCount),
                            0,
                            {params->constant_n, params->constant_k});
  if (!weight) {
    return std::unexpected(std::format("Gemm B: {}", weight.error()));
  }
  layer.add_weight(std::move(*weight));
  if (params->broadcast_c == 4) {
    auto bias =
      load_weight(cursor, params->constant_n, 0, {params->constant_n});
    if (!bias) {
      return std::unexpected(std::format("Gemm C: {}", bias.error()));
    }
    auto promoted = promote_float16_to_float32(*bias);
    if (!promoted) {
      return std::unexpected(std::format("Gemm C: {}", promoted.error()));
    }
    layer.add_weight(std::move(*promoted));
  }
  if (params->int8_scale_term != 0) {
    auto scale = load_weight(cursor, 1, 1, {1});
    if (!scale) {
      return std::unexpected(
        std::format("Gemm B int8 scale: {}", scale.error()));
    }
    layer.add_weight(std::move(*scale));
  }
  return {};
}

std::expected<void, std::string> load_memory_data_weights(Layer& layer,
                                                          BinCursor& cursor) {
  auto params = decode_memory_data_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  std::vector<std::int64_t> shape;
  if (params->depth != 0) {
    shape = {params->channels, params->depth, params->height, params->width};
  } else if (params->channels != 0) {
    shape = {params->channels, params->height, params->width};
  } else if (params->height != 0) {
    shape = {params->height, params->width};
  } else {
    shape = {params->width};
  }
  std::int64_t elementCount = 1;
  for (std::int64_t extent : shape) {
    if (elementCount > std::numeric_limits<std::int64_t>::max() / extent) {
      return std::unexpected("MemoryData element count overflows int64_t");
    }
    elementCount *= extent;
  }
  auto data = load_weight(
    cursor, elementCount, static_cast<int>(params->load_type), shape);
  if (!data) {
    return std::unexpected(std::format("MemoryData: {}", data.error()));
  }
  layer.add_weight(std::move(*data));
  return {};
}

std::expected<void, std::string> load_quantize_weights(Layer& layer,
                                                       BinCursor& cursor) {
  auto params = decode_quantize_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  auto scale =
    load_weight(cursor, params->scale_count, 1, {params->scale_count});
  if (!scale) {
    return std::unexpected(std::format("Quantize scale: {}", scale.error()));
  }
  layer.add_weight(std::move(*scale));
  return {};
}

std::expected<void, std::string> load_dequantize_weights(Layer& layer,
                                                         BinCursor& cursor) {
  auto params = decode_dequantize_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  auto scale =
    load_weight(cursor, params->scale_count, 1, {params->scale_count});
  if (!scale) {
    return std::unexpected(std::format("Dequantize scale: {}", scale.error()));
  }
  layer.add_weight(std::move(*scale));
  if (params->bias_count != 0) {
    auto bias =
      load_weight(cursor, params->bias_count, 1, {params->bias_count});
    if (!bias) {
      return std::unexpected(std::format("Dequantize bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  return {};
}

std::expected<void, std::string> load_requantize_weights(Layer& layer,
                                                         BinCursor& cursor) {
  auto params = decode_requantize_params(layer.get_params());
  if (!params) {
    return std::unexpected(params.error());
  }
  for (auto [count, name] :
       {std::pair{params->input_scale_count, "input scale"},
        std::pair{params->output_scale_count, "output scale"}}) {
    auto scale = load_weight(cursor, count, 1, {count});
    if (!scale) {
      return std::unexpected(
        std::format("Requantize {}: {}", name, scale.error()));
    }
    layer.add_weight(std::move(*scale));
  }
  if (params->bias_count != 0) {
    auto bias =
      load_weight(cursor, params->bias_count, 1, {params->bias_count});
    if (!bias) {
      return std::unexpected(std::format("Requantize bias: {}", bias.error()));
    }
    layer.add_weight(std::move(*bias));
  }
  return {};
}

using WeightLoader = std::expected<void, std::string> (*)(Layer&, BinCursor&);

struct WeightLoaderEntry {
  std::string_view type;
  WeightLoader loader;
};

std::span<const WeightLoaderEntry> weight_loaders() noexcept {
  static constexpr std::array kWeightLoaders{
    WeightLoaderEntry{.type = "Convolution",
                      .loader = load_convolution_weights},
    WeightLoaderEntry{.type = "ConvolutionDepthWise",
                      .loader = load_convolution_depthwise_weights},
    WeightLoaderEntry{.type = "Deconvolution",
                      .loader = load_deconvolution_weights},
    WeightLoaderEntry{.type = "InnerProduct",
                      .loader = load_inner_product_weights},
    WeightLoaderEntry{.type = "BatchNorm", .loader = load_batch_norm_weights},
    WeightLoaderEntry{.type = "PReLU", .loader = load_prelu_weights},
    WeightLoaderEntry{.type = "Gemm", .loader = load_gemm_weights},
    WeightLoaderEntry{.type = "MemoryData", .loader = load_memory_data_weights},
    WeightLoaderEntry{.type = "Quantize", .loader = load_quantize_weights},
    WeightLoaderEntry{.type = "Dequantize", .loader = load_dequantize_weights},
    WeightLoaderEntry{.type = "Requantize", .loader = load_requantize_weights},
  };
  return kWeightLoaders;
}

std::span<const WeightLoaderEntry> graph_only_weight_loaders() noexcept {
  static constexpr std::array kWeightLoaders{
    WeightLoaderEntry{.type = "LayerNorm", .loader = load_layer_norm_weights},
    WeightLoaderEntry{.type = "Embed", .loader = load_embed_weights},
    WeightLoaderEntry{.type = "MultiHeadAttention",
                      .loader = load_multi_head_attention_weights},
  };
  return kWeightLoaders;
}

std::expected<void, std::string> load_layer_weights(Layer& layer,
                                                    BinCursor& cursor) {
  for (const auto entries : {weight_loaders(), graph_only_weight_loaders()}) {
    const auto entry =
      std::ranges::find(entries, layer.get_type(), &WeightLoaderEntry::type);
    if (entry != entries.end()) {
      return entry->loader(layer, cursor);
    }
  }
  return {};
}

std::vector<std::string> split_ws(std::string_view text) {
  std::vector<std::string> tokens;
  std::size_t position = 0;
  while (position < text.size()) {
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position]))) {
      ++position;
    }
    if (position >= text.size()) {
      break;
    }
    std::size_t end = position;
    while (end < text.size() &&
           !std::isspace(static_cast<unsigned char>(text[end]))) {
      ++end;
    }
    tokens.emplace_back(text.substr(position, end - position));
    position = end;
  }
  return tokens;
}

std::string_view get_dtype_name(DataType dtype) {
  switch (dtype) {
    case DataType::Unknown:
      return "unknown";
    case DataType::Float32:
      return "f32";
    case DataType::Float16:
      return "f16";
    case DataType::BFloat16:
      return "bf16";
    case DataType::Int8:
      return "i8";
  }
  return "unknown";
}

}  // namespace

bool has_weight_loader(std::string_view layer_type) noexcept {
  for (const auto entries : {weight_loaders(), graph_only_weight_loaders()}) {
    if (std::ranges::find(entries, layer_type, &WeightLoaderEntry::type) !=
        entries.end()) {
      return true;
    }
  }
  return false;
}

std::size_t get_weight_loader_count() noexcept {
  return weight_loaders().size() + graph_only_weight_loaders().size();
}

std::expected<Graph, std::string> Graph::load(std::string_view param_path,
                                              std::string_view bin_path) {
  auto lines = read_lines(std::filesystem::path(param_path));
  if (!lines) {
    return std::unexpected(lines.error());
  }
  if (lines->empty()) {
    return std::unexpected("empty param file");
  }

  std::size_t line_index = 0;
  auto first_line = split_ws((*lines)[0]);
  if (first_line.size() == 1) {
    auto magic = parse_integer<std::int64_t>(first_line[0], "param magic");
    if (!magic) {
      return std::unexpected(magic.error());
    }
    if (*magic != 7767517) {
      return std::unexpected("param magic mismatch (expected 7767517)");
    }
    line_index = 1;
  } else if (first_line.size() != 2) {
    return std::unexpected("bad param magic/header line");
  }
  if (line_index >= lines->size()) {
    return std::unexpected("missing layer/blob count header");
  }

  auto header = split_ws((*lines)[line_index]);
  if (header.size() != 2) {
    return std::unexpected("bad layer/blob count header");
  }
  auto layer_count_result = parse_nonnegative_int(header[0], "layer count");
  auto blob_count_result = parse_nonnegative_int(header[1], "blob count");
  if (!layer_count_result) {
    return std::unexpected(layer_count_result.error());
  }
  if (!blob_count_result) {
    return std::unexpected(blob_count_result.error());
  }
  int layer_count = *layer_count_result;
  int blob_count = *blob_count_result;
  ++line_index;
  if (static_cast<std::size_t>(layer_count) > lines->size() - line_index) {
    return std::unexpected("param truncated before all layers read");
  }

  bool bin_supplied = !bin_path.empty();
  std::vector<std::byte> bin_data;
  if (bin_supplied) {
    auto data = read_file_bytes(std::filesystem::path(bin_path));
    if (!data) {
      return std::unexpected(data.error());
    }
    bin_data = std::move(*data);
  }
  BinCursor cursor(bin_data);

  std::vector<Layer> layers;
  layers.reserve(static_cast<std::size_t>(layer_count));
  std::vector<Blob> blobs;
  std::vector<std::pair<std::string, int>> blob_indices;
  auto find_or_allocate_blob =
    [&](const std::string& name) -> std::expected<int, std::string> {
    auto iterator = std::ranges::find(
      blob_indices, name, [](const auto& entry) { return entry.first; });
    if (iterator != blob_indices.end()) {
      return iterator->second;
    }
    if (blobs.size() >= static_cast<std::size_t>(blob_count)) {
      return std::unexpected("model creates more blobs than declared");
    }
    if (!std::in_range<int>(blobs.size())) {
      return std::unexpected("blob index does not fit int");
    }
    int index = static_cast<int>(blobs.size());
    blobs.emplace_back(name);
    blob_indices.emplace_back(name, index);
    return index;
  };

  for (int index = 0; index < layer_count; ++index) {
    auto tokens = split_ws((*lines)[line_index++]);
    if (tokens.size() < 4) {
      return std::unexpected("layer line too short");
    }

    Layer layer;
    layer.set_type(tokens[0]);
    layer.set_name(tokens[1]);
    auto bottom_count_result = parse_nonnegative_int(tokens[2], "bottom count");
    auto top_count_result = parse_nonnegative_int(tokens[3], "top count");
    if (!bottom_count_result) {
      return std::unexpected(bottom_count_result.error());
    }
    if (!top_count_result) {
      return std::unexpected(top_count_result.error());
    }
    auto bottom_count = static_cast<std::size_t>(*bottom_count_result);
    auto top_count = static_cast<std::size_t>(*top_count_result);
    std::size_t offset = 4;
    if (bottom_count > tokens.size() - offset) {
      return std::unexpected("layer line missing bottom blob names");
    }
    for (std::size_t bottom = 0; bottom < bottom_count; ++bottom) {
      layer.add_input(tokens[offset++]);
    }
    if (top_count > tokens.size() - offset) {
      return std::unexpected("layer line missing top blob names");
    }
    for (std::size_t top = 0; top < top_count; ++top) {
      layer.add_output(tokens[offset++]);
    }

    std::string param_tail;
    for (std::size_t token_index = offset; token_index < tokens.size();
         ++token_index) {
      if (token_index > offset) {
        param_tail += ' ';
      }
      param_tail += tokens[token_index];
    }
    if (!param_tail.empty()) {
      auto params = parse_layer_params(param_tail);
      if (!params) {
        return std::unexpected(std::format(
          "layer {} ({}): {}", index, layer.get_type(), params.error()));
      }
      layer.set_params(std::move(*params));
    }

    for (const auto& name : layer.get_inputs()) {
      auto iterator = std::ranges::find(
        blob_indices, name, [](const auto& entry) { return entry.first; });
      if (iterator == blob_indices.end() ||
          blobs[iterator->second].get_producer() == -1) {
        return std::unexpected(
          std::format("layer {} has unresolved bottom blob: {}", index, name));
      }
      if (blobs[iterator->second].get_consumer() == -1) {
        blobs[iterator->second].set_consumer(index);
      }
    }
    for (const auto& name : layer.get_outputs()) {
      auto blob = find_or_allocate_blob(name);
      if (!blob) {
        return std::unexpected(blob.error());
      }
      if (blobs[*blob].get_producer() != -1) {
        return std::unexpected(
          std::format("blob has multiple producers: {}", name));
      }
      blobs[*blob].set_producer(index);
    }
    layers.push_back(std::move(layer));
  }
  if (line_index != lines->size()) {
    return std::unexpected("param contains trailing layer lines");
  }
  bool weights_loaded = false;
  if (bin_supplied) {
    for (std::size_t index = 0; index < layers.size(); ++index) {
      auto& layer = layers[index];
      auto result = load_layer_weights(layer, cursor);
      if (!result) {
        return std::unexpected(std::format("layer {} ({}, {}): {}",
                                           index,
                                           layer.get_type(),
                                           layer.get_name(),
                                           result.error()));
      }
    }
    if (cursor.get_remaining() != 0) {
      return std::unexpected(std::format(
        "bin size mismatch: consumed {} bytes of {} (difference {})",
        cursor.get_position(),
        cursor.get_size(),
        cursor.get_remaining()));
    }
    weights_loaded = true;
  }

  std::vector<std::string> input_blob_names;
  for (const auto& layer : layers) {
    if (layer.get_type() == "Input") {
      for (const auto& name : layer.get_outputs()) {
        input_blob_names.push_back(name);
      }
    }
  }
  std::vector<std::string> output_blob_names;
  for (const auto& blob : blobs) {
    if (blob.get_consumer() == -1) {
      output_blob_names.emplace_back(blob.get_name());
    }
  }

  Graph graph;
  graph.set_layers(std::move(layers));
  graph.set_blobs(std::move(blobs));
  graph.set_input_blob_names(std::move(input_blob_names));
  graph.set_output_blob_names(std::move(output_blob_names));
  graph.set_weights_loaded(weights_loaded);
  return graph;
}

std::string Graph::dump() const {
  std::ostringstream output;
  auto layers = get_layers();
  auto blobs = get_blobs();
  auto input_names = get_input_blob_names();
  auto output_names = get_output_blob_names();
  output << std::format(
    "ncnn_graph: {} layers, {} blobs\n", layers.size(), blobs.size());
  output << std::format("inputs: {}\n", input_names.size());
  for (const auto& name : input_names) {
    output << std::format("  - {}\n", name);
  }
  output << std::format("outputs: {}\n", output_names.size());
  for (const auto& name : output_names) {
    output << std::format("  - {}\n", name);
  }
  output << "layers:\n";

  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto& layer = layers[index];
    output << std::format("  [{:>3}] {:<14} {:<22} in=[",
                          index,
                          layer.get_type(),
                          layer.get_name());
    auto inputs = layer.get_inputs();
    for (std::size_t input = 0; input < inputs.size(); ++input) {
      if (input) {
        output << ",";
      }
      output << inputs[input];
    }
    output << "] out=[";
    auto outputs = layer.get_outputs();
    for (std::size_t result = 0; result < outputs.size(); ++result) {
      if (result) {
        output << ",";
      }
      output << outputs[result];
    }
    output << "]";

    auto entries = layer.get_params().get_entries();
    if (!entries.empty()) {
      output << " {";
      for (std::size_t entry = 0; entry < entries.size(); ++entry) {
        if (entry) {
          output << " ";
        }
        const auto& [id, value] = entries[entry];
        output << id << "=";
        switch (value.get_kind()) {
          case ParamValue::Kind::Int:
            output << *value.get_int();
            break;
          case ParamValue::Kind::Float:
            output << *value.get_float();
            break;
          case ParamValue::Kind::String:
            output << "\"" << *value.get_string() << "\"";
            break;
          case ParamValue::Kind::IntArray: {
            output << "[";
            auto values = *value.get_int_array();
            for (std::size_t item = 0; item < values.size(); ++item) {
              if (item) {
                output << ",";
              }
              output << values[item];
            }
            output << "]";
          } break;
          case ParamValue::Kind::FloatArray: {
            output << "[";
            auto values = *value.get_float_array();
            for (std::size_t item = 0; item < values.size(); ++item) {
              if (item) {
                output << ",";
              }
              output << values[item];
            }
            output << "]";
          } break;
        }
      }
      output << "}";
    }

    auto weights = layer.get_weights();
    if (!weights.empty()) {
      output << " w=[";
      for (std::size_t weight_index = 0; weight_index < weights.size();
           ++weight_index) {
        if (weight_index) {
          output << ",";
        }
        const auto& weight = weights[weight_index];
        output << "[";
        auto shape = weight.get_shape();
        for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
          if (dimension) {
            output << ",";
          }
          output << shape[dimension];
        }
        output << ":" << get_dtype_name(weight.get_dtype()) << ":"
               << weight.get_data().size() << "B]";
      }
      output << "]";
    }
    output << "\n";
  }
  return output.str();
}

std::size_t Graph::layer_count_of(std::string_view type) const {
  return static_cast<std::size_t>(
    std::ranges::count(get_layers(), type, &Layer::get_type));
}

}  // namespace ncnn_graph
