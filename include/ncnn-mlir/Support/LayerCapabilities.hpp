#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace ncnn_compiler {

enum class LayerCapability : std::uint8_t {
  None = 0,
  HasWeights = 1U << 0,
  NeedsNormalization = 1U << 1,
  Lowerable = 1U << 2,
};

constexpr LayerCapability operator|(LayerCapability left,
                                    LayerCapability right) noexcept {
  return static_cast<LayerCapability>(static_cast<std::uint8_t>(left) |
                                      static_cast<std::uint8_t>(right));
}

constexpr bool has_capability(LayerCapability capabilities,
                              LayerCapability capability) noexcept {
  return (static_cast<std::uint8_t>(capabilities) &
          static_cast<std::uint8_t>(capability)) != 0;
}

struct LayerDescriptor {
  std::string_view source_type;
  LayerCapability capabilities;
};

inline constexpr std::array kLayerDescriptors{
  LayerDescriptor{.source_type = "Input",
                  .capabilities = LayerCapability::None},
  LayerDescriptor{.source_type = "Convolution",
                  .capabilities = LayerCapability::HasWeights |
                                  LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "ConvolutionDepthWise",
                  .capabilities = LayerCapability::HasWeights |
                                  LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "ReLU",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Pooling",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Split",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Concat",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Dropout",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Softmax",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "HardSigmoid",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "HardSwish",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Reshape",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "BinaryOp",
                  .capabilities = LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "InnerProduct",
                  .capabilities = LayerCapability::HasWeights |
                                  LayerCapability::NeedsNormalization |
                                  LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "ShuffleChannel",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Slice",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Reduction",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "GELU",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Squeeze",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{
    .source_type = "BatchNorm",
    .capabilities = LayerCapability::HasWeights | LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "ExpandDims",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{.source_type = "Permute",
                  .capabilities = LayerCapability::Lowerable},
  LayerDescriptor{
    .source_type = "Gemm",
    .capabilities = LayerCapability::HasWeights | LayerCapability::Lowerable},
};

constexpr std::span<const LayerDescriptor> get_layer_descriptors() noexcept {
  return kLayerDescriptors;
}

}  // namespace ncnn_compiler
