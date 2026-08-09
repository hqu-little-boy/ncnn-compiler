#pragma once

namespace mlir::ncnn {

enum class ShapeMode {
  Static,
  FixedRankDynamic,
  DynamicRankSpecialization,
};

constexpr ShapeMode classifyShapeMode(bool dynamicRankSpecialization,
                                      bool hasDynamicInput) noexcept {
  if (dynamicRankSpecialization) {
    return ShapeMode::DynamicRankSpecialization;
  }
  return hasDynamicInput ? ShapeMode::FixedRankDynamic : ShapeMode::Static;
}

}  // namespace mlir::ncnn
