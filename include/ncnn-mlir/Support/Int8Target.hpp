#pragma once

#include <cstddef>
#include <string_view>

namespace ncnn_mlir {

// Capability from the selected backend's predefined macros, not from the host
// or a CPU-name substring. Callers must probe with the final target arguments.
enum class Int8DotTarget {
  Portable,
  AVXVNNI,
  AVX512VNNI,
};

inline Int8DotTarget resolve_int8_dot_target(std::string_view macros) {
  auto has = [&](std::string_view name) {
    std::size_t position = 0;
    const std::string_view define = "#define ";
    while ((position = macros.find(define, position)) !=
           std::string_view::npos) {
      position += define.size();
      const std::size_t end = macros.find('\n', position);
      std::size_t length =
        (end == std::string_view::npos ? macros.size() : end) - position;
      const std::size_t trailing =
        macros.substr(position, length).find_last_not_of(" \t\r");
      if (trailing != std::string_view::npos) {
        length = trailing + 1;
      }
      const std::string_view definition = macros.substr(position, length);
      if (definition.starts_with(name) && definition.size() > name.size() &&
          definition[name.size()] == ' ' &&
          definition.substr(name.size() + 1) == "1") {
        return true;
      }
    }
    return false;
  };
  if (!has("__x86_64__") || !has("__AVX2__")) {
    return Int8DotTarget::Portable;
  }
  if (has("__AVXVNNI__")) {
    return Int8DotTarget::AVXVNNI;
  }
  if (has("__AVX512F__") && has("__AVX512BW__") && has("__AVX512VL__") &&
      has("__AVX512VNNI__")) {
    return Int8DotTarget::AVX512VNNI;
  }
  return Int8DotTarget::Portable;
}

inline std::string_view int8_dot_target_name(Int8DotTarget target) {
  switch (target) {
    case Int8DotTarget::AVXVNNI:
      return "avx-vnni";
    case Int8DotTarget::AVX512VNNI:
      return "avx512-vnni";
    case Int8DotTarget::Portable:
      return "portable";
  }
  return "portable";
}

}  // namespace ncnn_mlir
