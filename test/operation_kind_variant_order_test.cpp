#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <variant>

#include "ncnn_frontend/OperationKind.hpp"
#include "ncnn_frontend/Operations.hpp"

namespace {

template <std::size_t I>
constexpr bool alternative_matches_kind() {
  using Alt = std::variant_alternative_t<I, ncnn_frontend::OperationAttributes>;
  return Alt::kind_v == static_cast<ncnn_frontend::OperationKind>(I);
}

template <std::size_t... Is>
constexpr bool all_alternatives_match(std::index_sequence<Is...>) {
  return (alternative_matches_kind<Is>() && ...);
}

constexpr std::size_t kVariantSize =
  std::variant_size_v<ncnn_frontend::OperationAttributes>;

static_assert(
  static_cast<std::size_t>(ncnn_frontend::OperationKind::Count) == kVariantSize,
  "OperationKind::Count sentinel must equal OperationAttributes alternative "
  "count — adding an enum without a variant alternative (or vice versa) "
  "breaks the index correspondence used by ir.cpp get_kind.");

static_assert(all_alternatives_match(std::make_index_sequence<kVariantSize>{}),
              "OperationAttributes alternative order must match OperationKind "
              "enum order (see ir.cpp get_kind).");

}  // namespace

TEST(OperationKindVariantOrderTest, AlternativeIndexMatchesKindEnum) {
  EXPECT_TRUE(all_alternatives_match(std::make_index_sequence<kVariantSize>{}))
    << "OperationAttributes alternatives must be declared in the same order as "
       "the OperationKind enum values";
}
