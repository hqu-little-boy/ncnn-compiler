#include <cstddef>
#include <set>
#include <string_view>

#include "ncnn-mlir/Graph/graph.hpp"
#include "ncnn-mlir/Importer/NCNNImporter.hpp"
#include "ncnn-mlir/Support/LayerCapabilities.hpp"
#include <gtest/gtest.h>

TEST(LayerRegistryTest, DescriptorsMatchRuntimeRegistries) {
  const auto descriptors = ncnn_compiler::get_layer_descriptors();
  std::set<std::string_view> source_types;
  std::size_t weighted_layers = 0;

  for (const auto& descriptor : descriptors) {
    EXPECT_TRUE(source_types.insert(descriptor.source_type).second)
      << "duplicate layer descriptor: " << descriptor.source_type;
    EXPECT_TRUE(ncnn_importer::has_layer_importer(descriptor.source_type))
      << "missing importer: " << descriptor.source_type;

    const bool declares_weights = ncnn_compiler::has_capability(
      descriptor.capabilities, ncnn_compiler::LayerCapability::HasWeights);
    EXPECT_EQ(ncnn_graph::has_weight_loader(descriptor.source_type),
              declares_weights)
      << "weight loader capability mismatch: " << descriptor.source_type;
    weighted_layers += static_cast<std::size_t>(declares_weights);
  }

  EXPECT_EQ(ncnn_importer::get_layer_importer_count(), descriptors.size())
    << "importer registry contains an undeclared or duplicate entry";
  EXPECT_EQ(ncnn_graph::get_weight_loader_count(), weighted_layers)
    << "weight loader registry contains an unexpected or duplicate entry";
  EXPECT_TRUE(ncnn_graph::has_weight_loader("ConvolutionDepthWise"));
  EXPECT_TRUE(ncnn_graph::has_weight_loader("InnerProduct"));
}
