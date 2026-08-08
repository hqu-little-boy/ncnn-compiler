#include "ncnn-mlir/Graph/parser.hpp"

#include <cmath>
#include <string_view>

#include <gtest/gtest.h>

namespace {

bool contains(std::string_view text, std::string_view fragment) {
  return text.find(fragment) != std::string_view::npos;
}

}  // namespace

TEST(ParserTest, ValidScalarAndArrayParameters) {
  auto params = ncnn_graph::parse_layer_params(
    "0=42 1=-1.25e2 2=relu 3=1,2,3 4=1.0,2,3e-1 "
    "-23305=3,7,8,9 -23306=2,0.5,1.5 0=99");
  ASSERT_TRUE(params.has_value()) << "valid scalar and array parameters parse";

  EXPECT_EQ(params->get_int(0), 99) << "duplicate IDs use the last value";
  EXPECT_LT(std::abs(params->get_float(1) + 125.0f), 0.001f)
    << "exponent float parses completely";
  auto string_value = params->get_string(2);
  EXPECT_TRUE(string_value && string_value->get() == "relu")
    << "string value remains owned by ParamDict";

  auto implicit_ints = params->get_int_array(3);
  EXPECT_TRUE(implicit_ints.size() == 3 && implicit_ints[0] == 1 &&
              implicit_ints[2] == 3)
    << "implicit integer array parses";
  auto implicit_floats = params->get_float_array(4);
  EXPECT_TRUE(implicit_floats.size() == 3 &&
              std::abs(implicit_floats[2] - 0.3f) < 0.001f)
    << "implicit mixed numeric array promotes to float";
  auto explicit_ints = params->get_int_array(5);
  EXPECT_TRUE(explicit_ints.size() == 3 && explicit_ints[1] == 8)
    << "encoded integer array ID and length parse";
  auto explicit_floats = params->get_float_array(6);
  EXPECT_TRUE(explicit_floats.size() == 2 && explicit_floats[0] == 0.5f)
    << "encoded float array parses";

  EXPECT_EQ(params->get_float(0, 7.0f), 7.0f)
    << "wrong-kind float lookup returns caller default";
  EXPECT_EQ(params->get_int(1, 13), 13)
    << "wrong-kind integer lookup returns caller default";
  EXPECT_FALSE(params->get_string(31))
    << "missing string lookup returns nullopt";
}

TEST(ParserTest, ZeroLengthExplicitArray) {
  auto empty_array = ncnn_graph::parse_layer_params("-23300=0");
  EXPECT_TRUE(empty_array && empty_array->get_int_array(0).empty())
    << "zero-length explicit array parses";
}

TEST(ParserTest, PreservesQuotedShapeExpression) {
  auto params = ncnn_graph::parse_layer_params("6=\"1w,1h,1c\" 7=\"+(1w, 2)\"");
  ASSERT_TRUE(params) << params.error();
  ASSERT_TRUE(params->get_string(6));
  EXPECT_EQ(params->get_string(6)->get(), "1w,1h,1c");
  ASSERT_TRUE(params->get_string(7));
  EXPECT_EQ(params->get_string(7)->get(), "+(1w, 2)");
}

TEST(ParserTest, ParamValueWrongKindReturnsNullopt) {
  auto integer = ncnn_graph::ParamValue::make_int(0);
  EXPECT_EQ(integer.get_int(), 0);
  EXPECT_FALSE(integer.get_float());
  EXPECT_FALSE(integer.get_int_array());
  EXPECT_FALSE(integer.get_float_array());
  EXPECT_FALSE(integer.get_string());

  auto empty_string = ncnn_graph::ParamValue::make_string("");
  ASSERT_TRUE(empty_string.get_string());
  EXPECT_TRUE(empty_string.get_string()->empty());
  EXPECT_FALSE(empty_string.get_int());

  auto empty_array = ncnn_graph::ParamValue::make_int_array({});
  ASSERT_TRUE(empty_array.get_int_array());
  EXPECT_TRUE(empty_array.get_int_array()->empty());
  EXPECT_FALSE(empty_array.get_float_array());
}

TEST(ParserTest, DecodesConvolutionParameters) {
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(4));
  params.set_value(1, ncnn_graph::ParamValue::make_int(3));
  params.set_value(2, ncnn_graph::ParamValue::make_int(2));
  params.set_value(3, ncnn_graph::ParamValue::make_int(2));
  params.set_value(4, ncnn_graph::ParamValue::make_int(1));
  params.set_value(5, ncnn_graph::ParamValue::make_int(1));
  params.set_value(6, ncnn_graph::ParamValue::make_int(108));
  params.set_value(8, ncnn_graph::ParamValue::make_int(101));

  auto decoded = ncnn_graph::decode_convolution_params(params);
  ASSERT_TRUE(decoded.has_value()) << decoded.error();
  EXPECT_EQ(decoded->output_channels, 4);
  EXPECT_EQ(decoded->kernel_w, 3);
  EXPECT_EQ(decoded->kernel_h, 3);
  EXPECT_EQ(decoded->dilation_w, 2);
  EXPECT_EQ(decoded->dilation_h, 2);
  EXPECT_EQ(decoded->stride_w, 2);
  EXPECT_EQ(decoded->stride_h, 2);
  EXPECT_EQ(decoded->pad_left, 1);
  EXPECT_EQ(decoded->pad_right, 1);
  EXPECT_EQ(decoded->pad_top, 1);
  EXPECT_EQ(decoded->pad_bottom, 1);
  EXPECT_TRUE(decoded->has_bias);
  EXPECT_EQ(decoded->weight_count, 108);
  EXPECT_EQ(decoded->int8_scale_term, 101);
  EXPECT_EQ(decoded->expected_weight_tensors(), 5U);
}

TEST(ParserTest, DecodesSequentialWeightLayerParameters) {
  ncnn_graph::ParamDict depthwise;
  depthwise.set_value(0, ncnn_graph::ParamValue::make_int(4));
  depthwise.set_value(1, ncnn_graph::ParamValue::make_int(3));
  depthwise.set_value(5, ncnn_graph::ParamValue::make_int(1));
  depthwise.set_value(6, ncnn_graph::ParamValue::make_int(36));
  depthwise.set_value(7, ncnn_graph::ParamValue::make_int(4));
  auto decoded_depthwise =
    ncnn_graph::decode_convolution_depthwise_params(depthwise);
  ASSERT_TRUE(decoded_depthwise) << decoded_depthwise.error();
  EXPECT_EQ(decoded_depthwise->kernel_h, 3);
  EXPECT_EQ(decoded_depthwise->group, 4);
  EXPECT_TRUE(decoded_depthwise->has_bias);
  EXPECT_EQ(decoded_depthwise->expected_weight_tensors(), 2U);

  ncnn_graph::ParamDict inner_product;
  inner_product.set_value(0, ncnn_graph::ParamValue::make_int(3));
  inner_product.set_value(1, ncnn_graph::ParamValue::make_int(1));
  inner_product.set_value(2, ncnn_graph::ParamValue::make_int(12));
  auto decoded_inner = ncnn_graph::decode_inner_product_params(inner_product);
  ASSERT_TRUE(decoded_inner) << decoded_inner.error();
  EXPECT_EQ(decoded_inner->output_channels, 3);
  EXPECT_EQ(decoded_inner->weight_count, 12);
  EXPECT_EQ(decoded_inner->expected_weight_tensors(), 2U);

  depthwise.set_value(7, ncnn_graph::ParamValue::make_int(3));
  EXPECT_FALSE(ncnn_graph::decode_convolution_depthwise_params(depthwise));
  inner_product.set_value(2, ncnn_graph::ParamValue::make_int(10));
  EXPECT_FALSE(ncnn_graph::decode_inner_product_params(inner_product));
}

TEST(ParserTest, DecodesAsymmetricDepthwiseSpatialParameters) {
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(4));
  params.set_value(1, ncnn_graph::ParamValue::make_int(3));
  params.set_value(11, ncnn_graph::ParamValue::make_int(2));
  params.set_value(2, ncnn_graph::ParamValue::make_int(2));
  params.set_value(12, ncnn_graph::ParamValue::make_int(3));
  params.set_value(3, ncnn_graph::ParamValue::make_int(4));
  params.set_value(13, ncnn_graph::ParamValue::make_int(5));
  params.set_value(4, ncnn_graph::ParamValue::make_int(6));
  params.set_value(15, ncnn_graph::ParamValue::make_int(7));
  params.set_value(14, ncnn_graph::ParamValue::make_int(8));
  params.set_value(16, ncnn_graph::ParamValue::make_int(9));
  params.set_value(6, ncnn_graph::ParamValue::make_int(24));
  params.set_value(7, ncnn_graph::ParamValue::make_int(4));

  auto decoded = ncnn_graph::decode_convolution_depthwise_params(params);
  ASSERT_TRUE(decoded) << decoded.error();
  EXPECT_EQ(decoded->kernel_w, 3);
  EXPECT_EQ(decoded->kernel_h, 2);
  EXPECT_EQ(decoded->dilation_w, 2);
  EXPECT_EQ(decoded->dilation_h, 3);
  EXPECT_EQ(decoded->stride_w, 4);
  EXPECT_EQ(decoded->stride_h, 5);
  EXPECT_EQ(decoded->pad_left, 6);
  EXPECT_EQ(decoded->pad_right, 7);
  EXPECT_EQ(decoded->pad_top, 8);
  EXPECT_EQ(decoded->pad_bottom, 9);
}

TEST(ParserTest, AppliesDepthwiseSpatialParameterFallbacks) {
  ncnn_graph::ParamDict params;
  params.set_value(0, ncnn_graph::ParamValue::make_int(2));
  params.set_value(1, ncnn_graph::ParamValue::make_int(3));
  params.set_value(2, ncnn_graph::ParamValue::make_int(2));
  params.set_value(3, ncnn_graph::ParamValue::make_int(4));
  params.set_value(4, ncnn_graph::ParamValue::make_int(6));
  params.set_value(14, ncnn_graph::ParamValue::make_int(8));
  params.set_value(6, ncnn_graph::ParamValue::make_int(18));
  params.set_value(7, ncnn_graph::ParamValue::make_int(2));

  auto decoded = ncnn_graph::decode_convolution_depthwise_params(params);
  ASSERT_TRUE(decoded) << decoded.error();
  EXPECT_EQ(decoded->dilation_h, 2);
  EXPECT_EQ(decoded->stride_h, 4);
  EXPECT_EQ(decoded->pad_right, 6);
  EXPECT_EQ(decoded->pad_top, 8);
  EXPECT_EQ(decoded->pad_bottom, 8);
}

namespace {

void expect_failure(std::string_view text,
                    std::string_view error_fragment,
                    std::string_view description) {
  auto result = ncnn_graph::parse_layer_params(text);
  EXPECT_TRUE(!result && contains(result.error(), error_fragment))
    << description;
}

}  // namespace

TEST(ParserTest, RejectsMalformedInput) {
  expect_failure("0", "missing '='", "missing assignment separator rejected");
  expect_failure(
    "0=1=2", "multiple '='", "multiple assignment separators rejected");
  expect_failure("0=", "empty param value", "empty scalar value rejected");
  expect_failure("0=12junk", "bad integer", "integer suffix garbage rejected");
  expect_failure("0=1.0junk", "bad float", "float suffix garbage rejected");
  expect_failure(
    "0=1,,2", "empty element", "consecutive array commas rejected");
  expect_failure("0=,1", "empty element", "leading array comma rejected");
  expect_failure("0=1,", "empty element", "trailing array comma rejected");
  expect_failure(
    "6=\"1w,1h", "unterminated quoted", "unterminated string rejected");
  expect_failure(
    "-23300=2,1", "length mismatch", "explicit array length mismatch rejected");
  expect_failure(
    "-23300=-1", "non-negative", "negative explicit array length rejected");
  expect_failure(
    "32=1", "param id out of range", "ordinary parameter ID above 31 rejected");
  expect_failure("-23332=1,0",
                 "param id out of range",
                 "encoded parameter ID below range rejected");
  expect_failure("999999999999999999999999=1",
                 "out of range",
                 "overflowing parameter ID rejected");
  expect_failure("0=999999999999999999999999",
                 "out of range",
                 "overflowing integer value rejected");
  expect_failure(
    "0=1e9999", "out of range", "overflowing float value rejected");
  expect_failure("0=nan", "non-finite", "non-finite float token rejected");
}
