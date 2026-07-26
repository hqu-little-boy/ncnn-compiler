// ncnn_graph: ncnn 模型计算图数据模型
// 参考 netron/source/ncnn.js 的解析逻辑，用 C++23 实现。
// 对应 ncnn 权威序列化格式：ncnn/src/{paramdict,modelbin,net,layer/*}.cpp

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ncnn_graph {

// ─────────────────────────────────────────────────────────────────────────
// ParamDict：一个层的参数集合。key 是参数位置 id（ncnn 用整型）。
// 值类型对应 ncnn ParamDictPrivate 的 7 种 type：
//   int(2/1) | float(3) | int-array(5) | float-array(4/6) | string(7)
// 参考 ncnn/src/paramdict.h
// ─────────────────────────────────────────────────────────────────────────
class ParamValue {
 public:
  // 与 ncnn type() 对应的标签
  enum class Kind { Int, Float, IntArray, FloatArray, String };

  static ParamValue make_int(std::int64_t v) {
    return ParamValue{Kind::Int, v, 0.0f, {}, {}, {}};
  }
  static ParamValue make_float(float v) {
    return ParamValue{Kind::Float, 0, v, {}, {}, {}};
  }
  static ParamValue make_int_array(std::vector<std::int64_t> v) {
    return ParamValue{Kind::IntArray, 0, 0.0f, std::move(v), {}, {}};
  }
  static ParamValue make_float_array(std::vector<float> v) {
    return ParamValue{Kind::FloatArray, 0, 0.0f, {}, std::move(v), {}};
  }
  static ParamValue make_string(std::string v) {
    return ParamValue{Kind::String, 0, 0.0f, {}, {}, std::move(v)};
  }

  Kind kind() const noexcept { return kind_; }
  std::int64_t as_int() const noexcept { return i_; }
  float as_float() const noexcept { return f_; }
  std::span<const std::int64_t> as_int_array() const noexcept { return iarr_; }
  std::span<const float> as_float_array() const noexcept { return farr_; }
  std::string_view as_string() const noexcept { return s_; }

 private:
  ParamValue(Kind k,
             std::int64_t i,
             float f,
             std::vector<std::int64_t> ia,
             std::vector<float> fa,
             std::string s)
    : kind_(k),
      i_(i),
      f_(f),
      iarr_(std::move(ia)),
      farr_(std::move(fa)),
      s_(std::move(s)) {}
  Kind kind_;
  std::int64_t i_;
  float f_;
  std::vector<std::int64_t> iarr_;
  std::vector<float> farr_;
  std::string s_;
};

class ParamDict {
 public:
  // 获取参数，带默认值。key 不存在返回 def（与 ncnn ParamDict::get(id, def)
  // 一致）
  std::int64_t get_int(int id, std::int64_t def = 0) const;
  float get_float(int id, float def = 0.0f) const;
  std::string_view get_string(int id, std::string_view def = "") const;
  std::span<const std::int64_t> get_int_array(int id) const;
  std::span<const float> get_float_array(int id) const;

  bool has(int id) const noexcept;
  void set(int id, ParamValue v);
  std::span<const std::pair<int, ParamValue>> entries() const noexcept {
    return entries_;
  }

 private:
  std::vector<std::pair<int, ParamValue>> entries_;
  const ParamValue* find(int id) const noexcept;
};

// ─────────────────────────────────────────────────────────────────────────
// Tensor：权重张量。从 .bin 读出的原始权重数据 + 形状 + 数据类型。
// 对应 ncnn ModelBin::load 产出的 Mat。参考 netron BlobReader.load 的 flag
// 解码。
// ─────────────────────────────────────────────────────────────────────────
enum class DataType { Float32, Float16, Int8 };

struct Tensor {
  std::vector<std::int64_t> shape;  // NHWC 或 NCHW，由 importer 约定
  DataType dtype = DataType::Float32;
  std::vector<std::byte> data;  // 原始字节，按 dtype 解释

  std::size_t element_count() const noexcept;
  std::size_t byte_size() const noexcept;
};

// ─────────────────────────────────────────────────────────────────────────
// Node/Blob/Graph：计算图。参考 netron ncnn.Graph/Node/Blob 的拓扑结构。
// ncnn 原生用 blob 名（字符串）连边，netron 同理。我们也用名字连边，便于对齐。
// ─────────────────────────────────────────────────────────────────────────
struct Blob {
  std::string name;
  int producer = -1;  // 产出该 blob 的 layer 索引（-1 = 模型输入）
  int consumer = -1;  // 消费该 blob 的 layer 索引（-1 = 模型输出）
};

struct Layer {
  std::string type;                  // "Convolution" / "ReLU" / ...
  std::string name;                  // 层名
  std::vector<std::string> inputs;   // 输入 blob 名
  std::vector<std::string> outputs;  // 输出 blob 名
  ParamDict params;                  // 层参数（ncnn 原生 id→值）
  std::vector<Tensor>
    weights;  // 按 load_model 顺序读出的权重（如 conv: weight, bias）

  // 便捷访问参数（语义名 → 值）。具体 id↔语义见各算子 load_param。
  std::int64_t p_int(int id, std::int64_t def = 0) const {
    return params.get_int(id, def);
  }
  float p_float(int id, float def = 0.0f) const {
    return params.get_float(id, def);
  }
};

class Graph {
 public:
  std::vector<Layer> layers;
  std::vector<Blob> blobs;
  std::vector<std::string> input_blob_names;   // 图入口 blob（Input 层的输出）
  std::vector<std::string> output_blob_names;  // 图出口 blob（consumer==-1）
  bool weights_loaded = false;                 // 是否成功加载并校验了 .bin 权重

  // 文本 .param 解析。param_path + bin_path（bin 可为空路径则不读权重）
  static std::expected<std::unique_ptr<Graph>, std::string> load(
    std::string_view param_path, std::string_view bin_path);

  // 输出人类可读的图描述（调试用）
  std::string dump() const;

  // 统计
  std::size_t layer_count_of(std::string_view type) const;
};

}  // namespace ncnn_graph
