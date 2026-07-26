#include "ncnn_graph/graph.hpp"

#include "ncnn_graph/parser.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace ncnn_graph {

// ───────────────────────── ParamDict ─────────────────────────
const ParamValue* ParamDict::find(int id) const noexcept {
  for (const auto& [k, v] : entries_)
    if (k == id)
      return &v;
  return nullptr;
}

bool ParamDict::has(int id) const noexcept {
  return find(id) != nullptr;
}

void ParamDict::set(int id, ParamValue v) {
  for (auto& [k, _] : entries_)
    if (k == id) {
      _ = std::move(v);
      return;
    }
  entries_.emplace_back(id, std::move(v));
}

std::int64_t ParamDict::get_int(int id, std::int64_t def) const {
  const auto* p = find(id);
  return p ? p->as_int() : def;
}
float ParamDict::get_float(int id, float def) const {
  const auto* p = find(id);
  return (p && p->kind() == ParamValue::Kind::Float) ? p->as_float() : def;
}
std::string_view ParamDict::get_string(int id, std::string_view def) const {
  const auto* p = find(id);
  return (p && p->kind() == ParamValue::Kind::String) ? p->as_string() : def;
}
std::span<const std::int64_t> ParamDict::get_int_array(int id) const {
  const auto* p = find(id);
  if (p && p->kind() == ParamValue::Kind::IntArray)
    return p->as_int_array();
  return {};
}
std::span<const float> ParamDict::get_float_array(int id) const {
  const auto* p = find(id);
  if (p && p->kind() == ParamValue::Kind::FloatArray)
    return p->as_float_array();
  return {};
}

// ───────────────────────── Tensor ─────────────────────────
std::size_t Tensor::element_count() const noexcept {
  std::size_t n = 1;
  for (auto d : shape)
    n *= static_cast<std::size_t>(d);
  return n;
}
std::size_t Tensor::byte_size() const noexcept {
  switch (dtype) {
    case DataType::Float32:
      return element_count() * 4;
    case DataType::Float16:
      return element_count() * 2;
    case DataType::Int8:
      return element_count();
  }
  return 0;
}

// ───────────────────────── Graph load ─────────────────────────
namespace {

// 读整个文件到字节向量
std::expected<std::vector<std::byte>, std::string> read_file_bytes(
  std::string_view path) {
  std::string p(path);
  std::ifstream f{p, std::ios::binary};
  if (!f)
    return std::unexpected(std::string("cannot open file: ") + p);
  f.seekg(0, std::ios::end);
  std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<std::byte> buf(static_cast<std::size_t>(n));
  if (n > 0)
    f.read(reinterpret_cast<char*>(buf.data()), n);
  if (!f)
    return std::unexpected(std::string("read error: ") + p);
  return buf;
}

// 按行读文本
std::expected<std::vector<std::string>, std::string> read_lines(
  std::string_view path) {
  std::string p(path);
  std::ifstream f{p};
  if (!f)
    return std::unexpected(std::string("cannot open param file: ") + p);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(f, line)) {
    // trim
    auto a = line.find_first_not_of(" \t\r\n");
    auto b = line.find_last_not_of(" \t\r\n");
    if (a == std::string::npos)
      continue;
    lines.push_back(line.substr(a, b - a + 1));
  }
  return lines;
}

// 从权重流读一个张量。type=0 自检 flag，type=1 原始 float32。
// 移植自 ncnn/src/modelbin.cpp ModelBinFromDataReader::load。
struct BinCursor {
  const std::byte* base = nullptr;
  std::size_t size = 0;
  std::size_t pos = 0;
  bool ok = true;
};

std::uint32_t read_u32_le(BinCursor& c) {
  if (c.pos + 4 > c.size) {
    c.ok = false;
    return 0;
  }
  std::uint32_t v;
  std::memcpy(&v, c.base + c.pos, 4);
  c.pos += 4;
  return v;
}
void read_bytes(BinCursor& c, std::byte* dst, std::size_t n) {
  if (c.pos + n > c.size) {
    c.ok = false;
    return;
  }
  std::memcpy(dst, c.base + c.pos, n);
  c.pos += n;
}
void align4(BinCursor& c) {
  std::size_t r = c.pos % 4;
  if (r)
    c.pos += (4 - r);
}

// 读一个权重张量。w = 元素数，type 0=自检 1=float32 原始
std::expected<Tensor, std::string> load_weight(BinCursor& c,
                                               std::int64_t w,
                                               int type) {
  Tensor t;
  if (w <= 0)
    return std::unexpected("non-positive weight size");
  if (type == 0) {
    std::uint32_t flag = read_u32_le(c);
    if (!c.ok)
      return std::unexpected("unexpected EOF reading weight flag");
    // ncnn flag 判定
    if (flag == 0x01306B47) {  // float16
      t.dtype = DataType::Float16;
      t.data.resize(static_cast<std::size_t>(w) * 2);
      read_bytes(c, t.data.data(), t.data.size());
      align4(c);
    } else if (flag == 0x000D4B38) {  // int8
      t.dtype = DataType::Int8;
      t.data.resize(static_cast<std::size_t>(w));
      read_bytes(c, t.data.data(), t.data.size());
      align4(c);
    } else if (flag == 0x0002C056 || flag == 0x00000000) {  // float32
      t.dtype = DataType::Float32;
      t.data.resize(static_cast<std::size_t>(w) * 4);
      read_bytes(c, t.data.data(), t.data.size());
    } else {
      // 量化（256 浮点查找表 + 索引）—— Phase-3 支持，先报错
      return std::unexpected(
        "quantized (lookup-table) weights not supported yet (flag)");
    }
    if (!c.ok)
      return std::unexpected("unexpected EOF reading weight data");
  } else if (type == 1) {
    t.dtype = DataType::Float32;
    t.data.resize(static_cast<std::size_t>(w) * 4);
    read_bytes(c, t.data.data(), t.data.size());
    if (!c.ok)
      return std::unexpected("unexpected EOF reading weight (type=1)");
  } else {
    return std::unexpected("unsupported weight load type");
  }
  return t;
}

// 按算子 load_model 顺序读权重。参考 ncnn/src/layer/*.cpp 各算子的 load_model。
// 这里只实现 squeezenet 用到的 8 算子中的有权重者：Convolution。
std::expected<void, std::string> load_layer_weights(Layer& layer,
                                                    BinCursor& cur) {
  if (layer.type == "Convolution") {
    // num_output(0) weight_data_size(6) bias_term(5)
    std::int64_t num_output = layer.p_int(0);
    std::int64_t weight_data_size = layer.p_int(6);
    int bias_term = static_cast<int>(layer.p_int(5));
    // weight: type 0
    auto wt = load_weight(cur, weight_data_size, 0);
    if (!wt)
      return std::unexpected("conv weight: " + wt.error());
    // weight 形状（NHWC 内核 hwcf）：[num_output, num_input, kh, kw]
    // num_input = weight_data_size / (num_output*kh*kw)
    int kernel_w = static_cast<int>(layer.p_int(1));
    int kernel_h = static_cast<int>(layer.p_int(11, kernel_w));
    std::int64_t denom = num_output * kernel_h * kernel_w;
    std::int64_t num_input = denom > 0 ? weight_data_size / denom : 0;
    wt->shape = {num_output, num_input, kernel_h, kernel_w};
    layer.weights.push_back(std::move(*wt));
    if (bias_term) {
      auto bt = load_weight(cur, num_output, 1);
      if (!bt)
        return std::unexpected("conv bias: " + bt.error());
      bt->shape = {num_output};
      layer.weights.push_back(std::move(*bt));
    }
  }
  // ReLU/Split/Concat/Pooling/Softmax/Dropout/Input：无权重（Phase-1）
  return {};
}

// 切分一行按空格
std::vector<std::string> split_ws(std::string_view s) {
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
    if (i >= s.size())
      break;
    std::size_t j = i;
    while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j])))
      ++j;
    out.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

}  // namespace

std::expected<std::unique_ptr<Graph>, std::string> Graph::load(
  std::string_view param_path, std::string_view bin_path) {
  auto lines = read_lines(param_path);
  if (!lines)
    return std::unexpected(lines.error());
  if (lines->empty())
    return std::unexpected("empty param file");

  // 第一行：魔数 7767517（或带 blob_count 的旧式？netron 也处理 header
  // 为两整数） ncnn 标准：第 1 行 magic，第 2 行 "layer_count blob_count"
  std::size_t li = 0;
  long magic = 0;
  try {
    magic = std::stol((*lines)[li]);
  } catch (...) {
    return std::unexpected("bad magic");
  }
  if (magic != 7767517) {
    // netron 兼容：若第一行就是 "layer blob" 两整数
    auto h = split_ws((*lines)[0]);
    if (h.size() == 2) { /* 视作 header，跳过 magic 检查 */
    } else
      return std::unexpected("param magic mismatch (expected 7767517)");
  } else {
    ++li;
  }
  if (li >= lines->size())
    return std::unexpected("missing layer/blob count header");
  auto hdr = split_ws((*lines)[li]);
  if (hdr.size() != 2)
    return std::unexpected("bad layer/blob count header");
  int layer_count = 0, blob_count = 0;
  try {
    layer_count = std::stoi(hdr[0]);
    blob_count = std::stoi(hdr[1]);
  } catch (...) {
    return std::unexpected("bad layer/blob count");
  }
  ++li;

  auto g = std::make_unique<Graph>();
  g->layers.reserve(static_cast<std::size_t>(layer_count));

  // 读权重 bin（若提供）
  std::vector<std::byte> bin_data;
  BinCursor cur{};
  if (!bin_path.empty()) {
    auto bd = read_file_bytes(bin_path);
    if (!bd)
      return std::unexpected(bd.error());
    bin_data = std::move(*bd);
    cur.base = bin_data.data();
    cur.size = bin_data.size();
    cur.pos = 0;
  }

  // blob 名 → 索引（与 ncnn/netron 同：按名字去重分配 blob 槽）
  std::unordered_map<std::string, int> blob_index;
  auto find_or_alloc_blob = [&](const std::string& name) -> int {
    auto it = blob_index.find(name);
    if (it != blob_index.end())
      return it->second;
    int idx = static_cast<int>(g->blobs.size());
    g->blobs.push_back({name, -1, -1});
    blob_index.emplace(name, idx);
    return idx;
  };

  for (int idx = 0; idx < layer_count; ++idx) {
    if (li >= lines->size())
      return std::unexpected("param truncated before all layers read");
    auto toks = split_ws((*lines)[li]);
    ++li;
    if (toks.size() < 4)
      return std::unexpected("layer line too short");
    Layer layer;
    layer.type = toks[0];
    layer.name = toks[1];
    int bottom_count = std::stoi(toks[2]);
    int top_count = std::stoi(toks[3]);
    std::size_t off = 4;
    if (toks.size() < off + static_cast<std::size_t>(bottom_count + top_count))
      return std::unexpected("layer line missing blob names");
    for (int b = 0; b < bottom_count; ++b) {
      layer.inputs.push_back(toks[off++]);
    }
    for (int t = 0; t < top_count; ++t) {
      layer.outputs.push_back(toks[off++]);
    }

    // 剩余 toks[off..] 拼成参数尾串
    std::string param_tail;
    for (std::size_t k = off; k < toks.size(); ++k) {
      if (k > off)
        param_tail += ' ';
      param_tail += toks[k];
    }
    if (!param_tail.empty()) {
      std::string err = parse_layer_params(param_tail, layer.params);
      if (!err.empty())
        return std::unexpected(
          std::format("layer {} ({}): {}", idx, layer.type, err));
    }

    // 连边：设置 producer/consumer
    for (const auto& bn : layer.inputs) {
      int bi = find_or_alloc_blob(bn);
      if (g->blobs[bi].consumer == -1)
        g->blobs[bi].consumer = idx;
    }
    for (const auto& bn : layer.outputs) {
      int bi = find_or_alloc_blob(bn);
      g->blobs[bi].producer = idx;
    }

    g->layers.push_back(std::move(layer));
  }

  // 读权重（按 layer 顺序，复刻 ncnn 的 load_model 调用顺序）
  bool weights_loaded = false;
  if (cur.size > 0) {
    for (auto& layer : g->layers) {
      auto r = load_layer_weights(layer, cur);
      if (!r)
        return std::unexpected(r.error());
    }
    // 校验：权重应正好读完整（pos == size），否则模型/解析不一致
    if (cur.pos != cur.size) {
      return std::unexpected(
        std::format("bin size mismatch: consumed {} bytes of {} (差 {})",
                    cur.pos,
                    cur.size,
                    cur.size - cur.pos));
    }
    weights_loaded = true;
  }
  g->weights_loaded = weights_loaded;

  // 图入口：Input 层的输出 blob = 模型输入；图出口：consumer==-1 的 blob =
  // 输出。 (ncnn 里 Input 是真正的层，其 outputs 是模型输入张量；裸
  // producer==-1 的 blob
  //  在合法模型中不存在，故不作为入口判定。)
  for (const auto& l : g->layers) {
    if (l.type == "Input") {
      for (const auto& bn : l.outputs)
        g->input_blob_names.push_back(bn);
    }
  }
  for (const auto& b : g->blobs) {
    if (b.consumer == -1)
      g->output_blob_names.push_back(b.name);
  }

  (void)blob_count;
  return g;
}

// ───────────────────────── dump ─────────────────────────
std::string Graph::dump() const {
  std::ostringstream os;
  os << std::format(
    "ncnn_graph: {} layers, {} blobs\n", layers.size(), blobs.size());
  os << std::format("inputs: {}\n", input_blob_names.size());
  for (const auto& n : input_blob_names)
    os << std::format("  - {}\n", n);
  os << std::format("outputs: {}\n", output_blob_names.size());
  for (const auto& n : output_blob_names)
    os << std::format("  - {}\n", n);
  os << "layers:\n";
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto& l = layers[i];
    os << std::format("  [{:>3}] {:<14} {:<22} in=[", i, l.type, l.name);
    for (std::size_t k = 0; k < l.inputs.size(); ++k) {
      if (k)
        os << ",";
      os << l.inputs[k];
    }
    os << "] out=[";
    for (std::size_t k = 0; k < l.outputs.size(); ++k) {
      if (k)
        os << ",";
      os << l.outputs[k];
    }
    os << "]";
    // 参数
    auto entries = l.params.entries();
    if (!entries.empty()) {
      os << " {";
      for (std::size_t k = 0; k < entries.size(); ++k) {
        if (k)
          os << " ";
        const auto& [id, v] = entries[k];
        os << id << "=";
        switch (v.kind()) {
          case ParamValue::Kind::Int:
            os << v.as_int();
            break;
          case ParamValue::Kind::Float:
            os << v.as_float();
            break;
          case ParamValue::Kind::String:
            os << "\"" << v.as_string() << "\"";
            break;
          case ParamValue::Kind::IntArray: {
            os << "[";
            auto a = v.as_int_array();
            for (std::size_t m = 0; m < a.size(); ++m) {
              if (m)
                os << ",";
              os << a[m];
            }
            os << "]";
          } break;
          case ParamValue::Kind::FloatArray: {
            os << "[";
            auto a = v.as_float_array();
            for (std::size_t m = 0; m < a.size(); ++m) {
              if (m)
                os << ",";
              os << a[m];
            }
            os << "]";
          } break;
        }
      }
      os << "}";
    }
    // 权重
    if (!l.weights.empty()) {
      os << " w=[";
      for (std::size_t k = 0; k < l.weights.size(); ++k) {
        if (k)
          os << ",";
        const auto& w = l.weights[k];
        os << "[";
        for (std::size_t m = 0; m < w.shape.size(); ++m) {
          if (m)
            os << ",";
          os << w.shape[m];
        }
        const char* dt = w.dtype == DataType::Float32   ? "f32"
                         : w.dtype == DataType::Float16 ? "f16"
                                                        : "i8";
        os << ":" << dt << ":" << w.data.size() << "B";
        os << "]";
      }
      os << "]";
    }
    os << "\n";
  }
  return os.str();
}

std::size_t Graph::layer_count_of(std::string_view type) const {
  return static_cast<std::size_t>(
    std::count_if(layers.begin(), layers.end(), [&](const Layer& l) {
      return l.type == type;
    }));
}

}  // namespace ncnn_graph
