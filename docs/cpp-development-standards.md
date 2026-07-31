# ncnn-mlir C++ 开发规范

本文档规定 ncnn-mlir 项目自有 C++ 代码的强制开发要求，适用于模型导入器、
计算图、MLIR dialect、转换 pass、编译驱动、命令行工具及测试代码。
第三方源码、系统头文件和构建生成文件不受本文档约束。

本文档中的“必须”“禁止”是合入要求；“应当”“建议”表示没有充分理由时必须遵循。
已有代码与本文档冲突时，应修改代码或工具配置，不能仅通过注释规避检查。

相关执行配置：

- `compiler/.clang-format`：格式、缩进、include 排序和控制流大括号。
- `compiler/.clang-tidy`：Google、现代 C++ 和可读性静态检查。
- `compiler/CMakeLists.txt`：语言标准、编译器、警告、测试和检查目标。

## 1. 语言与工具链

### 1.1 C++ 标准

- 项目自有 C++ 代码必须使用 C++23。
- CMake 必须设置 `CMAKE_CXX_STANDARD 23`、
  `CMAKE_CXX_STANDARD_REQUIRED ON` 和 `CMAKE_CXX_EXTENSIONS OFF`。
- 禁止依赖 GNU C++ 扩展或其他未显式声明的编译器扩展。
- 项目使用 LLVM/Clang 21 工具链，并与 MLIR/LLVM 21 保持版本一致。
- 当前工具路径为 `/usr/bin/clang-21`、`/usr/bin/clang++-21`、
  `/usr/bin/clang-format-21` 和 `/usr/bin/clang-tidy-21`。
- MLIR/LLVM 命令使用带 `-21` 后缀的版本，例如
  `/usr/bin/mlir-opt-21`、`/usr/bin/mlir-tblgen-21` 和
  `/usr/bin/FileCheck-21`。

`compiler/.clang-format` 中的 `Standard: c++20` 只表示 clang-format 21
可稳定识别的最高格式化语法模式，不改变项目的 C++23 编译标准；实际语言标准由
CMake 和 `compile_commands.json` 决定。

### 1.2 CMake

- 编译器必须在顶层 `project()` 调用之前设置。
- 必须导出 `compile_commands.json`，供 clang-tidy、clangd 和其他工具读取
  真实编译参数。
- 每个 target 必须通过 `target_include_directories`、
  `target_link_libraries`、`target_compile_definitions` 等 target 级命令声明依赖。
- 新增源文件、测试或工具时，必须加入对应 CMake target。
- 测试必须通过 `add_test` 注册到 CTest，不能只提供手工运行的可执行文件。
- 路径必须从 CMake 源目录、构建目录、命令行参数或
  `std::filesystem::path` 构造；禁止硬编码开发者机器的绝对路径。
- 编译必须至少启用 `-Wall -Wextra -Wpedantic`，项目代码必须零编译警告。

## 2. 基础代码风格

### 2.1 Google C++ 风格

项目以 Google C++ Style Guide 和 `BasedOnStyle: Google` 为基线：

- 缩进使用 2 个空格，禁止使用制表符缩进。
- 行宽上限为 80 列。
- 指针和引用符号靠近类型，例如 `const Tensor* tensor` 和
  `const Graph& graph`。
- include 由 clang-format 分组和排序；`.cpp` 的对应头文件置于最前。
- 命名应表达语义，避免含糊缩写；局部循环索引等惯用短名称除外。
- 注释解释约束、原因和不明显的格式语义，不复述代码表面行为。
- 项目不强制版权文件头。

所有格式以 `clang-format-21` 的实际结果为准。禁止通过局部格式关闭注释长期规避
项目格式；确有必要时必须说明原因。

### 2.2 所有控制流必须使用大括号

`if`、`else`、`for`、range-for、`while`、`do` 等控制流的主体必须使用
`{}`，即使主体为空或只有一条语句。`compiler/.clang-format` 通过
`InsertBraces: true` 执行该规则，clang-tidy 的 brace checks 同时保持启用。

合规：

```cpp
if (!result) {
  return std::unexpected(result.error());
}

for (const auto& layer : layers) {
  validate(layer);
}
```

不合规：

```cpp
if (!result)
  return std::unexpected(result.error());

for (const auto& layer : layers) validate(layer);
```

不得以“只有一行”“提前返回”或“宏展开”为理由省略控制流大括号。

## 3. 头文件与实现文件

- 头文件只放类型、函数和方法声明；方法实现必须放入 `.cpp` 文件。
- 禁止在 class 声明中编写 getter、setter、构造函数或其他方法的方法体。
- 非公开实现细节应放在 `.cpp` 的匿名 namespace 中。
- 头文件必须自包含，直接 include 自身所使用类型对应的标准库或项目头文件。
- 使用 `#pragma once` 作为项目头文件的 include guard。
- 禁止仅为传递依赖而依赖其他头文件的偶然 include。
- 模板或生成代码若因语言/工具要求必须在头文件中定义，应在引入前经过明确的
  设计评审；该例外不能用于普通非模板类。

合规：

```cpp
// tensor.hpp
class Tensor {
 public:
  Tensor();
  DataType get_dtype() const noexcept;

 private:
  DataType dtype_;
};
```

```cpp
// tensor.cpp
Tensor::Tensor() : dtype_(DataType::Unknown) {}

DataType Tensor::get_dtype() const noexcept {
  return dtype_;
}
```

不合规：

```cpp
class Tensor {
 public:
  Tensor() : dtype_(DataType::Unknown) {}
  DataType get_dtype() const { return dtype_; }

 private:
  DataType dtype_;
};
```

## 4. 类设计与状态管理

### 4.1 成员必须私有

- 所有类属性必须为 `private`。
- 禁止通过 public data member 暴露对象状态。
- 读取通过 `get_*`，替换通过 `set_*`，受控追加通过 `add_*`。
- getter 应尽可能为 `const noexcept`，且不得修改对象。
- 集合 getter 优先返回 `std::span<const T>`，字符串 getter 优先返回
  `std::string_view`，避免复制和暴露可变容器。
- setter 对取得所有权的值使用值传递后移动，例如
  `void set_name(std::string name)`。
- 不应提供返回可变容器引用的接口来绕过对象不变量。

### 4.2 所有成员在构造函数中显式初始化

- 禁止在成员声明处设置默认值。
- 每个构造函数必须在 member initializer list 中显式初始化每个直接数据成员。
- 空字符串、空 vector、空字典和布尔默认状态也必须显式初始化。
- 初始化顺序必须与成员声明顺序一致。
- `Tensor::Tensor()` 必须将 dtype 初始化为 `DataType::Unknown`，不能隐式假定
  `Float32`。
- `.clang-tidy` 特意关闭 `modernize-use-default-member-init` 和
  `readability-redundant-member-init`，不能据此省略构造函数初始化。

合规：

```cpp
class Graph {
 public:
  Graph();

 private:
  std::vector<Layer> layers_;
  std::vector<Blob> blobs_;
  bool weights_loaded_;
};

Graph::Graph() : layers_(), blobs_(), weights_loaded_(false) {}
```

不合规：

```cpp
class Graph {
 private:
  std::vector<Layer> layers_{};
  std::vector<Blob> blobs_;
  bool weights_loaded_ = false;
};
```

### 4.3 值语义优先

- 领域对象默认采用值语义，以对象、`std::vector<T>`、`std::optional<T>` 或
  `std::expected<T, E>` 表达所有权和结果。
- 禁止为可直接移动返回的对象引入无必要的堆分配，例如禁止使用
  `std::expected<std::unique_ptr<Graph>, std::string>` 返回 Graph。
- 工厂或加载函数应直接返回值：

```cpp
[[nodiscard]] std::expected<Graph, std::string> load_graph(
  const std::filesystem::path& path);
```

- 需要运行期 tagged union 时必须优先使用 `std::variant`，不得手工维护 tag
  与多个同时存在的 payload。
- `ParamValue` 一类对象必须保证任意时刻仅有一个活动值，并通过
  `std::get_if` 或 `std::visit` 类型安全访问。
- 非拥有连续区间使用 `std::span`；可缺省结果使用 `std::optional`；
  非拥有且不可为空的对象关系优先使用引用。
- 需要在 optional 中表达引用时，使用
  `std::optional<std::reference_wrapper<const T>>`。

## 5. 指针、引用与生命周期

- 禁止项目自有 API 使用裸指针表达所有权。
- 禁止新增可由引用、`std::span`、`std::string_view`、`std::optional` 或值
  表达的裸指针 observer API。
- 禁止手工 `new`/`delete`。
- 标准接口边界所要求的指针允许保留，包括：
  - `int main(int argc, char** argv)`；
  - `std::ifstream::read(char*, std::streamsize)`；
  - `std::from_chars` 的 pointer range；
  - `std::get_if` 返回的类型检查指针；
  - 标准 C ABI 导出函数的数组指针。
- C ABI 的每个数组指针必须同时有明确的元素数量或 shape 参数，并在接口文档中
  规定所有权、可变性、生命周期和错误返回方式。
- `std::span` 和 `std::string_view` 不拥有数据。返回或保存它们时必须保证底层对象
  生命周期更长，禁止返回指向局部对象或临时对象的 view。

## 6. 错误处理与事务性

### 6.1 使用 `std::expected`

- 可恢复的解析、I/O、验证、shape 设置和 lowering 失败必须通过
  `std::expected<T, std::string>` 或适当的结构化错误类型返回。
- 返回结果必须标记 `[[nodiscard]]`，调用方必须检查后再解引用。
- 禁止使用空错误字符串、输出参数或半初始化对象表达失败。
- 禁止让格式错误、范围错误或文件错误以异常形式逃出 importer 的 expected API。
- 错误信息必须包含可定位上下文，例如参数 ID、layer index/type、文件阶段、期望值
  与实际值。
- 不可能失败的轻量 getter 应标记 `noexcept`。

合规：

```cpp
[[nodiscard]] std::expected<ParamDict, std::string> parse_layer_params(
  std::string_view text);

const auto params = parse_layer_params(text);
if (!params) {
  return std::unexpected(
    std::format("layer {} ({}): {}", index, type, params.error()));
}
```

不合规：

```cpp
std::string parse_layer_params(std::string_view text, ParamDict* output);

ParamDict params;
const auto error = parse_layer_params(text, &params);
```

### 6.2 失败必须保持事务性

- 解析函数只有在完整输入验证成功后才能返回结果对象。
- 修改已有对象时，应先在局部值中完成全部验证，再一次性提交新状态。
- 失败的 `Tensor::set_shape()` 不得改变原 shape。
- 二进制 cursor 的读取和对齐操作必须先验证剩余范围，成功后才能推进位置。
- 加载失败不得对调用方暴露部分构建且看似有效的 Graph。

## 7. 文本解析与数值安全

### 7.1 严格数值解析

- 整数和浮点解析必须使用 `std::from_chars`。
- 必须同时检查 `std::errc` 和结束指针，要求 token 被完整消费。
- 必须区分格式错误与 `std::errc::result_out_of_range`。
- 浮点结果必须用 `std::isfinite` 拒绝 NaN 和无穷值。
- 禁止使用 `std::stoi`、`std::stol`、`std::strtod` 等异常式或可能部分消费
  token 的接口。
- 空 token、后缀垃圾、非法符号和超出目标范围的值必须明确失败。
- 输入的语法分类顺序必须经过测试，不能因为字符串中含 `e` 就错误分类为浮点数。

合规的核心检查：

```cpp
Value value{};
auto [end, error] =
  std::from_chars(token.data(), token.data() + token.size(), value);
if (error != std::errc{} || end != token.data() + token.size()) {
  return std::unexpected("invalid numeric token");
}
```

### 7.2 范围、转换和算术

- 所有来自文件的 count、ID、维度和字节数都视为不可信输入。
- 窄化前必须验证目标类型可表示范围，优先使用 `std::in_range`、
  `std::cmp_*` 或项目 checked conversion helper。
- 加法、乘法、对齐和元素数到字节数的计算必须在执行前检查溢出。
- 除法或取模前必须检查除数及相关正数不变量。
- 禁止依赖有符号整数溢出、负数取反或从大整数到 `size_t` 的未检查转换。
- checked helper 必须返回 `std::expected`，错误中说明参与计算的语义。

不合规：

```cpp
const auto bytes = static_cast<std::size_t>(count * element_size);
const int id = static_cast<int>(parsed_id);
```

合规：

```cpp
if (count > std::numeric_limits<std::size_t>::max() / element_size) {
  return std::unexpected("tensor byte size overflows size_t");
}
const std::size_t bytes = count * element_size;
```

## 8. 文件与二进制处理

### 8.1 文件系统和读取

- 路径和文件元数据使用 `std::filesystem::path` 与 `std::filesystem` API。
- 可能失败的 filesystem 操作应使用带 `std::error_code` 的重载，并把错误转换为
  expected error。
- 文件大小必须先验证可由 `std::size_t`、容器 `max_size()` 和
  `std::streamsize` 表示。
- 读取必须验证实际读取成功且字节数精确匹配，禁止把 `tellg()` 的负值直接转换为
  无符号大小。
- `reinterpret_cast<char*>` 仅限 `ifstream::read` 等标准接口必需边界。

### 8.2 显式处理字节序

- 模型二进制格式按小端解析，不能假定宿主机字节序。
- 32 位字段使用 `std::array<std::byte, 4>`、`std::bit_cast`、
  `std::endian` 和必要的 `std::byteswap` 解码。
- 禁止用 native-endian `memcpy` 后直接解释整数。
- 二进制 cursor 的每项操作必须返回 expected，例如：

```cpp
std::expected<std::uint32_t, std::string> read_u32_le();
std::expected<std::span<const std::byte>, std::string> read_bytes(
  std::size_t count);
std::expected<void, std::string> align4();
```

- 禁止使用 sticky `ok` 状态掩盖第一次错误；每次读取失败必须立即传播。
- 必须拒绝截断 payload、缺失 alignment padding、未知 type flag 和未消费尾随字节。
- 空 bin path 表示不加载权重；一旦提供路径，即使文件大小为零，也必须执行完整验证。

## 9. 模型对象不变量

### 9.1 Tensor

- 默认 dtype 必须为 `DataType::Unknown`。
- Unknown dtype 的 `byte_size()` 为 0，不得误标记为 Int8 或 Float32。
- shape 维度必须非负；零维度合法并产生零元素。
- 空 shape 表示标量，`element_count()` 为 1。
- 元素数及元素数乘 dtype 宽度必须可由 `std::size_t` 表示。
- `set_shape()` 必须在验证成功后才更新 shape。
- tensor data 的 dtype、shape 和 byte size 必须在权重加载阶段保持一致。

### 9.2 Graph 与拓扑

- 参数 ID、layer/blob/bottom/top count 必须严格解析并验证范围。
- 标准 param magic 必须精确匹配 `7767517`；若保留旧式 header 兼容，两个
  header token 也必须完整合法。
- 实际构造的 blob 数必须等于声明值。
- bottom blob 必须由更早的 layer 产生；禁止未解析 bottom。
- 一个 blob 只能有一个 producer；禁止重复 producer。
- layer 参数错误必须附带 layer index 和 type 上下文。
- 提供 bin 时，只有在所有期望权重读取完成且 cursor 完全消费后，才能设置
  `weights_loaded`。

### 9.3 算子权重

- 算子参数必须在用于 shape 或内存计算前验证语义范围。
- Convolution 的输出通道、kernel 和权重元素数必须为正，`bias_term` 只能为
  0 或 1。
- 权重数必须能被输出通道与 kernel 元素积整除，推导的输入通道必须为正。
- dynamic weight 算子不得消费静态 model-bin 权重。
- Int8 权重模式必须按 ncnn 格式完整读取 weight scale、bottom scale 和条件性的
  top scale。

## 10. API 设计

- 方法命名统一使用 `get_*`、`set_*` 和 `add_*`。
- 成功值可直接移动返回时，不使用输出参数。
- 返回值仅在忽略会导致错误或不完整操作时使用 `[[nodiscard]]`；expected API
  必须标记。
- getter 返回的默认值必须由调用方明确提供或由接口文档定义；类型不匹配不得返回
  另一种 payload 的未定义解释。
- 字符串缺失使用 optional 表达，禁止返回可能悬垂的 fallback string view。
- 枚举用于有限状态集合，例如 `DataType` 和参数种类；不要使用相互关联的 bool
  组合模拟枚举。
- 公共接口的所有权、错误语义、单位、shape/layout 和生命周期必须可从声明及文档
  明确判断。
- 导出的推理入口必须是标准 C ABI，以原生数组和 shape 信息传递数据；不得把
  C++ STL 类型或项目自定义 runtime 对象暴露到 ABI。

## 11. 格式化与静态检查

每次提交前必须满足：

- `clang-format-21 --dry-run --Werror` 对全部项目 `.hpp`/`.cpp` 通过。
- clang-tidy 使用 CMake 生成的 `compile_commands.json` 和 C++23 参数。
- clang-tidy 输出中不得有项目代码警告；仅看到命令成功退出不足以替代检查输出。
- 不得新增全局 `NOLINT`、大范围 check 关闭或第三方目录误扫描。
- 局部 `NOLINT` 只能用于已确认的工具误报，并必须写明具体 check 和原因。
- `tidy_fix` 会修改文件，只能在检查 diff 后使用，不能把自动修复当成语义验证。

常用命令：

```bash
cmake --build compiler/build --target format
cmake --build compiler/build --target format_check
cmake --build compiler/build --target tidy
```

## 12. 测试规范

### 12.1 覆盖要求

新增或修改行为必须配套测试，覆盖范围与风险相匹配：

- 正常路径：合法标量、数组、图拓扑、权重和真实模型。
- 边界值：空输入、零长度、零维度、最大可表示值和精确边界。
- 畸形输入：截断、后缀垃圾、范围溢出、负 count、非法 ID、未知 flag、尾随
  字节和缺失 padding。
- 类型错误：错误参数类型、错误 dtype 和不一致 shape。
- 事务性：失败后原对象状态不变，cursor 不越界，调用方拿不到半成品。
- 集成验证：至少保留 SqueezeNet v1.1 的层数、拓扑、权重 shape、dtype 和完整
  bin 消费验证。
- importer 或 lowering 扩展到新算子时，必须同时增加正常模型和畸形属性测试。

测试代码同样遵循本文档，包括构造函数初始化、私有状态和控制流大括号规则。
测试 fixture 必须创建在构建目录，不能污染源码目录或依赖开发者绝对路径。

### 12.2 Sanitizer

涉及解析、shape、索引、字节游标或内存布局的修改，必须运行 ASan 与 UBSan。
独立 sanitizer 构建示例：

```bash
cmake -S compiler -B compiler/build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build compiler/build-sanitize --parallel
ctest --test-dir compiler/build-sanitize --output-on-failure
```

Sanitizer 报告、崩溃、未定义行为或测试失败均为阻断问题，不能只根据最终进程返回值
忽略诊断。

## 13. 提交前验收

从项目根目录执行：

```bash
cmake -S compiler -B compiler/build -DCMAKE_BUILD_TYPE=Release
cmake --build compiler/build --parallel
ctest --test-dir compiler/build --output-on-failure
cmake --build compiler/build --target format_check
cmake --build compiler/build --target tidy
```

涉及不可信输入、数值计算或二进制读写时，再执行第 12.2 节 sanitizer 构建。

合入前还必须确认：

- 构建和全部 CTest 通过。
- format check 通过。
- clang-tidy 对项目代码零警告。
- 必需的 ASan/UBSan 测试通过。
- 没有新增 `stoi`、`stol`、`strtod`、手工 `new`/`delete`、未检查窄化、
  native-endian 解码或项目自有裸指针所有权/observer API。
- 没有省略控制流大括号、声明处成员默认值、public data member 或头文件中的普通
  方法实现。
- 错误路径包含足够上下文，失败不留下部分提交的状态。

自动化工具只能执行其中一部分规则。生命周期、值语义、算术安全、事务性、格式兼容
和 ABI 边界仍必须通过代码评审与针对性测试验证。

## 14. MLIR 工具约定

- `bin/` 下所有 `*-opt` 工具（`mlir-opt` 克隆类）必须基于 `mlir::MlirOptMain` 实现，
  禁止手写 parse→verify→print 的极简 main。否则标准选项（`-o`、`--mlir-print-op-generic`、
  `--verify-diagnostics`）与全部 pass 都会缺失，lit 的 `RUN` 行会报
  `Unknown command line argument`。
- opt 工具必须用 `registerAllDialects` / `registerAllExtensions` / `registerAllPasses`
  注册全部上游方言/扩展/pass，再叠加项目方言，保证能运行任意上游 pass。
- 方言注册集中在 `bin/RegisterNCNNDialects.h`，所有 opt 工具共用，保证方言集合一致。
- 链接 opt 工具必须引入 `MLIR_DIALECT_LIBS` / `MLIR_CONVERSION_LIBS` /
  `MLIR_EXTENSION_LIBS` 全局属性列出的库，加 `MLIRMlirOptMain`、`MLIRPass`，
  并以 `-fno-rtti` 编译（与 MLIR 构建一致）。
- 标准 main 模板、CMake 链接模板与注意事项见
  `docs/ncnn-mlir-development-guide.md` §8。
