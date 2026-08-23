#include "ncnn-mlir/Support/TargetVectorInfo.hpp"

namespace ncnn_mlir {
namespace {

// 收集 feature 开关：显式 feature 列表与 march 内嵌 "+feature" 段共同决定，
// 后出现的负向开关覆盖此前的正向开关。
void collect_features(llvm::StringRef march,
                      llvm::ArrayRef<llvm::StringRef> features,
                      llvm::SmallVectorImpl<llvm::StringRef>& tokens) {
  llvm::StringRef remaining = march;
  while (!remaining.empty()) {
    std::pair<llvm::StringRef, llvm::StringRef> segment = remaining.split("-");
    std::pair<llvm::StringRef, llvm::StringRef> plus = segment.first.split("+");
    if (!plus.second.empty()) {
      tokens.push_back(plus.second);
    }
    remaining = segment.second;
  }
  tokens.append(features.begin(), features.end());
}

llvm::StringRef feature_body(llvm::StringRef token) {
  return token.starts_with("+") ? token.drop_front() : token;
}

// 精确匹配或同族前缀匹配（avx512* 家族、sve/sve2 派生）。
bool enables(const llvm::SmallVectorImpl<llvm::StringRef>& tokens,
             llvm::StringRef prefix) {
  for (llvm::StringRef token : tokens) {
    const llvm::StringRef body = feature_body(token);
    if (body == prefix || body.starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

}  // namespace

TargetVectorInfo TargetVectorInfo::resolve(
  llvm::StringRef triple,
  llvm::StringRef march,
  llvm::ArrayRef<llvm::StringRef> features) {
  llvm::SmallVector<llvm::StringRef, 16> tokens;
  collect_features(march, features, tokens);

  const bool x86 = triple.contains("x86_64") || triple.contains("amd64") ||
                   triple.contains("i686") || triple.contains("i386");
  const bool aarch64 = triple.contains("aarch64") || triple.contains("arm64");
  const bool riscv = triple.contains("riscv");

  if (x86) {
    if (enables(tokens, "avx512")) {
      return TargetVectorInfo{.mode = Mode::FixedWidth, .lanes = 16};
    }
    if (enables(tokens, "avx2") || march == "native") {
      return TargetVectorInfo{.mode = Mode::FixedWidth, .lanes = 8};
    }
    // x86-64 基线含 SSE2。
    return TargetVectorInfo{.mode = Mode::FixedWidth, .lanes = 4};
  }
  if (aarch64) {
    if (enables(tokens, "sve")) {
      return TargetVectorInfo{.mode = Mode::Scalable, .lanes = 4};
    }
    return TargetVectorInfo{.mode = Mode::FixedWidth, .lanes = 4};
  }
  if (riscv) {
    bool vector_extension = false;
    for (llvm::StringRef token : tokens) {
      const llvm::StringRef body = feature_body(token);
      if (body == "v" || body.starts_with("zve") || body.starts_with("zvl")) {
        vector_extension = true;
      }
    }
    // march 形如 rv64gcv：单字母扩展直接扫描。
    for (char letter : march) {
      if (letter == 'v') {
        vector_extension = true;
      }
    }
    if (vector_extension) {
      return TargetVectorInfo{.mode = Mode::Scalable, .lanes = 4};
    }
    return TargetVectorInfo{.mode = Mode::Scalar, .lanes = 0};
  }
  return TargetVectorInfo{.mode = Mode::Scalar, .lanes = 0};
}

}  // namespace ncnn_mlir
