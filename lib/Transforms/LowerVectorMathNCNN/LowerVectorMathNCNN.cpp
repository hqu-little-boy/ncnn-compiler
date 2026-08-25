#include "ncnn-mlir/Transforms/LowerVectorMathNCNN/LowerVectorMathNCNN.hpp"

#include <string>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_LOWERVECTORMATHNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// 向量化数学库调用下降。上游 convert-math-to-libm 对向量 math op 逐 lane
// 拆成标量 libcall（vector<227xf32> exp → 227 个内联调用），行级向量化产生
// 的整行向量因此退化为伪 SIMD。本 pass 在 SCFToControlFlow 之前把与目标
// ABI 等宽的 f32 向量 math op 整体替换为向量数学库调用：
// - backend=libmvec：glibc libmvec，符号 _ZGV<isa>N<lanes><'v'*arity>_<name>
//   （二元 powf 为 _ZGVdN8vv_powf；tanhf/erff 需 GLIBC_2.35+）；
// - backend=sleef：vendored SLEEF 静态库 dispatch 入口 Sleef_<name><tag>。
// 宽度为 lanes 整数倍且不超过 unrollFactor 倍的整行展开为 slice/call/insert
// 链；其余形态（超限整行、奇数尾、f64/bf16、scalable、无向量变体的 erfc）
// 保持不动，交由紧随其后的标量 MathToLibm 兜底。
class LowerVectorMathNCNNPass final
  : public impl::LowerVectorMathNCNNPassBase<LowerVectorMathNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (backend.getValue() == "none") {
      return;
    }

    SmallVector<Operation*> candidates;
    module.walk([&](Operation* operation) {
      if (!isCandidate(operation)) {
        return;
      }
      candidates.push_back(operation);
    });
    if (candidates.empty()) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    for (Operation* operation : candidates) {
      rewriteOne(rewriter, module, *operation);
    }
  }

 private:
  static constexpr unsigned kMaxUnrollFactor = 4;

  struct LibrarySymbol {
    llvm::StringRef basename;
    unsigned arity;
  };

  // op → (库名基段, 参数个数)。erfcf 无任何后端的向量变体，刻意缺席。
  static std::optional<LibrarySymbol> librarySymbol(Operation& operation) {
    if (isa<math::ExpOp>(operation)) {
      return LibrarySymbol{.basename = "expf", .arity = 1};
    }
    if (isa<math::LogOp>(operation)) {
      return LibrarySymbol{.basename = "logf", .arity = 1};
    }
    if (isa<math::TanhOp>(operation)) {
      return LibrarySymbol{.basename = "tanhf", .arity = 1};
    }
    if (isa<math::ErfOp>(operation)) {
      return LibrarySymbol{.basename = "erff", .arity = 1};
    }
    if (isa<math::PowFOp>(operation)) {
      return LibrarySymbol{.basename = "powf", .arity = 2};
    }
    return std::nullopt;
  }

  bool isCandidate(Operation* operation) const {
    const std::optional<LibrarySymbol> symbol = librarySymbol(*operation);
    if (!symbol) {
      return false;
    }
    const Value result = operation->getResult(0);
    const auto vectorType = dyn_cast<VectorType>(result.getType());
    if (!vectorType || vectorType.isScalable() ||
        !vectorType.getElementType().isF32() || vectorType.getRank() != 1) {
      return false;
    }
    return isSplittableWidth(vectorType.getShape()[0]) ||
           std::cmp_equal(vectorType.getShape()[0], lanes.getValue());
  }

  bool isSplittableWidth(int64_t width) const {
    const auto laneCount = static_cast<int64_t>(lanes.getValue());
    return laneCount > 0 && width > laneCount && width % laneCount == 0 &&
           width <= laneCount * kMaxUnrollFactor;
  }

  void rewriteOne(IRRewriter& rewriter, ModuleOp module, Operation& operation) {
    auto resultType = cast<VectorType>(operation.getResult(0).getType());
    const int64_t width = resultType.getShape()[0];
    const auto laneCount = static_cast<int64_t>(lanes.getValue());
    if (width != laneCount && !isSplittableWidth(width)) {
      return;
    }

    const std::optional<LibrarySymbol> symbol = librarySymbol(operation);
    const std::string symbolName = mangle(symbol->basename, symbol->arity);
    // 声明签名用块宽度（等宽情形即结果宽度）：展开链的每片调用都走它。
    auto chunkType = VectorType::get({width == laneCount ? width : laneCount},
                                     resultType.getElementType());
    func::FuncOp declaration =
      ensureDeclaration(rewriter, module, symbolName, chunkType, symbol->arity);
    if (!declaration) {
      signalPassFailure();
      return;
    }

    rewriter.setInsertionPoint(&operation);
    if (width == laneCount) {
      replaceWithCall(rewriter, operation, declaration);
      return;
    }
    replaceWithUnrolledCalls(rewriter, operation, declaration, laneCount);
  }

  // libmvec：abi 片段（如 "_ZGVdN8"）+ 'v'*arity + '_' + 基名；
  // sleef："Sleef_" + 基名 + abi 标签（如 "U10withdispatch"）。
  std::string mangle(llvm::StringRef basename, unsigned arity) const {
    const std::string& fragment = abiFragment.getValue();
    if (backend.getValue() == "libmvec") {
      return fragment + std::string(arity, 'v') + "_" + basename.str();
    }
    return "Sleef_" + basename.str() + fragment;
  }

  func::FuncOp ensureDeclaration(IRRewriter& rewriter,
                                 ModuleOp module,
                                 const std::string& symbolName,
                                 VectorType vectorType,
                                 unsigned arity) const {
    if (auto existing = module.lookupSymbol<func::FuncOp>(symbolName)) {
      return existing;
    }
    rewriter.setInsertionPointToStart(module.getBody());
    auto functionType = rewriter.getFunctionType(
      SmallVector<Type>(arity, vectorType), {vectorType});
    auto declaration =
      rewriter.create<func::FuncOp>(module.getLoc(), symbolName, functionType);
    declaration.setPrivate();
    // 纯函数标注让后续 canonicalizer/CSE 可以围绕调用做提升与去重。
    declaration->setAttr("llvm.readnone", UnitAttr::get(module.getContext()));
    return declaration;
  }

  void replaceWithCall(IRRewriter& rewriter,
                       Operation& operation,
                       func::FuncOp declaration) const {
    Location location = operation.getLoc();
    auto call = rewriter.create<func::CallOp>(
      location, declaration, operation.getOperands());
    operation.getResult(0).replaceAllUsesWith(call.getResult(0));
    rewriter.eraseOp(&operation);
  }

  void replaceWithUnrolledCalls(IRRewriter& rewriter,
                                Operation& operation,
                                func::FuncOp declaration,
                                int64_t laneCount) const {
    Location location = operation.getLoc();
    auto fullType = cast<VectorType>(operation.getResult(0).getType());
    const int64_t width = fullType.getShape()[0];

    // 提取：每个操作数切成 lanes 宽的静态块。
    SmallVector<SmallVector<Value>> operandChunks(operation.getNumOperands());
    for (unsigned index : llvm::seq<unsigned>(operation.getNumOperands())) {
      for (int64_t offset = 0; offset < width; offset += laneCount) {
        operandChunks[index].push_back(
          rewriter.create<vector::ExtractStridedSliceOp>(
            location,
            operation.getOperand(index),
            ArrayRef<int64_t>{offset},
            ArrayRef<int64_t>{laneCount},
            ArrayRef<int64_t>{1}));
      }
    }

    // 调用：按块下标对齐各操作数。
    SmallVector<Value> results;
    for (size_t chunkIndex = 0; chunkIndex < operandChunks[0].size();
         ++chunkIndex) {
      SmallVector<Value> arguments;
      for (SmallVector<Value>& chunks : operandChunks) {
        arguments.push_back(chunks[chunkIndex]);
      }
      results.push_back(
        rewriter.create<func::CallOp>(location, declaration, arguments)
          .getResult(0));
    }

    // 拼装：零初始化后逐块整体覆盖。
    Value assembled = rewriter.create<arith::ConstantOp>(
      location, fullType, rewriter.getZeroAttr(fullType));
    for (int64_t offset = 0, chunkIndex = 0; offset < width;
         offset += laneCount, ++chunkIndex) {
      assembled =
        rewriter.create<vector::InsertStridedSliceOp>(location,
                                                      results[chunkIndex],
                                                      assembled,
                                                      ArrayRef<int64_t>{offset},
                                                      ArrayRef<int64_t>{1});
    }
    operation.getResult(0).replaceAllUsesWith(assembled);
    rewriter.eraseOp(&operation);
  }
};

}  // namespace

}  // namespace mlir::ncnn
