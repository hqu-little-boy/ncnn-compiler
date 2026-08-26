#include "ncnn-mlir/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::ncnn {

#define GEN_PASS_DEF_MATMULKERNELNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// A1b SIMD matmul 内核 + forall 路径标量热点清理。
//
// 1) matmul 内核：A1 策略层把卷积改写为 im2col+matmul 后，内层
//    linalg.matmul 由通用下降展开为标量循环，PP 系大图模型单次推理劣化
//    数十倍（3×640×640 DBNet：直接卷积 ~20s vs 改写后 >600s）。把
//    scf.forall 区域内的静态 memref matmul 改写为显式向量内核——M 外层
//    每行将 C 读入寄存器累加器，K 内层零存储往返做 broadcast(A[m,k]) ·
//    B[k] 行累加，行末一次写回；行宽仅影响 LLVM 合法化拆分。
// 2) 恒等自拷贝循环消除：融合流水线在无激活时留下「load X 后 store 回
//    X」的纯浪费嵌套，直接删除。
// 3) 行级 generic 向量化：静态、全恒等映射、纯 arith/math body 的
//    memref generic（典型为 relu/maximumf epilogue）改写为整行
//    transfer_read/write + 行向量化 body。
// 4) im2col gather 向量化：窗口映射 (0, sh*d0+dh*d2, sw*d1+dw*d3, d4)
//    的纯转发拷贝改写为四层循环 + 整 IC 行 transfer_read/write，消除
//    通用下降的逐元素 div/mod 标量循环。
class MatmulKernelNCNNPass final
  : public impl::MatmulKernelNCNNPassBase<MatmulKernelNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<linalg::GenericOp> rowGenerics;
    SmallVector<scf::ForOp> selfCopyLoops;
    module.walk([&](Operation* operation) {
      const bool inForall =
        operation->getParentOfType<scf::ForallOp>() != nullptr;
      if (auto matmul = dyn_cast<linalg::MatmulOp>(operation)) {
        // matmul 内核仅针对 forall 分块形态；未切分的保持通用下降。
        if (inForall && isKernelizable(matmul)) {
          matmuls.push_back(matmul);
        }
        return;
      }
      if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
        // gather 改写暂时禁用：实测在 PP-det 上引入回归，待重设计。
        if (isRowVectorizable(generic)) {
          rowGenerics.push_back(generic);
        }
        return;
      }
      if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        if (isPureSelfCopyLoop(loop)) {
          selfCopyLoops.push_back(loop);
        }
      }
    });
    if (matmuls.empty() && rowGenerics.empty() && selfCopyLoops.empty()) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    // 自拷贝循环先删，避免干扰后续改写的结构匹配。
    for (scf::ForOp loop : selfCopyLoops) {
      rewriter.eraseOp(loop);
    }
    for (linalg::GenericOp generic : rowGenerics) {
      vectorizeRowGeneric(rewriter, generic);
    }
    for (linalg::MatmulOp matmul : matmuls) {
      kernelize(rewriter, matmul);
    }
  }

 private:
  // 行级向量宽度上限：relu 等逐元 epilogue 的最内维覆盖范围；超宽保持
  // 标量。matmul 内核行分块另有 128 上限。
  static constexpr int64_t kMaxRowWidth = 1024;

  static SmallVector<Value> bufferOperands(linalg::LinalgOp linalgOp) {
    SmallVector<Value> operands;
    llvm::append_range(operands, linalgOp.getDpsInputs());
    llvm::append_range(operands, linalgOp.getDpsInits());
    return operands;
  }

  static bool isStaticF32Matrix(MemRefType type) {
    return type.hasStaticShape() && type.getRank() == 2 &&
           isa<FloatType>(type.getElementType());
  }

  static bool isKernelizable(linalg::MatmulOp matmul) {
    if (!matmul.hasPureBufferSemantics() || matmul.getInputs().size() != 2 ||
        matmul.getOutputs().size() != 1) {
      return false;
    }
    for (Value operand : {matmul.getInputs()[1], matmul.getOutputs().front()}) {
      const auto type = dyn_cast<MemRefType>(operand.getType());
      if (!isStaticF32Matrix(type) || type.getShape()[1] > kMaxRowWidth) {
        return false;
      }
    }
    for (Value operand : matmul->getOperands()) {
      if (!isStaticF32Matrix(dyn_cast<MemRefType>(operand.getType()))) {
        return false;
      }
    }
    return true;
  }

  // 静态、全恒等映射、单输出、纯 arith/math body 的 memref generic 可行
  // 向量化；最内维宽度 1 或超宽的跳过。
  static bool isRowVectorizable(linalg::GenericOp generic) {
    if (!generic.hasPureBufferSemantics() || generic.getNumDpsInits() != 1) {
      return false;
    }
    Value out = generic.getOutputs().front();
    const auto outType = dyn_cast<MemRefType>(out.getType());
    if (!outType || !outType.hasStaticShape() || outType.getRank() == 0 ||
        !isa<FloatType>(outType.getElementType())) {
      return false;
    }
    const int64_t width = outType.getShape().back();
    if (width < 2 || width > kMaxRowWidth) {
      return false;
    }
    const unsigned loops = generic.getNumLoops();
    for (AffineMap map : generic.getIndexingMapsArray()) {
      if (!map.isIdentity() || map.getNumDims() != loops ||
          map.getNumResults() != outType.getRank()) {
        return false;
      }
    }
    for (Value operand : bufferOperands(generic)) {
      const auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.hasStaticShape() ||
          type.getElementType() != outType.getElementType() ||
          type.getShape() != outType.getShape()) {
        return false;
      }
    }
    Block& block = generic.getRegion().front();
    if (!dyn_cast<linalg::YieldOp>(block.getTerminator())) {
      return false;
    }
    for (Operation& statement : block.without_terminator()) {
      if (!isa<arith::ConstantOp>(statement) &&
          statement.getName().getDialect()->getNamespace() != "arith" &&
          statement.getName().getDialect()->getNamespace() != "math") {
        return false;
      }
    }
    return true;
  }

  void vectorizeRowGeneric(IRRewriter& rewriter,
                           linalg::GenericOp generic) const {
    Value out = generic.getOutputs().front();
    const auto outType = cast<MemRefType>(out.getType());
    const int64_t rank = outType.getRank();
    const ArrayRef<int64_t> shape = outType.getShape();
    Type elementType = outType.getElementType();

    ImplicitLocOpBuilder builder(generic.getLoc(), rewriter);
    builder.setInsertionPoint(generic);

    auto vectorType = VectorType::get({shape.back()}, elementType);
    auto zero = builder.create<arith::ConstantIndexOp>(0);
    auto one = builder.create<arith::ConstantIndexOp>(1);

    // 外层循环链（leading dims），逐层下探到最内行。
    SmallVector<Value> indices(rank, zero);
    for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
      auto bound = builder.create<arith::ConstantIndexOp>(shape[dimension]);
      auto loop = builder.create<scf::ForOp>(zero, bound.getResult(), one);
      indices[dimension] = loop.getInductionVar();
      builder.setInsertionPointToStart(loop.getBody());
    }

    IRMapping mapping;
    unsigned inputIndex = 0;
    Block& block = generic.getRegion().front();
    for (Value input : generic.getDpsInputs()) {
      mapping.map(block.getArgument(inputIndex),
                  builder.create<vector::TransferReadOp>(
                    vectorType, input, indices, std::nullopt));
      ++inputIndex;
    }
    mapping.map(block.getArgument(generic.getNumDpsInputs()),
                builder.create<vector::TransferReadOp>(
                  vectorType, out, indices, std::nullopt));

    for (Operation& statement : block.without_terminator()) {
      if (auto constant = dyn_cast<arith::ConstantOp>(&statement)) {
        mapping.map(
          constant.getResult(),
          builder.create<arith::ConstantOp>(vectorType, constant.getValue()));
        continue;
      }
      SmallVector<Value> operands;
      operands.reserve(statement.getNumOperands());
      for (Value operand : statement.getOperands()) {
        if (mapping.contains(operand)) {
          operands.push_back(mapping.lookup(operand));
        } else {
          auto splat = builder.create<vector::SplatOp>(vectorType, operand);
          mapping.map(operand, splat);
          operands.push_back(splat);
        }
      }
      OperationState state(statement.getLoc(), statement.getName());
      state.addOperands(operands);
      state.addTypes(SmallVector<Type>(statement.getNumResults(), vectorType));
      state.addAttributes(SmallVector<NamedAttribute>(
        statement.getAttrs().begin(), statement.getAttrs().end()));
      Operation* lifted = builder.create(state);
      for (auto [oldResult, newResult] :
           llvm::zip(statement.getResults(), lifted->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }

    Value yielded = block.getTerminator()->getOperand(0);
    builder.create<vector::TransferWriteOp>(
      mapping.lookup(yielded), out, indices, std::nullopt);

    rewriter.eraseOp(generic);
  }

  // 恒等自拷贝循环：body 仅含「load X[i..] 后 store 回 X 同索引」，或唯
  // 一内层循环递归满足同一性质。
  static bool isPureSelfCopyLoop(scf::ForOp loop) {
    Block& body = loop.getRegion().front();
    SmallVector<Operation*> statements;
    for (Operation& statement : body.without_terminator()) {
      statements.push_back(&statement);
    }
    if (statements.size() == 1) {
      if (auto inner = dyn_cast<scf::ForOp>(statements.front())) {
        return isPureSelfCopyLoop(inner);
      }
      return false;
    }
    if (statements.size() == 2) {
      auto store = dyn_cast<memref::StoreOp>(statements[1]);
      if (!store) {
        return false;
      }
      auto load = store.getValueToStore().getDefiningOp<memref::LoadOp>();
      return load && load.getMemRef() == store.getMemRef() &&
             load.getIndices() == store.getIndices();
    }
    return false;
  }

  void kernelize(IRRewriter& rewriter, linalg::MatmulOp matmul) const {
    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    Value acc = matmul.getOutputs().front();
    const auto lhsType = cast<MemRefType>(lhs.getType());
    const auto rhsType = cast<MemRefType>(rhs.getType());
    const auto accType = cast<MemRefType>(acc.getType());
    const int64_t rows = lhsType.getShape()[0];
    const int64_t depth = lhsType.getShape()[1];
    const int64_t columns = rhsType.getShape()[1];
    Type elementType = accType.getElementType();

    ImplicitLocOpBuilder builder(matmul.getLoc(), rewriter);
    builder.setInsertionPoint(matmul);

    // 行分块：向量类型宽度封顶（超宽向量会引发 LLVM 合法化期的编译爆
    // 炸），宽行拆成多个 n 块逐块计算。
    const int64_t blockWidth = std::min<int64_t>(columns, 128);
    const bool splitColumns = columns > blockWidth;

    auto vectorType = VectorType::get({blockWidth}, elementType);
    auto zero = builder.create<arith::ConstantIndexOp>(0);
    auto one = builder.create<arith::ConstantIndexOp>(1);
    auto depthBound = builder.create<arith::ConstantIndexOp>(depth);
    auto rowBound = builder.create<arith::ConstantIndexOp>(rows);

    // M 外层（可选 N 分块）：每块先把 C 读进寄存器累加器，K 内层纯 FMA
    // 零存储往返，块末一次写回。A 按收缩维逐元素标量读取（列主序访问，
    // 向量化无收益），broadcast 后与 B 子行相乘累加。
    auto mLoop = builder.create<scf::ForOp>(zero, rowBound, one);
    builder.setInsertionPointToStart(mLoop.getBody());
    Value rowIndex = mLoop.getInductionVar();
    Value columnStart = zero;

    std::optional<scf::ForOp> nLoop;
    if (splitColumns) {
      auto columnBound = builder.create<arith::ConstantIndexOp>(columns);
      auto blockStep = builder.create<arith::ConstantIndexOp>(blockWidth);
      nLoop = builder.create<scf::ForOp>(
        zero, columnBound.getResult(), blockStep.getResult());
      columnStart = nLoop->getInductionVar();
      builder.setInsertionPointToStart(nLoop->getBody());
    }

    auto accumulatorInit = builder.create<vector::TransferReadOp>(
      vectorType, acc, ValueRange{rowIndex, columnStart}, std::nullopt);
    auto kLoop = builder.create<scf::ForOp>(
      zero, depthBound, one, ValueRange{accumulatorInit.getResult()});
    builder.setInsertionPointToStart(kLoop.getBody());
    Value kIndex = kLoop.getInductionVar();
    Value accumulator = kLoop.getRegionIterArgs()[0];

    auto aScalar =
      builder.create<memref::LoadOp>(lhs, ValueRange{rowIndex, kIndex});
    auto broadcast = builder.create<vector::BroadcastOp>(vectorType, aScalar);
    auto bRow = builder.create<vector::TransferReadOp>(
      vectorType, rhs, ValueRange{kIndex, columnStart}, std::nullopt);
    auto product = builder.create<arith::MulFOp>(broadcast, bRow);
    auto updated = builder.create<arith::AddFOp>(accumulator, product);
    builder.create<scf::YieldOp>(ValueRange{updated});

    // 写回仍在 n 块循环体内（kLoop 之后）：每块行段独立累加并落回。
    builder.setInsertionPointAfter(kLoop);
    builder.create<vector::TransferWriteOp>(
      kLoop.getResult(0), acc, ValueRange{rowIndex, columnStart}, std::nullopt);

    rewriter.eraseOp(matmul);
  }
};

}  // namespace

}  // namespace mlir::ncnn
