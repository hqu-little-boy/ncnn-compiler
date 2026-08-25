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

// A1b SIMD matmul 内核。A1 策略层把卷积改写为 im2col+matmul 后，内层
// linalg.matmul 由通用下降展开为标量循环，PP 系大图模型单次推理劣化数
// 十倍（3×640×640 DBNet：直接卷积 ~20s vs 改写后 >600s）。本 pass 在
// bufferize 之后、结果出参之前，把 scf.forall 区域内的静态 memref
// matmul 改写为显式向量内核：
//
//   for m in 0..M:
//     acc = transfer_read C[m, 0:N]          // 进寄存器，贯穿 K
//     for k in 0..K:                          // 收缩维标量循环
//       acc += broadcast(A[m,k]) * transfer_read B[k, 0:N]
//     transfer_write C[m, 0:N]
//
// 累加器以 scf.for 迭代变量驻留寄存器，K 循环内零存储往返；B 行在 m 外
// 层轮换间保持 L1 热复用；N 维由 LLVM 合法化拆到目标 ABI 宽度。forall
// 外层并行语义不变——内核只写本迭代私有的输出切片，outs 初值经首次读
// 参与求和，与原 matmul 累加语义一致。
class MatmulKernelNCNNPass final
  : public impl::MatmulKernelNCNNPassBase<MatmulKernelNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> candidates;
    module.walk([&](linalg::MatmulOp matmul) {
      if (matmul->getParentOfType<scf::ForallOp>() == nullptr) {
        return;
      }
      if (!isKernelizable(matmul)) {
        return;
      }
      candidates.push_back(matmul);
    });
    if (candidates.empty()) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    for (linalg::MatmulOp matmul : candidates) {
      rewriteOne(rewriter, matmul);
    }
  }

 private:
  static constexpr int64_t kMaxColumns = 64;

  static bool isStaticF32Row(MemRefType type) {
    return type.hasStaticShape() && type.getRank() == 2 &&
           isa<FloatType>(type.getElementType());
  }

  static bool isKernelizable(linalg::MatmulOp matmul) {
    if (!matmul.hasPureBufferSemantics() || matmul.getInputs().size() != 2 ||
        matmul.getOutputs().size() != 1) {
      return false;
    }
    // 列上限只约束 N（rhs 与 acc 的第二维）；lhs 的第二维是收缩维 K，
    // 由标量循环遍历，不设上限。
    for (Value operand : {matmul.getInputs()[1], matmul.getOutputs().front()}) {
      const auto type = dyn_cast<MemRefType>(operand.getType());
      if (!isStaticF32Row(type) || type.getShape()[1] > kMaxColumns) {
        return false;
      }
    }
    for (Value operand : matmul->getOperands()) {
      if (!isStaticF32Row(dyn_cast<MemRefType>(operand.getType()))) {
        return false;
      }
    }
    return true;
  }

  void rewriteOne(IRRewriter& rewriter, linalg::MatmulOp matmul) const {
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

    auto vectorType = VectorType::get({columns}, elementType);
    auto zero = builder.create<arith::ConstantIndexOp>(0);
    auto one = builder.create<arith::ConstantIndexOp>(1);
    auto depthBound = builder.create<arith::ConstantIndexOp>(depth);
    auto rowBound = builder.create<arith::ConstantIndexOp>(rows);

    // M 外层循环：每行先把 C 读进寄存器累加器，K 内层循环纯 FMA 零存储
    // 往返，行末一次写回。
    auto mLoop = builder.create<scf::ForOp>(zero, rowBound, one);
    builder.setInsertionPointToStart(mLoop.getBody());
    Value rowIndex = mLoop.getInductionVar();
    auto accumulatorInit = builder.create<vector::TransferReadOp>(
      vectorType, acc, ValueRange{rowIndex, zero}, std::nullopt);
    auto kLoop = builder.create<scf::ForOp>(
      zero, depthBound, one, ValueRange{accumulatorInit.getResult()});
    builder.setInsertionPointToStart(kLoop.getBody());
    Value kIndex = kLoop.getInductionVar();
    Value accumulator = kLoop.getRegionIterArgs()[0];

    // A 按收缩维逐元素标量读取（列主序访问，向量化无收益）；broadcast
    // 后与 B 行向量相乘累加。
    auto aScalar =
      builder.create<memref::LoadOp>(lhs, ValueRange{rowIndex, kIndex});
    auto broadcast = builder.create<vector::BroadcastOp>(vectorType, aScalar);
    auto bRow = builder.create<vector::TransferReadOp>(
      vectorType, rhs, ValueRange{kIndex, zero}, std::nullopt);
    auto product = builder.create<arith::MulFOp>(broadcast, bRow);
    auto updated = builder.create<arith::AddFOp>(accumulator, product);
    builder.create<scf::YieldOp>(ValueRange{updated});

    builder.setInsertionPointAfter(kLoop);
    builder.create<vector::TransferWriteOp>(
      kLoop.getResult(0), acc, ValueRange{rowIndex, zero}, std::nullopt);

    rewriter.eraseOp(matmul);
  }
};

}  // namespace

}  // namespace mlir::ncnn
