#include "ncnn-mlir/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
//    按 mTile 行一组将 C 行段读入寄存器 accumulator（M×N 寄存器分块），
//    K 内层零存储往返：每轮只读一次 B[k] 行，与各行 A[m+i,k] 标量的
//    broadcast 做 vector.fma（单舍入）独立累加，块末一次写回。B 行复用
//    mTile 次且 K 循环持有 mTile 条相互独立的 FMA 链——单链内核的发射
//    率被 FMA 延迟钉死，多链才能喂满端口；寄存器预算见
//    kAccumulatorFloatBudget。
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
    SmallVector<linalg::MatmulTransposeBOp> int8Matmuls;
    SmallVector<linalg::GenericOp> rowGenerics;
    SmallVector<linalg::GenericOp> gathers;
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
      if (auto transposed = dyn_cast<linalg::MatmulTransposeBOp>(operation)) {
        // int8 量化路径（P4）：strategy 以 matmul_transpose_b 呈现 B 面转
        // 置物化（[N,K] 存放，k 连续），本 pass 发射 row-dot 向量内核。
        if (inForall && isKernelizableInt8(transposed)) {
          int8Matmuls.push_back(transposed);
        }
        return;
      }
      if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
        if (isIm2colGatherCopy(generic)) {
          gathers.push_back(generic);
        } else if (isRowVectorizable(generic)) {
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
    if (matmuls.empty() && int8Matmuls.empty() && rowGenerics.empty() &&
        selfCopyLoops.empty() && gathers.empty()) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    // 自拷贝循环先删，避免干扰后续改写的结构匹配。
    for (scf::ForOp loop : selfCopyLoops) {
      rewriter.eraseOp(loop);
    }
    for (linalg::GenericOp generic : gathers) {
      gatherToParallelVectorCopy(rewriter, generic);
    }
    for (linalg::GenericOp generic : rowGenerics) {
      vectorizeRowGeneric(rewriter, generic);
    }
    for (linalg::MatmulOp matmul : matmuls) {
      kernelize(rewriter, matmul);
    }
    for (linalg::MatmulTransposeBOp transposed : int8Matmuls) {
      kernelizeInt8RowDot(rewriter, transposed);
    }
  }

 private:
  // 行级向量宽度上限：relu 等逐元 epilogue 的最内维覆盖范围；超宽保持
  // 标量。matmul 内核的向量 accumulator 宽度由 accColumns（默认 16）与
  // 其余数列块给出，天然有界。
  static constexpr int64_t kMaxRowWidth = 1024;

  // matmul 内核 M×N 寄存器分块的 accumulator 浮点预算：tileRows ×
  // accColumns ≤ 预算时，accumulator（默认 4×16 = 8 个 ymm）加 B 行与
  // broadcast 瞬态可容纳于 16 个 ymm 之内，避免 LLVM 寄存器溢出把分块
  // 的访存/延迟收益吃回去。
  static constexpr int64_t kAccumulatorFloatBudget = 64;

  // int8 row-dot 内核：每个 (m,n) 输出一个 v8i32 部分 和 accumulator
  // （8 个 k-对），tileRows × accColumns ≤ 预算即 8 个 ymm；k 向量宽
  // 8 lane（vpmaddwd 一条指令消费 8 个 i16 乘积）。
  static constexpr int64_t kInt8AccumulatorBudget = 8;
  static constexpr int64_t kInt8ChunkLanes = 8;

  // i8 im2col gather 的非 2 幂通道分块宽（32 字节 = 一个 ymm）。
  static constexpr int64_t kGatherChunkLanes = 32;

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

  // int8 row-dot 内核（P4）形态：A [M,K]、B [N,K]（strategy 的 B 面转置
  // 物化）、C [M,N] i32 全静态。i8×i8 乘积精确落入 i16/i32，整数加法结
  // 合律保证任何 k 归约序与标量 i32-MAC 逐位一致。
  static bool isKernelizableInt8(linalg::MatmulTransposeBOp matmul) {
    if (!matmul.hasPureBufferSemantics() || matmul.getInputs().size() != 2 ||
        matmul.getOutputs().size() != 1) {
      return false;
    }
    const auto lhsType = dyn_cast<MemRefType>(matmul.getInputs()[0].getType());
    const auto rhsType = dyn_cast<MemRefType>(matmul.getInputs()[1].getType());
    const auto accType =
      dyn_cast<MemRefType>(matmul.getOutputs().front().getType());
    if (!lhsType || !rhsType || !accType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape() || !accType.hasStaticShape()) {
      return false;
    }
    if (!lhsType.getElementType().isInteger(8) ||
        !rhsType.getElementType().isInteger(8) ||
        !accType.getElementType().isInteger(32)) {
      return false;
    }
    // N 是写回行宽；K 无上限（row-dot 内核沿 k 向量化，代码量与 K 无关）。
    if (accType.getShape()[1] > kMaxRowWidth) {
      return false;
    }
    const int64_t rows = lhsType.getShape()[0];
    const int64_t depth = lhsType.getShape()[1];
    const int64_t rhsRows = rhsType.getShape()[0];
    const int64_t rhsDepth = rhsType.getShape()[1];
    // matmul_transpose_b 语义：C[M,N] = A[M,K] · B[N,K]ᵀ，B 的行数是 N。
    return rhsDepth == depth && rhsRows == accType.getShape()[1] &&
           accType.getShape()[0] == rows;
  }

  // 静态、输出恒等映射、纯 arith/math body 的 memref generic 可行向
  // 量化；最内维宽度 1 或超宽的跳过。P4 扩展：输出元素类型不再限浮点
  // （覆盖 i8 量化/requant 尾部的混合类型 body），非输出操作数允许两
  // 类映射——最内维沿最后迭代维连续的"行"映射（requant 的恒等主值与
  // (0,0,0,d3) 常量广播），或全常量的"标量"映射（逐张量 scale）；其
  // 余形态保持标量下降。
  static bool isRowVectorizable(linalg::GenericOp generic) {
    if (!generic.hasPureBufferSemantics() || generic.getNumDpsInits() != 1) {
      return false;
    }
    Value out = generic.getOutputs().front();
    const auto outType = dyn_cast<MemRefType>(out.getType());
    if (!outType || !outType.hasStaticShape() || outType.getRank() == 0 ||
        !outType.getElementType().isIntOrFloat()) {
      return false;
    }
    const int64_t width = outType.getShape().back();
    if (width < 2 || width > kMaxRowWidth) {
      return false;
    }
    const unsigned loops = generic.getNumLoops();
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    const unsigned outOperand = maps.size() - 1;
    if (!maps[outOperand].isIdentity() ||
        maps[outOperand].getNumDims() != loops ||
        maps[outOperand].getNumResults() != outType.getRank()) {
      return false;
    }
    for (auto [index, operand] : llvm::enumerate(generic.getDpsInputs())) {
      const auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.hasStaticShape() || type.getRank() == 0 ||
          !type.getElementType().isIntOrFloat()) {
        return false;
      }
      AffineMap map = maps[index];
      if (map.getNumDims() != loops || map.getNumResults() != type.getRank()) {
        return false;
      }
      AffineExpr innermost = map.getResult(type.getRank() - 1);
      if (auto dim = dyn_cast<AffineDimExpr>(innermost)) {
        // 行模式：最内维沿最后迭代维连续；分块读的任何段都必须界内。
        if (dim.getPosition() != loops - 1 || type.getShape().back() < width) {
          return false;
        }
        continue;
      }
      if (isa<AffineConstantExpr>(innermost)) {
        // 标量模式：要求全部结果为常量（逐张量参数，循环不变）。
        bool allConstant = llvm::all_of(
          map.getResults(),
          [](AffineExpr result) { return isa<AffineConstantExpr>(result); });
        if (!allConstant) {
          return false;
        }
        continue;
      }
      return false;
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

  // 判定仿射表达式 ≡ s·strideDim + w·windowDim 形式的系数抽取（系数 1
  // 会折叠成裸维度，两种操作数顺序都接受）。
  static bool extractStrideDilation(AffineExpr expression,
                                    unsigned strideDim,
                                    unsigned windowDim,
                                    int64_t& stride,
                                    int64_t& dilation) {
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary || binary.getKind() != AffineExprKind::Add) {
      return false;
    }
    auto matchTerm = [](AffineExpr term,
                        unsigned dim,
                        bool allowOne) -> std::optional<int64_t> {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(term)) {
        if (allowOne && dimExpr.getPosition() == dim) {
          return 1;
        }
        return std::nullopt;
      }
      if (auto product = dyn_cast<AffineBinaryOpExpr>(term);
          product && product.getKind() == AffineExprKind::Mul) {
        for (auto [lhs, rhs] : {std::pair{product.getLHS(), product.getRHS()},
                                {product.getRHS(), product.getLHS()}}) {
          auto coefficient = dyn_cast<AffineConstantExpr>(lhs);
          auto dimExpr = dyn_cast<AffineDimExpr>(rhs);
          if (coefficient && dimExpr && dimExpr.getPosition() == dim) {
            return coefficient.getValue();
          }
        }
      }
      return std::nullopt;
    };
    for (auto [first, second] : {std::pair{binary.getLHS(), binary.getRHS()},
                                 {binary.getRHS(), binary.getLHS()}}) {
      auto strideMatch = matchTerm(first, strideDim, false);
      auto dilationMatch = matchTerm(second, windowDim, true);
      if (strideMatch && dilationMatch) {
        stride = *strideMatch;
        dilation = *dilationMatch;
        return true;
      }
    }
    return false;
  }

  // im2col gather 识别：5 层全并行、纯转发拷贝，ins 映射呈窗口形式
  //   (d0,d1,d2,d3,d4) -> (0, sh*d0 + dh*d2, sw*d1 + dw*d3, d4)
  // outs 恒等。
  static bool isIm2colGatherCopy(linalg::GenericOp generic) {
    if (!generic.hasPureBufferSemantics() || generic.getNumDpsInputs() != 1 ||
        generic.getNumDpsInits() != 1 || generic.getNumLoops() != 5) {
      return false;
    }
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity()) {
      return false;
    }
    Value input = generic.getDpsInputs().front();
    Value output = generic.getDpsInits().front();
    const auto inputType = dyn_cast<MemRefType>(input.getType());
    const auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || inputType.getRank() != 4 ||
        outputType.getRank() != 5 ||
        inputType.getElementType() != outputType.getElementType() ||
        !isa<FloatType>(inputType.getElementType())) {
      return false;
    }

    MLIRContext* context = generic.getContext();
    auto constantZero = dyn_cast<AffineConstantExpr>(maps[0].getResult(0));
    if (!constantZero || constantZero.getValue() != 0) {
      return false;
    }
    if (maps[0].getResult(3) !=
        getAffineDimExpr(generic.getNumLoops() - 1, context)) {
      return false;
    }
    int64_t strideHeight = 0;
    int64_t dilationHeight = 0;
    int64_t strideWidth = 0;
    int64_t dilationWidth = 0;
    if (!extractStrideDilation(
          maps[0].getResult(1), 0, 2, strideHeight, dilationHeight) ||
        !extractStrideDilation(
          maps[0].getResult(2), 1, 3, strideWidth, dilationWidth)) {
      return false;
    }
    const int64_t channels = inputType.getShape()[3];
    if (outputType.getShape()[4] != channels) {
      return false;
    }
    // 整行向量形态要求通道数是 2 的幂（≥8）——对任意元素类型成立；
    // i8 通道行只有 1 字节/元素，非 2 幂通道改走 32-lane 分块 + 标量尾
    // （P4：int8 im2col gather 的行宽普遍非 2 幂，如 IC=24/40/240）。
    if (channels >= 8 && channels <= kMaxRowWidth &&
        (channels & (channels - 1)) == 0) {
      return true;
    }
    return isa<IntegerType>(inputType.getElementType()) && channels >= 8;
  }

  // im2col gather → scf.parallel + 整 IC 行向量拷贝。并行维度沿用原
  // generic 的四个 leading 维，保留通用下降 scf.parallel 的 OpenMP 多
  // 线程；行内 IC 连续段为 SIMD 拷贝。步长/膨胀从映射重新抽取。
  void gatherToParallelVectorCopy(IRRewriter& rewriter,
                                  linalg::GenericOp generic) const {
    Value input = generic.getDpsInputs().front();
    Value output = generic.getDpsInits().front();
    const auto inputType = cast<MemRefType>(input.getType());
    SmallVector<int64_t> bounds = generic.getStaticLoopRanges();
    const int64_t channels = inputType.getShape()[3];

    AffineMap inputMap = generic.getIndexingMapsArray()[0];
    int64_t strideHeight = 1;
    int64_t dilationHeight = 1;
    int64_t strideWidth = 1;
    int64_t dilationWidth = 1;
    extractStrideDilation(
      inputMap.getResult(1), 0, 2, strideHeight, dilationHeight);
    extractStrideDilation(
      inputMap.getResult(2), 1, 3, strideWidth, dilationWidth);

    ImplicitLocOpBuilder builder(generic.getLoc(), rewriter);
    builder.setInsertionPoint(generic);

    // 拷贝计划：通道行 2 幂 ≥8 时整行一个向量；i8 非 2 幂通道按
    // kGatherChunkLanes 分块 + 标量尾。行内 IC 段连续，源/目标最内维
    // 偏移同步推进。
    const int64_t wholeRow = (channels >= 8 && channels <= kMaxRowWidth &&
                              (channels & (channels - 1)) == 0)
                               ? channels
                               : 0;
    const int64_t chunkWidth = wholeRow != 0 ? wholeRow : kGatherChunkLanes;
    const int64_t fullChunks = wholeRow != 0 ? 1 : channels / kGatherChunkLanes;
    const int64_t tailWidth = wholeRow != 0 ? 0 : channels % kGatherChunkLanes;

    auto vectorType = VectorType::get({chunkWidth}, inputType.getElementType());
    auto zeroIndex = builder.create<arith::ConstantIndexOp>(0);
    auto oneIndex = builder.create<arith::ConstantIndexOp>(1);

    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    for (int64_t dimension = 0; dimension < 4; ++dimension) {
      lowerBounds.push_back(
        builder.create<arith::ConstantIndexOp>(0).getResult());
      upperBounds.push_back(
        builder.create<arith::ConstantIndexOp>(bounds[dimension]).getResult());
      steps.push_back(builder.create<arith::ConstantIndexOp>(1).getResult());
    }
    auto parallel = builder.create<scf::ParallelOp>(
      generic.getLoc(), lowerBounds, upperBounds, steps);
    builder.setInsertionPointToStart(parallel.getBody());

    SmallVector<Value> indices(parallel.getInductionVars());

    auto heightIndex = builder.create<arith::MulIOp>(
      builder.create<arith::ConstantIndexOp>(strideHeight), indices[0]);
    auto heightOffset = builder.create<arith::MulIOp>(
      builder.create<arith::ConstantIndexOp>(dilationHeight), indices[2]);
    auto widthIndex = builder.create<arith::MulIOp>(
      builder.create<arith::ConstantIndexOp>(strideWidth), indices[1]);
    auto widthOffset = builder.create<arith::MulIOp>(
      builder.create<arith::ConstantIndexOp>(dilationWidth), indices[3]);
    auto sourceRow = builder.create<arith::AddIOp>(heightIndex, heightOffset);
    auto sourceColumn = builder.create<arith::AddIOp>(widthIndex, widthOffset);

    // 满块：源/目标最内维偏移 = 分块下标 × 分块宽。
    auto emitChunk = [&](Value innerOffset) {
      auto sourceInner =
        builder.create<arith::AddIOp>(sourceColumn, innerOffset);
      auto row = builder.create<vector::TransferReadOp>(
        vectorType,
        input,
        ValueRange{zeroIndex.getResult(),
                   sourceRow.getResult(),
                   sourceInner.getResult(),
                   zeroIndex.getResult()},
        std::nullopt);
      auto targetInner = builder.create<arith::AddIOp>(zeroIndex, innerOffset);
      builder.create<vector::TransferWriteOp>(
        row,
        output,
        ValueRange{indices[0],
                   indices[1],
                   indices[2],
                   indices[3],
                   targetInner.getResult()},
        std::nullopt);
    };

    if (wholeRow != 0) {
      emitChunk(zeroIndex);
    } else {
      if (fullChunks > 1) {
        auto chunkBound = builder.create<arith::ConstantIndexOp>(fullChunks);
        auto chunkStep =
          builder.create<arith::ConstantIndexOp>(kGatherChunkLanes);
        auto chunkLoop =
          builder.create<scf::ForOp>(zeroIndex, chunkBound, chunkStep);
        builder.setInsertionPointToStart(chunkLoop.getBody());
        emitChunk(chunkLoop.getInductionVar());
        builder.setInsertionPointAfter(chunkLoop);
      } else if (fullChunks == 1) {
        emitChunk(zeroIndex);
      }
      if (tailWidth > 0) {
        auto tailBase = builder.create<arith::ConstantIndexOp>(
          fullChunks * kGatherChunkLanes);
        auto tailBound = builder.create<arith::ConstantIndexOp>(tailWidth);
        auto tailLoop =
          builder.create<scf::ForOp>(zeroIndex, tailBound, oneIndex);
        builder.setInsertionPointToStart(tailLoop.getBody());
        Value inner =
          builder.create<arith::AddIOp>(tailBase, tailLoop.getInductionVar());
        auto byte = builder.create<memref::LoadOp>(
          input,
          ValueRange{
            zeroIndex.getResult(),
            sourceRow.getResult(),
            builder.create<arith::AddIOp>(sourceColumn, inner).getResult(),
            zeroIndex.getResult()});
        builder.create<memref::StoreOp>(
          byte,
          output,
          ValueRange{indices[0], indices[1], indices[2], indices[3], inner});
      }
    }

    rewriter.eraseOp(generic);
  }

  void vectorizeRowGeneric(IRRewriter& rewriter,
                           linalg::GenericOp generic) const {
    Value out = generic.getOutputs().front();
    const auto outType = cast<MemRefType>(out.getType());
    const int64_t rank = outType.getRank();
    const ArrayRef<int64_t> shape = outType.getShape();
    Type elementType = outType.getElementType();
    const int64_t rowWidth = shape.back();
    // 行宽分块（同 VectorizeNCNN 的行级向量化）：整行向量在末维极宽的
    // 模型上把 math op 推出 libmvec 等宽 ABI（被标量 convert-math-to-
    // libm 逐 lane 拆成 extract/call/insert 风暴），并把 LLVM 合法化期
    // 推向小时级——vector-mode=off 的应急路径同样必须安全。分块预算
    // ≤32 位元素取 4×rowChunkLanes、64 位取 rowChunkLanes；行宽不超过
    // 预算时保持整行形态；余数以标量 load/store 兜底。混合类型 body
    // （P4 requant 尾部）逐操作数按各自元素类型读入。
    const auto laneBudget = static_cast<int64_t>(rowChunkLanes);
    const int64_t chunkBudget =
      elementType.getIntOrFloatBitWidth() <= 32 ? 4 * laneBudget : laneBudget;
    const int64_t chunkWidth =
      (laneBudget > 1 && rowWidth > chunkBudget) ? chunkBudget : rowWidth;
    const int64_t fullChunks = rowWidth / chunkWidth;
    const int64_t tailWidth = rowWidth % chunkWidth;

    ImplicitLocOpBuilder builder(generic.getLoc(), rewriter);
    builder.setInsertionPoint(generic);

    auto vectorTypeOf = [&](Type type) {
      return VectorType::get({chunkWidth}, type);
    };
    auto outVectorType = vectorTypeOf(elementType);

    // 输入分类（isRowVectorizable 已保证只有两类）：行模式逐分块按映射
    // 索引读；标量模式循环不变，循环前一次性 load，块内 splat。
    SmallVector<bool> inputRowMode;
    SmallVector<Value> hoistedScalars;
    for (auto [index, input] : llvm::enumerate(generic.getDpsInputs())) {
      const auto inputType = cast<MemRefType>(input.getType());
      AffineMap map = generic.getIndexingMapsArray()[index];
      AffineExpr innermost = map.getResult(inputType.getRank() - 1);
      if (isa<AffineDimExpr>(innermost)) {
        inputRowMode.push_back(true);
        hoistedScalars.push_back(Value());
        continue;
      }
      inputRowMode.push_back(false);
      SmallVector<Value> constantIndices;
      for (AffineExpr result : map.getResults()) {
        constantIndices.push_back(builder.create<arith::ConstantIndexOp>(
          cast<AffineConstantExpr>(result).getValue()));
      }
      hoistedScalars.push_back(
        builder.create<memref::LoadOp>(input, constantIndices));
    }

    // leading dims 发射为 scf.parallel：保留原 generic 经
    // ConvertLinalgToParallelLoops 的 OpenMP 多线程语义（串行 scf.for
    // 链会在大图 epilogue 上丢失全部并行度）。
    SmallVector<Value> parallelIndices(rank - 1);
    if (rank > 1) {
      SmallVector<Value> lowerBounds;
      SmallVector<Value> upperBounds;
      SmallVector<Value> steps;
      for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
        lowerBounds.push_back(
          builder.create<arith::ConstantIndexOp>(0).getResult());
        upperBounds.push_back(
          builder.create<arith::ConstantIndexOp>(shape[dimension]).getResult());
        steps.push_back(builder.create<arith::ConstantIndexOp>(1).getResult());
      }
      auto parallel = builder.create<scf::ParallelOp>(
        generic.getLoc(), lowerBounds, upperBounds, steps);
      llvm::copy(parallel.getInductionVars(), parallelIndices.begin());
      builder.setInsertionPointToStart(parallel.getBody());
    }

    Value zeroIndex = builder.create<arith::ConstantIndexOp>(0);
    Value stepOneIndex = builder.create<arith::ConstantIndexOp>(1);

    // 输出行模式的读写索引 = 该输入映射作用在 [并行索引..., 最内维偏移]
    // 上的仿射求值。
    auto mappedIndices = [&](AffineMap map, Value lastDimOffset) {
      SmallVector<Value> operands(parallelIndices);
      operands.push_back(lastDimOffset);
      SmallVector<Value> indices;
      indices.reserve(map.getNumResults());
      for (AffineExpr result : map.getResults()) {
        auto resultMap =
          AffineMap::get(map.getNumDims(), 0, result, generic.getContext());
        indices.push_back(builder
                            .create<affine::AffineApplyOp>(
                              generic.getLoc(), resultMap, operands)
                            .getResult());
      }
      return indices;
    };

    // 提升一个分块：读入 chunkWidth 宽行段、向量化 body、写回。body
    // 引用的外部标量值提升为分块宽 splat，并缓存复用。
    auto emitChunk = [&](Value lastDimOffset) {
      SmallVector<Value> outIndices(parallelIndices);
      outIndices.push_back(lastDimOffset);
      IRMapping mapping;
      unsigned inputIndex = 0;
      Block& block = generic.getRegion().front();
      for (auto [index, input] : llvm::enumerate(generic.getDpsInputs())) {
        Value readValue;
        if (inputRowMode[index]) {
          const auto inputType = cast<MemRefType>(input.getType());
          readValue = builder.create<vector::TransferReadOp>(
            vectorTypeOf(inputType.getElementType()),
            input,
            mappedIndices(generic.getIndexingMapsArray()[index], lastDimOffset),
            std::nullopt,
            SmallVector<bool>(1, true));
        } else {
          readValue = builder.create<vector::SplatOp>(
            vectorTypeOf(cast<MemRefType>(input.getType()).getElementType()),
            hoistedScalars[index]);
        }
        mapping.map(block.getArgument(index), readValue);
        ++inputIndex;
      }
      (void)inputIndex;
      mapping.map(
        block.getArgument(generic.getNumDpsInputs()),
        builder.create<vector::TransferReadOp>(outVectorType,
                                               out,
                                               outIndices,
                                               std::nullopt,
                                               SmallVector<bool>(1, true)));

      for (Operation& statement : block.without_terminator()) {
        if (auto constant = dyn_cast<arith::ConstantOp>(&statement)) {
          mapping.map(constant.getResult(),
                      builder.create<arith::ConstantOp>(
                        vectorTypeOf(constant.getType()),
                        DenseElementsAttr::get(vectorTypeOf(constant.getType()),
                                               constant.getValue())));
          continue;
        }
        SmallVector<Value> operands;
        operands.reserve(statement.getNumOperands());
        for (Value operand : statement.getOperands()) {
          if (mapping.contains(operand)) {
            operands.push_back(mapping.lookup(operand));
          } else {
            auto splatType = vectorTypeOf(
              isa<ShapedType>(operand.getType())
                ? cast<ShapedType>(operand.getType()).getElementType()
                : operand.getType());
            auto splat = builder.create<vector::SplatOp>(splatType, operand);
            mapping.map(operand, splat);
            operands.push_back(splat);
          }
        }
        OperationState state(statement.getLoc(), statement.getName());
        state.addOperands(operands);
        for (Type resultType : statement.getResultTypes()) {
          state.addTypes(vectorTypeOf(resultType));
        }
        state.addAttributes(SmallVector<NamedAttribute>(
          statement.getAttrs().begin(), statement.getAttrs().end()));
        Operation* lifted = builder.create(state);
        for (auto [oldResult, newResult] :
             llvm::zip(statement.getResults(), lifted->getResults())) {
          mapping.map(oldResult, newResult);
        }
      }

      Value yielded = block.getTerminator()->getOperand(0);
      auto rowWrite = builder.create<vector::TransferWriteOp>(
        mapping.lookup(yielded), out, outIndices);
      rowWrite.setInBoundsAttr(
        builder.getBoolArrayAttr(SmallVector<bool>(1, true)));
    };

    // 标量兜底一个尾元素：memref.load 提升为标量 body（clone 的 mapping
    // 缺省映射让外部标量原样复用），结果直接 store 回。
    auto emitTailElement = [&](Value lastDimOffset) {
      SmallVector<Value> outIndices(parallelIndices);
      outIndices.push_back(lastDimOffset);
      IRMapping mapping;
      Block& block = generic.getRegion().front();
      for (auto [index, input] : llvm::enumerate(generic.getDpsInputs())) {
        if (inputRowMode[index]) {
          mapping.map(block.getArgument(index),
                      builder.create<memref::LoadOp>(
                        input,
                        mappedIndices(generic.getIndexingMapsArray()[index],
                                      lastDimOffset)));
        } else {
          mapping.map(block.getArgument(index), hoistedScalars[index]);
        }
      }
      mapping.map(block.getArgument(generic.getNumDpsInputs()),
                  builder.create<memref::LoadOp>(out, outIndices));
      for (Operation& statement : block.without_terminator()) {
        builder.clone(statement, mapping);
      }
      Value yielded = mapping.lookup(block.getTerminator()->getOperand(0));
      builder.create<memref::StoreOp>(yielded, out, outIndices);
    };

    if (fullChunks > 1) {
      Value fullBound = builder.create<arith::ConstantIndexOp>(fullChunks);
      Value chunkWidthIndex =
        builder.create<arith::ConstantIndexOp>(chunkWidth);
      // 无 iter_args 的 scf.for 由 build 自动保证空 yield 终结符，body
      // 内插入点从块首起，新 op 落在终结符之前。步长 1 遍历分块下标，
      // 块内偏移由下标 × 分块宽给出。
      auto chunkLoop =
        builder.create<scf::ForOp>(zeroIndex, fullBound, stepOneIndex);
      builder.setInsertionPointToStart(chunkLoop.getBody());
      Value offset = builder.create<arith::MulIOp>(chunkLoop.getInductionVar(),
                                                   chunkWidthIndex);
      emitChunk(offset);
      builder.setInsertionPointAfter(chunkLoop);
    } else {
      emitChunk(zeroIndex);
    }
    if (tailWidth > 0) {
      Value tailBase =
        builder.create<arith::ConstantIndexOp>(fullChunks * chunkWidth);
      Value tailBound = builder.create<arith::ConstantIndexOp>(tailWidth);
      auto tailLoop =
        builder.create<scf::ForOp>(zeroIndex, tailBound, stepOneIndex);
      builder.setInsertionPointToStart(tailLoop.getBody());
      Value offset =
        builder.create<arith::AddIOp>(tailBase, tailLoop.getInductionVar());
      emitTailElement(offset);
    }

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

    // M×N 寄存器分块（P3）：M 方向 tileRows 行 accumulator 驻留寄存器，
    // B 行每轮 K 只读一次、复用 tileRows 次（各行 A 标量 broadcast），
    // 并把 K 循环从单条 FMA 依赖链变为 tileRows 条独立链。列侧 accumulator
    // 向量宽 accColumns，n 块循环按其步距覆盖 N（余数列块用窄向量）；
    // 行侧满块步距 tileRows，余数行退单行单 accumulator 形态。寄存器
    // 预算约束见 kAccumulatorFloatBudget；tileRows=1 时即历史单行内核
    // （A/B 对照口径）。
    const int64_t accColumns =
      std::clamp<int64_t>(matmulAccColumns, 1, columns);
    const int64_t tileRows = std::max<int64_t>(
      1,
      std::min<int64_t>(
        {matmulMRows, rows, kAccumulatorFloatBudget / accColumns}));

    const int64_t fullColumnBlocks = columns / accColumns;
    const int64_t tailColumns = columns % accColumns;
    const int64_t fullRowExtent = rows / tileRows * tileRows;

    auto zero = builder.create<arith::ConstantIndexOp>(0);
    auto one = builder.create<arith::ConstantIndexOp>(1);

    // 一段 rowCount 行 × 一个 n 块的内核体：C 行段读入 rowCount 个向量
    // accumulator，K 循环内 B 行读一次、与各行 A 标量 broadcast 后
    // vector.fma 独立累加，块末写回。行段与列块边界都是静态推导的界内
    // 区域，transfer 全部标 inBounds——动态 n 块偏移下也下降为无掩码
    // 的 vector.load/store。
    auto emitTileBlock = [&](Value rowStart,
                             int64_t rowCount,
                             Value columnStart,
                             int64_t blockColumns) {
      auto vectorType = VectorType::get({blockColumns}, elementType);
      SmallVector<Value> rowIndices;
      rowIndices.reserve(rowCount);
      for (int64_t i = 0; i < rowCount; ++i) {
        rowIndices.push_back(
          i == 0 ? rowStart
                 : builder
                     .create<arith::AddIOp>(
                       rowStart, builder.create<arith::ConstantIndexOp>(i))
                     .getResult());
      }

      SmallVector<Value> accumulators;
      for (int64_t i = 0; i < rowCount; ++i) {
        accumulators.push_back(builder.create<vector::TransferReadOp>(
          vectorType,
          acc,
          ValueRange{rowIndices[i], columnStart},
          std::nullopt,
          SmallVector<bool>(1, true)));
      }
      auto depthBound = builder.create<arith::ConstantIndexOp>(depth);
      auto kLoop = builder.create<scf::ForOp>(
        zero, depthBound, one, ValueRange(accumulators));
      builder.setInsertionPointToStart(kLoop.getBody());
      Value kIndex = kLoop.getInductionVar();
      auto bRow =
        builder.create<vector::TransferReadOp>(vectorType,
                                               rhs,
                                               ValueRange{kIndex, columnStart},
                                               std::nullopt,
                                               SmallVector<bool>(1, true));
      // 单舍入 FMA：mul+add 分离会让 LLVM 侧因无 fastmath/contract 而无
      // 法合成 vfmadd（每个 MAC 双指令、依赖链延迟翻倍）；vector.fma 一
      // 步到位，舍入语义与 ncnn 的 FMA 内核一致，差异由数值黄金预算吸
      // 收。各 accumulator 链相互独立，K 循环体的发射率不再被单链延迟
      // 钉死。
      SmallVector<Value> updated;
      updated.reserve(rowCount);
      for (int64_t i = 0; i < rowCount; ++i) {
        auto aScalar = builder.create<memref::LoadOp>(
          lhs, ValueRange{rowIndices[i], kIndex});
        auto broadcast =
          builder.create<vector::BroadcastOp>(vectorType, aScalar);
        updated.push_back(builder.create<vector::FMAOp>(
          vectorType, broadcast, bRow, kLoop.getRegionIterArgs()[i]));
      }
      builder.create<scf::YieldOp>(updated);

      // 写回在 kLoop 之后：行段累加完毕一次落回。
      builder.setInsertionPointAfter(kLoop);
      for (int64_t i = 0; i < rowCount; ++i) {
        auto rowWrite = builder.create<vector::TransferWriteOp>(
          kLoop.getResult(i), acc, ValueRange{rowIndices[i], columnStart});
        rowWrite.setInBoundsAttr(
          builder.getBoolArrayAttr(SmallVector<bool>(1, true)));
      }
    };

    // 一个 M 行段扫完整列块再扫列尾块。
    auto emitRowBlock = [&](Value rowStart, int64_t rowCount) {
      if (fullColumnBlocks > 1) {
        auto columnBound =
          builder.create<arith::ConstantIndexOp>(fullColumnBlocks * accColumns);
        auto columnStep = builder.create<arith::ConstantIndexOp>(accColumns);
        auto columnLoop =
          builder.create<scf::ForOp>(zero, columnBound, columnStep);
        builder.setInsertionPointToStart(columnLoop.getBody());
        emitTileBlock(
          rowStart, rowCount, columnLoop.getInductionVar(), accColumns);
        builder.setInsertionPointAfter(columnLoop);
      } else {
        emitTileBlock(rowStart, rowCount, zero, accColumns);
      }
      if (tailColumns > 0) {
        auto tailStart =
          builder.create<arith::ConstantIndexOp>(fullColumnBlocks * accColumns);
        emitTileBlock(rowStart, rowCount, tailStart, tailColumns);
      }
    };

    if (fullRowExtent > 0) {
      auto rowBound = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowStep = builder.create<arith::ConstantIndexOp>(tileRows);
      auto rowLoop = builder.create<scf::ForOp>(zero, rowBound, rowStep);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), tileRows);
      builder.setInsertionPointAfter(rowLoop);
    }
    if (const int64_t remainderRows = rows % tileRows) {
      auto rowStart = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowBound = builder.create<arith::ConstantIndexOp>(rows);
      auto rowLoop = builder.create<scf::ForOp>(rowStart, rowBound, one);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), 1);
    }

    rewriter.eraseOp(matmul);
  }

  // int8 row-dot 内核（P4）：B 已按 [N,K] 物化（strategy 的常量转置）。
  // 发射形态刻意保持标量 i32-MAC——每个 (m,n) 输出一条独立累加链
  // （tileRows×accColumns 条，寄存器预算 8），K 循环每轮每行一个 A
  // 字节、每列一个 B 字节。clang -O3 的循环向量化器对该形态稳定生成
  // vpmovsxbw + vpmaddwd（ncnn AVX2 int8 内核同款 MAC；实测显式向量
  // 内核的 mul+add 树依赖中端展开+重关联的偶然配对，稳定落回
  // vpmulld——本机无 avx_vnni_int8，s8×s8 也无 vpdpbusd 下降，tier 定
  // 档见 docs/ncnn-performance-parity-plan.md §3-P4）。整数加法结合律
  // 保证任何归约序与串行 i32-MAC 逐位一致。
  void kernelizeInt8RowDot(IRRewriter& rewriter,
                           linalg::MatmulTransposeBOp matmul) const {
    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    Value acc = matmul.getOutputs().front();
    const auto lhsType = cast<MemRefType>(lhs.getType());
    const auto rhsType = cast<MemRefType>(rhs.getType());
    const auto accType = cast<MemRefType>(acc.getType());
    const int64_t rows = lhsType.getShape()[0];
    const int64_t depth = lhsType.getShape()[1];
    const int64_t columns = accType.getShape()[1];

    ImplicitLocOpBuilder builder(matmul.getLoc(), rewriter);
    builder.setInsertionPoint(matmul);

    // 寄存器预算：每个 (m,n) 一条标量链（LV 向量化后一个 ymm 累加器），
    // tileRows × accColumns ≤ kInt8AccumulatorBudget。
    const int64_t accColumns =
      std::clamp<int64_t>(matmulI8AccColumns, 1, columns);
    const int64_t tileRows = std::max<int64_t>(
      1,
      std::min<int64_t>(
        {matmulI8Rows, rows, kInt8AccumulatorBudget / accColumns}));

    const int64_t fullColumnBlocks = columns / accColumns;
    const int64_t tailColumns = columns % accColumns;
    const int64_t fullRowExtent = rows / tileRows * tileRows;

    auto zero = builder.create<arith::ConstantIndexOp>(0);
    auto one = builder.create<arith::ConstantIndexOp>(1);
    auto i32Type = rewriter.getIntegerType(32);

    // 一段 rowCount 行 × 一个 n 块：每 (i,j) 一条标量 i32-MAC 链。
    auto emitTileBlock = [&](Value rowStart,
                             int64_t rowCount,
                             Value columnStart,
                             int64_t blockColumns) {
      SmallVector<Value> rowIndices;
      rowIndices.reserve(rowCount);
      for (int64_t i = 0; i < rowCount; ++i) {
        rowIndices.push_back(
          i == 0 ? rowStart
                 : builder
                     .create<arith::AddIOp>(
                       rowStart, builder.create<arith::ConstantIndexOp>(i))
                     .getResult());
      }
      SmallVector<Value> columnIndices;
      columnIndices.reserve(blockColumns);
      for (int64_t j = 0; j < blockColumns; ++j) {
        columnIndices.push_back(
          j == 0 ? columnStart
                 : builder
                     .create<arith::AddIOp>(
                       columnStart, builder.create<arith::ConstantIndexOp>(j))
                     .getResult());
      }

      // 累加链初值 = C 初始化（linalg.matmul 语义 C += A·B）。
      auto flatIndex = [&](int64_t i, int64_t j) {
        return (i * blockColumns) + j;
      };
      SmallVector<Value> accumulators;
      accumulators.reserve(rowCount * blockColumns);
      for (int64_t i = 0; i < rowCount; ++i) {
        for (int64_t j = 0; j < blockColumns; ++j) {
          accumulators.push_back(builder.create<memref::LoadOp>(
            acc, ValueRange{rowIndices[i], columnIndices[j]}));
        }
      }

      auto depthBound = builder.create<arith::ConstantIndexOp>(depth);
      auto kLoop =
        builder.create<scf::ForOp>(zero, depthBound, one, accumulators);
      builder.setInsertionPointToStart(kLoop.getBody());
      Value kIndex = kLoop.getInductionVar();
      SmallVector<Value> aScalars(rowCount);
      for (int64_t i = 0; i < rowCount; ++i) {
        auto aByte = builder.create<memref::LoadOp>(
          lhs, ValueRange{rowIndices[i], kIndex});
        aScalars[i] = builder.create<arith::ExtSIOp>(i32Type, aByte);
      }
      SmallVector<Value> bScalars(blockColumns);
      for (int64_t j = 0; j < blockColumns; ++j) {
        auto bByte = builder.create<memref::LoadOp>(
          rhs, ValueRange{columnIndices[j], kIndex});
        bScalars[j] = builder.create<arith::ExtSIOp>(i32Type, bByte);
      }
      SmallVector<Value> updated;
      updated.reserve(accumulators.size());
      for (int64_t i = 0; i < rowCount; ++i) {
        for (int64_t j = 0; j < blockColumns; ++j) {
          auto product =
            builder.create<arith::MulIOp>(aScalars[i], bScalars[j]);
          updated.push_back(builder.create<arith::AddIOp>(
            kLoop.getRegionIterArgs()[flatIndex(i, j)], product));
        }
      }
      builder.create<scf::YieldOp>(updated);

      builder.setInsertionPointAfter(kLoop);
      for (int64_t i = 0; i < rowCount; ++i) {
        for (int64_t j = 0; j < blockColumns; ++j) {
          builder.create<memref::StoreOp>(
            kLoop.getResult(flatIndex(i, j)),
            acc,
            ValueRange{rowIndices[i], columnIndices[j]});
        }
      }
    };

    auto emitRowBlock = [&](Value rowStart, int64_t rowCount) {
      if (fullColumnBlocks > 1) {
        auto columnBound =
          builder.create<arith::ConstantIndexOp>(fullColumnBlocks * accColumns);
        auto columnStep = builder.create<arith::ConstantIndexOp>(accColumns);
        auto columnLoop =
          builder.create<scf::ForOp>(zero, columnBound, columnStep);
        builder.setInsertionPointToStart(columnLoop.getBody());
        emitTileBlock(
          rowStart, rowCount, columnLoop.getInductionVar(), accColumns);
        builder.setInsertionPointAfter(columnLoop);
      } else {
        emitTileBlock(rowStart, rowCount, zero, accColumns);
      }
      if (tailColumns > 0) {
        auto tailStart =
          builder.create<arith::ConstantIndexOp>(fullColumnBlocks * accColumns);
        emitTileBlock(rowStart, rowCount, tailStart, tailColumns);
      }
    };

    if (fullRowExtent > 0) {
      auto rowBound = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowStep = builder.create<arith::ConstantIndexOp>(tileRows);
      auto rowLoop = builder.create<scf::ForOp>(zero, rowBound, rowStep);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), tileRows);
      builder.setInsertionPointAfter(rowLoop);
    }
    if (const int64_t remainderRows = rows % tileRows) {
      auto rowStart = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowBound = builder.create<arith::ConstantIndexOp>(rows);
      auto rowLoop = builder.create<scf::ForOp>(rowStart, rowBound, one);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), 1);
    }

    rewriter.eraseOp(matmul);
  }
};

}  // namespace

}  // namespace mlir::ncnn
