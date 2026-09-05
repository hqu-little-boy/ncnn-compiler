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
//    B[k] 行 FMA 累加（vector.fma 单舍入），行末一次写回；行宽仅影响
//    LLVM 合法化拆分。
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
    if (matmuls.empty() && rowGenerics.empty() && selfCopyLoops.empty() &&
        gathers.empty()) {
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
    // 仅当通道数是 2 的幂（≥8）时改写：保证整行向量宽度合法；否则保持
    // 原 generic 下降（其行为等价于基线）。
    return channels >= 8 && channels <= kMaxRowWidth &&
           (channels & (channels - 1)) == 0 &&
           outputType.getShape()[4] == channels;
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

    auto vectorType = VectorType::get({channels}, inputType.getElementType());
    auto zeroIndex = builder.create<arith::ConstantIndexOp>(0);

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

    auto row = builder.create<vector::TransferReadOp>(
      vectorType,
      input,
      ValueRange{zeroIndex.getResult(),
                 sourceRow.getResult(),
                 sourceColumn.getResult(),
                 zeroIndex.getResult()},
      std::nullopt);
    builder.create<vector::TransferWriteOp>(
      row,
      output,
      ValueRange{
        indices[0], indices[1], indices[2], indices[3], zeroIndex.getResult()},
      std::nullopt);

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
    // 预算时保持整行形态；余数以标量 load/store 兜底。
    const auto laneBudget = static_cast<int64_t>(rowChunkLanes);
    // 分块预算：≤32 位元素取 4×lanes（LowerVectorMathNCNN 可拆分上限，
    // math op 拆成 ≤4 段等宽向量库调用；i8/i16 行不再退化为 8 字节级
    // SIMD）；64 位元素无向量数学库覆盖，保持 lanes 等宽。
    const int64_t chunkBudget =
      elementType.getIntOrFloatBitWidth() <= 32 ? 4 * laneBudget : laneBudget;
    const int64_t chunkWidth =
      (laneBudget > 1 && rowWidth > chunkBudget) ? chunkBudget : rowWidth;
    const int64_t fullChunks = rowWidth / chunkWidth;
    const int64_t tailWidth = rowWidth % chunkWidth;

    ImplicitLocOpBuilder builder(generic.getLoc(), rewriter);
    builder.setInsertionPoint(generic);

    auto vectorType = VectorType::get({chunkWidth}, elementType);

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

    // 读写索引 = leading parallel 索引 + 最内维偏移。
    auto rowIndices = [&](Value lastDimOffset) {
      SmallVector<Value> indices(parallelIndices);
      indices.push_back(lastDimOffset);
      return indices;
    };

    // 提升一个分块：读入 chunkWidth 宽行段、向量化 body、写回。body
    // 引用的外部标量值提升为分块宽 splat，并缓存复用。
    auto emitChunk = [&](Value lastDimOffset) {
      const SmallVector<Value> indices = rowIndices(lastDimOffset);
      IRMapping mapping;
      unsigned inputIndex = 0;
      Block& block = generic.getRegion().front();
      for (Value input : generic.getDpsInputs()) {
        mapping.map(
          block.getArgument(inputIndex),
          builder.create<vector::TransferReadOp>(vectorType,
                                                 input,
                                                 indices,
                                                 std::nullopt,
                                                 SmallVector<bool>(1, true)));
        ++inputIndex;
      }
      mapping.map(
        block.getArgument(generic.getNumDpsInputs()),
        builder.create<vector::TransferReadOp>(
          vectorType, out, indices, std::nullopt, SmallVector<bool>(1, true)));

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
        state.addTypes(
          SmallVector<Type>(statement.getNumResults(), vectorType));
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
        mapping.lookup(yielded), out, indices);
      rowWrite.setInBoundsAttr(
        builder.getBoolArrayAttr(SmallVector<bool>(1, true)));
    };

    // 标量兜底一个尾元素：memref.load 提升为标量 body（clone 的 mapping
    // 缺省映射让外部标量原样复用），结果直接 store 回。
    auto emitTailElement = [&](Value lastDimOffset) {
      const SmallVector<Value> indices = rowIndices(lastDimOffset);
      IRMapping mapping;
      unsigned inputIndex = 0;
      Block& block = generic.getRegion().front();
      for (Value input : generic.getDpsInputs()) {
        mapping.map(block.getArgument(inputIndex),
                    builder.create<memref::LoadOp>(input, indices));
        ++inputIndex;
      }
      mapping.map(block.getArgument(generic.getNumDpsInputs()),
                  builder.create<memref::LoadOp>(out, indices));
      for (Operation& statement : block.without_terminator()) {
        builder.clone(statement, mapping);
      }
      Value yielded = mapping.lookup(block.getTerminator()->getOperand(0));
      builder.create<memref::StoreOp>(yielded, out, indices);
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
    // 单舍入 FMA：mul+add 分离会让 LLVM 侧因无 fastmath/contract 而无法
    // 合成 vfmadd（每个 MAC 双指令、K 链延迟翻倍）；vector.fma 一步到位，
    // 舍入语义与 ncnn 的 FMA 内核一致，差异由数值黄金预算吸收。
    auto fused =
      builder.create<vector::FMAOp>(vectorType, broadcast, bRow, accumulator);
    builder.create<scf::YieldOp>(ValueRange{fused.getResult()});

    // 写回仍在 n 块循环体内（kLoop 之后）：每块行段独立累加并落回。
    builder.setInsertionPointAfter(kLoop);
    builder.create<vector::TransferWriteOp>(
      kLoop.getResult(0), acc, ValueRange{rowIndex, columnStart}, std::nullopt);

    rewriter.eraseOp(matmul);
  }
};

}  // namespace

}  // namespace mlir::ncnn
