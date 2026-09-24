#include "ncnn-mlir/Transforms/MatmulKernelNCNN/MatmulKernelNCNN.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_MATMULKERNELNCNNPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

void copyConvContract(Operation* source, Operation* target) {
  for (StringRef attribute : {contract::kOperationFamily,
                              contract::kImplementation,
                              contract::kKernelStatic,
                              contract::kKernelHeight,
                              contract::kKernelWidth,
                              contract::kStrideHeight,
                              contract::kStrideWidth,
                              contract::kDilationHeight,
                              contract::kDilationWidth,
                              contract::kInputChannels,
                              contract::kOutputChannels,
                              contract::kMultiplier,
                              contract::kFallback}) {
    if (Attribute value = source->getAttr(attribute)) {
      target->setAttr(attribute, value);
    }
  }
  for (StringRef attribute :
       {StringRef("ncnn.name"), StringRef("ncnn.source_layer")}) {
    if (Attribute value = source->getAttr(attribute)) {
      target->setAttr(attribute, value);
    }
  }
}

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
// 5) im2col 物化消除（P6 搬运削减）：A 面板由 im2col gather 独占填充
//    且唯一被折叠后的 matmul 消费时，内核 K 循环改为按窗口映射直取
//    源图（gather-free）——M×K 物化拷贝与中间缓冲整体消失。K 遍历序
//    （kh, kw, ic 字典序）与折叠 [[0,1],[2,3,4]] 展平严格一致，FMA 累
//    加链数值逐位不变。
class MatmulKernelNCNNPass final
  : public impl::MatmulKernelNCNNPassBase<MatmulKernelNCNNPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if ((int8Kernel != "portable" && int8Kernel != "auto" &&
         int8Kernel != "vnni") ||
        (int8Target != "portable" && int8Target != "avx-vnni" &&
         int8Target != "avx-vnni-int8" && int8Target != "avx512-vnni") ||
        (int8Kernel == "vnni" &&
         (int8Target == "portable" || int8Target == "avx-vnni-int8"))) {
      module.emitError("invalid INT8 policy or unavailable VNNI target");
      signalPassFailure();
      return;
    }
    if (int8Kernel == "vnni") {
      if (Operation* existing =
            module.lookupSymbol("llvm.x86.avx512.vpdpbusd.256")) {
        auto declaration = dyn_cast<LLVM::LLVMFuncOp>(existing);
        auto words = VectorType::get({8}, IntegerType::get(&getContext(), 32));
        auto signature =
          LLVM::LLVMFunctionType::get(words, {words, words, words});
        if (!declaration || !declaration.isExternal() ||
            declaration.getFunctionType() != signature) {
          existing->emitError("incompatible VNNI intrinsic declaration");
          signalPassFailure();
          return;
        }
      }
    }
    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<linalg::MatmulTransposeBOp> int8Matmuls;
    SmallVector<linalg::BatchMatmulOp> batches;
    SmallVector<linalg::GenericOp> rowGenerics;
    SmallVector<linalg::GenericOp> gathers;
    SmallVector<scf::ForOp> selfCopyLoops;
    module.walk([&](Operation* operation) {
      const bool inForall =
        operation->getParentOfType<scf::ForallOp>() != nullptr;
      if (auto matmul = dyn_cast<linalg::MatmulOp>(operation)) {
        const bool packed =
          matmul->getAttrOfType<StringAttr>(contract::kPacking) &&
          matmul->getAttrOfType<StringAttr>(contract::kPacking).getValue() ==
            "prepacked_B";
        // A physically packed RHS is valid only for the explicit kernel path.
        // Never let an un-tiled or otherwise unsupported matmul silently reach
        // generic row-major lowering.
        if (packed && (!inForall || !isKernelizable(matmul))) {
          matmul.emitError(
            "prepacked_B matmul has no compatible packed kernel consumer");
          packingFailure = true;
          return;
        }
        // matmul 内核仅针对 forall 分块形态；未切分的保持通用下降。
        if (inForall && isKernelizable(matmul)) {
          matmuls.push_back(matmul);
        }
        return;
      }
      if (auto batch = dyn_cast<linalg::BatchMatmulOp>(operation)) {
        // P6-C：MHA 的 scores/context（heads 批维）不被裁剪或降维，
        // 通用下降成标量 mul+add + 逐 k 读写回的 C（medium_rec 29% 热
        // 点）。静态形状的批量收缩改为 b 外层循环 + 逐批 A1b 内核。
        if (batch.hasPureBufferSemantics() && isKernelizableBatch(batch)) {
          batches.push_back(batch);
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
    if (packingFailure) {
      signalPassFailure();
      return;
    }
    if (matmuls.empty() && int8Matmuls.empty() && batches.empty() &&
        rowGenerics.empty() && selfCopyLoops.empty() && gathers.empty()) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    // 自拷贝循环先删，避免干扰后续改写的结构匹配。
    for (scf::ForOp loop : selfCopyLoops) {
      rewriter.eraseOp(loop);
    }
    // P6-A：将被融合进 matmul 内核的 gather 先跳过（kernelize 直取源图
    // 后连 chain 带删除）；其余照旧改写为向量化窗口拷贝。融合判定就是
    // probe 的唯一消费/几何吻合条件，与 kernelize 内部一致。M/N 双切
    // 时同一 gather 服务多个 tile——各 tile 都在 fusedGathers 里才安全。
    llvm::SmallPtrSet<Operation*, 16> fusedGathers;
    SmallVector<Im2colGeometry> fusedChains;
    for (linalg::MatmulOp matmul : matmuls) {
      if (const std::optional<Im2colGeometry> im2col = probeIm2colSource(
            matmul.getInputs()[0],
            cast<MemRefType>(matmul.getInputs()[0].getType()).getShape()[0],
            cast<MemRefType>(matmul.getInputs()[0].getType()).getShape()[1],
            matmul.getOperation())) {
        fusedGathers.insert(im2col->gather);
        fusedChains.push_back(*im2col);
      }
    }
    for (linalg::GenericOp generic : gathers) {
      if (fusedGathers.contains(generic.getOperation())) {
        continue;
      }
      gatherToParallelVectorCopy(rewriter, generic);
    }
    for (linalg::GenericOp generic : rowGenerics) {
      // 融合目标 gather 不能被行级向量化抢先改写（窄通道 IC<8 的 gather
      // 不满足 isIm2colGatherCopy 的整行宽度下限，会落入本列表）。
      if (fusedGathers.contains(generic.getOperation())) {
        continue;
      }
      vectorizeRowGeneric(rewriter, generic);
    }
    for (linalg::MatmulOp matmul : matmuls) {
      kernelize(rewriter, matmul);
    }
    for (linalg::BatchMatmulOp batch : batches) {
      kernelizeBatchMatmul(rewriter, batch);
    }
    for (linalg::MatmulTransposeBOp transposed : int8Matmuls) {
      kernelizeInt8RowDot(rewriter, transposed);
    }
    // P6-A 收尾：全部融合内核就位后，物化缓冲生产链（gather → collapse
    // → alloc）成为死代码，逆序删除。tile-and-fuse 的副本对（共享 alloc
    // 的多份 gather/collapse）各自随其 matmul 失活；alloc 在其全部用户
    // 都死后才删。同一 gather 服务多个 tile 时去重后只删一次。
    llvm::SmallPtrSet<Operation*, 16> deleted;
    for (Im2colGeometry& chain : fusedChains) {
      if (deleted.contains(chain.gather)) {
        continue;
      }
      deleted.insert(chain.gather);
      if (chain.gather->use_empty()) {
        rewriter.eraseOp(chain.gather);
      }
      if (chain.collapse->use_empty()) {
        rewriter.eraseOp(chain.collapse);
      }
    }
    for (Im2colGeometry& chain : fusedChains) {
      if (chain.alloc->use_empty()) {
        rewriter.eraseOp(chain.alloc);
      }
    }
  }

 private:
  // 逐元素行向量化的宽度上限：relu 等逐元 epilogue 的最内维覆盖范围；
  // 超宽保持标量。matmul 内核的向量 accumulator 宽度由 accColumns（默认
  // 16）与其余数列块给出，天然有界，不受此限制。
  static constexpr int64_t kMaxElementwiseRowWidth = 1024;

  // matmul 内核 M×N 寄存器分块的 accumulator 浮点预算：tileRows ×
  // accColumns ≤ 预算时，accumulator（默认 4×16 = 8 个 ymm）加 B 行与
  // broadcast 瞬态可容纳于 16 个 ymm 之内，避免 LLVM 寄存器溢出把分块
  // 的访存/延迟收益吃回去。
  static constexpr int64_t kAccumulatorFloatBudget = 64;

  // int8 row-dot 内核：每个 (m,n) 输出一个 v8i32 部分 和 accumulator
  // （8 个 k-对），tileRows × accColumns ≤ 预算即 8 个 ymm；k 向量宽
  // 8 lane（vpmaddwd 一条指令消费 8 个 i16 乘积）。
  static constexpr int64_t kInt8AccumulatorBudget = 8;

  // i8 im2col gather 的非 2 幂通道分块宽（32 字节 = 一个 ymm）。
  static constexpr int64_t kGatherChunkLanes = 32;

  // A failed packed-layout probe cannot fall back to row-major addressing: the
  // RHS storage has already been physically reordered. Defer pass failure until
  // the traversal completes so the diagnostic points at the offending matmul.
  mutable bool packingFailure = false;

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
      if (!isStaticF32Matrix(type)) {
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
    // Canonical named-op 语义等价于 generalized body：accum + extsi(x)*extsi(y)
    // 的 wrapping MAC（P4 内核假设）。named op 的区域是惰性构建的——文本解析
    // 后 region 可能为空，此时语义由 tablegen regionBuilder 定义（canonical
    // signed MAC）；cast 属性会改变扩宽 signedness（如 cast_unsigned），custom
    // region 则可能是任意算术，两者都不匹配标量 signed-MAC，保持通用下降。
    if (matmul->getAttr("cast")) {
      return false;
    }
    if (!matmul.getRegion().empty()) {
      Block& body = matmul.getRegion().front();
      const auto statements = std::distance(body.without_terminator().begin(),
                                            body.without_terminator().end());
      if (body.getNumArguments() != 3 || statements != 4) {
        return false;
      }
      auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
      if (!yield || yield.getNumOperands() != 1) {
        return false;
      }
      // Canonical MAC：accumulator 在 add 的 LHS，乘积在 RHS（tablegen 区域
      // 即此序）。
      auto add = dyn_cast<arith::AddIOp>(yield.getOperand(0).getDefiningOp());
      if (!add || add.getLhs() != body.getArgument(2) ||
          add.getOverflowFlags() != arith::IntegerOverflowFlags::none) {
        return false;
      }
      auto multiply = dyn_cast<arith::MulIOp>(add.getRhs().getDefiningOp());
      if (!multiply ||
          multiply.getOverflowFlags() != arith::IntegerOverflowFlags::none) {
        return false;
      }
      auto lhsCast =
        dyn_cast<arith::ExtSIOp>(multiply.getLhs().getDefiningOp());
      auto rhsCast =
        dyn_cast<arith::ExtSIOp>(multiply.getRhs().getDefiningOp());
      if (!lhsCast || !rhsCast || lhsCast.getIn() != body.getArgument(0) ||
          rhsCast.getIn() != body.getArgument(1)) {
        return false;
      }
    }
    // N 是写回行宽；K 无上限（row-dot 内核沿 k 向量化，代码量与 K 无关）。
    const int64_t rows = lhsType.getShape()[0];
    const int64_t depth = lhsType.getShape()[1];
    const int64_t rhsRows = rhsType.getShape()[0];
    const int64_t rhsDepth = rhsType.getShape()[1];
    if (rows <= 0 || depth <= 0 || rhsRows <= 0 || rhsDepth <= 0 ||
        accType.getShape()[0] <= 0 || accType.getShape()[1] <= 0) {
      return false;
    }
    // matmul_transpose_b 语义：C[M,N] = A[M,K] · B[N,K]ᵀ，B 的行数是 N。
    return rhsDepth == depth && rhsRows == accType.getShape()[1] &&
           accType.getShape()[0] == rows;
  }

  // P4 遗留（requant epilogue 融合）的可融合判据：acc（i32 [M,N]）
  // 的唯一用户是静态 memref generic，输出 i8、主输入对 acc 恒等映射、
  // 其余输入全常量广播（逐张量 scale/bias——FuseQuantChain 的产物形
  // 态），body 纯 arith/math。
  static bool isFusableRequantEpilogue(linalg::GenericOp generic,
                                       linalg::MatmulTransposeBOp matmul) {
    if (!generic.hasPureBufferSemantics() || generic.getNumDpsInits() != 1 ||
        generic.getNumResults() != 0) {
      return false;
    }
    Value out = generic.getOutputs().front();
    const auto outType = dyn_cast<MemRefType>(out.getType());
    if (!outType || !outType.hasStaticShape() ||
        !outType.getElementType().isInteger(8)) {
      return false;
    }
    Value acc = matmul.getOutputs().front();
    const auto accType = cast<MemRefType>(acc.getType());
    // 输出与 acc 同形（requant 逐元素）。
    if (outType.getShape() != accType.getShape()) {
      return false;
    }
    const auto maps = generic.getIndexingMapsArray();
    if (maps.size() != static_cast<size_t>(generic.getNumDpsInputs() + 1U)) {
      return false;
    }
    // 主输入（acc）须恒等；其余输入全常量（逐张量标量）。
    if (!maps[0].isIdentity()) {
      return false;
    }
    for (unsigned index = 1; index < generic.getNumDpsInputs(); ++index) {
      if (!llvm::all_of(maps[index].getResults(), [](AffineExpr result) {
            return isa<AffineConstantExpr>(result);
          })) {
        return false;
      }
    }
    // body 纯 arith/math。
    Region& region = generic->getRegion(0);
    if (!region.hasOneBlock()) {
      return false;
    }
    Block& block = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1) {
      return false;
    }
    for (Operation& operation : block.without_terminator()) {
      const StringRef dialect =
        operation.getName().getDialect()->getNamespace();
      if (dialect != "arith" && dialect != "math") {
        return false;
      }
    }
    // emitRequantValue 会在 matmul 写回点克隆 body：所有 body 操作数必须
    // 封闭于块内（块参或块内定义），否则捕获值会被克隆到定义点之前。
    for (Operation& operation : block.without_terminator()) {
      for (Value operand : operation.getOperands()) {
        if (isa<BlockArgument>(operand)) {
          continue;
        }
        Operation* def = operand.getDefiningOp();
        if (def && def->getBlock() == &block) {
          continue;
        }
        // 捕获值必须定义于 matmul 所在块且严格位于其之前（克隆到写回点
        // 才是向前引用安全的支配保证；跨块捕获值一律拒绝，足够保守）。
        if (def && matmul->getBlock() &&
            def->getBlock() == matmul->getBlock() &&
            def->isBeforeInBlock(matmul)) {
          continue;
        }
        return false;
      }
    }
    return true;
  }

  // P6-C 批量收缩内核形态：[B,M,K]×[B,K,N]→[B,M,N] 全静态 f32。逐批
  // 视图（memref.subview 取 [b] 面）后与 A1b 内核同型。
  static bool isKernelizableBatch(linalg::BatchMatmulOp batch) {
    for (Value operand : batch->getOperands()) {
      const auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || !type.hasStaticShape() || type.getRank() != 3 ||
          !isa<FloatType>(type.getElementType())) {
        return false;
      }
    }
    const auto aType = cast<MemRefType>(batch.getInputs()[0].getType());
    const auto bType = cast<MemRefType>(batch.getInputs()[1].getType());
    const auto cType = cast<MemRefType>(batch.getOutputs().front().getType());
    for (ArrayRef<int64_t> shape :
         {aType.getShape(), bType.getShape(), cType.getShape()}) {
      for (int64_t extent : shape) {
        if (extent <= 0) {
          return false;
        }
      }
    }
    return bType.getShape()[0] == aType.getShape()[0] &&
           cType.getShape()[0] == aType.getShape()[0] &&
           bType.getShape()[1] == aType.getShape()[2] &&
           cType.getShape()[1] == aType.getShape()[1] &&
           cType.getShape()[2] == bType.getShape()[2];
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
    if (width < 2 || width > kMaxElementwiseRowWidth) {
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
    if (channels >= 8 && channels <= kMaxElementwiseRowWidth &&
        (channels & (channels - 1)) == 0) {
      return true;
    }
    return isa<IntegerType>(inputType.getElementType()) && channels >= 8;
  }

  // P6 im2col 物化消除的窗口几何：源图 4D 静态形状 + 步长/膨胀。K 遍
  // 历序按折叠 [[0,1],[2,3,4]] 展平 = (kh, kw, ic) 字典序，k =
  // kh·(KW·IC) + kw·IC + ic。
  struct Im2colGeometry {
    Value source;
    linalg::GenericOp gather;
    memref::CollapseShapeOp collapse;
    memref::AllocOp alloc;
    // forall M 向切分时的行偏移（subview 产出）；无 subview 时为空。
    Value rowOffset;
    int64_t outputHeight = 0;
    int64_t outputWidth = 0;
    int64_t kernelHeight = 0;
    int64_t kernelWidth = 0;
    int64_t channels = 0;
    int64_t strideHeight = 1;
    int64_t strideWidth = 1;
    int64_t dilationHeight = 1;
    int64_t dilationWidth = 1;
  };

  // OpFoldResult 是常量 index 且等于给定值。
  static bool isConstantIndex(OpFoldResult candidate, int64_t value) {
    auto attribute = dyn_cast_if_present<Attribute>(candidate);
    if (!attribute) {
      return false;
    }
    auto integer = dyn_cast<IntegerAttr>(attribute);
    return integer && integer.getValue() == value;
  }

  // A 面板来源探测：matmul 的 A 为 collapse_shape(alloc)，alloc 的唯一
  // 写者是 im2col 窗口 gather generic（与向量化路径同形态），collapse 结
  // 果只被本 matmul 消费，且折叠序 [[0,1],[2,3,4]] 与 K 展平序一致、行
  // 数/深度与窗口几何吻合。返回窗口几何与待删的 gather/collapse/alloc
  // 三件套；不满足任一条件返回 nullopt（内核保持常规形态）。宽容版匹
  // 配：不要求通道 2 幂（整行向量化的宽度限制对直取路径无意义，内核逐
  // k 标量取数）。
  static std::optional<Im2colGeometry> probeIm2colSource(
    Value lhs, int64_t rows, int64_t depth, Operation* matmulAnchor) {
    // forall 同时切 M/N 维时 A 是 collapse 的 M 向 subview（tile 行偏移
    // 动态、列偏移 0、形状 [rows, depth]）——穿透后融合，行号加回 tile
    // 偏移。
    Value rowOffset;
    if (auto subview = lhs.getDefiningOp<memref::SubViewOp>()) {
      auto source = subview.getSource();
      if (source.getDefiningOp<memref::CollapseShapeOp>() == nullptr) {
        return std::nullopt;
      }
      const auto sourceType = dyn_cast<MemRefType>(source.getType());
      const auto viewType = dyn_cast<MemRefType>(subview.getType());
      if (!sourceType || !viewType || !viewType.hasStaticShape() ||
          viewType.getRank() != 2 || viewType.getShape()[0] != rows ||
          viewType.getShape()[1] != depth) {
        return std::nullopt;
      }
      auto offsets = subview.getMixedOffsets();
      auto sizes = subview.getMixedSizes();
      auto strides = subview.getMixedStrides();
      if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2) {
        return std::nullopt;
      }
      // K 向偏移/步距必须恒 0/1（K 不在切分维度）；M 向步距 1。
      if (!isConstantIndex(strides[1], 1) || !isConstantIndex(strides[0], 1) ||
          !isConstantIndex(offsets[1], 0)) {
        return std::nullopt;
      }
      if (auto offset = offsets[0].dyn_cast<Value>()) {
        rowOffset = offset;
      } else if (!isConstantIndex(offsets[0], 0)) {
        return std::nullopt;
      }
      lhs = source;
    }
    auto collapse = lhs.getDefiningOp<memref::CollapseShapeOp>();
    if (!collapse) {
      return std::nullopt;
    }
    // 折叠序必须把 (oh, ow) 并为行维、(kh, kw, ic) 并为 K 维。
    SmallVector<ReassociationIndices> reassociation =
      llvm::to_vector(collapse.getReassociationIndices());
    if (reassociation.size() != 2 ||
        reassociation[0] != ReassociationIndices{0, 1} ||
        reassociation[1] != ReassociationIndices{2, 3, 4}) {
      return std::nullopt;
    }
    // collapse 结果的消费限制：A 是 collapse 本体时只允许被本 matmul
    // 消费；A 是 M 向 subview 时允许 collapse 被多个 tile subview 切
    // （各 tile 独立融合，全链删除由各 tile 的删除共同完成——gather/
    // collapse/alloc 只在最后一个消费者融合后变死代码，见 kernelize
    // 收尾的条件删除）。
    if (rowOffset == nullptr && !lhs.hasOneUse()) {
      return std::nullopt;
    }
    auto alloc = collapse.getSrc().getDefiningOp<memref::AllocOp>();
    if (!alloc) {
      return std::nullopt;
    }
    const auto windowType = dyn_cast<MemRefType>(collapse.getSrc().getType());
    if (!windowType || !windowType.hasStaticShape() ||
        windowType.getRank() != 5) {
      return std::nullopt;
    }
    // alloc 的消费限制：仅允许 gather 的 outs 与 collapse（无别名外
    // 泄）。tile-and-fuse 会把 gather/collapse 对按 M tile 复制多份共享
    // 同一 alloc——每份 collapse 找到自己的 gather 即可；副本清单记入
    // geometry，kernelize 后统一判死删除。多份 gather 写同一 alloc 时按
    // 「紧邻本 collapse 之前」的那一份配对（各副本链在 IR 中依次成对
    // 出现，取错会拿到别层的源图——融合内核直取的 pad 缓冲不支配
    // matmul 位置，yolov5 系实测在此炸 dominance）。
    SmallVector<Operation*> users(alloc->getUsers().begin(),
                                  alloc->getUsers().end());
    linalg::GenericOp gather;
    for (Operation* user : users) {
      auto generic = dyn_cast<linalg::GenericOp>(user);
      if (generic && generic.getDpsInits().front() == collapse.getSrc() &&
          generic->isBeforeInBlock(collapse)) {
        if (!gather || gather->isBeforeInBlock(generic)) {
          gather = generic;
        }
      }
    }
    if (!gather) {
      return std::nullopt;
    }
    // gather 形态：窗口映射 (0, sh*oh+dh*kh, sw*ow+dw*kw, ic) 纯转发。
    if (!gather.hasPureBufferSemantics() || gather.getNumDpsInputs() != 1 ||
        gather.getNumDpsInits() != 1 || gather.getNumLoops() != 5) {
      return std::nullopt;
    }
    SmallVector<AffineMap> maps = gather.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity()) {
      return std::nullopt;
    }
    Value input = gather.getDpsInputs().front();
    const auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 4 ||
        windowType.getElementType() != inputType.getElementType() ||
        !isa<FloatType>(inputType.getElementType())) {
      return std::nullopt;
    }
    MLIRContext* context = gather.getContext();
    auto constantZero = dyn_cast<AffineConstantExpr>(maps[0].getResult(0));
    if (!constantZero || constantZero.getValue() != 0) {
      return std::nullopt;
    }
    if (maps[0].getResult(3) !=
        getAffineDimExpr(gather.getNumLoops() - 1, context)) {
      return std::nullopt;
    }
    Im2colGeometry geometry;
    geometry.source = input;
    geometry.gather = gather;
    geometry.collapse = collapse;
    geometry.alloc = alloc;
    geometry.rowOffset = rowOffset;
    // 源图必须支配 matmul 位置（融合内核在 matmul 处直取源图；tile-
    // and-fuse 复制的 gather 副本链可能指向别的层的 pad 缓冲——IR 位置
    // 在 matmul 之后，直取会违反 SSA 支配）。
    if (Operation* sourceDefiner = input.getDefiningOp()) {
      if (!sourceDefiner->isBeforeInBlock(matmulAnchor)) {
        return std::nullopt;
      }
    }
    // 几何吻合校验用全局行数：A 为 collapse 本体时 rows 即全局；M 向
    // subview 时 rows 是 tile 行数，全局 = OH·OW（TileMatmulForall 的
    // 切分尺寸恒为维度因子，rows 整除全局行数且偏移为其倍数）。
    const int64_t globalRows = rowOffset != nullptr
                                 ? geometry.outputHeight * geometry.outputWidth
                                 : rows;
    if (!extractStrideDilation(maps[0].getResult(1),
                               0,
                               2,
                               geometry.strideHeight,
                               geometry.dilationHeight) ||
        !extractStrideDilation(maps[0].getResult(2),
                               1,
                               3,
                               geometry.strideWidth,
                               geometry.dilationWidth)) {
      return std::nullopt;
    }
    const ArrayRef<int64_t> windowShape = windowType.getShape();
    geometry.outputHeight = windowShape[0];
    geometry.outputWidth = windowShape[1];
    geometry.kernelHeight = windowShape[2];
    geometry.kernelWidth = windowShape[3];
    geometry.channels = windowShape[4];
    if (inputType.getShape()[3] != geometry.channels ||
        geometry.kernelHeight < 1 || geometry.kernelWidth < 1 ||
        geometry.channels < 1 || geometry.outputHeight < 1 ||
        geometry.outputWidth < 1 ||
        geometry.kernelHeight * geometry.kernelWidth * geometry.channels !=
          depth ||
        geometry.outputHeight * geometry.outputWidth != globalRows ||
        (rowOffset != nullptr && globalRows % rows != 0)) {
      return std::nullopt;
    }
    return geometry;
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
    const int64_t wholeRow =
      (channels >= 8 && channels <= kMaxElementwiseRowWidth &&
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

    // Top-level generics may use scf.parallel so the later SCF-to-OpenMP
    // conversion can distribute their leading dimensions.  A generic already
    // inside a tile forall must instead use nested serial scf.for loops;
    // otherwise it would create a second OpenMP team inside the outer one.
    const bool nestedInForall =
      generic->getParentOfType<scf::ForallOp>() != nullptr;
    if (auto forall = generic->getParentOfType<scf::ForallOp>()) {
      contract::setString(
        forall.getOperation(), contract::kParallel, "outer_tile+inner_simd");
      contract::setInteger(
        forall.getOperation(), contract::kSimdChunk, chunkWidth);
      contract::setString(
        forall.getOperation(), contract::kFma, "vector.transfer_elementwise");
    }
    SmallVector<Value> parallelIndices(rank - 1);
    SmallVector<scf::ForOp> serialLoops;
    Value zeroIndex = builder.create<arith::ConstantIndexOp>(0);
    Value stepOneIndex = builder.create<arith::ConstantIndexOp>(1);
    if (rank > 1) {
      if (nestedInForall) {
        for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
          Value upperBound =
            builder.create<arith::ConstantIndexOp>(shape[dimension]);
          auto loop =
            builder.create<scf::ForOp>(zeroIndex, upperBound, stepOneIndex);
          serialLoops.push_back(loop);
          parallelIndices[dimension] = loop.getInductionVar();
          builder.setInsertionPointToStart(loop.getBody());
        }
      } else {
        SmallVector<Value> lowerBounds;
        SmallVector<Value> upperBounds;
        SmallVector<Value> steps;
        for (int64_t dimension = 0; dimension < rank - 1; ++dimension) {
          lowerBounds.push_back(zeroIndex);
          upperBounds.push_back(
            builder.create<arith::ConstantIndexOp>(shape[dimension]));
          steps.push_back(stepOneIndex);
        }
        auto parallel = builder.create<scf::ParallelOp>(
          generic.getLoc(), lowerBounds, upperBounds, steps);
        llvm::copy(parallel.getInductionVars(), parallelIndices.begin());
        builder.setInsertionPointToStart(parallel.getBody());
      }
    }

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

    for (scf::ForOp loop : llvm::reverse(serialLoops)) {
      builder.setInsertionPointAfter(loop);
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

  struct PackedBSource final {
    Value flat;
    Value columnOffset;
    memref::GetGlobalOp globalAccess;
  };

  static std::optional<PackedBSource> preparePackedB(
    ImplicitLocOpBuilder& builder, Value rhs) {
    Value current = rhs;
    Value columnOffset = builder.create<arith::ConstantIndexOp>(0);
    memref::GetGlobalOp globalAccess;
    while (Operation* defining = current.getDefiningOp()) {
      if (auto global = dyn_cast<memref::GetGlobalOp>(defining)) {
        globalAccess = global;
        break;
      }
      if (auto subview = dyn_cast<memref::SubViewOp>(defining)) {
        auto offsets = subview.getMixedOffsets();
        auto strides = subview.getMixedStrides();
        if (offsets.size() != 2 || strides.size() != 2 ||
            !isConstantIndex(offsets[0], 0) ||
            !isConstantIndex(strides[0], 1) ||
            !isConstantIndex(strides[1], 1)) {
          // The packed address formula assumes the complete K dimension and a
          // unit-stride two-dimensional tile. In particular, a nonzero K
          // offset or strided view cannot be reinterpreted as panel-NK.
          return std::nullopt;
        }
        OpFoldResult offset = offsets[1];
        if (auto value = offset.dyn_cast<Value>()) {
          columnOffset = builder.create<arith::AddIOp>(columnOffset, value);
        } else if (auto integer =
                     dyn_cast<IntegerAttr>(offset.get<Attribute>())) {
          if (integer.getInt() != 0) {
            columnOffset = builder.create<arith::AddIOp>(
              columnOffset,
              builder.create<arith::ConstantIndexOp>(integer.getInt()));
          }
        } else {
          return std::nullopt;
        }
        current = subview.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(defining)) {
        auto sourceType = dyn_cast<MemRefType>(cast.getSource().getType());
        auto resultType = dyn_cast<MemRefType>(cast.getType());
        if (!sourceType || !resultType || sourceType.getRank() != 2 ||
            resultType.getRank() != 2 || !sourceType.hasStaticShape() ||
            !resultType.hasStaticShape() ||
            sourceType.getShape() != resultType.getShape()) {
          return std::nullopt;
        }
        current = cast.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(defining)) {
        auto sourceType = dyn_cast<MemRefType>(cast.getSource().getType());
        auto resultType = dyn_cast<MemRefType>(cast.getType());
        auto offsets = cast.getMixedOffsets();
        auto sizes = cast.getMixedSizes();
        auto strides = cast.getMixedStrides();
        if (!sourceType || !resultType || sourceType.getRank() != 2 ||
            resultType.getRank() != 2 || !sourceType.hasStaticShape() ||
            !resultType.hasStaticShape() ||
            sourceType.getShape() != resultType.getShape() ||
            offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2 ||
            !isConstantIndex(offsets[0], 0) ||
            !isConstantIndex(offsets[1], 0) ||
            !isConstantIndex(strides[1], 1)) {
          return std::nullopt;
        }
        current = cast.getSource();
        continue;
      }
      break;
    }
    auto type = dyn_cast<MemRefType>(current.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
        !globalAccess) {
      return std::nullopt;
    }
    SmallVector<ReassociationIndices> reassociation{{0, 1}};
    MemRefType flatType =
      memref::CollapseShapeOp::computeCollapsedType(type, reassociation);
    if (!flatType) {
      return std::nullopt;
    }
    Value flat = builder.create<memref::CollapseShapeOp>(
      builder.getLoc(), flatType, current, reassociation);
    return PackedBSource{
      .flat = flat, .columnOffset = columnOffset, .globalAccess = globalAccess};
  }

  static memref::GetGlobalOp findPackedInt8Global(Value value) {
    while (Operation* defining = value.getDefiningOp()) {
      if (auto global = dyn_cast<memref::GetGlobalOp>(defining)) {
        return global;
      }
      if (auto subview = dyn_cast<memref::SubViewOp>(defining)) {
        value = subview.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(defining)) {
        value = cast.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(defining)) {
        value = cast.getSource();
        continue;
      }
      break;
    }
    return {};
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

    // P6-A：A 面板由 im2col gather 独占物化时，内核直取源图窗口（gather-
    // free），M×K 物化缓冲与拷贝整体消除。probe 内部校验行数/深度与窗口
    // 几何吻合；失败返回 nullopt，内核保持从未物化缓冲读 A 的常规形态。
    const std::optional<Im2colGeometry> im2col =
      probeIm2colSource(lhs, rows, depth, matmul.getOperation());
    const bool fused = im2col.has_value();


    ImplicitLocOpBuilder builder(matmul.getLoc(), rewriter);
    builder.setInsertionPoint(matmul);
    auto parentForall = matmul->getParentOfType<scf::ForallOp>();
    // A tiled forall may contain multiple matmuls after producer/consumer
    // fusion. Its boundary contract describes the first matmul only, so the
    // physical B choice must come from the matmul being rewritten rather than
    // from the shared parent forall.
    bool packedB =
      matmul->getAttrOfType<StringAttr>(contract::kPacking) &&
      matmul->getAttrOfType<StringAttr>(contract::kPacking).getValue() ==
        "prepacked_B";
    std::optional<PackedBSource> packedBSource;
    if (packedB) {
      packedBSource = preparePackedB(builder, rhs);
      if (!packedBSource) {
        matmul.emitError(
          "cannot prove the memref view preserves the panel-NK packed layout");
        packingFailure = true;
        return;
      }
      if (packedBSource->globalAccess) {
        // Bufferization drops arbitrary tensor attributes while materializing
        // memref globals. Reattach the P20 identity to the unique global access
        // so execution-plan emission can account for physical storage once,
        // rather than once per tiled forall invocation.
        Operation* global = packedBSource->globalAccess.getOperation();
        contract::setBool(global, contract::kPackedWeight, true);
        if (auto bytes =
              matmul->getAttrOfType<IntegerAttr>(contract::kPackBytes)) {
          contract::setInteger(global, contract::kPackBytes, bytes.getInt());
        }
        for (StringRef attribute : {contract::kPackFactor,
                                    contract::kPackTileK,
                                    contract::kPackSchema,
                                    contract::kPackRuntime,
                                    contract::kAlignment,
                                    contract::kWeightLayout}) {
          if (Attribute value = matmul->getAttr(attribute)) {
            global->setAttr(attribute, value);
          }
        }
      }
    }

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

    if (auto forall = matmul->getParentOfType<scf::ForallOp>()) {
      copyConvContract(matmul.getOperation(), forall.getOperation());
      if (fused) {
        contract::annotateOperationFamily(
          forall.getOperation(), "conv", "gather_free");
      }
      const int64_t packedTileK =
        packedB && !fused &&
            matmul->getAttrOfType<IntegerAttr>(contract::kPackTileK)
          ? matmul->getAttrOfType<IntegerAttr>(contract::kPackTileK).getInt()
          : depth;
      const bool parentAlreadyPacked =
        forall->getAttrOfType<StringAttr>(contract::kPacking) &&
        forall->getAttrOfType<StringAttr>(contract::kPacking).getValue() ==
          "prepacked_B";
      // A mixed forall can contain a packed producer followed by an unpacked
      // small-shape producer. Preserve the parent boundary's packed metadata
      // for the former while selecting the physical layout independently per
      // matmul above; the latter must not overwrite that shared metadata.
      if (packedB || !parentAlreadyPacked) {
        contract::annotateTile(
          forall.getOperation(),
          packedB ? "f32_packed_mxn_fma" : "f32_mxn_fma",
          fused ? "nhwc_gather_free" : "identity",
          packedB ? "panel_nk" : "row_major_kxn",
          "identity",
          tileRows,
          accColumns,
          packedB ? std::min<int64_t>(packedTileK, depth) : depth,
          "outer_tile+inner_simd",
          tailColumns > 0 || rows % tileRows != 0 ? "scalar_tail" : "none");
        if (!packedB) {
          contract::annotatePacking(forall.getOperation(), "unpacked", 1, 0, 0);
        }
      }
      contract::setInteger(
        forall.getOperation(), contract::kSimdChunk, accColumns);
      contract::setString(forall.getOperation(), contract::kFma, "vector.fma");
    }

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

      // 单舍入 FMA：mul+add 分离会让 LLVM 侧因无 fastmath/contract 而无
      // 法合成 vfmadd（每个 MAC 双指令、依赖链延迟翻倍）；vector.fma 一
      // 步到位，舍入语义与 ncnn 的 FMA 内核一致，差异由数值黄金预算吸
      // 收。各 accumulator 链相互独立，K 循环体的发射率不再被单链延迟
      // 钉死。
      // The panel/lane portion of the packed B address is invariant across K.
      // Materialize it once per output block rather than rebuilding div/rem and
      // panel-base arithmetic in every reduction iteration.
      Value packedPanelBase;
      Value packedLaneOffset;
      Value packedPanelWidth;
      if (packedB) {
        const int64_t packN =
          matmul->getAttrOfType<IntegerAttr>(contract::kPackFactor)
            ? matmul->getAttrOfType<IntegerAttr>(contract::kPackFactor).getInt()
            : 16;
        Value globalColumn = builder.create<arith::AddIOp>(
          packedBSource->columnOffset, columnStart);
        packedPanelWidth = builder.create<arith::ConstantIndexOp>(packN);
        Value panelIndex =
          builder.create<arith::DivUIOp>(globalColumn, packedPanelWidth);
        Value panelStart =
          builder.create<arith::MulIOp>(panelIndex, packedPanelWidth);
        packedLaneOffset =
          builder.create<arith::RemUIOp>(globalColumn, packedPanelWidth);
        packedPanelBase = builder.create<arith::MulIOp>(
          panelStart, builder.create<arith::ConstantIndexOp>(depth));
      }

      // P6-A 融合时 A 标量直取源图窗口：k 拆 (kh, kw, ic) 由外层 kp
      // (kh·KW+kw) 与内层 ic 两层静态界循环给出——k = kp·IC + ic 与折叠
      // [[2,3,4]] 展平严格一致，FMA 累加链数值与常规内核逐位相同；行
      // m 拆 (oh, ow) 的小除法在 rowCount 个行上每行一次（行段首），循
      // 环内无除法。
      auto emitKBody = [&](Value kIndex,
                           Value windowRowPart,
                           Value windowColumnPart,
                           Value channelIndex,
                           SmallVector<Value> regionIterArgs) {
        Value bIndex;
        if (!packedB) {
          bIndex = builder.create<vector::TransferReadOp>(
            vectorType,
            rhs,
            ValueRange{kIndex, columnStart},
            std::nullopt,
            SmallVector<bool>(1, true));
        } else {
          Value kBase = builder.create<arith::MulIOp>(kIndex, packedPanelWidth);
          Value packedIndex = builder.create<arith::AddIOp>(
            builder.create<arith::AddIOp>(packedPanelBase, kBase),
            packedLaneOffset);
          bIndex =
            builder.create<vector::TransferReadOp>(vectorType,
                                                   packedBSource->flat,
                                                   ValueRange{packedIndex},
                                                   std::nullopt,
                                                   SmallVector<bool>(1, true));
        }
        Value bRow = bIndex;
        SmallVector<Value> updated;
        updated.reserve(rowCount);
        for (int64_t i = 0; i < rowCount; ++i) {
          Value aScalar;
          if (!fused) {
            aScalar = builder.create<memref::LoadOp>(
              lhs, ValueRange{rowIndices[i], kIndex});
          } else {
            // 窗口行/列 = sh·oh + dh·kh、sw·ow + dw·kw；oh/ow 由全局
            // 行号对 [OH, OW] 降维——M 向 forall 切分时行号 = tile 偏移
            // + 段内索引。
            Value m = rowIndices[i];
            if (im2col->rowOffset != nullptr) {
              m = builder.create<arith::AddIOp>(im2col->rowOffset, m);
            }
            Value oh = builder.create<arith::DivSIOp>(
              m, builder.create<arith::ConstantIndexOp>(im2col->outputWidth));
            Value ow = builder.create<arith::RemSIOp>(
              m, builder.create<arith::ConstantIndexOp>(im2col->outputWidth));
            Value windowRow = builder.create<arith::AddIOp>(
              builder.create<arith::MulIOp>(
                builder.create<arith::ConstantIndexOp>(im2col->strideHeight),
                oh),
              builder.create<arith::MulIOp>(
                builder.create<arith::ConstantIndexOp>(im2col->dilationHeight),
                windowRowPart));
            Value windowColumn = builder.create<arith::AddIOp>(
              builder.create<arith::MulIOp>(
                builder.create<arith::ConstantIndexOp>(im2col->strideWidth),
                ow),
              builder.create<arith::MulIOp>(
                builder.create<arith::ConstantIndexOp>(im2col->dilationWidth),
                windowColumnPart));
            aScalar = builder.create<memref::LoadOp>(
              im2col->source,
              ValueRange{zero, windowRow, windowColumn, channelIndex});
          }
          auto broadcast =
            builder.create<vector::BroadcastOp>(vectorType, aScalar);
          updated.push_back(builder.create<vector::FMAOp>(
            vectorType, broadcast, bRow, regionIterArgs[i]));
        }
        return updated;
      };

      // 写回在 kLoop 之后：行段累加完毕一次落回。
      SmallVector<Value> results;
      if (!fused) {
        auto depthBound = builder.create<arith::ConstantIndexOp>(depth);
        const int64_t packedKTile =
          packedB &&
              parentForall->getAttrOfType<IntegerAttr>(contract::kPackTileK)
            ? parentForall->getAttrOfType<IntegerAttr>(contract::kPackTileK)
                .getInt()
            : depth;
        const bool useKBlocking = packedB && packedKTile > 0 &&
                                  packedKTile < depth &&
                                  depth % packedKTile == 0;
        if (!useKBlocking) {
          auto kLoop = builder.create<scf::ForOp>(
            zero, depthBound, one, ValueRange(accumulators));
          builder.setInsertionPointToStart(kLoop.getBody());
          SmallVector<Value> updated =
            emitKBody(kLoop.getInductionVar(),
                      Value(),
                      Value(),
                      Value(),
                      SmallVector<Value>(kLoop.getRegionIterArgs().begin(),
                                         kLoop.getRegionIterArgs().end()));
          builder.create<scf::YieldOp>(updated);
          for (auto result : kLoop.getResults()) {
            results.push_back(result);
          }
          builder.setInsertionPointAfter(kLoop);
        } else {
          auto tileBound = builder.create<arith::ConstantIndexOp>(packedKTile);
          auto blockLoop = builder.create<scf::ForOp>(
            zero, depthBound, tileBound, ValueRange(accumulators));
          builder.setInsertionPointToStart(blockLoop.getBody());
          Value blockEnd = builder.create<arith::AddIOp>(
            blockLoop.getInductionVar(), tileBound);
          auto innerLoop = builder.create<scf::ForOp>(
            blockLoop.getInductionVar(),
            blockEnd,
            one,
            SmallVector<Value>(blockLoop.getRegionIterArgs().begin(),
                               blockLoop.getRegionIterArgs().end()));
          builder.setInsertionPointToStart(innerLoop.getBody());
          SmallVector<Value> updated =
            emitKBody(innerLoop.getInductionVar(),
                      Value(),
                      Value(),
                      Value(),
                      SmallVector<Value>(innerLoop.getRegionIterArgs().begin(),
                                         innerLoop.getRegionIterArgs().end()));
          builder.create<scf::YieldOp>(updated);
          builder.setInsertionPoint(blockLoop.getBody(),
                                    blockLoop.getBody()->end());
          builder.create<scf::YieldOp>(SmallVector<Value>(
            innerLoop.getResults().begin(), innerLoop.getResults().end()));
          for (auto result : blockLoop.getResults()) {
            results.push_back(result);
          }
          builder.setInsertionPointAfter(blockLoop);
        }
      } else {
        // 外层 kp 扫 KH·KW 个窗口位置，内层 ic 扫 IC 个通道；
        // k = kp·IC + ic 对齐折叠展平。kp 的 (kh, kw) 拆解用 div/mod——
        // 界 KH·KW 与 KW 都是编译期常量，IC 段内无除法；LLVM 循环反转
        // 后大宗是地址递推。
        auto kpBound = builder.create<arith::ConstantIndexOp>(
          im2col->kernelHeight * im2col->kernelWidth);
        auto outer = builder.create<scf::ForOp>(
          zero, kpBound, one, ValueRange(accumulators));
        builder.setInsertionPointToStart(outer.getBody());
        Value kp = outer.getInductionVar();
        Value windowRowPart = builder.create<arith::DivSIOp>(
          kp, builder.create<arith::ConstantIndexOp>(im2col->kernelWidth));
        Value windowColumnPart = builder.create<arith::RemSIOp>(
          kp, builder.create<arith::ConstantIndexOp>(im2col->kernelWidth));
        auto kpOffset = builder.create<arith::MulIOp>(
          kp, builder.create<arith::ConstantIndexOp>(im2col->channels));
        auto icBound = builder.create<arith::ConstantIndexOp>(im2col->channels);
        auto outerIterArgs = outer.getRegionIterArgs();
        auto inner = builder.create<scf::ForOp>(
          zero,
          icBound,
          one,
          SmallVector<Value>(outerIterArgs.begin(), outerIterArgs.end()));
        builder.setInsertionPointToStart(inner.getBody());
        Value kIndex =
          builder.create<arith::AddIOp>(kpOffset, inner.getInductionVar());
        auto innerIterArgs = inner.getRegionIterArgs();
        SmallVector<Value> updated = emitKBody(
          kIndex,
          windowRowPart,
          windowColumnPart,
          inner.getInductionVar(),
          SmallVector<Value>(innerIterArgs.begin(), innerIterArgs.end()));
        // 内层 ic 循环的 yield：插入点移到内层 body 末尾（在自动生成的
        // 空 yield 之前插入新值版本——scf.for 的 body 构建器自带空
        // yield 终结符，显式 yield 前插其后）。
        builder.setInsertionPoint(inner.getBody(), inner.getBody()->end());
        builder.create<scf::YieldOp>(updated);
        builder.setInsertionPoint(outer.getBody(), outer.getBody()->end());
        builder.create<scf::YieldOp>(SmallVector<Value>(
          inner.getResults().begin(), inner.getResults().end()));
        for (int64_t i = 0; i < rowCount; ++i) {
          results.push_back(outer.getResult(i));
        }
        builder.setInsertionPointAfter(outer);
      }
      for (int64_t i = 0; i < rowCount; ++i) {
        auto rowWrite = builder.create<vector::TransferWriteOp>(
          results[i], acc, ValueRange{rowIndices[i], columnStart});
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
    if (rows % tileRows) {
      auto rowStart = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowBound = builder.create<arith::ConstantIndexOp>(rows);
      auto rowLoop = builder.create<scf::ForOp>(rowStart, rowBound, one);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), 1);
    }

    rewriter.eraseOp(matmul);
  }

  // P6-C 批量收缩内核：[B,M,K]×[B,K,N]→[B,M,N] 改写为 forall(b) 网格
  // + 逐批 A1b 形态内核（M 行段 × N 列块、K 单链 FMA、寄存器 accumulator
  // 常驻——与 kernelize 同构，A/B/C 面板经 [b] 子视图取得）。批维互
  // 不依赖，forall 提供与卷积路径一致的 OpenMP 并行；K 归约保持在单
  // 批单 tile 内，累加顺序与逐批标量串行一致。
  void kernelizeBatchMatmul(IRRewriter& rewriter,
                            linalg::BatchMatmulOp batch) const {
    Value lhs = batch.getInputs()[0];
    Value rhs = batch.getInputs()[1];
    Value acc = batch.getOutputs().front();
    const auto lhsType = cast<MemRefType>(lhs.getType());
    const auto rhsType = cast<MemRefType>(rhs.getType());
    const auto accType = cast<MemRefType>(acc.getType());
    const int64_t batchCount = lhsType.getShape()[0];
    const int64_t rows = lhsType.getShape()[1];
    const int64_t depth = lhsType.getShape()[2];
    const int64_t columns = rhsType.getShape()[2];
    Type elementType = accType.getElementType();

    ImplicitLocOpBuilder builder(batch.getLoc(), rewriter);
    builder.setInsertionPoint(batch);

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

    // forall(b)：批维并行（对齐卷积路径 forall 的 OpenMP 语义）。
    SmallVector<OpFoldResult> batchBounds{builder.getIndexAttr(batchCount)};
    auto forall = builder.create<scf::ForallOp>(
      batch.getLoc(), batchBounds, ValueRange{}, std::nullopt);
    contract::annotateTile(
      forall.getOperation(),
      "batch_f32_mxn_fma",
      "batched_identity",
      "batched_row_major_kxn",
      "batched_identity",
      tileRows,
      accColumns,
      depth,
      "outer_batch+inner_simd",
      tailColumns > 0 || rows % tileRows != 0 ? "scalar_tail" : "none");
    contract::annotatePacking(forall.getOperation(), "unpacked", 1, 0, 0);
    contract::setInteger(
      forall.getOperation(), contract::kSimdChunk, accColumns);
    contract::setString(forall.getOperation(), contract::kFma, "vector.fma");
    builder.setInsertionPointToStart(forall.getBody());
    Value bIndex = forall.getInductionVar(0);

    // [b] 面板视图（静态偏移、零拷贝）：先 subview 取 [1,inner0,inner1]
    // 切片（批维偏移动态、stride 1），再 collapse 折掉批维；collapsed 类
    // 型由 computeCollapsedType 推断（保留 strided 布局）。
    auto slice2D = [&](Value panel, int64_t inner0, int64_t inner1) {
      SmallVector<OpFoldResult> offsets{
        bIndex, builder.getIndexAttr(0), builder.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1),
                                      builder.getIndexAttr(inner0),
                                      builder.getIndexAttr(inner1)};
      SmallVector<OpFoldResult> strides{builder.getIndexAttr(1),
                                        builder.getIndexAttr(1),
                                        builder.getIndexAttr(1)};
      auto sliced = builder.create<memref::SubViewOp>(
        batch.getLoc(), panel, offsets, sizes, strides);
      SmallVector<ReassociationIndices> groups{{0, 1}, {2}};
      auto collapsedType = memref::CollapseShapeOp::computeCollapsedType(
        cast<MemRefType>(sliced.getType()), groups);
      return builder.create<memref::CollapseShapeOp>(
        batch.getLoc(), collapsedType, sliced.getResult(), groups);
    };
    Value lhsPanel = slice2D(lhs, rows, depth);
    Value rhsPanel = slice2D(rhs, depth, columns);
    Value accPanel = slice2D(acc, rows, columns);

    // 以下与 kernelize 的行段 × 列块 × K 结构一致（A 直读 [b] 面板）。
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
          accPanel,
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
                                               rhsPanel,
                                               ValueRange{kIndex, columnStart},
                                               std::nullopt,
                                               SmallVector<bool>(1, true));
      SmallVector<Value> updated;
      updated.reserve(rowCount);
      for (int64_t i = 0; i < rowCount; ++i) {
        auto aScalar = builder.create<memref::LoadOp>(
          lhsPanel, ValueRange{rowIndices[i], kIndex});
        auto broadcast =
          builder.create<vector::BroadcastOp>(vectorType, aScalar);
        updated.push_back(builder.create<vector::FMAOp>(
          vectorType, broadcast, bRow, kLoop.getRegionIterArgs()[i]));
      }
      builder.create<scf::YieldOp>(updated);

      builder.setInsertionPointAfter(kLoop);
      for (int64_t i = 0; i < rowCount; ++i) {
        auto rowWrite = builder.create<vector::TransferWriteOp>(
          kLoop.getResult(i), accPanel, ValueRange{rowIndices[i], columnStart});
        rowWrite.setInBoundsAttr(
          builder.getBoolArrayAttr(SmallVector<bool>(1, true)));
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
    if (rows % tileRows) {
      auto rowStart = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowBound = builder.create<arith::ConstantIndexOp>(rows);
      auto rowLoop = builder.create<scf::ForOp>(rowStart, rowBound, one);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), 1);
    }

    rewriter.eraseOp(batch);
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
  // requant 内联求值：把 epilogue body 以 acc 标量为主实参克隆进当前
  // 插入点——block 参数 #0 接 i32 和，其余参数按常量下标从各自 memref
  // load（FuseQuantChain 保证这些是常量广播的逐张量 scale/bias）。
  Value emitRequantValue(ImplicitLocOpBuilder& builder,
                         linalg::GenericOp epilogue,
                         Value accValue) const {
    Block& block = epilogue->getRegion(0).front();
    IRMapping mapping;
    mapping.map(block.getArgument(0), accValue);
    SmallVector<Value> epilogueInputs;
    llvm::append_range(epilogueInputs, epilogue.getDpsInputs());
    for (auto [index, input] :
         llvm::enumerate(ArrayRef<Value>(epilogueInputs).drop_front())) {
      AffineMap map = epilogue.getIndexingMapsArray()[index + 1];
      SmallVector<Value> indices;
      for (AffineExpr result : map.getResults()) {
        indices.push_back(builder.create<arith::ConstantIndexOp>(
          cast<AffineConstantExpr>(result).getValue()));
      }
      mapping.map(block.getArgument(index + 1),
                  builder.create<memref::LoadOp>(input, indices));
    }
    for (Operation& operation : block.without_terminator()) {
      builder.clone(operation, mapping);
    }
    auto yield = cast<linalg::YieldOp>(block.getTerminator());
    return mapping.lookup(yield.getValues().front());
  }

  static bool hasContiguousK(Value value) {
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    return succeeded(cast<MemRefType>(value.getType())
                       .getStridesAndOffset(strides, offset)) &&
           strides.size() == 2 && strides.back() == 1;
  }

  static SmallVector<Value> emitVnniPartial(ImplicitLocOpBuilder& builder,
                                            ModuleOp module,
                                            Value lhs,
                                            Value rhs,
                                            ArrayRef<Value> rows,
                                            ArrayRef<Value> columns,
                                            int64_t depth,
                                            ArrayRef<Value> initial) {
    auto bytes = VectorType::get({32}, builder.getI8Type());
    auto words = VectorType::get({8}, builder.getI32Type());
    Value zeroVector = builder.create<arith::ConstantOp>(
      words, DenseElementsAttr::get(words, builder.getI32IntegerAttr(0)));
    Value signBytes = builder.create<arith::ConstantOp>(
      bytes, DenseElementsAttr::get(bytes, builder.getI8IntegerAttr(-128)));
    Value signWords = builder.create<vector::BitCastOp>(words, signBytes);
    const llvm::StringRef intrinsic = "llvm.x86.avx512.vpdpbusd.256";
    auto noMemory = LLVM::MemoryEffectsAttr::get(builder.getContext(),
                                                 LLVM::ModRefInfo::NoModRef,
                                                 LLVM::ModRefInfo::NoModRef,
                                                 LLVM::ModRefInfo::NoModRef);
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>(intrinsic)) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      moduleBuilder.setInsertionPointToStart(module.getBody());
      auto signature =
        LLVM::LLVMFunctionType::get(words, {words, words, words});
      auto declaration = moduleBuilder.create<LLVM::LLVMFuncOp>(
        builder.getLoc(), intrinsic, signature);
      declaration.setMemoryEffectsAttr(noMemory);
    }
    auto dot = [&](Value acc, Value a, Value b) -> Value {
      // Named calls implement the call interface needed by ownership-based
      // deallocation; the intrinsic takes only registers and accesses no
      // memory.
      auto call = builder.create<LLVM::CallOp>(
        words, builder.getStringAttr(intrinsic), ValueRange{acc, a, b});
      call.setMemoryEffectsAttr(noMemory);
      return call.getResult();
    };
    Value start = builder.create<arith::ConstantIndexOp>(0);
    Value bound = builder.create<arith::ConstantIndexOp>(depth / 32 * 32);
    Value step = builder.create<arith::ConstantIndexOp>(32);
    SmallVector<Value> seeds(initial.size(), zeroVector);
    auto loop = builder.create<scf::ForOp>(start, bound, step, seeds);
    builder.setInsertionPointToStart(loop.getBody());
    Value k = loop.getInductionVar();
    SmallVector<Value> aWords;
    for (Value row : rows) {
      Value loaded =
        builder.create<vector::LoadOp>(bytes, lhs, ValueRange{row, k});
      Value rebased = builder.create<arith::XOrIOp>(loaded, signBytes);
      aWords.push_back(builder.create<vector::BitCastOp>(words, rebased));
    }
    SmallVector<Value> bWords;
    SmallVector<Value> corrections;
    for (Value column : columns) {
      Value loaded =
        builder.create<vector::LoadOp>(bytes, rhs, ValueRange{column, k});
      Value packed = builder.create<vector::BitCastOp>(words, loaded);
      bWords.push_back(packed);
      // unsigned 128 * signed B; shared by every row in this tile.
      corrections.push_back(dot(zeroVector, signWords, packed));
    }
    SmallVector<Value> updated;
    for (size_t i = 0; i < rows.size(); ++i) {
      for (size_t j = 0; j < columns.size(); ++j) {
        Value sum = dot(loop.getRegionIterArgs()[(i * columns.size()) + j],
                        aWords[i],
                        bWords[j]);
        updated.push_back(builder.create<arith::SubIOp>(sum, corrections[j]));
      }
    }
    builder.create<scf::YieldOp>(updated);
    builder.setInsertionPointAfter(loop);
    SmallVector<Value> result;
    for (size_t i = 0; i < initial.size(); ++i) {
      Value sum = builder.create<vector::ReductionOp>(
        vector::CombiningKind::ADD, loop.getResult(i));
      result.push_back(builder.create<arith::AddIOp>(initial[i], sum));
    }
    return result;
  }

  void kernelizeInt8RowDot(IRRewriter& rewriter,
                           linalg::MatmulTransposeBOp matmul) const {
    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    Value acc = matmul.getOutputs().front();
    const auto lhsType = cast<MemRefType>(lhs.getType());
    const auto accType = cast<MemRefType>(acc.getType());
    const int64_t rows = lhsType.getShape()[0];
    const int64_t depth = lhsType.getShape()[1];
    const int64_t columns = accType.getShape()[1];
    const auto packSchema =
      matmul->getAttrOfType<StringAttr>(contract::kPackSchema);
    const bool packedInt8B =
      packSchema && packSchema.getValue() == contract::kInt8PanelPackSchema;
    const bool useVnni =
      int8Kernel == "vnni" &&
      (int8Target == "avx-vnni" || int8Target == "avx512-vnni") &&
      depth >= 32 && hasContiguousK(lhs) && hasContiguousK(rhs);
    memref::GetGlobalOp packedInt8Global;
    if (packedInt8B) {
      packedInt8Global = findPackedInt8Global(rhs);
      SmallVector<int64_t> rhsStrides;
      int64_t rhsOffset = 0;
      const auto rhsType = cast<MemRefType>(rhs.getType());
      if (!packedInt8Global ||
          failed(rhsType.getStridesAndOffset(rhsStrides, rhsOffset)) ||
          rhsStrides.size() != 2 || rhsStrides[1] != 1 ||
          rhsStrides[0] < depth || rhsStrides[0] % 64 != 0) {
        matmul.emitError(
          "cannot prove the INT8 K-padded row-panel global view");
        packingFailure = true;
        return;
      }
      contract::setBool(
        packedInt8Global.getOperation(), contract::kPackedWeight, true);
      for (StringRef attribute : {contract::kPackBytes,
                                  contract::kPackRawBytes,
                                  contract::kPackFactor,
                                  contract::kPackTileK}) {
        if (Attribute value = matmul->getAttr(attribute)) {
          packedInt8Global->setAttr(attribute, value);
        }
      }
      for (StringRef attribute : {contract::kPackSchema,
                                  contract::kPackRuntime,
                                  contract::kAlignment,
                                  contract::kWeightLayout}) {
        if (Attribute value = matmul->getAttr(attribute)) {
          packedInt8Global->setAttr(attribute, value);
        }
      }
    }

    // int8 requant epilogue 融合（P4 遗留 / parity-plan §3-P4 附注）：
    // matmul 的 i32 累加缓冲唯一用户是 requant generic（恒等主值 +
    // 常量广播 scale/bias 的 FuseQuantChain 产物）时，量化尾部内联进
    // 内核写回——K 归约在单 tile 内完整，i32 中间物化与独立全量 pass
    // 同时消失。数值逐位不变（body op 序/常量照抄，仅执行位置从
    // 独立循环移入写回点）。
    linalg::GenericOp requantEpilogue;
    if (acc.hasOneUse()) {
      if (auto user = dyn_cast<linalg::GenericOp>(*acc.getUsers().begin())) {
        if (isFusableRequantEpilogue(user, matmul)) {
          requantEpilogue = user;
        }
      }
    }

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

    if (auto forall = matmul->getParentOfType<scf::ForallOp>()) {
      contract::annotateTile(
        forall.getOperation(),
        useVnni ? "int8_vnni_row_dot" : "int8_row_dot",
        "identity",
        packedInt8B ? "panel_nk_kpad64" : "packed_nk",
        "identity",
        tileRows,
        accColumns,
        depth,
        "outer_tile+inner_simd",
        tailColumns > 0 || rows % tileRows != 0 || (useVnni && depth % 32 != 0)
          ? "scalar_tail"
          : "none");
      if (packedInt8B) {
        const auto packFactor =
          matmul->getAttrOfType<IntegerAttr>(contract::kPackFactor);
        const auto packBytes =
          matmul->getAttrOfType<IntegerAttr>(contract::kPackBytes);
        contract::annotatePacking(forall.getOperation(),
                                  "prepacked_B",
                                  packFactor ? packFactor.getInt() : 16,
                                  packBytes ? packBytes.getInt() : 0,
                                  0);
        for (StringRef attribute : {contract::kPackSchema,
                                    contract::kPackRawBytes,
                                    contract::kPackTileK,
                                    contract::kPackRuntime,
                                    contract::kAlignment,
                                    contract::kWeightLayout}) {
          if (Attribute value = matmul->getAttr(attribute)) {
            forall->setAttr(attribute, value);
          }
        }
      } else {
        contract::annotatePacking(
          forall.getOperation(), "prepacked_transpose_b", 1, 0, 0);
      }
      contract::setInteger(
        forall.getOperation(), contract::kSimdChunk, accColumns);
      contract::setString(
        forall.getOperation(),
        contract::kFma,
        useVnni ? "vpdpbusd_signed_correction" : "llvm_auto_vectorized_mac");
      if (useVnni) {
        Operation* kernel = forall.getOperation();
        contract::setString(kernel, contract::kInt8Isa, int8Target);
        contract::setString(kernel, contract::kInt8RequiredIsa, int8Target);
        contract::setString(kernel,
                            contract::kInt8Backend,
                            packedInt8B
                              ? "vpdpbusd_256_kpad64_row_nk_signed_mac"
                              : "vpdpbusd_256_u8s8_signed_mac");
        contract::setString(
          kernel, contract::kInt8Intrinsic, "llvm.x86.avx512.vpdpbusd.256");
        contract::setString(kernel,
                            contract::kInt8SignednessCorrection,
                            "xor_a_signbit_then_subtract_128_sum_b");
        contract::setInteger(
          kernel, contract::kInt8KAlignment, packedInt8B ? 64 : 32);
        contract::setInteger(kernel, contract::kInt8ReductionTail, depth % 32);
      } else if (int8Kernel == "vnni") {
        const StringRef reason = depth < 32 ? "reduction_k_lt_32"
                                 : !hasContiguousK(lhs) || !hasContiguousK(rhs)
                                   ? "non_contiguous_k"
                                   : "native_kernel_unavailable";
        contract::setString(forall.getOperation(), contract::kFallback, reason);
      }
      if (auto reason = matmul->getAttrOfType<StringAttr>(contract::kFallback);
          reason && !forall->hasAttr(contract::kFallback)) {
        contract::setString(
          forall.getOperation(), contract::kFallback, reason.getValue());
      }
    }

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

      Value kStart = zero;
      if (useVnni) {
        accumulators = emitVnniPartial(builder,
                                       matmul->getParentOfType<ModuleOp>(),
                                       lhs,
                                       rhs,
                                       rowIndices,
                                       columnIndices,
                                       depth,
                                       accumulators);
        kStart = builder.create<arith::ConstantIndexOp>(depth / 32 * 32);
      }
      auto depthBound = builder.create<arith::ConstantIndexOp>(depth);
      auto kLoop =
        builder.create<scf::ForOp>(kStart, depthBound, one, accumulators);
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
          Value result = kLoop.getResult(flatIndex(i, j));
          if (requantEpilogue) {
            // requant 内联：主值 = 本 tile 的 i32 和；其余输入按常量
            // 下标 load（循环不变，逐 (i,j) 重复 load 会被 LICM 折叠）。
            Value quantized =
              emitRequantValue(builder, requantEpilogue, result);
            builder.create<memref::StoreOp>(
              quantized,
              requantEpilogue.getOutputs().front(),
              ValueRange{rowIndices[i], columnIndices[j]});
          } else {
            builder.create<memref::StoreOp>(
              result, acc, ValueRange{rowIndices[i], columnIndices[j]});
          }
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
    if (rows % tileRows) {
      auto rowStart = builder.create<arith::ConstantIndexOp>(fullRowExtent);
      auto rowBound = builder.create<arith::ConstantIndexOp>(rows);
      auto rowLoop = builder.create<scf::ForOp>(rowStart, rowBound, one);
      builder.setInsertionPointToStart(rowLoop.getBody());
      emitRowBlock(rowLoop.getInductionVar(), 1);
    }

    // 融合的 epilogue 全量被内核写回覆盖，独立 pass 删除。
    if (requantEpilogue) {
      rewriter.eraseOp(requantEpilogue);
    }
    rewriter.eraseOp(matmul);
  }
};

}  // namespace

}  // namespace mlir::ncnn
