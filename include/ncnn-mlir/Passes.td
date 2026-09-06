#ifndef NCNN_MLIR_PASSES_TD
#define NCNN_MLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertNCNNModelToFuncPass
    : Pass<"convert-ncnn-model-to-func", "mlir::ModuleOp"> {
  let summary = "Build a function boundary for each ncnn.model";
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect"];
}

def ConvertNCNNToTosaPass
    : Pass<"convert-ncnn-to-tosa", "mlir::ModuleOp"> {
  let summary = "Convert the SqueezeNet ncnn op set to TOSA";
  let dependentDialects = [
    "arith::ArithDialect", "func::FuncDialect", "linalg::LinalgDialect",
    "math::MathDialect", "tensor::TensorDialect", "tosa::TosaDialect"
  ];
}

def BufferizeNCNNPass : Pass<"bufferize-ncnn", "mlir::ModuleOp"> {
  let summary = "One-shot bufferize with copies represented as Linalg operations";
  let dependentDialects = [
    "bufferization::BufferizationDialect", "func::FuncDialect",
    "linalg::LinalgDialect", "memref::MemRefDialect"
  ];
}

def RewriteLinalgCopiesPass : Pass<"rewrite-linalg-copies", "mlir::ModuleOp"> {
  let summary =
    "Rewrite identity linalg.generic copies into sequential copy loops";
  let dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"];
}

def GenerateCAPIPass : Pass<"generate-ncnn-c-api", "mlir::ModuleOp"> {
  let summary = "Prepare a typed model-specific C ABI for an ncnn entry";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"exportName", "export-name", "std::string", "",
           "Exported C function name">,
    Option<"manifestPath", "manifest-path", "std::string", "",
           "Optional JSON ABI manifest output path">
  ];
}

def FinalizeCAPIPass : Pass<"finalize-ncnn-c-api", "mlir::ModuleOp"> {
  let summary = "Materialize a prepared typed ncnn wrapper in LLVM dialect";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def NormalizeNCNNPass : Pass<"normalize-ncnn", "mlir::ModuleOp"> {
  let summary = "Normalize target-independent ncnn semantics";
}

def FoldLinalgConstantTransposePass
    : Pass<"fold-linalg-constant-transpose", "mlir::ModuleOp"> {
  let summary =
    "Fold Linalg transposes of constant tensors into materialized constants";
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect"];
}

def FoldNCNNBatchNormPass
    : Pass<"fold-ncnn-batchnorm", "mlir::ModuleOp"> {
  let summary = "Fold constant BatchNorm layers into preceding convolutions";
  let dependentDialects = ["arith::ArithDialect"];
}

def StrategyNCNNPass
    : Pass<"strategy-ncnn", "mlir::ModuleOp"> {
  let summary =
    "Rewrite convolutions into matmul forms ahead of vectorization";
  let description = [{
    Operator-shape strategy layer (A1). Convolutions whose window indexing
    maps cannot reach vector.contract are rewritten into matmul forms:
    1x1 stride-1 convs become collapsed-view matmuls; other kxk convs with
    static spatial dims and an ncnn-style prefer-gemm verdict become
    im2col+matmul. A single-user identity elementwise epilogue is lifted
    into the collapsed 2-D domain so activations stay attached to the
    matmul main loop. Depthwise convolutions lower to a dedicated op and
    never enter this pass. Winograd is a reserved switch only.
  }];
  let dependentDialects = [
    "arith::ArithDialect", "linalg::LinalgDialect",
    "math::MathDialect", "tensor::TensorDialect"
  ];
  let options = [
    Option<"strategy", "strategy", "std::string", /*default=*/"\"auto\"",
           "Convolution lowering strategy: auto, gemm, conv, winograd">,
    Option<"gemmL2Bytes", "conv-gemm-l2-bytes", "int64_t",
           /*default=*/"524288",
           "L2 cache byte budget for the ncnn-style prefer-gemm heuristic">
  ];
}

def FuseQuantChainNCNNPass
    : Pass<"fuse-quant-chain-ncnn", "mlir::ModuleOp"> {
  let summary = "Fuse quantize/dequantize elementwise chains into one generic";
  let description = [{
    Int8 lowering emits the per-layer requant/dequant tail as a chain of
    separate elementwise ops on the full tensor (sitofp map, scale mul,
    bias add, activation, requantize quantize). Each op used to stay an
    independent full-tensor pass with mixed element types that the row
    vectorizers reject. This pass inlines every single-use all-parallel
    elementwise producer (linalg.generic or linalg.map) into its
    elementwise consumer, collapsing the whole chain into one generic
    whose body is pure arith/math and whose operands keep their affine
    maps. Op order and constants are preserved, so numerics are
    bit-identical to the chained form
    (docs/ncnn-performance-parity-plan.md §3-P4).
  }];
  let dependentDialects = [
    "arith::ArithDialect", "linalg::LinalgDialect", "math::MathDialect"
  ];
}

def FuseLinalgEpiloguePass
    : Pass<"fuse-linalg-epilogue", "mlir::ModuleOp"> {
  let summary =
    "Fuse elementwise epilogues into tiled convolution and matmul loops";
  let dependentDialects = [
    "arith::ArithDialect", "linalg::LinalgDialect", "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
  let options = [
    Option<"tileWidth", "tile-width", "int64_t", /*default=*/"16",
           "Output column tiling width for fused producers">
  ];
}

def VectorizeNCNNPass
    : Pass<"vectorize-ncnn", "mlir::ModuleOp"> {
  let summary = "Vectorize Linalg computations to the target vector ABI";
  let dependentDialects = [
    "arith::ArithDialect", "linalg::LinalgDialect", "scf::SCFDialect",
    "tensor::TensorDialect", "vector::VectorDialect"
  ];
  let options = [
    Option<"lanes", "lanes", "unsigned", /*default=*/"0",
           "Target SIMD lane count enabling vectorization; 0 disables">,
    Option<"scalable", "scalable", "bool", /*default=*/"false",
           "Reserved for scalable-vector emission (SVE/RVV)">,
    Option<"unroll", "unroll", "bool", /*default=*/"false",
           "Reserved for lane-width contraction splitting">
  ];
}

def LowerVectorTransfersNCNNPass
    : Pass<"lower-vector-transfers-ncnn", "mlir::ModuleOp"> {
  let summary =
    "Normalize vector transfers into VectorToLLVM-lowerable form";
  let dependentDialects = ["memref::MemRefDialect", "vector::VectorDialect"];
  let options = [
    Option<"targetBitwidth", "target-bitwidth", "int64_t",
           /*default=*/"512",
           "Flattening cap for n-D transfer bitwidth">
  ];
}

def LowerVectorMathNCNNPass
    : Pass<"lower-vector-math-ncnn", "mlir::ModuleOp"> {
  let summary = "Lower vector math ops to vectorized math library calls";
  let description = [{
    Replaces the per-lane scalarization that convert-math-to-libm applies
    to vector math ops with whole-vector calls into a vector math backend.
    Fixed-shape f32 vectors matching the target ABI lane count become
    direct library calls; lane-multiple rows within the unroll budget
    expand into slice/call/insert chains. Odd widths, scalable vectors,
    non-f32 element types, and ops without a backend vector variant stay
    untouched for the scalar MathToLibm fallback.
  }];
  let dependentDialects = [
    "arith::ArithDialect", "func::FuncDialect", "math::MathDialect",
    "vector::VectorDialect"
  ];
  let options = [
    Option<"backend", "backend", "std::string", /*default=*/"\"none\"",
           "Vector math backend: none, libmvec, sleef">,
    Option<"lanes", "lanes", "unsigned", /*default=*/"0",
           "Target SIMD lane count for chunking decisions">,
    Option<"abiFragment", "abi", "std::string", /*default=*/"\"\"",
           "Backend ABI fragment (libmvec ISA prefix such as _ZGVdN8, "
           "or SLEEF accuracy/dispatch tag)">
  ];
}

def MatmulKernelNCNNPass : Pass<"matmul-kernel-ncnn", "mlir::ModuleOp"> {
  let summary = "Emit explicit SIMD kernels for forall-tiled matmuls";
  let description = [{
    Rewrites static memref linalg.matmul ops inside scf.forall regions
    (the tiles produced by tile-matmul-forall) into explicit vector FMA
    loop nests: the K loop streams one B row vector per iteration and a
    group of M rows keeps that many vector accumulators resident in
    registers (MxN register blocking), each accumulating
    broadcast(A[m+i,k]) * B-row via an independent FMA chain. Multiple
    resident rows both reuse each B row and break the single FMA
    dependency chain that otherwise pins the K loop issue rate to the
    FMA latency. Without this pass the generic matmul expansion
    degenerates to scalar loops, which dominates end-to-end latency on
    wide PP-family models.
  }];
  let options = [
    Option<"matmulMRows", "matmul-m-rows", "int64_t", /*default=*/"4",
           "M-direction register tile rows of the matmul kernel (each row "
           "keeps one vector accumulator across the K loop); clamped to the "
           "M extent and the accumulator float budget; 1 restores the "
           "legacy single-row kernel for A/B comparison">,
    Option<"matmulAccColumns", "matmul-acc-columns", "int64_t",
           /*default=*/"16",
           "Vector accumulator width in columns for the matmul kernel; "
           "N is swept in blocks of this width (remainder block uses a "
           "narrower vector); clamped to the N extent">,
    Option<"rowChunkLanes", "row-chunk-lanes", "unsigned", /*default=*/"8",
           "Lane budget for the row-generic vectorization chunks (0 keeps "
           "the historical whole-row form). Elements of up to 32 bits are "
           "chunked at 4x this budget (still within the vector-math "
           "splittable ABI; 64-bit elements at the budget itself), which "
           "pins math ops to vector library calls and decouples LLVM "
           "legalization size from the innermost extent; whole-row forms "
           "were the compile-explosion mechanism on wide models "
           "(docs/ncnn-performance-parity-plan.md §3-P2).">,
    Option<"matmulI8Rows", "matmul-i8-rows", "int64_t", /*default=*/"2",
           "M-direction register tile rows of the int8 row-dot kernel "
           "(each (row, column) pair keeps one vector accumulator of K-pair "
           "partial sums); clamped to the M extent and the int8 accumulator "
           "budget">,
    Option<"matmulI8AccColumns", "matmul-i8-acc-columns", "int64_t",
           /*default=*/"4",
           "Column-block width (outputs per tile) of the int8 row-dot "
           "kernel; each output keeps a dedicated K-vector accumulator, "
           "so tileRows x accColumns is bounded by the accumulator budget">
  ];
  let dependentDialects = [
    "arith::ArithDialect", "linalg::LinalgDialect", "memref::MemRefDialect",
    "scf::SCFDialect", "vector::VectorDialect"
  ];
}

def VerifyBufferizedModelPass
    : Pass<"verify-bufferized-model", "mlir::ModuleOp"> {
  let summary = "Verify the bufferized ncnn entry point and its buffer ownership";
  let dependentDialects = ["memref::MemRefDialect"];
}

def VerifyModelShapeContractsPass
    : Pass<"verify-ncnn-model-shape-contracts", "mlir::ModuleOp"> {
  let summary = "Verify final static and dynamic ncnn model shape contracts";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect"];
}

def VerifyNoNCNNOpsPass : Pass<"verify-no-ncnn-ops", "mlir::ModuleOp"> {
  let summary = "Fail if any ncnn dialect operation remains";
}

def VerifyNoTosaOpsPass : Pass<"verify-no-tosa-ops", "mlir::ModuleOp"> {
  let summary = "Fail if any TOSA dialect operation remains";
}

def VerifyNoSCFForallPass : Pass<"verify-no-scf-forall", "mlir::ModuleOp"> {
  let summary =
    "Fail if any SCF forall or parallel loop remains after parallel lowering";
  let dependentDialects = ["scf::SCFDialect"];
}

def TileMatmulForallPass : Pass<"tile-matmul-forall", "mlir::ModuleOp"> {
  let summary =
    "Tile top-level static matmuls into scf.forall grids along M/N";
  let dependentDialects = [
    "arith::ArithDialect", "linalg::LinalgDialect", "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
  let options = [
    Option<"tileRows", "tile-rows", "int64_t", /*default=*/"32",
           "Row (M) tile size; K contraction is never tiled">,
    Option<"tileColumns", "tile-columns", "int64_t", /*default=*/"32",
           "Column (N) tile size">
  ];
}

def ForallizeDisjointTileLoopsPass
    : Pass<"forallize-disjoint-tile-loops", "mlir::ModuleOp"> {
  let summary =
    "Rewrite disjoint-window tiled scf.for loops into shared_outs scf.forall";
  let dependentDialects = ["arith::ArithDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}


#endif
