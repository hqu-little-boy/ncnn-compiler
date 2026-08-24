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
