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

#endif
