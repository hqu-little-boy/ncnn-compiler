#ifndef NCNN_ATTRS
#define NCNN_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "ncnn-mlir/Dialect/NCNN/IR/NCNNDialect.td"

class NCNN_Attr<string name, string attrMnemonic>
  : AttrDef<NCNN_Dialect, name> {
  let mnemonic = attrMnemonic;
}

def NCNN_DimConstraintAttr : NCNN_Attr<"DimConstraint", "dim_constraint"> {
  let summary = "Runtime constraint on one input tensor dimension";
  let parameters = (ins
    "uint32_t":$input,
    "uint32_t":$dim,
    "int64_t":$min,
    "int64_t":$multiple_of
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif  // NCNN_ATTRS
