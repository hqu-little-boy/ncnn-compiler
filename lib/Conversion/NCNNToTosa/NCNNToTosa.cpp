// NCNN -> TOSA 转换 pass 壳。
//
// 职责
//   只做三件事：配置 TypeConverter（rank-3 <-> NHWC 的双向
//   materialization）、按族调用 populate*Patterns 注册 pattern、设定
//   ConversionTarget 的合法/非法集合。pattern 本体在各族 TU 里。
//
// 不变量
//   * 各族 pattern 匹配的 op 名互不重叠，注册顺序不选择竞争 pattern；
//   * materialization 只做 CHW<->NHWC 布局转换，不改数值；
//   * 未登记的 ncnn.* 算子名一律按 illegal 处理（addIllegalOp 名单）。
//
// 顺序依赖
//   * 必须在 NormalizeNCNN 之后、BufferizeNCNN 之前；
//   * 依赖的 dialect 见 Passes.td 的 dependentDialects。
//
// 明确不做
//   * 不承载任何具体 lowering 逻辑；
//   * 不做 pattern benefit 调优（全部默认 benefit）。

#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosa.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ncnn-mlir/Conversion/NCNNToTosa/NCNNToTosaPatterns.hpp"
#include "ncnn-mlir/Conversion/NCNNToTosa/TosaLoweringUtils.hpp"
#include "ncnn-mlir/Dialect/NCNN/IR/NCNNOps.hpp"
#include "ncnn-mlir/Support/ConstantFold.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/ModelLedger.hpp"
#include "ncnn-mlir/Support/Precision.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_CONVERTNCNNTOTOSAPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

// Shared lowering helpers used by the type converter.
using tosa_lowering::convertCHWToNHWC;
using tosa_lowering::convertNHWCToCHW;
using tosa_lowering::getNHWCType;

class ConvertNCNNToTosaPass final
  : public impl::ConvertNCNNToTosaPassBase<ConvertNCNNToTosaPass> {
 public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (!module.getOps<ModelOp>().empty()) {
      module.emitError(
        "convert-ncnn-to-tosa requires ncnn.model to be "
        "converted to func.func first");
      signalPassFailure();
      return;
    }

    MLIRContext* context = module.getContext();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return type.getRank() == 3 ? getNHWCType(type) : type;
    });
    typeConverter.addTargetMaterialization([](OpBuilder& builder,
                                              RankedTensorType resultType,
                                              ValueRange inputs,
                                              Location location,
                                              Type) -> Value {
      if (inputs.size() != 1 || resultType.getRank() != 4) {
        return {};
      }
      auto inputType = dyn_cast<RankedTensorType>(inputs.front().getType());
      if (!inputType || inputType.getRank() != 3 ||
          getNHWCType(inputType) != resultType) {
        return {};
      }
      return convertCHWToNHWC(builder, location, inputs.front());
    });
    typeConverter.addSourceMaterialization([](OpBuilder& builder,
                                              RankedTensorType resultType,
                                              ValueRange inputs,
                                              Location location) -> Value {
      if (inputs.size() != 1 || resultType.getRank() != 3) {
        return {};
      }
      auto inputType = dyn_cast<RankedTensorType>(inputs.front().getType());
      if (!inputType || inputType.getRank() != 4 ||
          getNHWCType(resultType) != inputType) {
        return {};
      }
      return convertNHWCToCHW(builder, location, inputs.front(), resultType);
    });

    RewritePatternSet patterns(context);
    populateConvToTosaPatterns(patterns, typeConverter, context);
    populateDeconvToTosaPatterns(patterns, typeConverter, context);
    populatePoolingToTosaPatterns(patterns, typeConverter, context);
    populateAttentionToTosaPatterns(patterns, typeConverter, context);
    populateSDPAToTosaPatterns(patterns, typeConverter, context);
    populateQuantToTosaPatterns(patterns, typeConverter, context);
    populateGemmToTosaPatterns(patterns, typeConverter, context);
    populateElementwiseToTosaPatterns(patterns, typeConverter, context);
    populateDetectionToTosaPatterns(patterns, typeConverter, context);
    populateLayoutToTosaPatterns(patterns, typeConverter, context);
    populateReduceToTosaPatterns(patterns, typeConverter, context);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect,
                           func::FuncDialect,
                           linalg::LinalgDialect,
                           math::MathDialect,
                           scf::SCFDialect,
                           tensor::TensorDialect,
                           tosa::TosaDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<ConvolutionOp>();
    target.addDynamicallyLegalOp<PoolingOp>([](PoolingOp operation) {
      auto inputType = cast<RankedTensorType>(operation.getInput().getType());
      return operation.getKind() == static_cast<int64_t>(PoolKind::Average) &&
             operation.getIncludePad() && inputType.hasStaticShape();
    });
    target.addIllegalOp<ReluOp,
                        QuantizeOp,
                        DequantizeOp,
                        RequantizeOp,
                        CastOp,
                        ZeroPointCastOp,
                        DetectionOutputOp,
                        SplitOp,
                        ConcatOp,
                        DropoutOp,
                        SoftmaxOp,
                        ShuffleChannelOp,
                        SliceOp,
                        ReductionOp,
                        SwishOp,
                        TanHOp,
                        LayerNormOp,
                        EmbedOp,
                        MultiHeadAttentionOp,
                        SDPAOp,
                        GridSampleOp,
                        GELUOp,
                        BatchNormOp,
                        PermuteOp,
                        GemmOp>();
    for (StringRef name : {contract::kLayerHardSigmoid,
                           contract::kLayerHardSwish,
                           contract::kLayerConvolutionDepthwise,
                           contract::kLayerReshape,
                           contract::kLayerSqueeze,
                           contract::kLayerExpandDims,
                           contract::kLayerBinary,
                           contract::kLayerUnary,
                           contract::kLayerInnerProduct,
                           contract::kLayerPadding,
                           contract::kLayerInterp,
                           contract::kLayerDeconvolution,
                           contract::kLayerSigmoid}) {
      target.addIllegalOp(OperationName(name, context));
    }

    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPartialConversion(module, target, frozen))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
