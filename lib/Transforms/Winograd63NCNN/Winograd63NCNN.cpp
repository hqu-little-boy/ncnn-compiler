// Winograd F(6,3) 改写的 IR 发射。矩阵、判据与数值契约说明见
// ncnn-mlir/Support/Winograd63.hpp；本 TU 只负责 IR 构造。
//
// 职责
//   把 conv2d_nhwc_hwcf 改写为 pad → 两段输入变换 → batch_matmul →
//   两段输出变换 → 裁剪 + bias 逐元素加（全 tensor 层，向量化/并行化由
//   既有后续 pass 接管）。
//
// 不变量
//   * 只吃 3×3 s1 d1、N=1、全静态 f32 的卷积；判据与 StrategyNCNN 共用
//     winograd63::eligible，避免「选了却不改写」；
//   * 临时缓冲共享策略层的单张量元素预算（由调用方传入），超出即放弃
//     且 convolution 保持原样（改写事务性）；
//   * 权重变换用 APFloat + 显式 RN-even 舍入，不得改成 float 运算
//     （-ffp-contract 会改变权重常量比特）。
//
// 顺序依赖
//   * 由 StrategyNCNN 的 winograd 分派调用；
//   * 中央 linalg.batch_matmul 交给 MatmulKernelNCNN 的
//     kernelizeBatchMatmul 接管。
//
// 明确不做
//   * 不做 epilogue 提升（bias 直接逐元素补加）；
//   * 不做 int8、不做批维 > 1——这些由 rewriteConvolution 的既有路径承担。

#include "ncnn-mlir/Transforms/Winograd63NCNN/Winograd63NCNN.hpp"

#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "ncnn-mlir/Support/CheckedMath.hpp"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/Winograd63.hpp"

namespace mlir::ncnn::winograd63 {

// Bounds proofs all run through one implementation (see CheckedMath.hpp).
using ncnn_mlir::checkedAdd;
using ncnn_mlir::checkedMul;
using ncnn_mlir::checkedProduct;

// 编译期权重变换：G·w·Gᵀ 折叠 [64, OC, IC] 常量（ncnn create_pipeline
// 预变换的编译期等价物）。输入 [3,3,IC,OC]（MLIR 布局，kh-major），
// 输出批维最外与 batch_matmul 的 A 面板一致（行内 oc*IC+ic）。
DenseFPElementsAttr transformWeight(RankedTensorType weightType,
                                    DenseFPElementsAttr weights) {
  const int64_t inputChannels = weightType.getShape()[2];
  const int64_t outputChannels = weightType.getShape()[3];
  const int64_t depth = inputChannels * outputChannels;
  FloatType elementType = cast<FloatType>(weightType.getElementType());
  const llvm::fltSemantics& semantics = elementType.getFloatSemantics();
  SmallVector<APFloat> transformed(kBatch * depth, APFloat(semantics));
  auto values = weights.getValues<APFloat>();
  // 先行段（kw 方向）：W1[kh][jp][ic][oc] = Σ_kw G[jp][kw]·w[kh][kw][ic][oc]
  SmallVector<APFloat> stage1(3 * kAlpha * depth, APFloat(semantics));
  auto at1 = [&](int64_t kh, int64_t jp, int64_t index) -> APFloat& {
    return stage1[(((kh * kAlpha) + jp) * depth) + index];
  };
  int64_t sourceIndex = 0;
  for (int64_t kh = 0; kh < 3; ++kh) {
    for (int64_t kw = 0; kw < 3; ++kw) {
      for (int64_t ic = 0; ic < inputChannels; ++ic) {
        for (int64_t oc = 0; oc < outputChannels; ++oc) {
          const APFloat& value = values[sourceIndex++];
          for (int64_t jp = 0; jp < kAlpha; ++jp) {
            const float coefficient = kG[jp][kw];
            if (coefficient != 0.0F) {
              APFloat product(value);
              product.multiply(APFloat(coefficient),
                               llvm::APFloat::rmNearestTiesToEven);
              APFloat& slot = at1(kh, jp, (ic * outputChannels) + oc);
              slot.add(product, llvm::APFloat::rmNearestTiesToEven);
            }
          }
        }
      }
    }
  }
  // 再列段（kh 方向）：V[i·8+jp][oc][ic] = Σ_kh G[i][kh]·W1[kh][jp][ic][oc]
  for (int64_t kh = 0; kh < 3; ++kh) {
    for (int64_t jp = 0; jp < kAlpha; ++jp) {
      for (int64_t ic = 0; ic < inputChannels; ++ic) {
        for (int64_t oc = 0; oc < outputChannels; ++oc) {
          const APFloat& value = at1(kh, jp, (ic * outputChannels) + oc);
          for (int64_t i = 0; i < kAlpha; ++i) {
            const float coefficient = kG[i][kh];
            if (coefficient != 0.0F) {
              APFloat product(value);
              product.multiply(APFloat(coefficient),
                               llvm::APFloat::rmNearestTiesToEven);
              APFloat& slot = transformed[(((i * kAlpha) + jp) * depth) +
                                          ((oc * inputChannels) + ic)];
              slot.add(product, llvm::APFloat::rmNearestTiesToEven);
            }
          }
        }
      }
    }
  }
  auto transformedType = RankedTensorType::get(
    {kBatch, outputChannels, inputChannels}, weightType.getElementType());
  return DenseFPElementsAttr::get(transformedType, transformed);
}

// P7 Winograd F(6,3) 改写。IR 结构（全部 tensor 层，向量化/并行化由
// 既有后续 pass 接管）：
//   1. pad 图像到 tile 网格 [1, TH·6+2, TW·6+2, IC]（tile 对齐余量 +
//      右/下各 2 的变换窗口越界；输入已含 TosaToLinalg 显式 pad）；
//   2. 输入变换两段 generic（8-tap 全展开直线 body）：U = Bᵀ·(d·B)，
//      iterator (th, tw, i, jp, c) 全 parallel、c 最内——归约形态每内
//      层迭代只有 1 MAC 无法向量化（实测 vmovss 16k 条、resnet18
//      winograd 变体 129ms vs 基线 32ms），展开后 8 次 mul+add 由
//      clang auto-vec 命中；
//   3. 中心 linalg.batch_matmul [64,OC,IC]×[64,IC,T]→[64,OC,T]——
//      命中 P6-C kernelizeBatchMatmul 的 forall(b)+A1b 形态内核
//      （复用 P3 M×N 寄存器分块微内核）；
//   4. 输出变换两段 generic（同展开形态），[TH,6,TW,6,OC] 交错空间
//      布局（collapse {{0,1},{2,3},{4}} 直接折出 [TH·6,TW·6,OC]）；
//   5. 折回裁剪到 [1,OH,OW,OC]，与 conv init（逐通道 bias）逐元素加。
// 数值：f32 全链，相对 ncnn 直接卷积 ~1e-3 量级（F(6,3) 有理插值点），
// 数值契约由启用侧的 golden 预算承担（默认不启用，见 dispatch）。
bool rewriteWinograd(RewriterBase& rewriter,
                     linalg::Conv2DNhwcHwcfOp convolution,
                     int64_t maxElements) {
  auto imageType =
    dyn_cast<RankedTensorType>(convolution.getInputs()[0].getType());
  auto weightType =
    dyn_cast<RankedTensorType>(convolution.getInputs()[1].getType());
  auto resultType =
    dyn_cast<RankedTensorType>(convolution.getResult(0).getType());
  auto initType = dyn_cast<RankedTensorType>(
    convolution.getDpsInitOperand(0)->get().getType());
  if (!imageType || !weightType || !resultType || !initType ||
      imageType.getRank() != 4 || weightType.getRank() != 4 ||
      resultType.getRank() != 4 || initType.getRank() != 4 ||
      !imageType.hasStaticShape() || !resultType.hasStaticShape() ||
      !initType.hasStaticShape() || !weightType.hasStaticShape() ||
      !imageType.getElementType().isF32() ||
      !weightType.getElementType().isF32() ||
      !resultType.getElementType().isF32() || initType != resultType) {
    return false;
  }
  const ArrayRef<int64_t> imageShape = imageType.getShape();
  const ArrayRef<int64_t> resultShape = resultType.getShape();
  const int64_t inputChannels = weightType.getShape()[2];
  const int64_t outputChannels = weightType.getShape()[3];
  const int64_t batches = imageShape[0];
  const int64_t outputHeight = resultShape[1];
  const int64_t outputWidth = resultShape[2];
  const auto strides = convolution.getStrides().getValues<int64_t>();
  const auto dilations = convolution.getDilations().getValues<int64_t>();
  if (batches != 1 || inputChannels < 1 || outputChannels < 1 ||
      imageShape[3] != inputChannels || resultShape[0] != 1 ||
      resultShape[3] != outputChannels || strides.size() != 2 ||
      dilations.size() != 2 || strides[0] != 1 || strides[1] != 1 ||
      dilations[0] != 1 || dilations[1] != 1) {
    // tile 域以 N=1 折叠展开；批维实例回退既有路径。
    return false;
  }
  auto weightConstant =
    dyn_cast<arith::ConstantOp>(convolution.getInputs()[1].getDefiningOp());
  auto weightElements =
    weightConstant
      ? dyn_cast<DenseFPElementsAttr>(weightConstant.getValueAttr())
      : nullptr;
  if (!weightElements || !weightElements.getElementType().isF32()) {
    return false;
  }
  contract::annotateConvContract(convolution,
                                 "winograd",
                                 3,
                                 3,
                                 strides[0],
                                 strides[1],
                                 dilations[0],
                                 dilations[1],
                                 inputChannels,
                                 outputChannels);

  auto roundedHeightProof = checkedAdd(outputHeight, kTile - 1);
  auto roundedWidthProof = checkedAdd(outputWidth, kTile - 1);
  if (failed(roundedHeightProof) || failed(roundedWidthProof)) {
    return false;
  }
  const int64_t tileRows = *roundedHeightProof / kTile;
  const int64_t tileColumns = *roundedWidthProof / kTile;
  auto tileHeightProof = checkedMul(tileRows, kTile);
  auto tileWidthProof = checkedMul(tileColumns, kTile);
  if (failed(tileHeightProof) || failed(tileWidthProof)) {
    return false;
  }
  auto paddedHeightProof = checkedAdd(*tileHeightProof, 2);
  auto paddedWidthProof = checkedAdd(*tileWidthProof, 2);
  auto tilesProof = checkedMul(tileRows, tileColumns);
  if (failed(paddedHeightProof) || failed(paddedWidthProof) ||
      failed(tilesProof)) {
    return false;
  }
  const int64_t paddedHeight = *paddedHeightProof;
  const int64_t paddedWidth = *paddedWidthProof;
  const int64_t tiles = *tilesProof;
  // Winograd temporary buffers share the per-tensor strategy budget.
  auto safeShape = [maxElements](ArrayRef<int64_t> shape) {
    return succeeded(checkedProduct(shape, maxElements));
  };
  if (!safeShape(
        ArrayRef<int64_t>{1, paddedHeight, paddedWidth, inputChannels}) ||
      !safeShape(ArrayRef<int64_t>{
        tileRows, tileColumns, kAlpha, kAlpha, inputChannels}) ||
      !safeShape(ArrayRef<int64_t>{tiles, kBatch, inputChannels}) ||
      !safeShape(ArrayRef<int64_t>{kBatch, inputChannels, tiles}) ||
      !safeShape(ArrayRef<int64_t>{kBatch, outputChannels, inputChannels}) ||
      !safeShape(ArrayRef<int64_t>{kBatch, outputChannels, tiles}) ||
      !safeShape(ArrayRef<int64_t>{tiles, outputChannels, kBatch}) ||
      !safeShape(ArrayRef<int64_t>{
        tileRows, tileColumns, outputChannels, kAlpha, kAlpha}) ||
      !safeShape(ArrayRef<int64_t>{
        tileRows, kTile, tileColumns, kTile, outputChannels})) {
    return false;
  }
  const int64_t inputHeight = imageShape[1];
  const int64_t inputWidth = imageShape[2];
  if (paddedHeight < inputHeight || paddedWidth < inputWidth) {
    return false;
  }

  Location location = convolution.getLoc();
  rewriter.setInsertionPoint(convolution);
  MLIRContext* context = rewriter.getContext();
  const auto elementType = imageType.getElementType();
  auto zeroScalar = [&]() {
    return rewriter
      .create<arith::ConstantOp>(location,
                                 rewriter.getFloatAttr(elementType, 0.0))
      .getResult();
  };
  auto dim = [&](int64_t position) {
    return getAffineDimExpr(position, context);
  };
  auto zeroInit = [&](ArrayRef<int64_t> shape) {
    return rewriter
      .create<linalg::FillOp>(location,
                              ValueRange{zeroScalar()},
                              ValueRange{rewriter.create<tensor::EmptyOp>(
                                location, shape, elementType)})
      .getResult(0);
  };

  // 1. pad 到 tile 网格。
  auto pad = tensor::PadOp::create(
    rewriter,
    location,
    RankedTensorType::get({1, paddedHeight, paddedWidth, inputChannels},
                          elementType),
    convolution.getInputs()[0],
    SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(0)),
    SmallVector<OpFoldResult>{rewriter.getIndexAttr(0),
                              rewriter.getIndexAttr(paddedHeight - inputHeight),
                              rewriter.getIndexAttr(paddedWidth - inputWidth),
                              rewriter.getIndexAttr(0)},
    zeroScalar());
  // 批维折掉变 [PH, PW, IC]（纯视图），供 affine 投影消费。
  Value inputRows = rewriter.create<tensor::CollapseShapeOp>(
    location,
    RankedTensorType::get({paddedHeight, paddedWidth, inputChannels},
                          elementType),
    pad,
    SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}});

  // 2. 输入变换两段，8-tap 全展开直线 body（Σ_k input_k·coeff[k]，
  //    系数为 body 标量字面常量），iterator 全 parallel、c 最内维。
  //    发射器 transformRows：对输出行 r 逐行发射一个 generic——行维
  //    （source 的 rowDim 维）以常量 row 出现在所有 tap 投影中，从迭
  //    代空间完全移除；K 个 tap 各占一个 operand 槽（linalg generic
  //    one-map-per-operand，同一 source 重复 K 次），输出域 = 去掉
  //    rowDim 维的 outputDomain，assemble 时 insert_slice 拼回。
  //    rowTapMaps(row) 返回 K 个 4 维 map（迭代器 = outputDomain 去
  //    rowDim 维，c 最内）；输出的恒等投影由本发射器补齐。
  auto transformRows =
    [&](Value source,
        ArrayRef<int64_t> outputDomain,
        int64_t rows,
        unsigned rowDim,
        const std::function<SmallVector<AffineMap>(int64_t)>& rowTapMaps,
        const std::function<ArrayRef<float>(int64_t)>& coefficientsForRow)
    -> Value {
    Value assembled =
      rewriter.create<tensor::EmptyOp>(location, outputDomain, elementType)
        .getResult();
    SmallVector<int64_t> rowDomain(outputDomain);
    rowDomain.erase(rowDomain.begin() + static_cast<std::ptrdiff_t>(rowDim));
    const auto iteratorCount = static_cast<unsigned>(rowDomain.size());
    for (int64_t row = 0; row < rows; ++row) {
      SmallVector<AffineMap> maps = rowTapMaps(row);
      maps.push_back(AffineMap::getMultiDimIdentityMap(iteratorCount, context));
      auto outputType = RankedTensorType::get(rowDomain, elementType);
      auto init = zeroInit(rowDomain);
      SmallVector<Value> tapOperands(maps.size() - 1, source);
      auto generic = rewriter.create<linalg::GenericOp>(
        location,
        TypeRange{outputType},
        tapOperands,
        ValueRange{init},
        maps,
        SmallVector<utils::IteratorType>(iteratorCount,
                                         utils::IteratorType::parallel),
        [&, row](OpBuilder& bodyBuilder,
                 Location bodyLocation,
                 ValueRange bodyValues) {
          ArrayRef<float> coefficients = coefficientsForRow(row);
          Value accumulator;
          for (auto [tap, coefficient] : llvm::enumerate(coefficients)) {
            auto product = bodyBuilder.create<arith::MulFOp>(
              bodyLocation,
              bodyValues[tap],
              bodyBuilder.create<arith::ConstantOp>(
                bodyLocation,
                bodyBuilder.getFloatAttr(elementType, coefficient)));
            accumulator = tap == 0 ? product.getResult()
                                   : bodyBuilder
                                       .create<arith::AddFOp>(
                                         bodyLocation, accumulator, product)
                                       .getResult();
          }
          bodyBuilder.create<linalg::YieldOp>(bodyLocation, accumulator);
        });
      SmallVector<OpFoldResult> offsets(outputDomain.size(),
                                        rewriter.getIndexAttr(0));
      offsets[rowDim] = rewriter.getIndexAttr(row);
      SmallVector<OpFoldResult> sizes;
      for (std::size_t dimension = 0; dimension < outputDomain.size();
           ++dimension) {
        sizes.push_back(rewriter.getIndexAttr(
          dimension == rowDim ? 1 : outputDomain[dimension]));
      }
      SmallVector<OpFoldResult> strides(outputDomain.size(),
                                        rewriter.getIndexAttr(1));
      assembled =
        rewriter
          .create<tensor::InsertSliceOp>(
            location, generic->getResult(0), assembled, offsets, sizes, strides)
          ->getResult(0);
    }
    return assembled;
  };

  // 2a. 第一段（行方向）：M1[th,tw,i,jp,c] = Σ_j B[j,jp]·d[th·6+i,
  //     tw·6+k, c]。tap k 投影 d[(th·6+i), (tw·6+k), c]；输出行 jp 的
  //     系数 = B[:,jp] = kBt[jp,:]（B = kBtᵀ，bRowMajor 转置序）。
  const auto inputDomain =
    SmallVector<int64_t>{tileRows, tileColumns, kAlpha, kAlpha, inputChannels};
  // B（bRowMajor）行主序 B[j][jp] = kBt[jp][j]；输出行 jp 的 tap 系
  // 数 = B 的第 jp 列（k 序）。
  SmallVector<float> bColumnsFlat(kAlpha * kAlpha);
  for (int64_t jp = 0; jp < kAlpha; ++jp) {
    for (int64_t k = 0; k < kAlpha; ++k) {
      bColumnsFlat[(jp * kAlpha) + k] = kBt[jp][k];
    }
  }
  Value stage1 = transformRows(
    inputRows,
    inputDomain,
    kAlpha,
    3,
    [&](int64_t) {
      SmallVector<AffineMap> maps;
      for (int64_t tap = 0; tap < kAlpha; ++tap) {
        maps.push_back(AffineMap::get(
          4, 0, {(dim(0) * 6) + dim(2), (dim(1) * 6) + tap, dim(3)}, context));
      }
      return maps;
    },
    [&](int64_t row) {
      return ArrayRef<float>(bColumnsFlat.data() + (row * kAlpha), kAlpha);
    });
  // 2b. 第二段（列方向）：U[i',jp',c] = Σ_i Bt[i',i]·M1[i, jp', c]。
  //     tap k 投影 M1[th, tw, k, jp', c]；输出行 i' 的系数 = kBt[i',:]。
  SmallVector<float> btRowsFlat(kAlpha * kAlpha);
  for (int64_t row = 0; row < kAlpha; ++row) {
    for (int64_t column = 0; column < kAlpha; ++column) {
      btRowsFlat[(row * kAlpha) + column] = kBt[row][column];
    }
  }
  Value inputTransformed = transformRows(
    stage1,
    inputDomain,
    kAlpha,
    2,
    [&](int64_t) {
      SmallVector<AffineMap> maps;
      for (int64_t tap = 0; tap < kAlpha; ++tap) {
        maps.push_back(AffineMap::get(
          4,
          0,
          {dim(0), dim(1), getAffineConstantExpr(tap, context), dim(2), dim(3)},
          context));
      }
      return maps;
    },
    [&](int64_t row) {
      return ArrayRef<float>(btRowsFlat.data() + (row * kAlpha), kAlpha);
    });

  // 3. 中心 batch_matmul：U 折叠 [T,64,IC] → transpose [64,IC,T] →
  //    与常量 V [64,OC,IC] 批量收缩 → Y [64,OC,T]。
  Value uTiles = rewriter.create<tensor::CollapseShapeOp>(
    location,
    RankedTensorType::get({tiles, kBatch, inputChannels}, elementType),
    inputTransformed,
    SmallVector<ReassociationIndices>{{0, 1}, {2, 3}, {4}});
  Value uInit = rewriter.create<tensor::EmptyOp>(
    location,
    RankedTensorType::get({kBatch, inputChannels, tiles}, elementType)
      .getShape(),
    elementType);
  Value uBatched = rewriter
                     .create<linalg::TransposeOp>(
                       location, uTiles, uInit, ArrayRef<int64_t>{1, 2, 0})
                     ->getResult(0);
  Value weightTransformed =
    rewriter
      .create<arith::ConstantOp>(
        location,
        winograd63::transformWeight(weightType,
                                    cast<DenseFPElementsAttr>(weightElements)))
      .getResult();
  // linalg 收缩语义：outs 是初始累加器，必须零填充（tensor.empty 未
  // 初始化会把堆内存并进结果）。
  Value yInit = zeroInit(SmallVector<int64_t>{kBatch, outputChannels, tiles});
  Value yBatched = rewriter
                     .create<linalg::BatchMatmulOp>(
                       location,
                       TypeRange{RankedTensorType::get(
                         {kBatch, outputChannels, tiles}, elementType)},
                       ValueRange{weightTransformed, uBatched},
                       ValueRange{yInit})
                     .getResult(0);
  contract::copyConvContract(convolution, yBatched.getDefiningOp());

  // 4. 输出变换两段（8-tap 展开形态，输出域交错布局）。Y [64,OC,T]
  //    → [T,OC,64] 转置 → expand [TH,TW,OC,8,8] → 转置换轴到
  //    [TH,TW,i,j,OC]。
  Value yTiled = rewriter.create<tensor::EmptyOp>(
    location,
    RankedTensorType::get({tiles, outputChannels, kBatch}, elementType)
      .getShape(),
    elementType);
  Value ySwapped = rewriter
                     .create<linalg::TransposeOp>(
                       location, yBatched, yTiled, ArrayRef<int64_t>{2, 1, 0})
                     ->getResult(0);
  Value ySpatial = rewriter.create<tensor::ExpandShapeOp>(
    location,
    RankedTensorType::get(
      {tileRows, tileColumns, outputChannels, kAlpha, kAlpha}, elementType),
    ySwapped,
    SmallVector<ReassociationIndices>{{0, 1}, {2}, {3, 4}});
  Value yChannelInnerInit = rewriter.create<tensor::EmptyOp>(
    location,
    RankedTensorType::get(
      {tileRows, tileColumns, kAlpha, kAlpha, outputChannels}, elementType)
      .getShape(),
    elementType);
  Value yTiles =
    rewriter
      .create<linalg::TransposeOp>(
        location, ySpatial, yChannelInnerInit, ArrayRef<int64_t>{0, 1, 3, 4, 2})
      ->getResult(0);

  // 4a. 列段：P[th,tw,i,c,oc] = Σ_jp Y[i,jp,oc]·At[c,jp]（kAt 行主
  //     序 [6,8]）。输出域 [TH,6,TW,6,OC] 交错（含 tw 维在 c 前）。
  //     注意本段的输出域 = 最终交错布局（列段直接产 [TH,c,TW? …]：
  //     行段再换轴代价更高，改为列段产 [TH,TW,i,c,oc]、行段产交错）。
  const auto outputMidDomain =
    SmallVector<int64_t>{tileRows, tileColumns, kAlpha, kTile, outputChannels};
  // 空间交错布局 [TH, r, TW, c, OC]：collapse {{0,1},{2,3},{4}} 直接
  // 折出 [TH·6, TW·6, OC]（无中间 transpose）。
  const auto outputDomain =
    SmallVector<int64_t>{tileRows, kTile, tileColumns, kTile, outputChannels};
  SmallVector<float> atRowMajor(kTile * kAlpha);
  for (int64_t r = 0; r < kTile; ++r) {
    for (int64_t c = 0; c < kAlpha; ++c) {
      atRowMajor[(r * kAlpha) + c] = kAt[r][c];
    }
  }
  Value outputStage1 = transformRows(
    yTiles,
    outputMidDomain,
    kTile,
    3,
    [&](int64_t) {
      SmallVector<AffineMap> maps;
      for (int64_t tap = 0; tap < kAlpha; ++tap) {
        maps.push_back(AffineMap::get(
          4,
          0,
          {dim(0), dim(1), dim(2), getAffineConstantExpr(tap, context), dim(3)},
          context));
      }
      return maps;
    },
    [&](int64_t row) {
      return ArrayRef<float>(atRowMajor.data() + (row * kAlpha), kAlpha);
    });
  // 4b. 行段：O[th,r,tw,c,oc] = Σ_i P[th,tw,i,c,oc]·At[r,i]（kAt
  //     行主序）。输出域 [TH,6,TW,6,OC]，rowDim=1（r 维）从迭代空
  // 间移除，迭代器 (th, tw, c, oc)；P 投影 [th, tw, tap, c, oc]（5
  // 结果，P 域 [TH,TW,8,6,OC]）。
  Value outputTransformed = transformRows(
    outputStage1,
    outputDomain,
    kTile,
    1,
    [&](int64_t) {
      SmallVector<AffineMap> maps;
      for (int64_t tap = 0; tap < kAlpha; ++tap) {
        maps.push_back(AffineMap::get(
          4,
          0,
          {dim(0), dim(1), getAffineConstantExpr(tap, context), dim(2), dim(3)},
          context));
      }
      return maps;
    },
    [&](int64_t row) {
      return ArrayRef<float>(atRowMajor.data() + (row * kAlpha), kAlpha);
    });

  // 5. 折回 [1, TH·6, TW·6, OC] 后裁剪到 [1, OH, OW, OC]。
  Value outputRows = rewriter.create<tensor::CollapseShapeOp>(
    location,
    RankedTensorType::get(
      {(tileRows * kTile), (tileColumns * kTile), outputChannels}, elementType),
    outputTransformed,
    SmallVector<ReassociationIndices>{{0, 1}, {2, 3}, {4}});
  Value outputExpanded = rewriter.create<tensor::ExpandShapeOp>(
    location,
    RankedTensorType::get(
      {1, (tileRows * kTile), (tileColumns * kTile), outputChannels},
      elementType),
    outputRows,
    SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}});
  auto trimmed = rewriter.create<tensor::ExtractSliceOp>(
    location,
    RankedTensorType::get({1, outputHeight, outputWidth, outputChannels},
                          elementType),
    outputExpanded,
    SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(0)),
    SmallVector<OpFoldResult>{rewriter.getIndexAttr(1),
                              rewriter.getIndexAttr(outputHeight),
                              rewriter.getIndexAttr(outputWidth),
                              rewriter.getIndexAttr(outputChannels)},
    SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1)),
    std::nullopt);
  // bias 补加：conv 的 init operand 含逐通道 bias（NCNNToTosa 以广播
  // 填充进入 conv 的 outs）。中心 GEMM 域是变换域不含空间维，在空间
  // 折叠恢复后与 init 逐元素相加（等价于直接卷积的 acc+bias 语义）。
  Value convInit = convolution.getDpsInitOperand(0)->get();
  Value biased =
    rewriter
      .create<linalg::GenericOp>(
        location,
        TypeRange{trimmed->getResult(0).getType()},
        ValueRange{trimmed->getResult(0), convInit},
        ValueRange{rewriter.create<tensor::EmptyOp>(
          location,
          SmallVector<int64_t>{1, outputHeight, outputWidth, outputChannels},
          elementType)},
        SmallVector<AffineMap>{AffineMap::getMultiDimIdentityMap(4, context),
                               AffineMap::getMultiDimIdentityMap(4, context),
                               AffineMap::getMultiDimIdentityMap(4, context)},
        SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
        [&](OpBuilder& bodyBuilder,
            Location bodyLocation,
            ValueRange bodyValues) {
          auto sum = bodyBuilder.create<arith::AddFOp>(
            bodyLocation, bodyValues[0], bodyValues[1]);
          bodyBuilder.create<linalg::YieldOp>(bodyLocation, sum.getResult());
        })
      ->getResult(0);
  // 类型一致（conv 输出即裁剪目标形状），CastOp 在 canonicalize 折叠。
  rewriter.replaceOpWithNewOp<tensor::CastOp>(convolution, resultType, biased);
  return true;
}

}  // namespace mlir::ncnn::winograd63
