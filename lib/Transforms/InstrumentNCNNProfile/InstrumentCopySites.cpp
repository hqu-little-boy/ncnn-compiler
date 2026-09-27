// InstrumentCopySites：拷贝事件插桩。
//
// 职责
//   在 memref.copy 与带 ncnn.copy_contract 的算子之后插入
//   __ncnn_profile_copy(id, bytes)。
//
// 不变量
//   * 事件插在算子**之后**（拷贝完成才是一次完整搬运）；
//   * bytes 取源 memref 的静态字节；静态尺寸取不到就报 -1，由运行期
//     记成「未知」而不是猜；
//   * 带 copy_contract 但不是 memref.copy 的算子（RewriteLinalgCopies 的
//     vectorized / fallback 根）走 kCopyBytes 属性取字节数。
//
// 插入点
//   自行 setInsertionPointAfter(operation)。注意既有行为：后续类别再次
//   setInsertionPointAfter 会把本调用往 IR 后段推（倒序效应），这是刻意
//   保留的，不要"整理"。
//
// 明确不做
//   * 不区分 memcpy / elementwise 搬运的代价（归因层自己看 copy_kind）。

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNSites.hpp"

namespace mlir::ncnn::sites {

void instrumentCopySites(const SiteContext& context,
                         Operation* operation,
                         Value idValue) {
  IRRewriter& rewriter = context.rewriter;
  if (auto copyOp = dyn_cast<memref::CopyOp>(operation)) {
    const auto sourceType = dyn_cast<MemRefType>(copyOp.getSource().getType());
    const std::int64_t bytes = sourceType ? staticByteSize(sourceType) : -1;
    Value byteValue = emitConstant(rewriter, operation->getLoc(), bytes);
    rewriter.setInsertionPointAfter(operation);
    emitCall(rewriter, operation->getLoc(), context.copy, {idValue, byteValue});
  }
  if (operation->hasAttr(contract::kCopyContract) &&
      !isa<memref::CopyOp>(operation)) {
    const auto bytesAttr =
      operation->getAttrOfType<IntegerAttr>(contract::kCopyBytes);
    const std::int64_t bytes = bytesAttr ? bytesAttr.getInt() : -1;
    Value byteValue = emitConstant(rewriter, operation->getLoc(), bytes);
    rewriter.setInsertionPointAfter(operation);
    emitCall(rewriter, operation->getLoc(), context.copy, {idValue, byteValue});
  }
}

}  // namespace mlir::ncnn::sites
