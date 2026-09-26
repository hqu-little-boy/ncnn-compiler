// InstrumentNCNNProfile：为归因报告插入运行期计数器站点。
//
// 职责
//   在分配 / 拷贝 / 算子 / worker / fusion 各站点插入 profile_runtime 的
//   事件调用，并派生稳定 profile_id（stableHash64）。
//
// 不变量
//   * 插桩是 opt-in（NCNN_NATIVE_INT8_PROFILE 等开关关闭时全不插）；
//   * profile_id 由 stableHash64 派生——换哈希算法即 identity 变化；
//   * 站点 source-op 溯源必须唯一；来源不明的 worker 站点宁可不插；
//   * 计时边界不得跨 worker join（P27/P28 的归因 bug 即源于此）。
//
// 顺序依赖
//   * 必须在全部改写与内核化之后（站点要落在最终形态上）；
//   * 必须在 EmitModelPlan 之前（plan 记录的 profile_id 要与插桩同源）。
//
// 明确不做
//   * 不改变计算语义（只加旁路调用）；
//   * 不引入未定义符号白名单之外的依赖（profile_allowed 约束）；
//   * 不做数值采样——只出整数计数与时钟。

#include "ncnn-mlir/Transforms/InstrumentNCNNProfile/InstrumentNCNNProfile.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"
#include "ncnn-mlir/Support/StableHash.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_INSTRUMENTNCNNPROFILEPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

constexpr StringLiteral kBegin = "__ncnn_profile_event_begin";
constexpr StringLiteral kEnd = "__ncnn_profile_event_end";
constexpr StringLiteral kWorkerBegin = "__ncnn_profile_worker_event_begin";
constexpr StringLiteral kWorkerEnd = "__ncnn_profile_worker_event_end";
constexpr StringLiteral kAlloc = "__ncnn_profile_alloc";
constexpr StringLiteral kDealloc = "__ncnn_profile_dealloc";
constexpr StringLiteral kCopy = "__ncnn_profile_copy";
constexpr StringLiteral kMovement = "__ncnn_profile_movement";
constexpr StringLiteral kMaterialized = "__ncnn_profile_materialized";
constexpr StringLiteral kFlush = "__ncnn_profile_flush";

// Categories are part of the private profile ABI.  They intentionally remain
// numeric so the generated module does not depend on a compiler-side enum.
enum class EventCategory : std::int64_t {
  Operation = 0,
  Allocation = 1,
  Deallocation = 2,
  Copy = 3,
  Parallel = 4,
  Transpose = 5,
  Pack = 6,
  Unpack = 7,
  MaterializedWrite = 8,
  MaterializedRead = 9,
  FusionSite = 10,
  WorkerOperation = 11,
};

struct MaterializedEvent {
  std::uint64_t id;
  std::int64_t bytes;
  std::int64_t expectedReaders;
};

func::FuncOp declare(IRRewriter& rewriter,
                     ModuleOp module,
                     StringRef name,
                     ArrayRef<Type> argumentTypes) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name)) {
    return existing;
  }
  rewriter.setInsertionPointToStart(module.getBody());
  auto type = rewriter.getFunctionType(argumentTypes, {});
  auto function = rewriter.create<func::FuncOp>(module.getLoc(), name, type);
  function.setPrivate();
  function->setAttr(contract::kProfileRuntime,
                    UnitAttr::get(module.getContext()));
  return function;
}

Value constant(IRRewriter& rewriter, Location location, std::int64_t value) {
  return rewriter.create<arith::ConstantOp>(
    location,
    rewriter.getI64Type(),
    rewriter.getIntegerAttr(rewriter.getI64Type(), value));
}

void call(IRRewriter& rewriter,
          Location location,
          func::FuncOp callee,
          ArrayRef<Value> arguments) {
  rewriter.create<func::CallOp>(location, callee, arguments);
}

bool isViewLike(Operation& operation) {
  return isa<memref::CastOp,
             memref::SubViewOp,
             memref::ReinterpretCastOp,
             memref::CollapseShapeOp,
             memref::ExpandShapeOp,
             memref::MemorySpaceCastOp,
             memref::ReshapeOp,
             memref::TransposeOp,
             memref::ViewOp,
             memref::ExtractStridedMetadataOp,
             memref::AssumeAlignmentOp>(operation);
}

std::int64_t staticByteSize(ShapedType type) {
  if (!type || !type.hasStaticShape() ||
      (isa<MemRefType>(type) &&
       !cast<MemRefType>(type).getLayout().isIdentity()) ||
      !type.getElementType().isIntOrFloat()) {
    return -1;
  }
  std::uint64_t count = 1;
  for (std::int64_t extent : type.getShape()) {
    if (extent < 0) {
      return -1;
    }
    if (extent == 0) {
      count = 0;
      continue;
    }
    if (count > std::numeric_limits<std::uint64_t>::max() /
                  static_cast<std::uint64_t>(extent)) {
      return -1;
    }
    count *= static_cast<std::uint64_t>(extent);
  }
  const unsigned width =
    type.getElementType().isInteger()
      ? (type.getElementType().getIntOrFloatBitWidth() + 7) / 8
      : type.getElementType().getIntOrFloatBitWidth() / 8;
  if (width == 0 || count > static_cast<std::uint64_t>(
                              std::numeric_limits<std::int64_t>::max()) /
                              width) {
    return -1;
  }
  return static_cast<std::int64_t>(count * width);
}

bool isStaticIdentityMemRef(MemRefType type) {
  return type && type.hasStaticShape() && type.getLayout().isIdentity() &&
         staticByteSize(type) > 0;
}

bool isWholeBufferView(Operation& operation) {
  Value source;
  Value result;
  if (auto cast = dyn_cast<memref::CastOp>(operation)) {
    source = cast.getSource();
    result = cast.getResult();
  } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(operation)) {
    source = collapse.getSrc();
    result = collapse.getResult();
  } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(operation)) {
    source = expand.getSrc();
    result = expand.getResult();
  } else if (auto subview = dyn_cast<memref::SubViewOp>(operation)) {
    source = subview.getSource();
    result = subview.getResult();
  } else {
    return false;
  }

  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto resultType = dyn_cast<MemRefType>(result.getType());
  if (!isStaticIdentityMemRef(sourceType) ||
      !isStaticIdentityMemRef(resultType) ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getMemorySpace() != resultType.getMemorySpace() ||
      staticByteSize(sourceType) != staticByteSize(resultType)) {
    return false;
  }

  if (isa<memref::CastOp>(operation)) {
    return sourceType.getShape() == resultType.getShape();
  }
  if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(operation)) {
    return true;
  }

  auto subview = cast<memref::SubViewOp>(operation);
  if (sourceType.getRank() != resultType.getRank()) {
    return false;
  }
  ArrayRef<int64_t> offsets = subview.getStaticOffsets();
  ArrayRef<int64_t> sizes = subview.getStaticSizes();
  ArrayRef<int64_t> strides = subview.getStaticStrides();
  if (static_cast<std::int64_t>(offsets.size()) != sourceType.getRank() ||
      static_cast<std::int64_t>(sizes.size()) != sourceType.getRank() ||
      static_cast<std::int64_t>(strides.size()) != sourceType.getRank()) {
    return false;
  }
  for (int64_t dimension = 0; dimension < sourceType.getRank(); ++dimension) {
    if (offsets[dimension] != 0 ||
        sizes[dimension] != sourceType.getDimSize(dimension) ||
        sizes[dimension] != resultType.getDimSize(dimension) ||
        strides[dimension] != 1) {
      return false;
    }
  }
  return true;
}

std::optional<Operation*> traceWholeAllocation(
  Value value, SmallVectorImpl<Value>& aliases) {
  while (value) {
    aliases.push_back(value);
    if (auto allocation = value.getDefiningOp<memref::AllocOp>()) {
      return allocation.getOperation();
    }
    Operation* defining = value.getDefiningOp();
    if (!defining || !isWholeBufferView(*defining) ||
        defining->getNumOperands() == 0) {
      return std::nullopt;
    }
    value = defining->getOperand(0);
  }
  return std::nullopt;
}

bool hasFullAccessMap(linalg::LinalgOp operation, OpOperand* operand) {
  auto type = dyn_cast<ShapedType>(operand->get().getType());
  if (!type || !type.hasStaticShape() ||
      !isa<MemRefType, RankedTensorType>(type)) {
    return false;
  }
  AffineMap map = operation.getMatchingIndexingMap(operand);
  if (!map || !map.isProjectedPermutation() ||
      map.getNumResults() != type.getRank()) {
    return false;
  }

  SmallVector<int64_t> loopRanges = operation.getStaticLoopRanges();
  if (llvm::any_of(loopRanges, [](int64_t range) { return range <= 0; })) {
    return false;
  }
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
    auto loopDimension = dyn_cast<AffineDimExpr>(map.getResult(dimension));
    if (!loopDimension || loopDimension.getPosition() >= loopRanges.size() ||
        loopRanges[loopDimension.getPosition()] != type.getDimSize(dimension)) {
      return false;
    }
  }
  return true;
}

bool isProfileCandidate(Operation& operation) {
  if (isa<func::FuncOp, func::ReturnOp, memref::DeallocOp>(operation) ||
      operation.hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  if (isa<memref::AllocOp,
          memref::CopyOp,
          memref::TransposeOp,
          scf::ForallOp,
          scf::ParallelOp>(operation)) {
    return true;
  }
  StringRef name = operation.getName().getStringRef();
  return name.starts_with("linalg.") || name.starts_with("vector.") ||
         name.starts_with("scf.") || name.starts_with("omp.");
}

bool isDirectParallelWorkerCandidate(Operation& operation) {
  if (!isa<scf::ForallOp, scf::ParallelOp, omp::ParallelOp>(
        operation.getParentOp()) ||
      !isProfileCandidate(operation) ||
      isa<memref::AllocOp, memref::CopyOp, memref::TransposeOp>(operation)) {
    return false;
  }
  StringRef name = operation.getName().getStringRef();
  // Time coarse per-worker loop/kernel scopes only. Instrumenting each vector
  // lane or scalar operation would distort the profile and dominate tiny ops.
  return isa<scf::ForOp>(operation) || name.starts_with("linalg.");
}

class InstrumentNCNNFusionSitesPass final
  : public PassWrapper<InstrumentNCNNFusionSitesPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "instrument-ncnn-fusion-sites"; }

  StringRef getDescription() const final {
    return "Bracket selected fusion sites before canonicalization";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(contract::kProfileFusionSitesInstrumented)) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    auto begin = declare(rewriter, module, kBegin, {i64, i64});
    auto end = declare(rewriter, module, kEnd, {i64});
    struct FusionSite {
      Operation* operation;
      std::int64_t profileId;
    };
    SmallVector<FusionSite> fusionSites;
    module.walk([&](Operation* operation) {
      auto profileId =
        operation->getAttrOfType<IntegerAttr>(contract::kFusionSiteId);
      if (profileId && operation->getParentOfType<func::FuncOp>()) {
        fusionSites.push_back(
          FusionSite{.operation = operation, .profileId = profileId.getInt()});
      }
    });

    for (const FusionSite& site : fusionSites) {
      rewriter.setInsertionPoint(site.operation);
      Value eventId =
        constant(rewriter, site.operation->getLoc(), site.profileId);
      Value category =
        constant(rewriter,
                 site.operation->getLoc(),
                 static_cast<std::int64_t>(EventCategory::FusionSite));
      call(rewriter, site.operation->getLoc(), begin, {eventId, category});
      rewriter.setInsertionPointAfter(site.operation);
      Value endId =
        constant(rewriter, site.operation->getLoc(), site.profileId);
      call(rewriter, site.operation->getLoc(), end, {endId});
    }
    module->setAttr(contract::kProfileFusionSitesInstrumented,
                    UnitAttr::get(module.getContext()));
  }
};

class InstrumentNCNNMaterializedSitesPass final
  : public PassWrapper<InstrumentNCNNMaterializedSitesPass,
                       OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final {
    return "instrument-ncnn-materialized-sites";
  }

  StringRef getDescription() const final {
    return "Bracket complete static Linalg producer-consumer materializations";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(contract::kProfileMaterializedSitesInstrumented)) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    auto materialized =
      declare(rewriter, module, kMaterialized, {i64, i64, i64, i64});

    struct Site {
      Operation* writer;
      Operation* reader;
      std::uint64_t id;
      std::int64_t bytes;
    };
    SmallVector<Site> sites;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (function->hasAttr(contract::kProfileRuntime)) {
        continue;
      }
      std::size_t ordinal = 0;
      function.walk([&](linalg::LinalgOp writer) {
        Operation* writerOp = writer.getOperation();
        if (writerOp->getParentOp() != function.getOperation()) {
          return;
        }
        const int64_t resultCount =
          std::min<int64_t>(writer->getNumResults(), writer.getNumDpsInits());
        for (int64_t resultNumber = 0; resultNumber < resultCount;
             ++resultNumber) {
          Value result = writer->getResult(resultNumber);
          auto resultType = dyn_cast<RankedTensorType>(result.getType());
          const std::int64_t bytes =
            resultType ? staticByteSize(resultType) : -1;
          if (bytes <= 0 || !result.hasOneUse() ||
              !hasFullAccessMap(writer,
                                writer.getDpsInitOperand(resultNumber))) {
            continue;
          }

          OpOperand* use = &*result.getUses().begin();
          auto reader = dyn_cast<linalg::LinalgOp>(use->getOwner());
          if (!reader ||
              !llvm::is_contained(reader.getDpsInputOperands(), use) ||
              !hasFullAccessMap(reader, use) ||
              reader->getParentOp() != function.getOperation() ||
              reader->getBlock() != writerOp->getBlock() ||
              !writerOp->isBeforeInBlock(reader)) {
            continue;
          }

          const std::string siteName = function.getName().str() +
                                       "/materialized#" +
                                       std::to_string(ordinal++);
          sites.push_back(Site{
            .writer = writerOp,
            .reader = reader.getOperation(),
            .id = ncnn_mlir::stableHash64(siteName),
            .bytes = bytes,
          });
        }
      });
    }

    for (const Site& site : sites) {
      rewriter.setInsertionPointAfter(site.writer);
      Value writeId = constant(
        rewriter, site.writer->getLoc(), static_cast<std::int64_t>(site.id));
      Value writeKind = constant(rewriter, site.writer->getLoc(), 0);
      Value writeBytes = constant(rewriter, site.writer->getLoc(), site.bytes);
      Value expectedReaders = constant(rewriter, site.writer->getLoc(), 1);
      call(rewriter,
           site.writer->getLoc(),
           materialized,
           {writeId, writeKind, writeBytes, expectedReaders});

      rewriter.setInsertionPoint(site.reader);
      Value readId = constant(
        rewriter, site.reader->getLoc(), static_cast<std::int64_t>(site.id));
      Value readKind = constant(rewriter, site.reader->getLoc(), 1);
      Value readBytes = constant(rewriter, site.reader->getLoc(), site.bytes);
      Value noExpectedReaders = constant(rewriter, site.reader->getLoc(), 0);
      call(rewriter,
           site.reader->getLoc(),
           materialized,
           {readId, readKind, readBytes, noExpectedReaders});
    }
    module->setAttr(contract::kProfileMaterializedSitesInstrumented,
                    UnitAttr::get(module.getContext()));
  }
};

class InstrumentNCNNProfilePass final
  : public impl::InstrumentNCNNProfilePassBase<InstrumentNCNNProfilePass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithDialect,
                    func::FuncDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    omp::OpenMPDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module->hasAttr(contract::kProfileInstrumented)) {
      return;
    }

    IRRewriter rewriter(module.getContext());
    const Type i64 = rewriter.getI64Type();
    const SmallVector<Type> oneId{i64};
    const SmallVector<Type> twoIds{i64, i64};
    const SmallVector<Type> threeIds{i64, i64, i64};
    const SmallVector<Type> fourIds{i64, i64, i64, i64};
    auto begin = declare(rewriter, module, kBegin, {i64, i64});  // id, category
    auto end = declare(rewriter, module, kEnd, oneId);
    auto workerBegin = declare(rewriter, module, kWorkerBegin, oneId);
    auto workerEnd = declare(rewriter, module, kWorkerEnd, oneId);
    auto alloc = declare(rewriter, module, kAlloc, twoIds);
    auto dealloc = declare(rewriter, module, kDealloc, oneId);
    auto copy = declare(rewriter, module, kCopy, twoIds);
    auto movement = declare(rewriter, module, kMovement, threeIds);
    auto materialized = declare(rewriter, module, kMaterialized, fourIds);
    auto flush = declare(rewriter, module, kFlush, {});

    SmallVector<func::FuncOp> functions;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr(contract::kProfileRuntime)) {
        functions.push_back(function);
      }
    }

    for (func::FuncOp function : functions) {
      std::map<Operation*, std::uint64_t> operation_ids;
      std::map<std::string, unsigned> operation_ordinals;
      function.walk([&](Operation* operation) {
        const std::string kind = operation->getName().getStringRef().str();
        const std::string ordinal_key = function.getName().str() + "/" + kind;
        const unsigned ordinal = operation_ordinals[ordinal_key]++;
        const auto fusionProfileId =
          operation->getAttrOfType<IntegerAttr>(contract::kFusionSiteId);
        operation_ids.emplace(
          operation,
          fusionProfileId ? static_cast<std::uint64_t>(fusionProfileId.getInt())
                          : ncnn_mlir::stableHash64(ordinal_key + "#" +
                                                    std::to_string(ordinal)));
      });

      std::map<Operation*, SmallVector<Operation*>> producers;
      std::map<Operation*, SmallVector<Operation*>> consumers;
      std::map<Operation*, std::map<Operation*, unsigned>> producerUses;
      std::map<Operation*, std::map<Operation*, unsigned>> consumerUses;
      std::map<Operation*, std::map<Operation*, bool>> producerCoverage;
      std::map<Operation*, std::map<Operation*, bool>> consumerCoverage;
      std::map<Operation*, SmallVector<Value>> allocationAliases;
      std::map<Operation*, std::set<OpOperand*>> trackedLinalgUses;
      auto recordLinalgAccess =
        [&](linalg::LinalgOp linalgOp, OpOperand* operand, bool isProducer) {
          SmallVector<Value> aliases;
          auto allocation = traceWholeAllocation(operand->get(), aliases);
          if (!allocation) {
            return;
          }
          Operation* root = *allocation;
          for (Value alias : aliases) {
            if (!llvm::is_contained(allocationAliases[root], alias)) {
              allocationAliases[root].push_back(alias);
            }
          }
          trackedLinalgUses[root].insert(operand);

          Operation* operation = linalgOp.getOperation();
          auto& accesses = isProducer ? producers[root] : consumers[root];
          if (!llvm::is_contained(accesses, operation)) {
            accesses.push_back(operation);
          }
          auto& counts = isProducer ? producerUses[root] : consumerUses[root];
          ++counts[operation];
          auto& coverage =
            isProducer ? producerCoverage[root] : consumerCoverage[root];
          auto [it, inserted] = coverage.try_emplace(
            operation, hasFullAccessMap(linalgOp, operand));
          if (!inserted) {
            it->second &= hasFullAccessMap(linalgOp, operand);
          }
        };
      function.walk([&](linalg::LinalgOp linalgOp) {
        for (int64_t i = 0; i < linalgOp.getNumDpsInits(); ++i) {
          recordLinalgAccess(linalgOp, linalgOp.getDpsInitOperand(i), true);
        }
        for (OpOperand* operand : linalgOp.getDpsInputOperands()) {
          recordLinalgAccess(linalgOp, operand, false);
        }
      });

      // Prove the entire allocation alias chain consists only of whole-buffer
      // views, complete Linalg accesses, and deallocation. Any escape or
      // partial view makes the byte evidence incomplete and is rejected.
      std::map<Operation*, bool> completeAliasUses;
      for (const auto& [allocation, initialAliases] : allocationAliases) {
        SmallVector<Value> aliases(initialAliases.begin(),
                                   initialAliases.end());
        bool complete = true;
        for (std::size_t i = 0; i < aliases.size() && complete; ++i) {
          for (OpOperand& use : aliases[i].getUses()) {
            Operation* user = use.getOwner();
            if (isa<memref::DeallocOp>(user)) {
              continue;
            }
            if (use.getOperandNumber() == 0 && isWholeBufferView(*user) &&
                user->getNumResults() == 1) {
              Value result = user->getResult(0);
              if (!llvm::is_contained(aliases, result)) {
                aliases.push_back(result);
              }
              continue;
            }
            auto tracked = trackedLinalgUses.find(allocation);
            if (tracked != trackedLinalgUses.end() &&
                tracked->second.contains(&use)) {
              continue;
            }
            complete = false;
            break;
          }
        }
        completeAliasUses[allocation] = complete;
      }

      // Bracket only static allocations with one full-map Linalg writer and
      // full-map readers that follow it in the same block. Events report the
      // complete logical allocation footprint, not scalar load/store traffic.
      std::map<Operation*, SmallVector<MaterializedEvent>> materializedWrites;
      std::map<Operation*, SmallVector<MaterializedEvent>> materializedReads;
      for (const auto& [allocation, writeOps] : producers) {
        if (!completeAliasUses[allocation] || writeOps.size() != 1) {
          continue;
        }
        Operation* writer = writeOps.front();
        if (producerUses[allocation][writer] != 1 ||
            !producerCoverage[allocation][writer]) {
          continue;
        }
        auto allocOp = dyn_cast<memref::AllocOp>(allocation);
        if (!allocOp) {
          continue;
        }
        const std::int64_t bytes = staticByteSize(allocOp.getType());
        if (bytes <= 0) {
          continue;
        }

        SmallVector<Operation*> readOps;
        bool hasCompleteConsumers = true;
        if (auto it = consumers.find(allocation); it != consumers.end()) {
          for (Operation* consumer : it->second) {
            if (consumer == writer) {
              continue;
            }
            if (consumerUses[allocation][consumer] != 1 ||
                !consumerCoverage[allocation][consumer] ||
                consumer->getBlock() != writer->getBlock() ||
                !writer->isBeforeInBlock(consumer)) {
              hasCompleteConsumers = false;
              break;
            }
            readOps.push_back(consumer);
          }
        }
        if (readOps.empty() || !hasCompleteConsumers) {
          continue;
        }

        const std::uint64_t id = operation_ids.at(allocation);
        materializedWrites[writer].push_back(MaterializedEvent{
          .id = id,
          .bytes = bytes,
          .expectedReaders = static_cast<std::int64_t>(readOps.size()),
        });
        for (Operation* consumer : readOps) {
          materializedReads[consumer].push_back(
            MaterializedEvent{.id = id, .bytes = bytes, .expectedReaders = 0});
        }
      }

      SmallVector<Operation*> candidates;
      function.walk([&](Operation* operation) {
        // Nested memory events need no region wrapping. Keep duration timers
        // at function scope: worker TLS spans cannot be subtracted from the
        // parent parallel span, and timing individual lanes is too expensive.
        if (isa<memref::AllocOp, memref::CopyOp>(operation) ||
            operation->hasAttr(contract::kCopyContract) ||
            materializedWrites.contains(operation) ||
            materializedReads.contains(operation) ||
            ((operation->getParentOp() == function.getOperation() &&
              isProfileCandidate(*operation) &&
              !(module->hasAttr(contract::kProfileFusionSitesInstrumented) &&
                operation->hasAttr(contract::kFusionSiteId))) ||
             isDirectParallelWorkerCandidate(*operation))) {
          candidates.push_back(operation);
        }
      });

      std::optional<std::uint64_t> root_id;
      if (function->hasAttr(contract::kEntryPoint)) {
        if (auto root = operation_ids.find(function.getOperation());
            root != operation_ids.end()) {
          root_id = root->second;
        }
      }
      if (root_id && !function.getBody().empty()) {
        rewriter.setInsertionPointToStart(&function.getBody().front());
        Value idValue = constant(
          rewriter, function.getLoc(), static_cast<std::int64_t>(*root_id));
        Value categoryValue =
          constant(rewriter,
                   function.getLoc(),
                   static_cast<std::int64_t>(EventCategory::Operation));
        call(rewriter, function.getLoc(), begin, {idValue, categoryValue});
      }

      for (Operation* operation : candidates) {
        const std::uint64_t id = operation_ids.at(operation);
        if (auto it = materializedReads.find(operation);
            it != materializedReads.end()) {
          rewriter.setInsertionPoint(operation);
          for (const MaterializedEvent& event : it->second) {
            Value eventId = constant(rewriter,
                                     operation->getLoc(),
                                     static_cast<std::int64_t>(event.id));
            Value kind = constant(rewriter, operation->getLoc(), 1);
            Value bytes = constant(rewriter, operation->getLoc(), event.bytes);
            Value expectedReaders =
              constant(rewriter, operation->getLoc(), event.expectedReaders);
            call(rewriter,
                 operation->getLoc(),
                 materialized,
                 {eventId, kind, bytes, expectedReaders});
          }
        }
        const auto category =
          isa<memref::AllocOp>(*operation)       ? EventCategory::Allocation
          : isa<memref::CopyOp>(*operation)      ? EventCategory::Copy
          : isa<memref::TransposeOp>(*operation) ? EventCategory::Transpose
          : isa<scf::ForallOp, scf::ParallelOp, omp::ParallelOp>(*operation)
            ? EventCategory::Parallel
            : EventCategory::Operation;
        rewriter.setInsertionPoint(operation);
        Value idValue = constant(
          rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
        const bool workerTimed = isDirectParallelWorkerCandidate(*operation);
        const bool timed = operation->getParentOp() == function.getOperation();
        if (workerTimed) {
          call(rewriter, operation->getLoc(), workerBegin, {idValue});
        } else if (timed) {
          Value categoryValue = constant(
            rewriter, operation->getLoc(), static_cast<std::int64_t>(category));
          call(rewriter, operation->getLoc(), begin, {idValue, categoryValue});
        }

        if (auto allocOp = dyn_cast<memref::AllocOp>(operation)) {
          const std::int64_t bytes = staticByteSize(allocOp.getType());
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          call(rewriter, operation->getLoc(), alloc, {idValue, byteValue});
        }
        if (auto copyOp = dyn_cast<memref::CopyOp>(operation)) {
          const auto sourceType =
            dyn_cast<MemRefType>(copyOp.getSource().getType());
          const std::int64_t bytes =
            sourceType ? staticByteSize(sourceType) : -1;
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          rewriter.setInsertionPointAfter(operation);
          call(rewriter, operation->getLoc(), copy, {idValue, byteValue});
        }
        if (operation->hasAttr(contract::kCopyContract) &&
            !isa<memref::CopyOp>(operation)) {
          const auto bytesAttr =
            operation->getAttrOfType<IntegerAttr>(contract::kCopyBytes);
          const std::int64_t bytes = bytesAttr ? bytesAttr.getInt() : -1;
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          rewriter.setInsertionPointAfter(operation);
          call(rewriter, operation->getLoc(), copy, {idValue, byteValue});
        }
        if (auto transposeOp = dyn_cast<memref::TransposeOp>(operation)) {
          const auto resultType =
            dyn_cast<MemRefType>(transposeOp.getResult().getType());
          const std::int64_t bytes =
            resultType ? staticByteSize(resultType) : -1;
          Value kind = constant(rewriter, operation->getLoc(), 0);
          Value byteValue = constant(rewriter, operation->getLoc(), bytes);
          call(rewriter,
               operation->getLoc(),
               movement,
               {idValue, kind, byteValue});
        }

        if (auto it = materializedWrites.find(operation);
            it != materializedWrites.end()) {
          rewriter.setInsertionPointAfter(operation);
          for (const MaterializedEvent& event : it->second) {
            Value eventId = constant(rewriter,
                                     operation->getLoc(),
                                     static_cast<std::int64_t>(event.id));
            Value kind = constant(rewriter, operation->getLoc(), 0);
            Value bytes = constant(rewriter, operation->getLoc(), event.bytes);
            Value expectedReaders =
              constant(rewriter, operation->getLoc(), event.expectedReaders);
            call(rewriter,
                 operation->getLoc(),
                 materialized,
                 {eventId, kind, bytes, expectedReaders});
          }
        }
        if (workerTimed || timed) {
          rewriter.setInsertionPointAfter(operation);
          Value endId = constant(
            rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
          call(rewriter,
               operation->getLoc(),
               workerTimed ? workerEnd : end,
               {endId});
        }
      }

      // Deallocation has no useful duration, but it is still an explicit
      // event.  Keep it separate from the operation timer so allocation bytes
      // cannot be mistaken for arithmetic time.
      SmallVector<Operation*> deallocations;
      function.walk([&](memref::DeallocOp operation) {
        deallocations.push_back(operation.getOperation());
      });
      auto resolveAllocation = [&](Value value) -> Operation* {
        // Follow the SSA forwarding performed by scf.for in addition to
        // view-like definitions. A loop result (or an iter arg used by a
        // nested dealloc) still denotes the original allocation.
        while (true) {
          if (auto allocOp = value.getDefiningOp<memref::AllocOp>()) {
            return allocOp.getOperation();
          }
          if (Operation* defining = value.getDefiningOp()) {
            if (isViewLike(*defining) && defining->getNumOperands() != 0) {
              value = defining->getOperand(0);
              continue;
            }
            if (auto forOp = dyn_cast<scf::ForOp>(defining)) {
              auto result = dyn_cast<OpResult>(value);
              if (!result ||
                  result.getResultNumber() >= forOp.getInitArgs().size()) {
                return nullptr;
              }
              value = forOp.getInitArgs()[result.getResultNumber()];
              continue;
            }
            return nullptr;
          }
          auto blockArgument = dyn_cast<BlockArgument>(value);
          if (!blockArgument) {
            return nullptr;
          }
          auto forOp =
            dyn_cast<scf::ForOp>(blockArgument.getOwner()->getParentOp());
          const unsigned argumentNumber = blockArgument.getArgNumber();
          if (!forOp || argumentNumber == 0 ||
              argumentNumber - 1 >= forOp.getInitArgs().size()) {
            return nullptr;
          }
          value = forOp.getInitArgs()[argumentNumber - 1];
        }
      };
      for (Operation* operation : deallocations) {
        std::uint64_t id = operation_ids.at(operation);
        if (operation->getNumOperands() == 1) {
          if (Operation* allocation =
                resolveAllocation(operation->getOperand(0))) {
            id = operation_ids.at(allocation);
          }
        }
        rewriter.setInsertionPoint(operation);
        Value idValue = constant(
          rewriter, operation->getLoc(), static_cast<std::int64_t>(id));
        call(rewriter, operation->getLoc(), dealloc, {idValue});
      }

      if (root_id) {
        SmallVector<func::ReturnOp> returns;
        function.walk(
          [&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
        for (func::ReturnOp returnOp : returns) {
          rewriter.setInsertionPoint(returnOp);
          Value idValue = constant(
            rewriter, returnOp.getLoc(), static_cast<std::int64_t>(*root_id));
          call(rewriter, returnOp.getLoc(), end, {idValue});
          call(rewriter, returnOp.getLoc(), flush, {});
        }
      }
    }
    module->setAttr(contract::kProfileInstrumented,
                    UnitAttr::get(module.getContext()));
  }
};

}  // namespace

std::unique_ptr<Pass> createInstrumentNCNNFusionSitesPass() {
  return std::make_unique<InstrumentNCNNFusionSitesPass>();
}

std::unique_ptr<Pass> createInstrumentNCNNMaterializedSitesPass() {
  return std::make_unique<InstrumentNCNNMaterializedSitesPass>();
}

}  // namespace mlir::ncnn
