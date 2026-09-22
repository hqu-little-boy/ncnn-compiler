#include "ncnn-mlir/Transforms/EmitModelPlan/EmitModelPlan.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn {

#define GEN_PASS_DEF_EMITMODELPLANPASS
#include "ncnn-mlir/Passes.h.inc"

namespace {

using JsonObject = llvm::json::Object;
using JsonArray = llvm::json::Array;

struct ByteSize final {
  std::optional<std::int64_t> bytes;
  bool dynamic = false;
  bool overflow = false;
  bool unknown_element_type = false;
};

std::string printType(Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  return result;
}

std::optional<std::uint64_t> elementByteWidth(Type type) {
  if (type.isInteger(1)) {
    return 1;
  }
  if (auto integer = dyn_cast<IntegerType>(type)) {
    return (integer.getWidth() + 7U) / 8U;
  }
  if (type.isF16() || type.isBF16()) {
    return 2;
  }
  if (type.isF32()) {
    return 4;
  }
  if (type.isF64()) {
    return 8;
  }
  return std::nullopt;
}

ByteSize checkedByteSize(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank()) {
    return {.bytes = std::nullopt, .dynamic = true};
  }
  const auto element_width = elementByteWidth(shaped.getElementType());
  if (!element_width) {
    return {.bytes = std::nullopt, .unknown_element_type = true};
  }
  std::uint64_t elements = 1;
  for (std::int64_t extent : shaped.getShape()) {
    if (ShapedType::isDynamic(extent)) {
      return {.bytes = std::nullopt, .dynamic = true};
    }
    if (extent < 0) {
      return {.bytes = std::nullopt, .overflow = true};
    }
    if (elements == 0) {
      continue;
    }
    if (static_cast<std::uint64_t>(extent) >
        std::numeric_limits<std::uint64_t>::max() / elements) {
      return {.bytes = std::nullopt, .overflow = true};
    }
    elements *= static_cast<std::uint64_t>(extent);
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / *element_width) {
    return {.bytes = std::nullopt, .overflow = true};
  }
  const std::uint64_t bytes = elements * *element_width;
  if (bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return {.bytes = std::nullopt, .overflow = true};
  }
  return {.bytes = static_cast<std::int64_t>(bytes)};
}

JsonArray shapeArray(Type type) {
  JsonArray shape;
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank()) {
    return shape;
  }
  for (std::int64_t extent : shaped.getShape()) {
    if (ShapedType::isDynamic(extent)) {
      shape.push_back(nullptr);
    } else {
      shape.push_back(extent);
    }
  }
  return shape;
}

std::string operationKind(Operation& operation) {
  return operation.getName().getStringRef().str();
}

std::string printAttributes(Operation& operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation.getAttrDictionary().print(stream);
  return result;
}

bool isNamed(Operation& operation, StringRef name) {
  return operation.getName().getStringRef() == name;
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

std::uint64_t profileId(StringRef operationId) {
  // FNV-1a is deterministic across processes and does not expose addresses.
  std::uint64_t result = 14695981039346656037ULL;
  for (unsigned char character : operationId.bytes()) {
    result ^= character;
    result *= 1099511628211ULL;
  }
  return result;
}

// Audit only the INT8 kernel contracts visible at this pass boundary. In
// particular, a module capability is not evidence that an operation uses VNNI,
// and forall operands need not retain the original i8 input types.
std::optional<JsonObject> lowPrecisionAudit(Operation* operation) {
  auto stringValue = [&](StringRef name) -> StringRef {
    if (auto value = operation->getAttrOfType<StringAttr>(name)) {
      return value.getValue();
    }
    return {};
  };
  const StringRef kernel = stringValue(contract::kKernel);
  const bool vnni = kernel == "int8_vnni_row_dot";
  const bool portable = kernel == "int8_row_dot";
  const bool depthwise = kernel == "depthwise_simd" &&
                         (stringValue(contract::kFma) == "extsi_muli_addi" ||
                          operation->hasAttr(contract::kInt8Isa));
  if (!vnni && !portable && !depthwise &&
      !operation->hasAttr(contract::kInt8Isa)) {
    return std::nullopt;
  }

  JsonObject result;
  auto copyString = [&](StringRef attribute, StringRef field) {
    StringRef value = stringValue(attribute);
    if (!value.empty()) {
      result[field.str()] = value.str();
    } else {
      result[field.str()] = nullptr;
    }
  };
  copyString(contract::kKernel, "kernel");
  copyString(contract::kContract, "status");
  copyString(contract::kFallback, "fallback_reason");
  copyString(contract::kFma, "arithmetic");
  copyString(contract::kInt8Isa, "required_isa");
  if (portable && stringValue(contract::kInt8Isa).empty()) {
    // Portable row-dot imposes no dot-product ISA requirement; backend
    // auto-vectorization is deliberately not advertised here.
    result["required_isa"] = "portable";
  }
  result["signedness"] = nullptr;
  result["signedness_correction"] = nullptr;
  result["accumulator"] = nullptr;
  if (vnni || portable || depthwise) {
    result["signedness"] = "signed8*signed8";
    result["signedness_correction"] =
      vnni ? "xor_0x80_subtract_128_times_rhs_sum" : "sign_extend_i8_to_i32";
    result["accumulator"] = "modulo32";
  }

  // Tail counts are static extents per kernel, never invocation counts. The
  // row-dot ncnn.tail describes M/N only: VNNI also has a separate K % 32 tail.
  JsonObject tails;
  tails["output_policy"] = nullptr;
  if (StringRef policy = stringValue(contract::kTail); !policy.empty()) {
    tails["output_policy"] = policy.str();
  }
  tails["reduction_elements"] = nullptr;
  if (auto explicitTail =
        operation->getAttrOfType<IntegerAttr>(contract::kInt8ReductionTail);
      explicitTail && explicitTail.getInt() >= 0) {
    tails["reduction_elements"] = explicitTail.getInt();
  } else if (vnni) {
    if (auto depth = operation->getAttrOfType<IntegerAttr>(contract::kTileK);
        depth && depth.getInt() >= 0) {
      tails["reduction_elements"] = depth.getInt() % 32;
    }
  }
  tails["channel_elements"] = nullptr;
  if (depthwise) {
    auto channels =
      operation->getAttrOfType<IntegerAttr>(contract::kInputChannels);
    auto chunk = operation->getAttrOfType<IntegerAttr>(contract::kSimdChunk);
    if (channels && channels.getInt() >= 0 && chunk && chunk.getInt() > 0) {
      tails["channel_elements"] = channels.getInt() % chunk.getInt();
    }
  }
  result["static_tails"] = std::move(tails);
  return result;
}

struct Lifetime final {
  Operation* allocation = nullptr;
  std::string id;
  std::optional<std::int64_t> bytes;
  std::int64_t allocationPosition = 0;
  std::optional<std::int64_t> firstUse;
  std::optional<std::int64_t> lastUse;
  std::optional<std::int64_t> deallocation;
  bool aliasUnknown = false;
};

class EmitModelPlanPass final
  : public impl::EmitModelPlanPassBase<EmitModelPlanPass> {
 public:
  using Base::Base;

  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<func::FuncDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    omp::OpenMPDialect,
                    scf::SCFDialect,
                    vector::VectorDialect>();
  }

  // This pass intentionally centralizes the complete static-plan walk so that
  // all counters and the plan hash are derived from one deterministic visit.
  // NOLINTNEXTLINE(google-readability-function-size)
  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (path.empty()) {
      module.emitError("execution-plan output path is empty");
      signalPassFailure();
      return;
    }

    JsonArray functions;
    JsonArray operations;
    JsonArray buffers;
    JsonArray regions;
    JsonArray contracts;
    JsonArray conv_depthwise_operations;
    JsonArray fusions;
    JsonArray attention_segments;
    JsonArray low_precision_operations;
    JsonArray unknown_fields;
    JsonObject summary;
    std::int64_t allocation_count = 0;
    std::int64_t deallocation_count = 0;
    std::int64_t copy_count = 0;
    std::int64_t transpose_count = 0;
    std::int64_t matmul_count = 0;
    std::int64_t batch_matmul_count = 0;
    std::int64_t vector_fma_count = 0;
    std::int64_t parallel_region_count = 0;
    std::int64_t kernel_contract_count = 0;
    std::int64_t kernel_contract_fallback_count = 0;
    std::int64_t conv_operation_count = 0;
    std::int64_t depthwise_operation_count = 0;
    std::int64_t conv_fallback_count = 0;
    std::int64_t depthwise_fallback_count = 0;
    std::int64_t conv_unknown_count = 0;
    std::int64_t depthwise_unknown_count = 0;
    std::int64_t layout_island_count = 0;
    std::int64_t layout_island_selected_count = 0;
    std::int64_t layout_island_rejected_count = 0;
    std::int64_t layout_pack4_count = 0;
    std::int64_t layout_pack8_count = 0;
    std::map<std::string, std::int64_t> conv_implementations;
    std::map<std::string, std::int64_t> depthwise_implementations;
    std::map<std::string, std::int64_t> conv_fallback_reasons;
    std::map<std::string, std::int64_t> depthwise_fallback_reasons;
    std::int64_t nested_openmp_count = 0;
    std::int64_t workspace_slot_count = 0;
    std::int64_t workspace_reused_allocation_count = 0;
    std::int64_t workspace_fallback_count = 0;
    std::int64_t workspace_slot_bytes = 0;
    bool workspace_slot_bytes_unknown = false;
    std::set<std::string> workspace_slot_ids;
    std::set<std::string> workspace_fallback_reasons;
    std::int64_t packed_buffer_bytes = 0;
    bool packed_buffer_bytes_unknown = false;
    std::int64_t packed_constant_bytes = 0;
    bool packed_constant_bytes_unknown = false;
    std::int64_t packed_constant_count = 0;
    std::set<std::string> packed_constant_ids;
    std::int64_t packed_operation_count = 0;
    std::int64_t static_pack_bytes = 0;
    bool static_pack_bytes_unknown = false;
    std::int64_t unpacked_operation_count = 0;
    std::int64_t static_unpack_bytes = 0;
    bool static_unpack_bytes_unknown = false;
    std::optional<std::int64_t> static_buffer_bytes = 0;
    std::optional<std::int64_t> static_peak_live_bytes;
    bool peak_workspace_unknown = false;
    std::int64_t dynamic_buffer_count = 0;
    std::int64_t static_copy_bytes = 0;
    bool has_unknown_copy_bytes = false;
    std::int64_t static_transpose_bytes = 0;
    bool has_unknown_transpose_bytes = false;
    const bool fusion_enabled =
      module->getAttrOfType<BoolAttr>(contract::kFusionEnabled)
        ? module->getAttrOfType<BoolAttr>(contract::kFusionEnabled).getValue()
        : true;
    const std::int64_t fusion_selected_count =
      module->getAttrOfType<IntegerAttr>(contract::kFusionSelectedCount)
        ? module->getAttrOfType<IntegerAttr>(contract::kFusionSelectedCount)
            .getInt()
        : 0;
    const std::int64_t fusion_residual_count =
      module->getAttrOfType<IntegerAttr>(contract::kFusionResidualCount)
        ? module->getAttrOfType<IntegerAttr>(contract::kFusionResidualCount)
            .getInt()
        : 0;
    const std::int64_t fusion_rejected_count =
      module->getAttrOfType<IntegerAttr>(contract::kFusionRejectedCount)
        ? module->getAttrOfType<IntegerAttr>(contract::kFusionRejectedCount)
            .getInt()
        : 0;
    const std::string fusion_rejection_reasons =
      module->getAttrOfType<StringAttr>(contract::kFusionRejectionReasons)
        ? module->getAttrOfType<StringAttr>(contract::kFusionRejectionReasons)
            .getValue()
            .str()
        : "";
    const auto module_fusion_records =
      module->getAttrOfType<ArrayAttr>(contract::kFusionRecords);
    const bool has_module_fusion_records =
      module_fusion_records && !module_fusion_records.empty();
    std::set<std::string> unknown_reasons;
    auto add_unknown = [&](StringRef reason) {
      if (unknown_reasons.insert(reason.str()).second) {
        unknown_fields.push_back(reason.str());
      }
    };

    const auto module_attention_records =
      module->getAttrOfType<ArrayAttr>(contract::kAttentionSegments);
    const std::string attention_revision =
      module->getAttrOfType<StringAttr>(contract::kAttentionRevision)
        ? module->getAttrOfType<StringAttr>(contract::kAttentionRevision)
            .getValue()
            .str()
        : "attention-segment-v1";
    std::string attention_hash_input =
      "attention-revision=" + attention_revision;
    std::int64_t attention_selected_count = 0;
    std::int64_t attention_fallback_count = 0;
    std::int64_t attention_unknown_count = 0;
    if (module_attention_records) {
      std::int64_t recordOrdinal = 0;
      for (Attribute attribute : module_attention_records) {
        auto record = dyn_cast<DictionaryAttr>(attribute);
        if (!record) {
          add_unknown("attention_record_malformed");
          continue;
        }
        JsonObject segment;
        auto copyString = [&](StringRef attributeName, StringRef fieldName) {
          if (auto value = record.getAs<StringAttr>(attributeName)) {
            segment[fieldName.str()] = value.getValue().str();
            attention_hash_input +=
              "|" + fieldName.str() + "=" + value.getValue().str();
          } else {
            segment[fieldName.str()] = nullptr;
          }
        };
        auto copyInteger = [&](StringRef attributeName, StringRef fieldName) {
          if (auto value = record.getAs<IntegerAttr>(attributeName)) {
            segment[fieldName.str()] = value.getInt();
            attention_hash_input +=
              "|" + fieldName.str() + "=" + std::to_string(value.getInt());
          } else {
            segment[fieldName.str()] = nullptr;
          }
        };
        auto copyBool = [&](StringRef attributeName, StringRef fieldName) {
          if (auto value = record.getAs<BoolAttr>(attributeName)) {
            segment[fieldName.str()] = value.getValue();
            attention_hash_input += "|" + fieldName.str() + "=" +
                                    (value.getValue() ? "true" : "false");
          } else {
            segment[fieldName.str()] = nullptr;
          }
        };

        copyString("id", "id");
        copyString("function", "function");
        copyString("source_operation", "source_operation");
        copyInteger("attention_ordinal", "attention_ordinal");
        copyString("phase", "phase");
        copyInteger("heads", "heads");
        copyInteger("M", "M");
        copyInteger("K", "K");
        copyInteger("N", "N");
        copyString("kernel_status", "kernel_status");
        copyString("reason", "reason");
        copyString("layout", "layout");
        copyString("layout_producer", "layout_producer");
        copyString("layout_consumer", "layout_consumer");
        copyString("parallel_policy", "parallel_policy");
        copyString("tile_policy", "tile_policy");
        copyString("transpose_elided_reason", "transpose_elided_reason");
        copyString("softmax_strategy", "softmax_strategy");
        copyInteger("transpose_count", "transpose_count");
        copyInteger("transpose_bytes", "transpose_bytes");
        copyBool("transpose_bytes_known", "transpose_bytes_known");
        copyInteger("copy_count", "copy_count");
        copyInteger("copy_bytes", "copy_bytes");
        copyBool("copy_bytes_known", "copy_bytes_known");

        if (auto status = record.getAs<StringAttr>("kernel_status")) {
          if (status.getValue() == "selected") {
            ++attention_selected_count;
          } else if (status.getValue() == "fallback") {
            ++attention_fallback_count;
          } else {
            ++attention_unknown_count;
          }
        } else {
          ++attention_unknown_count;
          add_unknown("attention_kernel_status_unknown");
        }
        attention_segments.push_back(std::move(segment));
        attention_hash_input += "|ordinal=" + std::to_string(recordOrdinal++);
      }
    }
    unsigned region_ordinal = 0;
    JsonArray static_liveness;
    JsonArray provenance;

    // Low-precision identity: serialize the module-level low-precision
    // attributes (requested policy, resolved capability, depthwise/cast-chain
    // switches) so any change flips the plan hash and, transitively, the
    // build identity that the attribution join enforces.
    const auto int8TargetAttr =
      module->getAttrOfType<StringAttr>(contract::kInt8Target);
    const auto int8KernelAttr =
      module->getAttrOfType<StringAttr>(contract::kInt8Kernel);
    const auto int8DepthwiseAttr =
      module->getAttrOfType<BoolAttr>(contract::kInt8Depthwise);
    const auto int8CastChainAttr =
      module->getAttrOfType<BoolAttr>(contract::kInt8CastChain);
    const std::string int8_target =
      int8TargetAttr ? int8TargetAttr.getValue().str() : "unknown";
    const std::string int8_kernel =
      int8KernelAttr ? int8KernelAttr.getValue().str() : "unknown";
    const bool int8_depthwise =
      int8DepthwiseAttr ? int8DepthwiseAttr.getValue() : false;
    const bool int8_cast_chain =
      int8CastChainAttr ? int8CastChainAttr.getValue() : false;
    const auto packedConvDepthwiseAttr =
      module->getAttrOfType<BoolAttr>("ncnn.packed_conv_depthwise");
    const bool packed_conv_depthwise =
      packedConvDepthwiseAttr && packedConvDepthwiseAttr.getValue();
    const std::string layout_island_revision = "layout-island-v1";
    std::string layout_island_hash_input =
      "layout-island-revision=" + layout_island_revision +
      "|packed-conv-depthwise=" + std::to_string(packed_conv_depthwise);
    const std::string low_precision_revision = "int8-target-v1";
    std::string low_precision_hash_input =
      "low-precision-revision=" + low_precision_revision +
      "|int8-target=" + int8_target + "|int8-kernel=" + int8_kernel +
      "|int8-depthwise=" + std::to_string(int8_depthwise) +
      "|int8-cast-chain=" + std::to_string(int8_cast_chain) +
      "|int8-policy-status=" +
      (int8_kernel == "auto" && int8_target == "portable"
         ? "fallback=auto_target_unsupported"
       : int8_kernel == "auto" ? "pending_defaultization"
                               : int8_kernel);

    std::string tuning_hash_input =
      "tuning-revision=tuning-v1|profile=" + tuningProfile.getValue() +
      "|status=" + tuningStatus.getValue() +
      "|fallback-reason=" + tuningFallbackReason.getValue() +
      "|matmul-m-rows=" + std::to_string(matmulMRows.getValue()) +
      "|matmul-acc-columns=" + std::to_string(matmulAccColumns.getValue()) +
      "|row-chunk-lanes=" + std::to_string(rowChunkLanes.getValue()) +
      "|matmul-i8-rows=" + std::to_string(matmulI8Rows.getValue()) +
      "|matmul-i8-acc-columns=" + std::to_string(matmulI8AccColumns.getValue());

    constexpr StringLiteral attribution_revision = "attribution-v1";
    std::string plan_hash_input =
      "static-v1|layout-kernel-v1|workspace-slot-v1|fusion-v1|conv-depthwise-"
      "v1|packed-gemm-v1|" +
      attribution_revision.str() + "|" + model + "|" + targetTriple + "|" +
      std::to_string(threads) + "|" + std::to_string(vectorLanes) + "|" +
      std::to_string(vectorScalable.getValue()) + "|" +
      std::to_string(vectorTail.getValue()) + "|codegen=" + codegenIdentity +
      "|fusion-enabled=" + std::to_string(fusion_enabled) +
      "|fusion-selected=" + std::to_string(fusion_selected_count) +
      "|fusion-residual=" + std::to_string(fusion_residual_count) +
      "|fusion-rejected=" + std::to_string(fusion_rejected_count) +
      "|fusion-reasons=" + fusion_rejection_reasons + "|" +
      low_precision_hash_input + "|" + layout_island_hash_input + "|" +
      tuning_hash_input + "|" + attention_hash_input;

    // The static plan does not execute a runner or collect runtime counters.
    // Prepared and allocation-audit modes are reported by the numerical
    // harness, not by this compiler-side artifact.
    add_unknown("runtime_counters_not_collected");

    if (module_fusion_records) {
      std::int64_t recordOrdinal = 0;
      for (Attribute attribute : module_fusion_records) {
        auto record = dyn_cast<DictionaryAttr>(attribute);
        if (!record) {
          add_unknown("fusion_record_malformed");
          continue;
        }
        JsonObject entry;
        auto copyString = [&](StringRef attributeName, StringRef fieldName) {
          if (auto value = record.getAs<StringAttr>(attributeName)) {
            entry[fieldName.str()] = value.getValue().str();
          }
        };
        auto copyInteger = [&](StringRef attributeName, StringRef fieldName) {
          if (auto value = record.getAs<IntegerAttr>(attributeName)) {
            entry[fieldName.str()] = value.getInt();
          }
        };
        std::string functionName;
        if (auto value = record.getAs<StringAttr>("function")) {
          functionName = value.getValue().str();
        }
        std::string kind;
        if (auto value = record.getAs<StringAttr>("fusion_kind")) {
          kind = value.getValue().str();
        }
        entry["id"] = "fusion/" + functionName + "/" + kind + "#" +
                      std::to_string(recordOrdinal++);
        copyString("function", "function");
        copyString("operation", "operation");
        copyString("fusion_status", "fusion_status");
        copyString("fusion_kind", "fusion_kind");
        copyString("fusion_producer", "fusion_producer");
        copyInteger("fusion_residual_inputs", "fusion_residual_inputs");
        copyInteger("fusion_tile_width", "fusion_tile_width");
        copyInteger("fusion_intermediate_bytes", "fusion_intermediate_bytes");
        copyInteger("fusion_saved_bytes", "fusion_saved_bytes");
        fusions.push_back(std::move(entry));
        plan_hash_input += "|fusion-record=" + functionName + "|" + kind;
      }
    }

    auto add_size_fields = [&](JsonObject& object, Type type) {
      object["shape"] = shapeArray(type);
      object["element_type"] = printType(
        dyn_cast<ShapedType>(type) ? dyn_cast<ShapedType>(type).getElementType()
                                   : type);
      ByteSize size = checkedByteSize(type);
      if (size.bytes) {
        object["bytes"] = *size.bytes;
      } else {
        object["bytes"] = nullptr;
        if (size.dynamic) {
          ++dynamic_buffer_count;
        } else if (size.overflow) {
          add_unknown("buffer_bytes_overflow");
        } else if (size.unknown_element_type) {
          add_unknown("buffer_element_type_size_unknown");
        }
      }
    };

    module.walk([&](func::FuncOp function) {
      const std::string function_name = function.getName().str();
      std::map<std::string, unsigned> operation_ordinals;
      std::map<Operation*, std::string> operation_ids;
      std::map<Operation*, std::int64_t> operation_positions;
      std::int64_t operation_position = 0;
      const bool straight_line =
        function.getBody().hasOneBlock() &&
        llvm::all_of(function.getBody().front(), [](Operation& operation) {
          return operation.getNumRegions() == 0;
        });
      function.walk([&](Operation* operation) {
        const std::string kind = operationKind(*operation);
        const std::string ordinal_key = function_name + "/" + kind;
        const unsigned ordinal = operation_ordinals[ordinal_key]++;
        const std::string operation_id =
          function_name + "/" + kind + "#" + std::to_string(ordinal);
        operation_ids.emplace(operation, operation_id);
        operation_positions.emplace(operation, operation_position++);
      });

      llvm::DenseMap<Operation*, Lifetime> lifetimes;
      SmallVector<Operation*> allocation_order;
      for (auto& [operation, operation_id] : operation_ids) {
        if (!isa<memref::AllocOp>(operation)) {
          continue;
        }
        Lifetime lifetime;
        lifetime.allocation = operation;
        lifetime.id = operation_id;
        lifetime.allocationPosition = operation_positions[operation];
        const ByteSize size =
          checkedByteSize(cast<memref::AllocOp>(operation).getType());
        lifetime.bytes = size.bytes;
        if (!size.bytes) {
          lifetime.aliasUnknown = true;
        }
        lifetimes[operation] = std::move(lifetime);
      }
      function.walk([&](Operation* operation) {
        if (isa<memref::AllocOp>(operation)) {
          allocation_order.push_back(operation);
        }
      });

      for (Operation* allocation : allocation_order) {
        Lifetime& lifetime = lifetimes.find(allocation)->second;
        SmallPtrSet<Value, 16> visited;
        std::function<void(Value)> visit = [&](Value value) {
          if (!visited.insert(value).second) {
            return;
          }
          for (Operation* user : value.getUsers()) {
            const auto position = operation_positions.find(user);
            if (position == operation_positions.end()) {
              lifetime.aliasUnknown = true;
              continue;
            }
            if (isa<memref::DeallocOp>(user)) {
              if (lifetime.deallocation &&
                  *lifetime.deallocation != position->second) {
                lifetime.aliasUnknown = true;
              } else {
                lifetime.deallocation = position->second;
              }
              continue;
            }
            if (isViewLike(*user)) {
              for (Value result : user->getResults()) {
                visit(result);
              }
              continue;
            }
            // A call or return can retain or expose the allocation beyond this
            // function.  A raw-pointer extraction has the same ambiguity.  Do
            // not claim a local lifetime proof merely because a dealloc
            // follows.
            if (isa<func::CallOp, func::ReturnOp>(user) ||
                isNamed(*user, "memref.extract_aligned_pointer_as_index")) {
              lifetime.aliasUnknown = true;
            }
            if (!lifetime.firstUse || position->second < *lifetime.firstUse) {
              lifetime.firstUse = position->second;
            }
            if (!lifetime.lastUse || position->second > *lifetime.lastUse) {
              lifetime.lastUse = position->second;
            }
          }
        };
        visit(cast<memref::AllocOp>(allocation).getResult());

        JsonObject lifetime_object;
        lifetime_object["id"] = lifetime.id;
        lifetime_object["profile_id"] = profileId(lifetime.id);
        if (lifetime.bytes) {
          lifetime_object["bytes"] = *lifetime.bytes;
        } else {
          lifetime_object["bytes"] = nullptr;
        }
        lifetime_object["allocation_position"] = lifetime.allocationPosition;
        if (lifetime.firstUse) {
          lifetime_object["first_use_position"] = *lifetime.firstUse;
        } else {
          lifetime_object["first_use_position"] = nullptr;
        }
        if (lifetime.lastUse) {
          lifetime_object["last_use_position"] = *lifetime.lastUse;
        } else {
          lifetime_object["last_use_position"] = nullptr;
        }
        if (lifetime.deallocation) {
          lifetime_object["deallocation_position"] = *lifetime.deallocation;
        } else {
          lifetime_object["deallocation_position"] = nullptr;
        }
        const bool proven = straight_line && lifetime.bytes &&
                            lifetime.firstUse && lifetime.lastUse &&
                            lifetime.deallocation && !lifetime.aliasUnknown;
        lifetime_object["status"] = proven ? "proven" : "unknown";
        auto allocation_op = cast<memref::AllocOp>(allocation);
        auto copy_workspace_string = [&](StringRef attribute, StringRef field) {
          if (auto value =
                allocation_op->getAttrOfType<StringAttr>(attribute)) {
            lifetime_object[field.str()] = value.getValue().str();
          }
        };
        auto copy_workspace_integer = [&](StringRef attribute,
                                          StringRef field) {
          if (auto value =
                allocation_op->getAttrOfType<IntegerAttr>(attribute)) {
            lifetime_object[field.str()] = value.getInt();
          }
        };
        copy_workspace_string("ncnn.workspace_reuse_status",
                              "workspace_reuse_status");
        copy_workspace_string("ncnn.workspace_fallback_reason",
                              "workspace_fallback_reason");
        copy_workspace_string("ncnn.workspace_slot_owner",
                              "workspace_slot_owner");
        copy_workspace_string("ncnn.workspace_slot_thread_visibility",
                              "workspace_slot_thread_visibility");
        copy_workspace_integer("ncnn.workspace_slot", "workspace_slot");
        copy_workspace_integer("ncnn.workspace_slot_lifetime_begin",
                               "workspace_slot_lifetime_begin");
        copy_workspace_integer("ncnn.workspace_slot_lifetime_end",
                               "workspace_slot_lifetime_end");
        copy_workspace_integer("ncnn.workspace_slot_bytes",
                               "workspace_slot_bytes");
        copy_workspace_integer("ncnn.workspace_slot_alignment",
                               "workspace_slot_alignment");
        copy_workspace_integer("ncnn.workspace_reuse_count",
                               "workspace_reuse_count");
        if (!proven) {
          add_unknown("buffer_liveness_unknown");
        }
        static_liveness.push_back(std::move(lifetime_object));
      }

      if (!lifetimes.empty()) {
        bool function_peak_known = straight_line;
        std::vector<std::pair<std::int64_t, std::int64_t>> events;
        if (!straight_line) {
          add_unknown("peak_workspace_control_flow_unknown");
        }
        for (Operation* allocation : allocation_order) {
          const Lifetime& lifetime = lifetimes.find(allocation)->second;
          if (!lifetime.bytes || !lifetime.deallocation ||
              lifetime.aliasUnknown ||
              (lifetime.lastUse &&
               *lifetime.deallocation < *lifetime.lastUse)) {
            function_peak_known = false;
            continue;
          }
          events.emplace_back(lifetime.allocationPosition, *lifetime.bytes);
          events.emplace_back(*lifetime.deallocation, -*lifetime.bytes);
        }
        if (function_peak_known && !events.empty()) {
          std::ranges::sort(events, [](const auto& left, const auto& right) {
            if (left.first == right.first) {
              return left.second < right.second;
            }
            return left.first < right.first;
          });
          std::int64_t live = 0;
          std::int64_t function_peak = 0;
          for (const auto& event : events) {
            const std::int64_t delta = event.second;
            if (delta > 0) {
              if (live > std::numeric_limits<std::int64_t>::max() - delta) {
                function_peak_known = false;
                break;
              }
              live += delta;
              function_peak = std::max(function_peak, live);
            } else {
              if (live < -delta) {
                function_peak_known = false;
                break;
              }
              live += delta;
            }
          }
          if (function_peak_known && !peak_workspace_unknown) {
            static_peak_live_bytes =
              std::max(static_peak_live_bytes.value_or(0), function_peak);
          }
        }
        if (!function_peak_known) {
          static_peak_live_bytes = std::nullopt;
          peak_workspace_unknown = true;
          add_unknown("peak_workspace_liveness_unknown");
        }
      }

      JsonObject function_object;
      function_object["id"] = function_name;
      function_object["name"] = function_name;
      JsonArray arguments;
      for (Type type : function.getArgumentTypes()) {
        JsonObject argument;
        add_size_fields(argument, type);
        arguments.push_back(std::move(argument));
      }
      JsonArray results;
      for (Type type : function.getResultTypes()) {
        JsonObject result;
        add_size_fields(result, type);
        results.push_back(std::move(result));
      }
      function_object["arguments"] = std::move(arguments);
      function_object["results"] = std::move(results);
      functions.push_back(std::move(function_object));

      JsonObject body_region;
      body_region["id"] =
        function_name + "/region#" + std::to_string(region_ordinal++);
      body_region["function"] = function_name;
      body_region["kind"] = "function_body";
      regions.push_back(std::move(body_region));

      function.walk([&](Operation* operation) {
        const std::string kind = operationKind(*operation);
        const std::string& operation_id = operation_ids[operation];
        plan_hash_input += "|" + operation_id + "|" + kind;
        for (Value operand : operation->getOperands()) {
          plan_hash_input += "|" + printType(operand.getType());
        }
        for (Type result : operation->getResultTypes()) {
          plan_hash_input += "|" + printType(result);
        }
        plan_hash_input += "|attrs=" + printAttributes(*operation);
        if (operation->hasAttr(contract::kPackedWeight)) {
          std::string packedId = operation_id;
          if (auto global = dyn_cast<memref::GetGlobalOp>(operation)) {
            packedId = "global/" + global.getName().str();
          }
          if (packed_constant_ids.insert(packedId).second) {
            ++packed_constant_count;
            Type packedType = operation->getNumResults() != 0
                                ? operation->getResult(0).getType()
                                : Type();
            ByteSize packedSize = checkedByteSize(packedType);
            if (packedSize.bytes && !packed_constant_bytes_unknown &&
                packed_constant_bytes <=
                  std::numeric_limits<std::int64_t>::max() -
                    *packedSize.bytes) {
              packed_constant_bytes += *packedSize.bytes;
            } else {
              packed_constant_bytes_unknown = true;
              add_unknown("packed_constant_bytes_unknown");
            }
          }
        }
        if (auto source =
              operation->getAttrOfType<IntegerAttr>("ncnn.source_layer")) {
          plan_hash_input += "|source-layer=" + std::to_string(source.getInt());
        }
        if (auto name = operation->getAttrOfType<StringAttr>("ncnn.name")) {
          plan_hash_input += "|source-name=" + name.getValue().str();
        }

        const bool has_contract =
          operation->hasAttr(contract::kContract) ||
          operation->hasAttr(contract::kKernel) ||
          operation->hasAttr(contract::kPacking) ||
          operation->hasAttr(contract::kFallback) ||
          operation->hasAttr(contract::kOperationFamily) ||
          operation->hasAttr(contract::kImplementation) ||
          operation->hasAttr(contract::kFusion) ||
          operation->hasAttr(contract::kLayoutIslandId);
        auto make_contract = [&]() {
          JsonObject result;
          auto copy_string = [&](StringRef attribute, StringRef field) {
            if (auto value = operation->getAttrOfType<StringAttr>(attribute)) {
              result[field.str()] = value.getValue().str();
            }
          };
          auto copy_integer = [&](StringRef attribute, StringRef field) {
            if (auto value = operation->getAttrOfType<IntegerAttr>(attribute)) {
              result[field.str()] = value.getInt();
            }
          };
          auto copy_bool = [&](StringRef attribute, StringRef field) {
            if (auto value = operation->getAttrOfType<BoolAttr>(attribute)) {
              result[field.str()] = value.getValue();
            }
          };
          copy_string(contract::kLayout, "layout");
          copy_string(contract::kInputLayout, "input_layout");
          copy_string(contract::kWeightLayout, "weight_layout");
          copy_string(contract::kOutputLayout, "output_layout");
          copy_string(contract::kPacking, "packing");
          copy_integer(contract::kPackFactor, "pack_factor");
          copy_integer(contract::kPackBytes, "pack_bytes");
          copy_integer(contract::kUnpackBytes, "unpack_bytes");
          copy_string(contract::kPackSchema, "pack_schema");
          copy_integer(contract::kPackRawBytes, "pack_raw_bytes");
          copy_integer(contract::kPackTileK, "pack_tile_k");
          copy_string(contract::kPackRuntime, "pack_runtime");
          copy_bool(contract::kPackedWeight, "packed_weight");
          copy_integer(contract::kTileM, "tile_m");
          copy_integer(contract::kTileN, "tile_n");
          copy_integer(contract::kTileK, "tile_k");
          copy_string(contract::kKernel, "kernel");
          copy_string(contract::kParallel, "parallel");
          copy_integer(contract::kSimdLanes, "simd_lanes");
          copy_integer(contract::kSimdChunk, "simd_chunk");
          copy_string(contract::kFma, "fma");
          copy_string(contract::kTail, "tail");
          copy_string(contract::kAlignment, "alignment");
          copy_string(contract::kAlias, "alias");
          copy_string(contract::kContract, "status");
          copy_string(contract::kFallback, "fallback_reason");
          copy_string(contract::kOperationFamily, "operation_family");
          copy_string(contract::kImplementation, "implementation");
          copy_bool(contract::kKernelStatic, "kernel_static");
          copy_integer(contract::kKernelHeight, "kernel_h");
          copy_integer(contract::kKernelWidth, "kernel_w");
          copy_integer(contract::kStrideHeight, "stride_h");
          copy_integer(contract::kStrideWidth, "stride_w");
          copy_integer(contract::kDilationHeight, "dilation_h");
          copy_integer(contract::kDilationWidth, "dilation_w");
          copy_integer(contract::kInputChannels, "input_channels");
          copy_integer(contract::kOutputChannels, "output_channels");
          copy_integer(contract::kMultiplier, "multiplier");
          copy_string(contract::kLayoutIslandId, "layout_island_id");
          copy_string(contract::kLayoutIslandStatus, "layout_island_status");
          copy_string(contract::kLayoutIslandEntry, "layout_island_entry");
          copy_string(contract::kLayoutIslandExit, "layout_island_exit");
          copy_string(contract::kLayoutIslandCost, "layout_island_cost");
          copy_string(contract::kLayoutIslandReason, "layout_island_reason");
          copy_integer(contract::kLayoutPackFactor, "layout_pack_factor");
          copy_integer(contract::kLayoutChannelBlocks, "layout_channel_blocks");
          copy_string(contract::kFusion, "fusion_status");
          copy_string(contract::kFusionKind, "fusion_kind");
          copy_string(contract::kFusionProducer, "fusion_producer");
          copy_integer(contract::kFusionResidualInputs,
                       "fusion_residual_inputs");
          copy_integer(contract::kFusionTileWidth, "fusion_tile_width");
          copy_integer(contract::kFusionIntermediateBytes,
                       "fusion_intermediate_bytes");
          copy_integer(contract::kFusionSavedBytes, "fusion_saved_bytes");
          return result;
        };
        auto family =
          operation->getAttrOfType<StringAttr>(contract::kOperationFamily);
        if (family &&
            (family.getValue() == "conv" || family.getValue() == "depthwise")) {
          const bool isDepthwise = family.getValue() == "depthwise";
          const std::string familyName = family.getValue().str();
          std::string implementation = "unknown";
          if (auto value = operation->getAttrOfType<StringAttr>(
                contract::kImplementation)) {
            implementation = value.getValue().str();
          }
          auto& implementationCounts =
            isDepthwise ? depthwise_implementations : conv_implementations;
          ++implementationCounts[implementation];
          if (isDepthwise) {
            ++depthwise_operation_count;
          } else {
            ++conv_operation_count;
          }
          if (implementation == "fallback") {
            if (isDepthwise) {
              ++depthwise_fallback_count;
            } else {
              ++conv_fallback_count;
            }
            std::string reason = "unknown";
            if (auto value =
                  operation->getAttrOfType<StringAttr>(contract::kFallback)) {
              reason = value.getValue().str();
            }
            auto& reasonCounts =
              isDepthwise ? depthwise_fallback_reasons : conv_fallback_reasons;
            ++reasonCounts[reason];
          } else if (implementation == "unknown") {
            if (isDepthwise) {
              ++depthwise_unknown_count;
            } else {
              ++conv_unknown_count;
            }
          }
          JsonObject familyEntry = make_contract();
          familyEntry["id"] = operation_id;
          familyEntry["operation"] = kind;
          familyEntry["function"] = function_name;
          familyEntry["family"] = familyName;
          if (auto source =
                operation->getAttrOfType<IntegerAttr>("ncnn.source_layer")) {
            familyEntry["source_layer"] = source.getInt();
          } else {
            familyEntry["source_layer"] = nullptr;
          }
          if (auto name = operation->getAttrOfType<StringAttr>("ncnn.name")) {
            familyEntry["source_name"] = name.getValue().str();
          } else {
            familyEntry["source_name"] = nullptr;
          }
          conv_depthwise_operations.push_back(std::move(familyEntry));
        }
        if (auto island =
              operation->getAttrOfType<StringAttr>(contract::kLayoutIslandId)) {
          ++layout_island_count;
          layout_island_hash_input += "|id=" + island.getValue().str();
          if (auto status = operation->getAttrOfType<StringAttr>(
                contract::kLayoutIslandStatus)) {
            layout_island_hash_input += "|status=" + status.getValue().str();
            if (status.getValue() == "selected") {
              ++layout_island_selected_count;
            } else if (status.getValue() == "rejected") {
              ++layout_island_rejected_count;
            }
          }
          if (auto factor = operation->getAttrOfType<IntegerAttr>(
                contract::kLayoutPackFactor)) {
            if (factor.getInt() == 4) {
              ++layout_pack4_count;
            } else if (factor.getInt() == 8) {
              ++layout_pack8_count;
            }
          }
        }
        if (has_contract) {
          ++kernel_contract_count;
          if (auto status =
                operation->getAttrOfType<StringAttr>(contract::kContract);
              status && status.getValue() == "fallback") {
            ++kernel_contract_fallback_count;
          }
          if (auto packing =
                operation->getAttrOfType<StringAttr>(contract::kPacking);
              packing && packing.getValue() != "unpacked" &&
              packing.getValue() != "none") {
            ++packed_operation_count;
            const bool compileTimeB =
              packing.getValue() == "prepacked_B" &&
              operation->getAttrOfType<StringAttr>(contract::kPackRuntime) &&
              operation->getAttrOfType<StringAttr>(contract::kPackRuntime)
                  .getValue() == "compile_time_B";
            // A compile-time B panel is accounted for by its deduplicated
            // packed global below, not once per tiled forall operation.
            if (!compileTimeB) {
              if (auto bytes = operation->getAttrOfType<IntegerAttr>(
                    contract::kPackBytes)) {
                if (!packed_buffer_bytes_unknown && bytes.getInt() > 0 &&
                    packed_buffer_bytes <=
                      std::numeric_limits<std::int64_t>::max() -
                        bytes.getInt()) {
                  packed_buffer_bytes += bytes.getInt();
                } else {
                  packed_buffer_bytes_unknown = true;
                }
                if (!static_pack_bytes_unknown && bytes.getInt() >= 0 &&
                    static_pack_bytes <=
                      std::numeric_limits<std::int64_t>::max() -
                        bytes.getInt()) {
                  static_pack_bytes += bytes.getInt();
                } else {
                  static_pack_bytes_unknown = true;
                }
              } else {
                packed_buffer_bytes_unknown = true;
                static_pack_bytes_unknown = true;
              }
            }
          }
          if (auto bytes =
                operation->getAttrOfType<IntegerAttr>(contract::kUnpackBytes);
              bytes) {
            if (bytes.getInt() > 0) {
              ++unpacked_operation_count;
            }
            if (!static_unpack_bytes_unknown && bytes.getInt() >= 0 &&
                static_unpack_bytes <=
                  std::numeric_limits<std::int64_t>::max() - bytes.getInt()) {
              static_unpack_bytes += bytes.getInt();
            } else {
              static_unpack_bytes_unknown = true;
            }
          }
          JsonObject contract_entry = make_contract();
          contract_entry["id"] = operation_id;
          contract_entry["operation"] = kind;
          contract_entry["function"] = function_name;
          const auto operationFusionStatus =
            operation->getAttrOfType<StringAttr>(contract::kFusion);
          const bool operationFusionIsSelected =
            operationFusionStatus &&
            operationFusionStatus.getValue() == "selected";
          if (operation->hasAttr(contract::kFusion) &&
              (!operationFusionIsSelected || !has_module_fusion_records)) {
            JsonObject fusion_entry = make_contract();
            fusion_entry["id"] = operation_id;
            fusion_entry["operation"] = kind;
            fusion_entry["function"] = function_name;
            fusions.push_back(std::move(fusion_entry));
          }
          contracts.push_back(std::move(contract_entry));
        }
        if (auto lowPrecision = lowPrecisionAudit(operation)) {
          JsonObject low_precision_entry = std::move(*lowPrecision);
          low_precision_entry["id"] = operation_id;
          low_precision_entry["operation"] = kind;
          low_precision_entry["function"] = function_name;
          low_precision_operations.push_back(std::move(low_precision_entry));
          plan_hash_input += "|low-precision-op=" + operation_id;
        }
        JsonObject operation_object;
        operation_object["id"] = operation_id;
        operation_object["profile_id"] = profileId(operation_id);
        if (auto source =
              operation->getAttrOfType<IntegerAttr>("ncnn.source_layer")) {
          operation_object["source_layer"] = source.getInt();
        }
        if (auto name = operation->getAttrOfType<StringAttr>("ncnn.name")) {
          operation_object["source_name"] = name.getValue().str();
        }
        if (operation->hasAttr("ncnn.source_layer") ||
            operation->hasAttr("ncnn.name")) {
          JsonObject source;
          source["operation"] = operation_id;
          if (auto layer =
                operation->getAttrOfType<IntegerAttr>("ncnn.source_layer")) {
            source["layer"] = layer.getInt();
          }
          if (auto name = operation->getAttrOfType<StringAttr>("ncnn.name")) {
            source["name"] = name.getValue().str();
          }
          provenance.push_back(std::move(source));
        }
        operation_object["function"] = function_name;
        operation_object["kind"] = kind;
        JsonArray operand_types;
        for (Type type : operation->getOperandTypes()) {
          operand_types.push_back(printType(type));
        }
        JsonArray result_types;
        for (Type type : operation->getResultTypes()) {
          result_types.push_back(printType(type));
        }
        operation_object["operand_types"] = std::move(operand_types);
        operation_object["result_types"] = std::move(result_types);
        if (has_contract) {
          operation_object["kernel_contract"] = make_contract();
        }
        operations.push_back(std::move(operation_object));

        if (isa<memref::AllocOp>(*operation)) {
          ++allocation_count;
          auto alloc = cast<memref::AllocOp>(*operation);
          JsonObject buffer;
          buffer["id"] = operation_id;
          buffer["function"] = function_name;
          buffer["ownership"] = "internal";
          buffer["type"] = printType(alloc.getType());
          buffer["profile_id"] = profileId(operation_id);
          add_size_fields(buffer, alloc.getType());
          const ByteSize size = checkedByteSize(alloc.getType());
          buffer["bytes_known"] = size.bytes.has_value();
          buffer["workspace_join_status"] = "not_applicable";
          buffer["workspace_join_reason"] = "no_workspace_slot";
          auto lifetime = lifetimes.find(operation);
          if (lifetime != lifetimes.end()) {
            const bool proven =
              straight_line && lifetime->second.bytes &&
              lifetime->second.firstUse && lifetime->second.lastUse &&
              lifetime->second.deallocation && !lifetime->second.aliasUnknown;
            buffer["liveness_status"] = proven ? "proven" : "unknown";
          }
          if (auto status = alloc->getAttrOfType<StringAttr>(
                "ncnn.workspace_reuse_status")) {
            buffer["workspace_reuse_status"] = status.getValue().str();
            if (status.getValue() == "fallback") {
              ++workspace_fallback_count;
              buffer["workspace_join_status"] = "fallback";
              if (auto reason = alloc->getAttrOfType<StringAttr>(
                    "ncnn.workspace_fallback_reason")) {
                const std::string reason_value = reason.getValue().str();
                workspace_fallback_reasons.insert(reason_value);
                buffer["workspace_join_reason"] = reason_value;
              } else {
                buffer["workspace_join_reason"] = "unknown";
              }
            }
          }
          if (auto slot =
                alloc->getAttrOfType<IntegerAttr>("ncnn.workspace_slot")) {
            buffer["workspace_slot"] = slot.getInt();
            buffer["workspace_join_status"] = "joined";
            buffer["workspace_join_reason"] = nullptr;
            const std::string slotId =
              function_name + "/" + std::to_string(slot.getInt());
            if (workspace_slot_ids.insert(slotId).second) {
              ++workspace_slot_count;
              if (auto bytes = alloc->getAttrOfType<IntegerAttr>(
                    "ncnn.workspace_slot_bytes")) {
                if (!workspace_slot_bytes_unknown && bytes.getInt() >= 0 &&
                    workspace_slot_bytes <=
                      std::numeric_limits<std::int64_t>::max() -
                        bytes.getInt()) {
                  workspace_slot_bytes += bytes.getInt();
                } else {
                  workspace_slot_bytes_unknown = true;
                }
              } else {
                workspace_slot_bytes_unknown = true;
              }
            }
          }
          auto copy_buffer_workspace_string = [&](StringRef attribute,
                                                  StringRef field) {
            if (auto value = alloc->getAttrOfType<StringAttr>(attribute)) {
              buffer[field.str()] = value.getValue().str();
            }
          };
          auto copy_buffer_workspace_integer = [&](StringRef attribute,
                                                   StringRef field) {
            if (auto value = alloc->getAttrOfType<IntegerAttr>(attribute)) {
              buffer[field.str()] = value.getInt();
            }
          };
          copy_buffer_workspace_string("ncnn.workspace_slot_owner",
                                       "workspace_slot_owner");
          copy_buffer_workspace_string("ncnn.workspace_slot_thread_visibility",
                                       "workspace_slot_thread_visibility");
          copy_buffer_workspace_integer("ncnn.workspace_slot_lifetime_begin",
                                        "workspace_slot_lifetime_begin");
          copy_buffer_workspace_integer("ncnn.workspace_slot_lifetime_end",
                                        "workspace_slot_lifetime_end");
          if (auto bytes = alloc->getAttrOfType<IntegerAttr>(
                "ncnn.workspace_slot_bytes")) {
            buffer["workspace_slot_bytes"] = bytes.getInt();
          }
          if (auto alignment = alloc->getAttrOfType<IntegerAttr>(
                "ncnn.workspace_slot_alignment")) {
            buffer["workspace_slot_alignment"] = alignment.getInt();
          }
          if (auto reuseCount = alloc->getAttrOfType<IntegerAttr>(
                "ncnn.workspace_reuse_count")) {
            buffer["workspace_reuse_count"] = reuseCount.getInt();
            if (reuseCount.getInt() > 1 &&
                workspace_reused_allocation_count <=
                  std::numeric_limits<std::int64_t>::max() -
                    (reuseCount.getInt() - 1)) {
              workspace_reused_allocation_count += reuseCount.getInt() - 1;
            }
          }
          if (!size.bytes) {
            static_buffer_bytes = std::nullopt;
          } else if (static_buffer_bytes &&
                     *static_buffer_bytes <=
                       std::numeric_limits<std::int64_t>::max() - *size.bytes) {
            *static_buffer_bytes += *size.bytes;
          } else if (static_buffer_bytes) {
            static_buffer_bytes = std::nullopt;
            add_unknown("static_buffer_bytes_overflow");
          }
          buffers.push_back(std::move(buffer));
        } else if (isa<memref::DeallocOp>(*operation)) {
          ++deallocation_count;
        } else if (isa<memref::CopyOp>(*operation)) {
          ++copy_count;
          auto copy = cast<memref::CopyOp>(*operation);
          const ByteSize size = checkedByteSize(copy.getSource().getType());
          if (size.bytes &&
              static_copy_bytes <=
                std::numeric_limits<std::int64_t>::max() - *size.bytes) {
            static_copy_bytes += *size.bytes;
          } else {
            has_unknown_copy_bytes = true;
            add_unknown("copy_bytes_unknown");
          }
        }
        if (isa<linalg::MatmulOp>(*operation) ||
            isNamed(*operation, "linalg.matmul")) {
          ++matmul_count;
        }
        if (isa<linalg::BatchMatmulOp>(*operation) ||
            isNamed(*operation, "linalg.batch_matmul")) {
          ++batch_matmul_count;
        }
        if (isa<vector::FMAOp>(*operation) ||
            isNamed(*operation, "vector.fma")) {
          ++vector_fma_count;
        }
        if (isa<scf::ForallOp, scf::ParallelOp, omp::ParallelOp>(*operation)) {
          ++parallel_region_count;
        }
        if (isa<omp::ParallelOp>(*operation) &&
            operation->getParentOfType<omp::ParallelOp>() != nullptr) {
          ++nested_openmp_count;
        }
        if (isa<memref::TransposeOp>(*operation)) {
          ++transpose_count;
          if (!has_unknown_transpose_bytes) {
            Type size_type;
            if (operation->getNumResults() != 0) {
              size_type = operation->getResult(0).getType();
            } else if (operation->getNumOperands() != 0) {
              size_type = operation->getOperand(0).getType();
            }
            const ByteSize size = checkedByteSize(size_type);
            if (size.bytes &&
                static_transpose_bytes <=
                  std::numeric_limits<std::int64_t>::max() - *size.bytes) {
              static_transpose_bytes += *size.bytes;
            } else {
              has_unknown_transpose_bytes = true;
              add_unknown("transpose_bytes_unknown");
            }
          }
        }
        if (operation == function.getOperation()) {
          return;
        }
        for (Region& region : operation->getRegions()) {
          (void)region;
          JsonObject region_object;
          region_object["id"] =
            function_name + "/region#" + std::to_string(region_ordinal++);
          region_object["function"] = function_name;
          region_object["owner"] = operation_id;
          regions.push_back(std::move(region_object));
        }
      });
    });

    if (packed_constant_count != 0) {
      if (packed_constant_bytes_unknown) {
        packed_buffer_bytes_unknown = true;
        static_pack_bytes_unknown = true;
      } else {
        if (!packed_buffer_bytes_unknown &&
            packed_buffer_bytes <= std::numeric_limits<std::int64_t>::max() -
                                     packed_constant_bytes) {
          packed_buffer_bytes += packed_constant_bytes;
        } else {
          packed_buffer_bytes_unknown = true;
        }
        if (!static_pack_bytes_unknown &&
            static_pack_bytes <= std::numeric_limits<std::int64_t>::max() -
                                   packed_constant_bytes) {
          static_pack_bytes += packed_constant_bytes;
        } else {
          static_pack_bytes_unknown = true;
        }
      }
    }

    summary["allocation_count"] = allocation_count;
    summary["deallocation_count"] = deallocation_count;
    summary["copy_count"] = copy_count;
    summary["transpose_count"] = transpose_count;
    summary["matmul_count"] = matmul_count;
    summary["batch_matmul_count"] = batch_matmul_count;
    summary["vector_fma_count"] = vector_fma_count;
    summary["parallel_region_count"] = parallel_region_count;
    summary["kernel_contract_count"] = kernel_contract_count;
    summary["kernel_contract_fallback_count"] = kernel_contract_fallback_count;
    summary["layout_island_count"] = layout_island_count;
    summary["layout_island_selected_count"] = layout_island_selected_count;
    summary["layout_island_rejected_count"] = layout_island_rejected_count;
    summary["layout_pack4_count"] = layout_pack4_count;
    summary["layout_pack8_count"] = layout_pack8_count;
    summary["packed_conv_depthwise"] = packed_conv_depthwise;
    auto mapToJson = [](const std::map<std::string, std::int64_t>& values) {
      JsonObject result;
      for (const auto& [key, value] : values) {
        result[key] = value;
      }
      return result;
    };
    JsonObject convDepthwiseSummary;
    convDepthwiseSummary["conv_operation_count"] = conv_operation_count;
    convDepthwiseSummary["depthwise_operation_count"] =
      depthwise_operation_count;
    convDepthwiseSummary["conv_fallback_count"] = conv_fallback_count;
    convDepthwiseSummary["depthwise_fallback_count"] = depthwise_fallback_count;
    convDepthwiseSummary["conv_unknown_count"] = conv_unknown_count;
    convDepthwiseSummary["depthwise_unknown_count"] = depthwise_unknown_count;
    convDepthwiseSummary["conv_implementations"] =
      mapToJson(conv_implementations);
    convDepthwiseSummary["depthwise_implementations"] =
      mapToJson(depthwise_implementations);
    convDepthwiseSummary["conv_fallback_reasons"] =
      mapToJson(conv_fallback_reasons);
    convDepthwiseSummary["depthwise_fallback_reasons"] =
      mapToJson(depthwise_fallback_reasons);
    summary["conv_depthwise"] = std::move(convDepthwiseSummary);
    summary["fusion_enabled"] = fusion_enabled;
    summary["fusion_selected_count"] = fusion_selected_count;
    summary["fusion_residual_count"] = fusion_residual_count;
    summary["fusion_rejected_count"] = fusion_rejected_count;
    summary["fusion_rejection_reasons"] = fusion_rejection_reasons;
    summary["fusion_contract_count"] =
      static_cast<std::int64_t>(fusions.size());
    summary["nested_openmp_count"] = nested_openmp_count;
    summary["workspace_slot_count"] = workspace_slot_count;
    summary["workspace_reused_allocation_count"] =
      workspace_reused_allocation_count;
    summary["workspace_fallback_count"] = workspace_fallback_count;
    summary["attention_segment_count"] =
      static_cast<std::int64_t>(attention_segments.size());
    summary["attention_selected_count"] = attention_selected_count;
    summary["attention_fallback_count"] = attention_fallback_count;
    summary["attention_unknown_count"] = attention_unknown_count;
    summary["attention_revision"] = attention_revision;
    if (workspace_slot_bytes_unknown) {
      summary["workspace_slot_bytes"] = nullptr;
      summary["workspace_slot_bytes_known"] = false;
    } else {
      summary["workspace_slot_bytes"] = workspace_slot_bytes;
      summary["workspace_slot_bytes_known"] = true;
    }
    if (packed_buffer_bytes_unknown) {
      summary["packed_buffer_bytes"] = nullptr;
      add_unknown("packed_buffer_bytes_unknown");
    } else {
      summary["packed_buffer_bytes"] = packed_buffer_bytes;
    }
    summary["packed_constant_count"] = packed_constant_count;
    if (packed_constant_bytes_unknown) {
      summary["packed_constant_bytes"] = nullptr;
      summary["packed_constant_bytes_known"] = false;
      add_unknown("packed_constant_bytes_unknown");
    } else {
      summary["packed_constant_bytes"] = packed_constant_bytes;
      summary["packed_constant_bytes_known"] = true;
    }
    summary["packed_constant_status"] =
      packed_constant_count == 0      ? "not_applicable"
      : packed_constant_bytes_unknown ? "unknown"
                                      : "known";
    if (nested_openmp_count > 0) {
      add_unknown("nested_openmp_present");
    }
    if (static_buffer_bytes) {
      summary["static_buffer_bytes"] = *static_buffer_bytes;
    } else {
      summary["static_buffer_bytes"] = nullptr;
    }
    summary["dynamic_buffer_count"] = dynamic_buffer_count;
    if (copy_count == 0 || has_unknown_copy_bytes) {
      summary["static_copy_bytes"] = nullptr;
    } else {
      summary["static_copy_bytes"] = static_copy_bytes;
    }
    summary["static_copy_bytes_known"] =
      copy_count != 0 && !has_unknown_copy_bytes;
    summary["attribution_revision"] = attribution_revision.str();
    summary["static_copy_count"] = copy_count;
    summary["static_copy_status"] = copy_count == 0          ? "not_applicable"
                                    : has_unknown_copy_bytes ? "unknown"
                                                             : "known";
    summary["static_transpose_count"] = transpose_count;
    if (transpose_count == 0 || has_unknown_transpose_bytes) {
      summary["static_transpose_bytes"] = nullptr;
    } else {
      summary["static_transpose_bytes"] = static_transpose_bytes;
    }
    summary["static_transpose_bytes_known"] =
      transpose_count != 0 && !has_unknown_transpose_bytes;
    summary["static_transpose_status"] = transpose_count == 0 ? "not_applicable"
                                         : has_unknown_transpose_bytes
                                           ? "unknown"
                                           : "known";
    summary["static_pack_count"] = packed_operation_count;
    if (packed_operation_count == 0 || static_pack_bytes_unknown) {
      summary["static_pack_bytes"] = nullptr;
    } else {
      summary["static_pack_bytes"] = static_pack_bytes;
    }
    summary["static_pack_bytes_known"] =
      packed_operation_count != 0 && !static_pack_bytes_unknown;
    summary["static_pack_status"] = packed_operation_count == 0
                                      ? "not_applicable"
                                    : static_pack_bytes_unknown ? "unknown"
                                                                : "known";
    summary["static_unpack_count"] = unpacked_operation_count;
    if (unpacked_operation_count == 0 || static_unpack_bytes_unknown) {
      summary["static_unpack_bytes"] = nullptr;
    } else {
      summary["static_unpack_bytes"] = static_unpack_bytes;
    }
    summary["static_unpack_bytes_known"] =
      unpacked_operation_count != 0 && !static_unpack_bytes_unknown;
    summary["static_unpack_status"] = unpacked_operation_count == 0
                                        ? "not_applicable"
                                      : static_unpack_bytes_unknown ? "unknown"
                                                                    : "known";
    if (static_peak_live_bytes && !peak_workspace_unknown) {
      summary["peak_workspace_bytes"] = *static_peak_live_bytes;
      summary["peak_workspace_proven"] = true;
    } else {
      summary["peak_workspace_bytes"] = nullptr;
      summary["peak_workspace_proven"] = false;
      add_unknown("peak_workspace_not_proven");
    }

    JsonObject fusion;
    fusion["enabled"] = fusion_enabled;
    fusion["selected_count"] = fusion_selected_count;
    fusion["residual_count"] = fusion_residual_count;
    fusion["rejected_count"] = fusion_rejected_count;
    fusion["rejection_reasons"] = fusion_rejection_reasons;
    fusion["contract_count"] = static_cast<std::int64_t>(fusions.size());

    JsonObject low_precision;
    low_precision["revision"] = low_precision_revision;
    low_precision["requested_policy"] = int8_kernel;
    low_precision["capability"] = int8_target;
    low_precision["depthwise_enabled"] = int8_depthwise;
    low_precision["cast_chain_enabled"] = int8_cast_chain;
    // auto is not yet a selection: it is pending defaultization until a
    // performance decision pins it, and separately the probed target may
    // simply have no dot-product ISA.  Report these distinctly; never
    // silently present auto as a selected VNNI capability.
    if (int8_kernel == "auto" && int8_target == "portable") {
      low_precision["requested_policy_status"] = "fallback";
      low_precision["requested_policy_fallback_reason"] =
        "auto_target_unsupported";
    } else if (int8_kernel == "auto") {
      low_precision["requested_policy_status"] = "pending_defaultization";
      low_precision["requested_policy_fallback_reason"] = nullptr;
    } else if (int8_kernel == "unknown") {
      low_precision["requested_policy_status"] = "unknown";
      low_precision["requested_policy_fallback_reason"] = nullptr;
    } else {
      low_precision["requested_policy_status"] = "selected";
      low_precision["requested_policy_fallback_reason"] = nullptr;
    }
    low_precision["operations"] = std::move(low_precision_operations);

    JsonObject tuning;
    tuning["revision"] = "tuning-v1";
    tuning["profile"] = tuningProfile.getValue();
    tuning["status"] = tuningStatus.getValue();
    if (tuningFallbackReason.getValue().empty()) {
      tuning["fallback_reason"] = nullptr;
    } else {
      tuning["fallback_reason"] = tuningFallbackReason.getValue();
    }
    tuning["matmul_m_rows"] = matmulMRows.getValue();
    tuning["matmul_acc_columns"] = matmulAccColumns.getValue();
    tuning["row_chunk_lanes"] =
      static_cast<std::int64_t>(rowChunkLanes.getValue());
    tuning["matmul_i8_rows"] = matmulI8Rows.getValue();
    tuning["matmul_i8_acc_columns"] = matmulI8AccColumns.getValue();

    JsonObject target;
    target["triple"] = targetTriple;
    target["threads"] = static_cast<std::int64_t>(threads);
    target["vector_lanes"] = static_cast<std::int64_t>(vectorLanes);
    target["vector_scalable"] = vectorScalable.getValue();
    target["vector_tail"] = vectorTail.getValue();

    JsonObject diagnostics;
    diagnostics["runtime_counters"] = "not_collected";
    diagnostics["prepared_runner"] = "available_in_performance_harness";
    JsonArray workspace_fallbacks;
    for (const std::string& reason : workspace_fallback_reasons) {
      workspace_fallbacks.push_back(reason);
    }
    diagnostics["workspace_fallback_reasons"] = std::move(workspace_fallbacks);
    diagnostics["unknown_fields"] = std::move(unknown_fields);

    const std::string plan_hash = std::to_string(profileId(plan_hash_input));
    JsonObject root;
    root["schema_version"] = 1;
    root["plan_revision"] =
      "static-v1|workspace-slot-v1|fusion-v1|attention-segment-v1|conv-"
      "depthwise-v1|packed-gemm-v1|layout-island-v1|int8-target-v1|tuning-v1|"
      "attribution-v1";
    root["contract_revision"] =
      "layout-kernel-v1|workspace-slot-v1|fusion-v1|attention-segment-v1|conv-"
      "depthwise-v1|packed-gemm-v1|layout-island-v1|int8-target-v1|tuning-v1|"
      "attribution-v1";
    root["attribution_revision"] = attribution_revision.str();
    root["plan_hash"] = plan_hash;
    // This identity is deliberately derived from the complete plan/codegen
    // hash, so profile/performance rows cannot join across code-generation
    // variants that happen to share model and thread names.
    root["build_identity"] = plan_hash;
    root["codegen_identity"] = codegenIdentity;
    root["codegen_identity_encoding"] = "hex-utf8-or-empty";
    root["kind"] = "ncnn.model_execution_plan";
    root["model"] = model;
    root["target"] = std::move(target);
    root["fusion"] = std::move(fusion);
    root["low_precision"] = std::move(low_precision);
    root["tuning"] = std::move(tuning);
    root["attention_revision"] = attention_revision;
    root["attention_segments"] = std::move(attention_segments);
    root["functions"] = std::move(functions);
    root["operations"] = std::move(operations);
    root["contracts"] = std::move(contracts);
    root["conv_depthwise_operations"] = std::move(conv_depthwise_operations);
    root["fusions"] = std::move(fusions);
    root["buffers"] = std::move(buffers);
    root["regions"] = std::move(regions);
    root["static_liveness"] = std::move(static_liveness);
    root["provenance"] = std::move(provenance);
    root["summary"] = std::move(summary);
    root["diagnostics"] = std::move(diagnostics);

    std::error_code error;
    llvm::raw_fd_ostream output(path, error);
    if (error) {
      module.emitError() << "cannot write execution plan '" << path
                         << "': " << error.message();
      signalPassFailure();
      return;
    }
    output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
    output.flush();
    if (output.has_error()) {
      module.emitError() << "cannot finish execution plan '" << path << "'";
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::ncnn
