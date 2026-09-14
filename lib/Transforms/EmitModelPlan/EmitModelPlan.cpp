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
    std::int64_t nested_openmp_count = 0;
    std::int64_t packed_buffer_bytes = 0;
    bool packed_buffer_bytes_unknown = false;
    std::optional<std::int64_t> static_buffer_bytes = 0;
    std::optional<std::int64_t> static_peak_live_bytes;
    bool peak_workspace_unknown = false;
    std::int64_t dynamic_buffer_count = 0;
    std::int64_t static_copy_bytes = 0;
    bool has_unknown_copy_bytes = false;
    std::set<std::string> unknown_reasons;
    auto add_unknown = [&](StringRef reason) {
      if (unknown_reasons.insert(reason.str()).second) {
        unknown_fields.push_back(reason.str());
      }
    };
    unsigned region_ordinal = 0;
    JsonArray static_liveness;
    JsonArray provenance;
    std::string plan_hash_input =
      "static-v1|layout-kernel-v1|" + model + "|" + targetTriple + "|" +
      std::to_string(threads) + "|" + std::to_string(vectorLanes) + "|" +
      std::to_string(vectorScalable.getValue()) + "|" +
      std::to_string(vectorTail.getValue()) + "|codegen=" + codegenIdentity;

    // The static plan does not execute a runner or collect runtime counters.
    // Prepared and allocation-audit modes are reported by the numerical
    // harness, not by this compiler-side artifact.
    add_unknown("runtime_counters_not_collected");

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
        if (auto source =
              operation->getAttrOfType<IntegerAttr>("ncnn.source_layer")) {
          plan_hash_input += "|source-layer=" + std::to_string(source.getInt());
        }
        if (auto name = operation->getAttrOfType<StringAttr>("ncnn.name")) {
          plan_hash_input += "|source-name=" + name.getValue().str();
        }

        const bool has_contract = operation->hasAttr(contract::kContract) ||
                                  operation->hasAttr(contract::kKernel) ||
                                  operation->hasAttr(contract::kPacking) ||
                                  operation->hasAttr(contract::kFallback);
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
          copy_string(contract::kLayout, "layout");
          copy_string(contract::kInputLayout, "input_layout");
          copy_string(contract::kWeightLayout, "weight_layout");
          copy_string(contract::kOutputLayout, "output_layout");
          copy_string(contract::kPacking, "packing");
          copy_integer(contract::kPackFactor, "pack_factor");
          copy_integer(contract::kPackBytes, "pack_bytes");
          copy_integer(contract::kUnpackBytes, "unpack_bytes");
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
          return result;
        };
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
            if (auto bytes =
                  operation->getAttrOfType<IntegerAttr>(contract::kPackBytes)) {
              if (!packed_buffer_bytes_unknown && bytes.getInt() > 0 &&
                  packed_buffer_bytes <=
                    std::numeric_limits<std::int64_t>::max() - bytes.getInt()) {
                packed_buffer_bytes += bytes.getInt();
              } else {
                packed_buffer_bytes_unknown = true;
              }
            } else {
              packed_buffer_bytes_unknown = true;
            }
          }
          JsonObject contract_entry = make_contract();
          contract_entry["id"] = operation_id;
          contract_entry["operation"] = kind;
          contract_entry["function"] = function_name;
          contracts.push_back(std::move(contract_entry));
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
          auto lifetime = lifetimes.find(operation);
          if (lifetime != lifetimes.end()) {
            const bool proven =
              straight_line && lifetime->second.bytes &&
              lifetime->second.firstUse && lifetime->second.lastUse &&
              lifetime->second.deallocation && !lifetime->second.aliasUnknown;
            buffer["liveness_status"] = proven ? "proven" : "unknown";
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
        if (kind.find("transpose") != std::string::npos) {
          ++transpose_count;
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
    summary["nested_openmp_count"] = nested_openmp_count;
    if (packed_buffer_bytes_unknown) {
      summary["packed_buffer_bytes"] = nullptr;
      add_unknown("packed_buffer_bytes_unknown");
    } else {
      summary["packed_buffer_bytes"] = packed_buffer_bytes;
    }
    if (nested_openmp_count > 0) {
      add_unknown("nested_openmp_present");
    }
    if (static_buffer_bytes) {
      summary["static_buffer_bytes"] = *static_buffer_bytes;
    } else {
      summary["static_buffer_bytes"] = nullptr;
    }
    summary["dynamic_buffer_count"] = dynamic_buffer_count;
    if (has_unknown_copy_bytes) {
      summary["static_copy_bytes"] = nullptr;
    } else {
      summary["static_copy_bytes"] = static_copy_bytes;
    }
    summary["static_copy_bytes_known"] = !has_unknown_copy_bytes;
    if (static_peak_live_bytes && !peak_workspace_unknown) {
      summary["peak_workspace_bytes"] = *static_peak_live_bytes;
      summary["peak_workspace_proven"] = true;
    } else {
      summary["peak_workspace_bytes"] = nullptr;
      summary["peak_workspace_proven"] = false;
      add_unknown("peak_workspace_not_proven");
    }

    JsonObject target;
    target["triple"] = targetTriple;
    target["threads"] = static_cast<std::int64_t>(threads);
    target["vector_lanes"] = static_cast<std::int64_t>(vectorLanes);
    target["vector_scalable"] = vectorScalable.getValue();
    target["vector_tail"] = vectorTail.getValue();

    JsonObject diagnostics;
    diagnostics["runtime_counters"] = "not_collected";
    diagnostics["prepared_runner"] = "available_in_performance_harness";
    diagnostics["unknown_fields"] = std::move(unknown_fields);

    const std::string plan_hash = std::to_string(profileId(plan_hash_input));
    JsonObject root;
    root["schema_version"] = 1;
    root["plan_revision"] = "static-v1";
    root["contract_revision"] = "layout-kernel-v1";
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
    root["functions"] = std::move(functions);
    root["operations"] = std::move(operations);
    root["contracts"] = std::move(contracts);
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
