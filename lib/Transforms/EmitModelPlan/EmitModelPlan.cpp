#include "ncnn-mlir/Transforms/EmitModelPlan/EmitModelPlan.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

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
    if (extent < 0 || static_cast<std::uint64_t>(extent) >
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

bool isNamed(Operation& operation, StringRef name) {
  return operation.getName().getStringRef() == name;
}

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
    std::optional<std::int64_t> static_buffer_bytes = 0;
    std::int64_t dynamic_buffer_count = 0;
    std::map<std::string, unsigned> operation_ordinals;
    unsigned region_ordinal = 0;

    unknown_fields.push_back("runtime_counters_not_collected");
    unknown_fields.push_back("prepared_runner_not_supported");
    unknown_fields.push_back("peak_workspace_not_proven");

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
          unknown_fields.push_back("buffer_bytes_overflow");
        } else if (size.unknown_element_type) {
          unknown_fields.push_back("buffer_element_type_size_unknown");
        }
      }
    };

    module.walk([&](func::FuncOp function) {
      const std::string function_name = function.getName().str();
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
        const std::string ordinal_key = function_name + "/" + kind;
        const unsigned ordinal = operation_ordinals[ordinal_key]++;
        const std::string operation_id =
          function_name + "/" + kind + "#" + std::to_string(ordinal);
        JsonObject operation_object;
        operation_object["id"] = operation_id;
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
        operations.push_back(std::move(operation_object));

        if (isa<memref::AllocOp>(*operation)) {
          ++allocation_count;
          auto alloc = cast<memref::AllocOp>(*operation);
          JsonObject buffer;
          buffer["id"] = operation_id;
          buffer["function"] = function_name;
          buffer["ownership"] = "internal";
          buffer["type"] = printType(alloc.getType());
          add_size_fields(buffer, alloc.getType());
          const ByteSize size = checkedByteSize(alloc.getType());
          if (!size.bytes) {
            static_buffer_bytes = std::nullopt;
          } else if (static_buffer_bytes &&
                     *static_buffer_bytes <=
                       std::numeric_limits<std::int64_t>::max() - *size.bytes) {
            *static_buffer_bytes += *size.bytes;
          } else if (static_buffer_bytes) {
            static_buffer_bytes = std::nullopt;
            unknown_fields.push_back("static_buffer_bytes_overflow");
          }
          buffers.push_back(std::move(buffer));
        } else if (isa<memref::DeallocOp>(*operation)) {
          ++deallocation_count;
        } else if (isa<memref::CopyOp>(*operation)) {
          ++copy_count;
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
        if (kind.find("transpose") != std::string::npos) {
          ++transpose_count;
        }
        for (Region& region : operation->getRegions()) {
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
    if (static_buffer_bytes) {
      summary["static_buffer_bytes"] = *static_buffer_bytes;
    } else {
      summary["static_buffer_bytes"] = nullptr;
    }
    summary["dynamic_buffer_count"] = dynamic_buffer_count;
    summary["peak_workspace_bytes"] = nullptr;

    JsonObject target;
    target["triple"] = targetTriple;
    target["threads"] = static_cast<std::int64_t>(threads);
    target["vector_lanes"] = static_cast<std::int64_t>(vectorLanes);
    target["vector_scalable"] = vectorScalable.getValue();
    target["vector_tail"] = vectorTail.getValue();

    JsonObject diagnostics;
    diagnostics["runtime_counters"] = "not_collected";
    diagnostics["prepared_runner"] = "not_supported";
    diagnostics["unknown_fields"] = std::move(unknown_fields);

    JsonObject root;
    root["schema_version"] = 1;
    root["kind"] = "ncnn.model_execution_plan";
    root["model"] = model;
    root["target"] = std::move(target);
    root["functions"] = std::move(functions);
    root["operations"] = std::move(operations);
    root["buffers"] = std::move(buffers);
    root["regions"] = std::move(regions);
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
