#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "ncnn-mlir/Support/KernelContract.hpp"

namespace mlir::ncnn::contract {

// The structured side channels passes use to hand analysis results across pass
// boundaries.  They are attributes on the module or the function because that
// is the only thing that survives without reordering the pipeline.
//
// This is the conservative consolidation (roadmap T-S3): one read / write /
// clear so no pass has to spell these keys itself and the attribute set is
// named in exactly one place.  It deliberately does NOT introduce a single
// `ncnn.ledger` dictionary attribute -- that would rename every field and
// break every existing reader and lit test for no behavioural gain.
//
// Lifecycle, by group:
//   * records (copyRecords / fusionRecords / attentionSegments) -- appended by
//     their producing passes mid-pipeline, read by EmitModelPlan at the end.
//   * shape contract (shapeConstraints / inputDimRelations / rankVariant /
//     dynamicRank) -- written by the importer, read by the C ABI, removed by
//     GenerateCAPI once exported.
struct ModelLedger {
  // Append-only record arrays produced mid-pipeline.
  ArrayAttr copyRecords;
  ArrayAttr fusionRecords;
  ArrayAttr attentionSegments;

  // Shape contract carried from the importer to the C ABI.
  ArrayAttr shapeConstraints;
  ArrayAttr inputDimRelations;
  IntegerAttr rankVariant;
  // UnitAttr; null means the module/function is not flagged dynamic rank.
  UnitAttr dynamicRank;

  // Snapshot the ledger currently carried by `moduleOrFunc`.  Absent fields
  // stay null; reading a module that carries none of them is not an error.
  static ModelLedger read(Operation* moduleOrFunc) {
    ModelLedger ledger;
    ledger.copyRecords = moduleOrFunc->getAttrOfType<ArrayAttr>(kCopyRecords);
    ledger.fusionRecords =
      moduleOrFunc->getAttrOfType<ArrayAttr>(kFusionRecords);
    ledger.attentionSegments =
      moduleOrFunc->getAttrOfType<ArrayAttr>(kAttentionSegments);
    ledger.shapeConstraints =
      moduleOrFunc->getAttrOfType<ArrayAttr>(kShapeConstraints);
    ledger.inputDimRelations =
      moduleOrFunc->getAttrOfType<ArrayAttr>(kInputDimRelations);
    ledger.rankVariant = moduleOrFunc->getAttrOfType<IntegerAttr>(kRankVariant);
    ledger.dynamicRank = moduleOrFunc->getAttrOfType<UnitAttr>(kDynamicRank);
    return ledger;
  }

  // Write back only the fields that are non-null.  Read-modify-write is the
  // intended use: read, change one field, write.  That keeps a pass from
  // silently dropping a side channel it does not care about.
  void write(Operation* moduleOrFunc) const {
    if (copyRecords) {
      moduleOrFunc->setAttr(kCopyRecords, copyRecords);
    }
    if (fusionRecords) {
      moduleOrFunc->setAttr(kFusionRecords, fusionRecords);
    }
    if (attentionSegments) {
      moduleOrFunc->setAttr(kAttentionSegments, attentionSegments);
    }
    if (shapeConstraints) {
      moduleOrFunc->setAttr(kShapeConstraints, shapeConstraints);
    }
    if (inputDimRelations) {
      moduleOrFunc->setAttr(kInputDimRelations, inputDimRelations);
    }
    if (rankVariant) {
      moduleOrFunc->setAttr(kRankVariant, rankVariant);
    }
    if (dynamicRank) {
      moduleOrFunc->setAttr(kDynamicRank, dynamicRank);
    }
  }

  // Drop the dimension constraints once the C ABI has consumed them.
  //
  // Deliberately does NOT touch rankVariant / dynamicRank: those describe the
  // exported function itself and stay on it.  The two lifecycles end at
  // different passes, which is precisely why the clears are named rather than
  // inlined as a removeAttr list at the call site.
  static void clearDimensionConstraints(Operation* moduleOrFunc) {
    moduleOrFunc->removeAttr(kShapeConstraints);
    moduleOrFunc->removeAttr(kInputDimRelations);
  }

  // Drop just the record ledgers, once the plan has been emitted from them.
  static void clearRecords(Operation* moduleOrFunc) {
    moduleOrFunc->removeAttr(kCopyRecords);
    moduleOrFunc->removeAttr(kFusionRecords);
    moduleOrFunc->removeAttr(kAttentionSegments);
  }

  // Drop the whole ledger, rank included.  Teardown of a scratch module; the
  // normal pipeline uses the two narrower clears above.
  static void clear(Operation* moduleOrFunc) {
    clearDimensionConstraints(moduleOrFunc);
    clearRecords(moduleOrFunc);
    moduleOrFunc->removeAttr(kRankVariant);
    moduleOrFunc->removeAttr(kDynamicRank);
  }
};

}  // namespace mlir::ncnn::contract
