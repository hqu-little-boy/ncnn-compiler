#pragma once

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/xxhash.h"

namespace ncnn_mlir {

// One stable 64-bit hash for plan identity and profile site ids.
//
// The algorithm lives here and nowhere else on purpose.  Three call sites must
// agree bit-for-bit on the same input string: the plan's `operations[].
// profile_id`, the profile runtime's event `id`, and the `ncnn.fusion_site_id`
// attribute.  They are written by three different passes.  If any one of them
// hashed differently, `tools/attr/attr_join.py` would quietly drop the
// operation into the attribution report's `unattributed` bucket rather than
// fail, so the single home is load-bearing.
//
// Wraps `llvm::xxHash64` (LLVM 21: `uint64_t xxHash64(StringRef)` — no seed
// parameter, symbol from LLVMSupport).  Deterministic across processes and
// never exposes addresses, which is what the hand-written FNV-1a copies this
// replaces were relied on for.
//
// This is a *stable* hash, not a *persistent* one: swapping the algorithm
// changes every `plan_hash` / `build_identity` / `profile_id`, which is an
// identity change and invalidates cross-period plan-hash joins.  See
// docs/refactor/p3-2026-09-26/identity-changes.md.
inline std::uint64_t stableHash64(llvm::StringRef data) {
  return llvm::xxHash64(data);
}

}  // namespace ncnn_mlir
