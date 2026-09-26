#define _POSIX_C_SOURCE 200809L

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NCNN_PROFILE_DEFAULT_MODEL
#define NCNN_PROFILE_DEFAULT_MODEL ""
#endif
#ifndef NCNN_PROFILE_DEFAULT_TARGET
#define NCNN_PROFILE_DEFAULT_TARGET ""
#endif
#ifndef NCNN_PROFILE_DEFAULT_THREADS
#define NCNN_PROFILE_DEFAULT_THREADS ""
#endif
#ifndef NCNN_PROFILE_DEFAULT_PLAN_HASH
#define NCNN_PROFILE_DEFAULT_PLAN_HASH ""
#endif
#ifndef NCNN_PROFILE_DEFAULT_BUILD_IDENTITY
#define NCNN_PROFILE_DEFAULT_BUILD_IDENTITY ""
#endif
#ifndef NCNN_PROFILE_DEFAULT_PLAN_REVISION
#define NCNN_PROFILE_DEFAULT_PLAN_REVISION ""
#endif
#ifndef NCNN_PROFILE_DEFAULT_ATTRIBUTION_REVISION
#define NCNN_PROFILE_DEFAULT_ATTRIBUTION_REVISION "attribution-v4"
#endif

#define NCNN_PROFILE_MAX_RECORDS 4096
#define NCNN_PROFILE_MAX_STACK 128
#define NCNN_PROFILE_MAX_ALLOCS 4096
#define NCNN_PROFILE_MAX_SLOTS 256
#define NCNN_PROFILE_HASH_SIZE 8192
#define NCNN_PROFILE_DEFAULT_INTERVAL_CAP 262144
#define NCNN_PROFILE_DEFAULT_WINDOW_NS 8000000ULL

enum {
  NCNN_PROFILE_OPERATION = 0,
  NCNN_PROFILE_ALLOCATION = 1,
  NCNN_PROFILE_DEALLOCATION = 2,
  NCNN_PROFILE_COPY = 3,
  NCNN_PROFILE_PARALLEL = 4,
  NCNN_PROFILE_TRANSPOSE = 5,
  NCNN_PROFILE_PACK = 6,
  NCNN_PROFILE_UNPACK = 7,
  NCNN_PROFILE_MATERIALIZED_WRITE = 8,
  NCNN_PROFILE_MATERIALIZED_READ = 9,
  NCNN_PROFILE_FUSION_SITE = 10,
  NCNN_PROFILE_WORKER_OPERATION = 11,
};

typedef struct {
  uint64_t id;
  uint64_t calls;
  uint64_t inclusive_ns;
  uint64_t exclusive_ns;
  uint64_t bytes;
  uint64_t worker_wall_union_ns;
  uint64_t worker_wall_attributed_ns;
  uint64_t worker_wall_attributed_estimated_ns;
  uint64_t worker_sampled_calls;
  uint64_t worker_covered_wall_ns;
  uint64_t worker_covered_wall_estimated_ns;
  uint64_t sampled_window_wall_ns;
  uint64_t region_wall_ns;
  uint64_t wall_exclusive_estimated_ns;
  uint64_t region_worker_spans;
  int64_t category;
  int exclusive_known;
  int worker_wall_union_known;
  int worker_wall_attributed_known;
  int wall_coverage_known;
  int bytes_known;
} ncnn_profile_record;

typedef struct {
  uint64_t id;
  uint64_t start_ns;
  uint64_t child_ns;
  uint64_t window_limit;
  int64_t category;
  int record;
  int sampled;
} ncnn_profile_frame;

/* Worker spans are collected per thread and merged at flush.  This keeps the
   parallel hot path free of the global profile lock; only thread registration
   and flush take it. */
typedef struct {
  uint64_t id;
  uint64_t calls;
  uint64_t inclusive_ns;
  uint64_t exclusive_ns;
} ncnn_profile_worker_acc;

typedef struct {
  uint64_t id;
  uint64_t start_ns;
  uint64_t end_ns;
} ncnn_profile_span;

typedef struct {
  ncnn_profile_worker_acc* accs;
  unsigned acc_count;
  unsigned acc_cap;
  int acc_hash[NCNN_PROFILE_HASH_SIZE];
  ncnn_profile_span* spans;
  unsigned span_count;
  unsigned span_cap;
  unsigned countdown;
  int used;
  int span_overflow;
} ncnn_profile_slot;

typedef struct {
  uint64_t id;
  uint64_t start_ns;
  uint64_t end_ns;
} ncnn_profile_region;

typedef struct {
  uint64_t id;
  int64_t bytes;
  uint64_t active;
} ncnn_profile_allocation;

static ncnn_profile_record records[NCNN_PROFILE_MAX_RECORDS];
static unsigned record_count;
static int record_overflow;
static int record_hash[NCNN_PROFILE_HASH_SIZE];
static ncnn_profile_slot* slots[NCNN_PROFILE_MAX_SLOTS];
static unsigned slot_count;
static _Thread_local ncnn_profile_slot* my_slot;
static ncnn_profile_region regions[NCNN_PROFILE_MAX_RECORDS];
static unsigned region_count;
static unsigned sampling_duty = 1;
static unsigned sampling_window_ns = NCNN_PROFILE_DEFAULT_WINDOW_NS;
static int sampling_parsed;
static int interval_overflow;
static uint64_t attribution_window_total;
static double attribution_projection = 1.0;
static ncnn_profile_allocation allocations[NCNN_PROFILE_MAX_ALLOCS];
static unsigned allocation_count;
static int allocation_overflow;
static atomic_flag profile_lock = ATOMIC_FLAG_INIT;
static atomic_uint_fast64_t open_worker_spans = 0;
static _Thread_local ncnn_profile_frame stack[NCNN_PROFILE_MAX_STACK];
static _Thread_local unsigned stack_depth;
static uint64_t allocation_events;
static uint64_t allocation_bytes;
static int allocation_bytes_known = 1;
static uint64_t deallocation_events;
static uint64_t deallocation_bytes;
static int deallocation_bytes_known = 1;
static uint64_t copy_events;
static uint64_t copy_bytes;
static int copy_bytes_known = 1;
static uint64_t materialized_write_events;
static uint64_t materialized_write_bytes;
static int materialized_write_bytes_known = 1;
static uint64_t materialized_read_events;
static uint64_t materialized_read_bytes;
static int materialized_read_bytes_known = 1;
static uint64_t materialized_expected_read_bytes;
static int materialized_expected_read_bytes_known = 1;
static uint64_t transpose_events;
static uint64_t transpose_write_bytes;
static int transpose_write_bytes_known = 1;
static uint64_t pack_events;
static uint64_t pack_bytes;
static int pack_bytes_known;
static int pack_bytes_observed;
static uint64_t unpack_events;
static uint64_t unpack_bytes;
static int unpack_bytes_known;
static int unpack_bytes_observed;
static uint64_t parallel_events;
static uint64_t live_bytes;
static uint64_t peak_live_bytes;
static int live_bytes_known = 1;
static uint64_t mismatch_events;
static uint64_t top_level_time_ns;
static int top_level_time_known = 1;
static uint64_t flush_count;
static uint64_t invocation_sequence;
static uint64_t active_roots;
static int invocation_active;
static int invocation_complete = 1;
static int v2_output_initialized;

static unsigned profile_schema_version(void) {
  const char* schema = getenv("NCNN_PROFILE_SCHEMA");
  if (schema && strcmp(schema, "3") == 0) {
    return 3;
  }
  if (schema && strcmp(schema, "2") == 0) {
    return 2;
  }
  return 1;
}

static int profile_v2_enabled(void) {
  return profile_schema_version() >= 2;
}

static void reset_profile_state_locked(void) {
  memset(records, 0, sizeof(records));
  record_count = 0;
  record_overflow = 0;
  memset(record_hash, 0, sizeof(record_hash));
  for (unsigned index = 0; index < slot_count; ++index) {
    if (slots[index]) {
      slots[index]->acc_count = 0;
      slots[index]->span_count = 0;
      slots[index]->span_overflow = 0;
      memset(slots[index]->acc_hash, 0, sizeof(slots[index]->acc_hash));
    }
  }
  region_count = 0;
  interval_overflow = 0;
  atomic_store_explicit(&open_worker_spans, 0, memory_order_relaxed);
  memset(allocations, 0, sizeof(allocations));
  allocation_count = 0;
  allocation_overflow = 0;
  allocation_events = 0;
  allocation_bytes = 0;
  allocation_bytes_known = 1;
  deallocation_events = 0;
  deallocation_bytes = 0;
  deallocation_bytes_known = 1;
  copy_events = 0;
  copy_bytes = 0;
  copy_bytes_known = 1;
  materialized_write_events = 0;
  materialized_write_bytes = 0;
  materialized_write_bytes_known = 1;
  materialized_read_events = 0;
  materialized_read_bytes = 0;
  materialized_read_bytes_known = 1;
  materialized_expected_read_bytes = 0;
  materialized_expected_read_bytes_known = 1;
  transpose_events = 0;
  transpose_write_bytes = 0;
  transpose_write_bytes_known = 1;
  pack_events = 0;
  pack_bytes = 0;
  pack_bytes_known = 0;
  pack_bytes_observed = 0;
  unpack_events = 0;
  unpack_bytes = 0;
  unpack_bytes_known = 0;
  unpack_bytes_observed = 0;
  parallel_events = 0;
  live_bytes = 0;
  peak_live_bytes = 0;
  live_bytes_known = 1;
  mismatch_events = 0;
  top_level_time_ns = 0;
  top_level_time_known = 1;
}

static void lock_profile(void) {
  while (
    atomic_flag_test_and_set_explicit(&profile_lock, memory_order_acquire)) {}
}

static void unlock_profile(void) {
  atomic_flag_clear_explicit(&profile_lock, memory_order_release);
}

static uint64_t profile_now(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return 0;
  }
  return ((uint64_t)value.tv_sec * 1000000000ULL) + (uint64_t)value.tv_nsec;
}

/* Sampling gate only.  The coarse clock is cheap enough to read on every
   worker call and, being process-wide, makes all threads agree on which
   windows are sampled.  Exact span timestamps still use CLOCK_MONOTONIC. */
static uint64_t profile_now_coarse(void) {
#ifdef CLOCK_MONOTONIC_COARSE
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC_COARSE, &value) == 0) {
    return ((uint64_t)value.tv_sec * 1000000000ULL) + (uint64_t)value.tv_nsec;
  }
#endif
  return profile_now();
}

static unsigned record_hash_slot(uint64_t id, int64_t category) {
  uint64_t key = id ^ ((uint64_t)category * 0x9E3779B97F4A7C15ULL);
  key ^= key >> 32;
  key *= 0xBF58476D1CE4E5B9ULL;
  key ^= key >> 29;
  return (unsigned)(key & (NCNN_PROFILE_HASH_SIZE - 1));
}

static int record_for(uint64_t id, int64_t category) {
  unsigned slot = record_hash_slot(id, category);
  for (unsigned probe = 0; probe < NCNN_PROFILE_HASH_SIZE; ++probe) {
    int entry = record_hash[slot];
    if (entry == 0) {
      break;
    }
    unsigned index = (unsigned)(entry - 1);
    if (records[index].id == id && records[index].category == category) {
      return (int)index;
    }
    slot = (slot + 1) & (NCNN_PROFILE_HASH_SIZE - 1);
  }
  if (record_count == NCNN_PROFILE_MAX_RECORDS) {
    record_overflow = 1;
    return -1;
  }
  unsigned index = record_count++;
  record_hash[slot] = (int)index + 1;
  records[index].id = id;
  records[index].calls = 0;
  records[index].inclusive_ns = 0;
  records[index].exclusive_ns = 0;
  records[index].bytes = 0;
  records[index].worker_wall_union_ns = 0;
  records[index].worker_wall_attributed_ns = 0;
  records[index].worker_wall_attributed_estimated_ns = 0;
  records[index].worker_sampled_calls = 0;
  records[index].worker_covered_wall_ns = 0;
  records[index].worker_covered_wall_estimated_ns = 0;
  records[index].sampled_window_wall_ns = 0;
  records[index].region_wall_ns = 0;
  records[index].wall_exclusive_estimated_ns = 0;
  records[index].region_worker_spans = 0;
  records[index].category = category;
  records[index].exclusive_known = 1;
  records[index].worker_wall_union_known = 1;
  records[index].worker_wall_attributed_known = 1;
  records[index].wall_coverage_known = 1;
  records[index].bytes_known =
    category == NCNN_PROFILE_ALLOCATION ||
    category == NCNN_PROFILE_DEALLOCATION || category == NCNN_PROFILE_COPY ||
    category == NCNN_PROFILE_TRANSPOSE || category == NCNN_PROFILE_PACK ||
    category == NCNN_PROFILE_UNPACK ||
    category == NCNN_PROFILE_MATERIALIZED_WRITE ||
    category == NCNN_PROFILE_MATERIALIZED_READ;
  return (int)index;
}

static void add_record_bytes(int record, int64_t bytes) {
  if (record < 0) {
    return;
  }
  if (bytes < 0 || !records[record].bytes_known) {
    records[record].bytes_known = 0;
    return;
  }
  if (records[record].bytes > UINT64_MAX - (uint64_t)bytes) {
    records[record].bytes_known = 0;
    return;
  }
  records[record].bytes += (uint64_t)bytes;
}

static int allocation_for(uint64_t id) {
  unsigned index;
  for (index = 0; index < allocation_count; ++index) {
    if (allocations[index].id == id) {
      return (int)index;
    }
  }
  if (allocation_count == NCNN_PROFILE_MAX_ALLOCS) {
    allocation_overflow = 1;
    return -1;
  }
  allocations[allocation_count].id = id;
  allocations[allocation_count].bytes = -1;
  allocations[allocation_count].active = 0;
  return (int)allocation_count++;
}

static void parse_sampling_options(void);
static uint64_t sampled_window_overlap(uint64_t start, uint64_t end);
static unsigned parse_unsigned(const char* value, int* known);

static int span_before(const ncnn_profile_span* left,
                       const ncnn_profile_span* right) {
  if (left->start_ns != right->start_ns) {
    return left->start_ns < right->start_ns;
  }
  return left->id < right->id;
}

static int span_id_before(const ncnn_profile_span* left,
                          const ncnn_profile_span* right) {
  if (left->id != right->id) {
    return left->id < right->id;
  }
  return left->start_ns < right->start_ns;
}

static void insertion_sort_spans(ncnn_profile_span* spans,
                                 unsigned count,
                                 int (*before)(const ncnn_profile_span*,
                                               const ncnn_profile_span*)) {
  for (unsigned index = 1; index < count; ++index) {
    ncnn_profile_span value = spans[index];
    unsigned position = index;
    while (position != 0 && before(&value, &spans[position - 1])) {
      spans[position] = spans[position - 1];
      --position;
    }
    spans[position] = value;
  }
}

static void merge_sort_spans(ncnn_profile_span* spans,
                             unsigned count,
                             int (*before)(const ncnn_profile_span*,
                                           const ncnn_profile_span*)) {
  if (count < 32) {
    insertion_sort_spans(spans, count, before);
    return;
  }
  unsigned middle = count / 2;
  merge_sort_spans(spans, middle, before);
  merge_sort_spans(spans + middle, count - middle, before);
  ncnn_profile_span* merged =
    (ncnn_profile_span*)malloc((size_t)count * sizeof(ncnn_profile_span));
  if (!merged) {
    insertion_sort_spans(spans, count, before);
    return;
  }
  unsigned left = 0;
  unsigned right = middle;
  unsigned out = 0;
  while (left < middle && right < count) {
    merged[out++] =
      before(&spans[right], &spans[left]) ? spans[right++] : spans[left++];
  }
  while (left < middle) {
    merged[out++] = spans[left++];
  }
  while (right < count) {
    merged[out++] = spans[right++];
  }
  memcpy(spans, merged, (size_t)count * sizeof(ncnn_profile_span));
  free(merged);
}

/* Merge lock-free worker accumulations and compute wall attribution.
   Worker spans are siblings inside one parallel region, so a sweep with an
   open-span list partitions region wall time additively: each instant is split
   equally among the spans executing at that instant.  Region wall time not
   covered by any worker span is explicit gap time, not attributed to an op. */
static void merge_worker_slots_locked(void) {
  unsigned total_spans = 0;
  int overflow = 0;
  for (unsigned index = 0; index < slot_count; ++index) {
    ncnn_profile_slot* slot = slots[index];
    if (!slot) {
      continue;
    }
    if (slot->span_overflow) {
      overflow = 1;
    }
    if (slot->span_count > UINT32_MAX - total_spans) {
      overflow = 1;
      break;
    }
    total_spans += slot->span_count;
    for (unsigned acc = 0; acc < slot->acc_count; ++acc) {
      const int record =
        record_for(slot->accs[acc].id, NCNN_PROFILE_WORKER_OPERATION);
      if (record < 0) {
        overflow = 1;
        continue;
      }
      records[record].calls += slot->accs[acc].calls;
      records[record].worker_sampled_calls += slot->accs[acc].calls;
      records[record].inclusive_ns += slot->accs[acc].inclusive_ns;
      records[record].exclusive_ns += slot->accs[acc].exclusive_ns;
    }
  }
  interval_overflow = overflow;
  if (total_spans == 0) {
    return;
  }
  ncnn_profile_span* spans =
    (ncnn_profile_span*)malloc((size_t)total_spans * sizeof(ncnn_profile_span));
  if (!spans) {
    interval_overflow = 1;
    return;
  }
  unsigned count = 0;
  for (unsigned index = 0; index < slot_count; ++index) {
    ncnn_profile_slot* slot = slots[index];
    if (!slot) {
      continue;
    }
    memcpy(spans + count,
           slot->spans,
           (size_t)slot->span_count * sizeof(ncnn_profile_span));
    count += slot->span_count;
  }
  merge_sort_spans(spans, count, span_before);

  ncnn_profile_region* sorted_regions = (ncnn_profile_region*)malloc(
    (size_t)(region_count ? region_count : 1) * sizeof(ncnn_profile_region));
  if (!sorted_regions) {
    free(spans);
    interval_overflow = 1;
    return;
  }
  memcpy(sorted_regions,
         regions,
         (size_t)region_count * sizeof(ncnn_profile_region));
  for (unsigned index = 1; index < region_count; ++index) {
    ncnn_profile_region value = sorted_regions[index];
    unsigned position = index;
    while (position != 0 &&
           sorted_regions[position - 1].start_ns > value.start_ns) {
      sorted_regions[position] = sorted_regions[position - 1];
      --position;
    }
    sorted_regions[position] = value;
  }

  unsigned span_index = 0;
  for (unsigned region_index = 0; region_index < region_count; ++region_index) {
    const ncnn_profile_region region = sorted_regions[region_index];
    while (span_index < count && spans[span_index].end_ns <= region.start_ns) {
      ++span_index;
    }
    const unsigned region_begin = span_index;
    while (span_index < count && spans[span_index].start_ns < region.end_ns) {
      ++span_index;
    }
    const unsigned region_end = span_index;
    const uint64_t window_ns =
      sampled_window_overlap(region.start_ns, region.end_ns);
    uint64_t covered_ns = 0;
    if (region_end > region_begin) {
      unsigned open[NCNN_PROFILE_MAX_SLOTS];
      unsigned open_count = 0;
      unsigned next = region_begin;
      uint64_t cursor = region.start_ns;
      while (next < region_end || open_count != 0) {
        uint64_t next_start =
          next < region_end ? spans[next].start_ns : UINT64_MAX;
        uint64_t next_end = UINT64_MAX;
        for (unsigned slot = 0; slot < open_count; ++slot) {
          if (spans[open[slot]].end_ns < next_end) {
            next_end = spans[open[slot]].end_ns;
          }
        }
        const uint64_t step = next_start < next_end ? next_start : next_end;
        if (step == UINT64_MAX) {
          break;
        }
        if (step > cursor) {
          const uint64_t delta = step - cursor;
          if (open_count != 0) {
            covered_ns += delta;
            const uint64_t share = delta / open_count;
            uint64_t remainder = delta % open_count;
            for (unsigned slot = 0; slot < open_count; ++slot) {
              const int record =
                record_for(spans[open[slot]].id, NCNN_PROFILE_WORKER_OPERATION);
              if (record < 0) {
                continue;
              }
              const uint64_t attributed = share + (remainder ? 1 : 0);
              if (remainder) {
                --remainder;
              }
              if (records[record].worker_wall_attributed_ns <=
                  UINT64_MAX - attributed) {
                records[record].worker_wall_attributed_ns += attributed;
              } else {
                records[record].worker_wall_attributed_known = 0;
              }
            }
          }
          cursor = step;
        }
        while (next < region_end && spans[next].start_ns <= cursor) {
          if (open_count == NCNN_PROFILE_MAX_SLOTS) {
            interval_overflow = 1;
          } else {
            open[open_count++] = next;
          }
          ++next;
        }
        for (unsigned slot = 0; slot < open_count;) {
          if (spans[open[slot]].end_ns <= cursor) {
            memmove(&open[slot],
                    &open[slot + 1],
                    (size_t)(open_count - slot - 1) * sizeof(unsigned));
            --open_count;
          } else {
            ++slot;
          }
        }
      }
    }
    const int record = record_for(region.id, NCNN_PROFILE_PARALLEL);
    if (record >= 0) {
      ncnn_profile_record* parallel_record = &records[record];
      const uint64_t region_wall =
        region.end_ns > region.start_ns ? region.end_ns - region.start_ns : 0;
      parallel_record->region_worker_spans += (region_end - region_begin);
      if (parallel_record->worker_covered_wall_ns <= UINT64_MAX - covered_ns &&
          parallel_record->sampled_window_wall_ns <=
            UINT64_MAX - window_ns &&
          parallel_record->region_wall_ns <= UINT64_MAX - region_wall) {
        parallel_record->worker_covered_wall_ns += covered_ns;
        parallel_record->sampled_window_wall_ns += window_ns;
        parallel_record->region_wall_ns += region_wall;
      } else {
        parallel_record->wall_coverage_known = 0;
      }
    }
  }
  free(sorted_regions);

  merge_sort_spans(spans, count, span_id_before);
  for (unsigned index = 0; index < count;) {
    const uint64_t id = spans[index].id;
    uint64_t run_start = spans[index].start_ns;
    uint64_t run_end = spans[index].end_ns;
    uint64_t union_total = 0;
    unsigned cursor = index + 1;
    while (cursor < count && spans[cursor].id == id) {
      if (spans[cursor].start_ns > run_end) {
        union_total += run_end - run_start;
        run_start = spans[cursor].start_ns;
        run_end = spans[cursor].end_ns;
      } else if (spans[cursor].end_ns > run_end) {
        run_end = spans[cursor].end_ns;
      }
      ++cursor;
    }
    union_total += run_end - run_start;
    const int record = record_for(id, NCNN_PROFILE_WORKER_OPERATION);
    if (record >= 0) {
      if (records[record].worker_wall_union_ns <= UINT64_MAX - union_total) {
        records[record].worker_wall_union_ns += union_total;
      } else {
        records[record].worker_wall_union_known = 0;
      }
    }
    index = cursor;
  }
  free(spans);
  parse_sampling_options();
  // One projection factor for the whole invocation keeps wall accounting
  // additive: every per-op attributed duration and every region covered
  // duration is scaled by the same region_wall / sampled_window ratio. At
  // duty=1 the factor is exactly 1 and nothing is estimated.
  double projection = 1.0;
  uint64_t window_total = 0;
  uint64_t region_wall_total = 0;
  for (unsigned index = 0; index < record_count; ++index) {
    ncnn_profile_record* record = &records[index];
    if (record->category != NCNN_PROFILE_PARALLEL ||
        !record->wall_coverage_known || interval_overflow ||
        record->region_worker_spans == 0 ||
        record->sampled_window_wall_ns == 0 ||
        record->worker_covered_wall_ns > record->sampled_window_wall_ns) {
      if (record->category == NCNN_PROFILE_PARALLEL) {
        record->wall_coverage_known = 0;
      }
      continue;
    }
    window_total += record->sampled_window_wall_ns;
    region_wall_total += record->inclusive_ns;
  }
  if (window_total != 0 && region_wall_total > window_total) {
    projection = (double)region_wall_total / (double)window_total;
  }
  attribution_window_total = window_total;
  attribution_projection = projection;
  for (unsigned index = 0; index < record_count; ++index) {
    ncnn_profile_record* record = &records[index];
    if (record->category == NCNN_PROFILE_WORKER_OPERATION &&
        record->worker_wall_attributed_known) {
      const double projected =
        (double)record->worker_wall_attributed_ns * projection;
      if (projected > (double)UINT64_MAX) {
        record->worker_wall_attributed_known = 0;
      } else {
        record->worker_wall_attributed_estimated_ns = (uint64_t)projected;
      }
    }
    if (record->category != NCNN_PROFILE_PARALLEL ||
        !record->wall_coverage_known) {
      continue;
    }
    const uint64_t window = record->sampled_window_wall_ns;
    const uint64_t covered_in_window = record->worker_covered_wall_ns;
    const uint64_t region_wall = record->inclusive_ns;
    const double covered_projected =
      (double)covered_in_window * projection;
    if (covered_projected > (double)region_wall) {
      record->worker_covered_wall_estimated_ns = region_wall;
      record->wall_exclusive_estimated_ns = 0;
    } else {
      record->worker_covered_wall_estimated_ns = (uint64_t)covered_projected;
      record->wall_exclusive_estimated_ns =
        region_wall - (uint64_t)covered_projected;
    }
    if (sampling_duty <= 1 && window == region_wall) {
      // Complete interval accounting with observed worker spans: region wall
      // not covered by any measured worker span is exactly the
      // non-overlapping remainder, matching the serial exclusive definition
      // when children do not overlap.
      record->exclusive_ns = region_wall - covered_in_window;
      record->exclusive_known = 1;
    }
  }
}

static void parse_sampling_options(void) {
  if (sampling_parsed) {
    return;
  }
  const char* duty = getenv("NCNN_PROFILE_SAMPLE_DUTY");
  if (duty && *duty) {
    int known = 0;
    const unsigned value = parse_unsigned(duty, &known);
    if (known && value >= 1) {
      sampling_duty = value;
    }
  }
  const char* window = getenv("NCNN_PROFILE_SAMPLE_WINDOW_NS");
  if (window && *window) {
    int known = 0;
    const unsigned value = parse_unsigned(window, &known);
    if (known && value >= 1) {
      sampling_window_ns = value;
    }
  }
  sampling_parsed = 1;
}

/* Coherent time-window duty sampling.  All threads evaluate the same window
   pattern, so cross-worker interval unions stay comparable and wall shares are
   normalized by the sampled-window duration rather than scaled by duty.  Only
   spans that start in a sampled window are timed; their ends are clipped to
   that window so every recorded span lies inside one sampled window. */
static int window_sampled(uint64_t now, uint64_t* window_limit) {
  parse_sampling_options();
  if (sampling_duty <= 1) {
    *window_limit = UINT64_MAX;
    return 1;
  }
  const uint64_t window_index = now / sampling_window_ns;
  *window_limit = (window_index + 1) * sampling_window_ns;
  return (window_index % sampling_duty) == 0;
}

/* Wall time of [start,end) lying inside sampled windows. */
static uint64_t sampled_window_overlap(uint64_t start, uint64_t end) {
  parse_sampling_options();
  if (sampling_duty <= 1) {
    return end > start ? end - start : 0;
  }
  if (end <= start) {
    return 0;
  }
  uint64_t first = start / sampling_window_ns;
  const uint64_t last = (end - 1) / sampling_window_ns;
  if (first % sampling_duty != 0) {
    first += sampling_duty - (first % sampling_duty);
  }
  uint64_t total = 0;
  for (uint64_t index = first; index <= last; index += sampling_duty) {
    const uint64_t window_start = index * sampling_window_ns;
    const uint64_t window_end = window_start + sampling_window_ns;
    const uint64_t from = start > window_start ? start : window_start;
    const uint64_t to = end < window_end ? end : window_end;
    if (to > from) {
      total += to - from;
    }
  }
  return total;
}

static ncnn_profile_slot* ensure_slot(void) {
  if (my_slot) {
    return my_slot;
  }
  ncnn_profile_slot* slot =
    (ncnn_profile_slot*)calloc(1, sizeof(ncnn_profile_slot));
  if (!slot) {
    return NULL;
  }
  lock_profile();
  if (slot_count == NCNN_PROFILE_MAX_SLOTS) {
    unlock_profile();
    free(slot);
    return NULL;
  }
  slots[slot_count++] = slot;
  unlock_profile();
  // Stagger the sampling phase per thread so a periodic workload cannot line
  // up with the same counter residue on every worker.
  parse_sampling_options();
  slot->countdown =
    sampling_duty > 1 ? ((slot_count - 1) % sampling_duty) + 1 : 0;
  my_slot = slot;
  return slot;
}

static ncnn_profile_worker_acc* slot_acc_for(ncnn_profile_slot* slot,
                                             uint64_t id) {
  unsigned bucket = record_hash_slot(id, NCNN_PROFILE_WORKER_OPERATION);
  for (unsigned probe = 0; probe < NCNN_PROFILE_HASH_SIZE; ++probe) {
    const int entry = slot->acc_hash[bucket];
    if (entry == 0) {
      break;
    }
    const unsigned index = (unsigned)(entry - 1);
    if (slot->accs[index].id == id) {
      return &slot->accs[index];
    }
    bucket = (bucket + 1) & (NCNN_PROFILE_HASH_SIZE - 1);
  }
  if (slot->acc_count == slot->acc_cap) {
    const unsigned next = slot->acc_cap ? slot->acc_cap * 2 : 64;
    if (next > NCNN_PROFILE_MAX_RECORDS) {
      return NULL;
    }
    ncnn_profile_worker_acc* grown = (ncnn_profile_worker_acc*)realloc(
      slot->accs, (size_t)next * sizeof(ncnn_profile_worker_acc));
    if (!grown) {
      return NULL;
    }
    slot->accs = grown;
    slot->acc_cap = next;
  }
  const unsigned index = slot->acc_count++;
  slot->acc_hash[bucket] = (int)index + 1;
  slot->accs[index].id = id;
  slot->accs[index].calls = 0;
  slot->accs[index].inclusive_ns = 0;
  slot->accs[index].exclusive_ns = 0;
  return &slot->accs[index];
}

static void slot_record_span(ncnn_profile_slot* slot,
                             uint64_t id,
                             uint64_t start,
                             uint64_t end) {
  if (slot->span_count == slot->span_cap) {
    const unsigned next = slot->span_cap ? slot->span_cap * 2 : 1024;
    if (next > NCNN_PROFILE_DEFAULT_INTERVAL_CAP * 8U) {
      slot->span_overflow = 1;
      return;
    }
    ncnn_profile_span* grown = (ncnn_profile_span*)realloc(
      slot->spans, (size_t)next * sizeof(ncnn_profile_span));
    if (!grown) {
      slot->span_overflow = 1;
      return;
    }
    slot->spans = grown;
    slot->span_cap = next;
  }
  slot->spans[slot->span_count].id = id;
  slot->spans[slot->span_count].start_ns = start;
  slot->spans[slot->span_count].end_ns = end;
  ++slot->span_count;
}

static void profile_event_begin(int64_t signed_id, int64_t category) {
  const uint64_t id = (uint64_t)signed_id;
  if (stack_depth == NCNN_PROFILE_MAX_STACK) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  const uint64_t start = profile_now();
  const int root = stack_depth == 0 && category == NCNN_PROFILE_OPERATION;
  lock_profile();
  if (root && profile_v2_enabled()) {
    if (active_roots == 0) {
      invocation_active = 1;
      invocation_complete = 1;
      ++invocation_sequence;
    } else {
      invocation_complete = 0;
    }
    ++active_roots;
  }
  const int record = record_for(id, category);
  if (record >= 0 && category != NCNN_PROFILE_ALLOCATION &&
      category != NCNN_PROFILE_COPY && category != NCNN_PROFILE_TRANSPOSE &&
      category != NCNN_PROFILE_PACK && category != NCNN_PROFILE_UNPACK) {
    records[record].calls++;
  }
  if (category == NCNN_PROFILE_PARALLEL) {
    parallel_events++;
  }
  unlock_profile();
  stack[stack_depth].id = id;
  stack[stack_depth].start_ns = start;
  stack[stack_depth].child_ns = 0;
  stack[stack_depth].window_limit = UINT64_MAX;
  stack[stack_depth].category = category;
  stack[stack_depth].record = record;
  stack[stack_depth].sampled = 1;
  stack_depth++;
}

void __ncnn_profile_event_begin(int64_t signed_id, int64_t category) {
  profile_event_begin(signed_id, category);
}

void __ncnn_profile_worker_event_begin(int64_t signed_id) {
  const uint64_t id = (uint64_t)signed_id;
  if (stack_depth == NCNN_PROFILE_MAX_STACK) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  ncnn_profile_slot* slot = ensure_slot();
  uint64_t window_limit = UINT64_MAX;
  const int tracked = slot != NULL &&
                      window_sampled(profile_now_coarse(), &window_limit);
  const uint64_t start = tracked ? profile_now() : 0;
  if (tracked) {
    atomic_fetch_add_explicit(&open_worker_spans, 1, memory_order_relaxed);
  }
  stack[stack_depth].id = id;
  stack[stack_depth].start_ns = start;
  stack[stack_depth].child_ns = 0;
  stack[stack_depth].window_limit = window_limit;
  stack[stack_depth].category = NCNN_PROFILE_WORKER_OPERATION;
  stack[stack_depth].record = -1;
  stack[stack_depth].sampled = tracked;
  stack_depth++;
}

static void profile_event_end(int64_t signed_id) {
  const uint64_t id = (uint64_t)signed_id;
  const uint64_t end = profile_now();
  if (stack_depth == 0 || stack[stack_depth - 1].id != id) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  ncnn_profile_frame frame = stack[--stack_depth];
  if (frame.category == NCNN_PROFILE_WORKER_OPERATION) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  const int root = stack_depth == 0 && frame.category == NCNN_PROFILE_OPERATION;
  const uint64_t elapsed = end >= frame.start_ns ? end - frame.start_ns : 0;
  const uint64_t exclusive =
    elapsed >= frame.child_ns ? elapsed - frame.child_ns : 0;
  lock_profile();
  if (frame.record >= 0) {
    ncnn_profile_record* record = &records[frame.record];
    record->inclusive_ns += elapsed;
    if (frame.category == NCNN_PROFILE_PARALLEL) {
      // Worker CPU time is emitted under its own category and is not a
      // subtractable child of this wall-clock region span.  The region wall
      // interval is retained so flush can compute worker coverage.
      record->exclusive_known = 0;
      if (region_count < NCNN_PROFILE_MAX_RECORDS) {
        regions[region_count].id = id;
        regions[region_count].start_ns = frame.start_ns;
        regions[region_count].end_ns = end;
        ++region_count;
      } else {
        record_overflow = 1;
      }
    } else if (record->exclusive_known) {
      record->exclusive_ns += exclusive;
    }
  }
  if (stack_depth != 0) {
    stack[stack_depth - 1].child_ns += elapsed;
  } else if (frame.category != NCNN_PROFILE_FUSION_SITE &&
             top_level_time_known) {
    if (top_level_time_ns <= UINT64_MAX - elapsed) {
      top_level_time_ns += elapsed;
    } else {
      top_level_time_known = 0;
    }
  }
  if (root && profile_v2_enabled() && active_roots != 0) {
    --active_roots;
    if (active_roots != 0) {
      invocation_complete = 0;
    }
  }
  unlock_profile();
}

void __ncnn_profile_event_end(int64_t signed_id) {
  profile_event_end(signed_id);
}

void __ncnn_profile_worker_event_end(int64_t signed_id) {
  const uint64_t id = (uint64_t)signed_id;
  if (stack_depth == 0 || stack[stack_depth - 1].id != id ||
      stack[stack_depth - 1].category != NCNN_PROFILE_WORKER_OPERATION) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  const ncnn_profile_frame frame = stack[--stack_depth];
  if (!frame.sampled) {
    return;
  }
  const uint64_t end = profile_now();
  atomic_fetch_sub_explicit(&open_worker_spans, 1, memory_order_relaxed);
  if (end <= frame.start_ns) {
    return;
  }
  // Wall metrics use spans clipped to the sampled window so unions are
  // comparable with sampled_window_overlap(). CPU metrics keep the full
  // instance duration: they are per-instance sums in a different time domain.
  const uint64_t span_end =
    end < frame.window_limit ? end : frame.window_limit;
  ncnn_profile_slot* slot = my_slot;
  if (!slot) {
    return;
  }
  const uint64_t elapsed = end - frame.start_ns;
  const uint64_t exclusive =
    elapsed >= frame.child_ns ? elapsed - frame.child_ns : 0;
  if (stack_depth != 0) {
    stack[stack_depth - 1].child_ns += elapsed;
  }
  ncnn_profile_worker_acc* acc = slot_acc_for(slot, id);
  if (acc) {
    ++acc->calls;
    acc->inclusive_ns += elapsed;
    acc->exclusive_ns += exclusive;
  } else {
    interval_overflow = 1;
  }
  if (span_end > frame.start_ns) {
    slot_record_span(slot, id, frame.start_ns, span_end);
  }
}

void __ncnn_profile_alloc(int64_t signed_id, int64_t bytes) {
  const uint64_t id = (uint64_t)signed_id;
  lock_profile();
  allocation_events++;
  if (bytes < 0) {
    allocation_bytes_known = 0;
    live_bytes_known = 0;
  } else if (allocation_bytes <= UINT64_MAX - (uint64_t)bytes) {
    allocation_bytes += (uint64_t)bytes;
  } else {
    allocation_bytes_known = 0;
  }
  const int slot = allocation_for(id);
  if (slot >= 0) {
    if (allocations[slot].active == 0) {
      allocations[slot].bytes = bytes;
    } else if (allocations[slot].bytes != bytes) {
      // Static equal-size instances are interchangeable. Unequal overlapping
      // sizes cannot be paired by ID alone, so retain the unknown result.
      allocations[slot].bytes = -1;
      live_bytes_known = 0;
    }
    if (allocations[slot].active == UINT64_MAX) {
      allocation_overflow = 1;
      allocations[slot].bytes = -1;
      live_bytes_known = 0;
    } else {
      allocations[slot].active++;
    }
    if (bytes >= 0 && live_bytes_known &&
        live_bytes <= UINT64_MAX - (uint64_t)bytes) {
      live_bytes += (uint64_t)bytes;
      if (live_bytes > peak_live_bytes) {
        peak_live_bytes = live_bytes;
      }
    } else {
      live_bytes_known = 0;
    }
  } else {
    live_bytes_known = 0;
  }
  const int record = record_for(id, NCNN_PROFILE_ALLOCATION);
  if (record >= 0) {
    records[record].calls++;
    add_record_bytes(record, bytes);
  }
  unlock_profile();
}

void __ncnn_profile_dealloc(int64_t signed_id) {
  const uint64_t id = (uint64_t)signed_id;
  int64_t record_bytes = -1;
  lock_profile();
  deallocation_events++;
  const int slot = allocation_for(id);
  if (slot >= 0 && allocations[slot].active) {
    if (allocations[slot].bytes >= 0) {
      record_bytes = allocations[slot].bytes;
      if (deallocation_bytes <=
          UINT64_MAX - (uint64_t)allocations[slot].bytes) {
        deallocation_bytes += (uint64_t)allocations[slot].bytes;
      } else {
        deallocation_bytes_known = 0;
      }
    } else {
      deallocation_bytes_known = 0;
    }
    if (allocations[slot].bytes >= 0 && live_bytes_known &&
        live_bytes >= (uint64_t)allocations[slot].bytes) {
      live_bytes -= (uint64_t)allocations[slot].bytes;
    } else {
      live_bytes_known = 0;
    }
    allocations[slot].active--;
  } else {
    deallocation_bytes_known = 0;
    live_bytes_known = 0;
  }
  const int record = record_for(id, NCNN_PROFILE_DEALLOCATION);
  if (record >= 0) {
    records[record].calls++;
    add_record_bytes(record, record_bytes);
  }
  unlock_profile();
}

void __ncnn_profile_copy(int64_t signed_id, int64_t bytes) {
  const uint64_t id = (uint64_t)signed_id;
  lock_profile();
  copy_events++;
  if (bytes < 0) {
    copy_bytes_known = 0;
  } else if (copy_bytes <= UINT64_MAX - (uint64_t)bytes) {
    copy_bytes += (uint64_t)bytes;
  } else {
    copy_bytes_known = 0;
  }
  const int record = record_for(id, NCNN_PROFILE_COPY);
  if (record >= 0) {
    records[record].calls++;
    add_record_bytes(record, bytes);
  }
  unlock_profile();
}

void __ncnn_profile_materialized(int64_t signed_id,
                                 int64_t kind,
                                 int64_t bytes,
                                 int64_t expected_readers) {
  const uint64_t id = (uint64_t)signed_id;
  lock_profile();
  uint64_t* events = NULL;
  uint64_t* total_bytes = NULL;
  int* bytes_known = NULL;
  int64_t category = NCNN_PROFILE_OPERATION;
  if (kind == 0) {
    events = &materialized_write_events;
    total_bytes = &materialized_write_bytes;
    bytes_known = &materialized_write_bytes_known;
    category = NCNN_PROFILE_MATERIALIZED_WRITE;
  } else if (kind == 1) {
    events = &materialized_read_events;
    total_bytes = &materialized_read_bytes;
    bytes_known = &materialized_read_bytes_known;
    category = NCNN_PROFILE_MATERIALIZED_READ;
  }
  if (events && total_bytes && bytes_known) {
    ++*events;
    if (bytes < 0) {
      *bytes_known = 0;
    } else if (*bytes_known && *total_bytes <= UINT64_MAX - (uint64_t)bytes) {
      *total_bytes += (uint64_t)bytes;
    } else {
      *bytes_known = 0;
    }
    if (kind == 0) {
      if (bytes < 0 || expected_readers < 0 ||
          (expected_readers != 0 &&
           (uint64_t)bytes > UINT64_MAX / (uint64_t)expected_readers)) {
        materialized_expected_read_bytes_known = 0;
      } else {
        const uint64_t expected = (uint64_t)bytes * (uint64_t)expected_readers;
        if (materialized_expected_read_bytes_known &&
            materialized_expected_read_bytes <= UINT64_MAX - expected) {
          materialized_expected_read_bytes += expected;
        } else {
          materialized_expected_read_bytes_known = 0;
        }
      }
    } else if (expected_readers != 0) {
      materialized_expected_read_bytes_known = 0;
    }
    const int record = record_for(id, category);
    if (record >= 0) {
      records[record].calls++;
      add_record_bytes(record, bytes);
    }
  } else {
    mismatch_events++;
  }
  unlock_profile();
}

void __ncnn_profile_movement(int64_t signed_id, int64_t kind, int64_t bytes) {
  const uint64_t id = (uint64_t)signed_id;
  lock_profile();
  uint64_t* events = NULL;
  uint64_t* total_bytes = NULL;
  int* bytes_known = NULL;
  int* bytes_observed = NULL;
  int64_t category = NCNN_PROFILE_OPERATION;
  if (kind == 0) {
    events = &transpose_events;
    total_bytes = &transpose_write_bytes;
    bytes_known = &transpose_write_bytes_known;
    category = NCNN_PROFILE_TRANSPOSE;
  } else if (kind == 1) {
    events = &pack_events;
    total_bytes = &pack_bytes;
    bytes_known = &pack_bytes_known;
    bytes_observed = &pack_bytes_observed;
    category = NCNN_PROFILE_PACK;
  } else if (kind == 2) {
    events = &unpack_events;
    total_bytes = &unpack_bytes;
    bytes_known = &unpack_bytes_known;
    bytes_observed = &unpack_bytes_observed;
    category = NCNN_PROFILE_UNPACK;
  }
  if (events && total_bytes && bytes_known) {
    const int had_observation =
      bytes_observed ? *bytes_observed : (*events != 0);
    ++*events;
    if (bytes_observed) {
      *bytes_observed = 1;
    }
    if (bytes < 0 || (had_observation && !*bytes_known)) {
      *bytes_known = 0;
    } else if (*total_bytes <= UINT64_MAX - (uint64_t)bytes) {
      *total_bytes += (uint64_t)bytes;
      *bytes_known = 1;
    } else {
      *bytes_known = 0;
    }
  } else {
    mismatch_events++;
  }
  const int record = record_for(id, category);
  if (record >= 0) {
    records[record].calls++;
    add_record_bytes(record, bytes);
  }
  unlock_profile();
}

static void write_json_string(FILE* file, const char* value) {
  const unsigned char* cursor = (const unsigned char*)(value ? value : "");
  fputc('"', file);
  while (*cursor) {
    switch (*cursor) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (*cursor < 0x20) {
          fprintf(file, "\\u%04x", (unsigned)*cursor);
        } else {
          fputc(*cursor, file);
        }
        break;
    }
    ++cursor;
  }
  fputc('"', file);
}

static const char* category_name(int64_t category) {
  switch (category) {
    case NCNN_PROFILE_ALLOCATION:
      return "allocation";
    case NCNN_PROFILE_DEALLOCATION:
      return "deallocation";
    case NCNN_PROFILE_COPY:
      return "copy";
    case NCNN_PROFILE_PARALLEL:
      return "parallel";
    case NCNN_PROFILE_TRANSPOSE:
      return "transpose";
    case NCNN_PROFILE_PACK:
      return "pack";
    case NCNN_PROFILE_UNPACK:
      return "unpack";
    case NCNN_PROFILE_MATERIALIZED_WRITE:
      return "materialized_write";
    case NCNN_PROFILE_MATERIALIZED_READ:
      return "materialized_read";
    case NCNN_PROFILE_FUSION_SITE:
      return "operation";
    case NCNN_PROFILE_WORKER_OPERATION:
      return "worker_operation";
    default:
      return "operation";
  }
}

static int record_before(const ncnn_profile_record* left,
                         const ncnn_profile_record* right) {
  if (left->id != right->id) {
    return left->id < right->id;
  }
  return left->category < right->category;
}

static const char* environment_or_default(const char* name,
                                          const char* fallback) {
  const char* value = getenv(name);
  return value && *value ? value : fallback;
}

static unsigned parse_unsigned(const char* value, int* known) {
  uint64_t result = 0;
  const unsigned char* cursor = (const unsigned char*)value;
  if (!cursor || !*cursor) {
    *known = 0;
    return 0;
  }
  while (*cursor) {
    if (*cursor < '0' || *cursor > '9' ||
        result > (UINT64_MAX - (uint64_t)(*cursor - '0')) / 10) {
      *known = 0;
      return 0;
    }
    result = (result * 10) + (uint64_t)(*cursor - '0');
    ++cursor;
  }
  if (result > UINT32_MAX) {
    *known = 0;
    return 0;
  }
  *known = 1;
  return (unsigned)result;
}

static void flush_profile_v2(const char* path) {
  ncnn_profile_record snapshot[NCNN_PROFILE_MAX_RECORDS];
  unsigned snapshot_count;
  uint64_t local_allocation_events;
  uint64_t local_allocation_bytes;
  int local_allocation_bytes_known;
  uint64_t local_deallocation_events;
  uint64_t local_deallocation_bytes;
  int local_deallocation_bytes_known;
  uint64_t local_copy_events;
  uint64_t local_copy_bytes;
  int local_copy_bytes_known;
  uint64_t local_materialized_write_events;
  uint64_t local_materialized_write_bytes;
  int local_materialized_write_bytes_known;
  uint64_t local_materialized_read_events;
  uint64_t local_materialized_read_bytes;
  int local_materialized_read_bytes_known;
  uint64_t local_materialized_expected_read_bytes;
  int local_materialized_expected_read_bytes_known;
  uint64_t local_transpose_events;
  uint64_t local_transpose_write_bytes;
  int local_transpose_write_bytes_known;
  uint64_t local_pack_events;
  uint64_t local_pack_bytes;
  int local_pack_bytes_known;
  uint64_t local_unpack_events;
  uint64_t local_unpack_bytes;
  int local_unpack_bytes_known;
  uint64_t local_parallel_events;
  uint64_t local_peak_live_bytes;
  int local_live_bytes_known;
  uint64_t local_mismatch_events;
  uint64_t local_top_level_time_ns;
  int local_top_level_time_known;
  int local_record_overflow;
  int local_allocation_overflow;
  uint64_t local_invocation_id;
  int local_complete;
  unsigned index;

  lock_profile();
  if (active_roots != 0 ||
      (!invocation_active && record_count == 0 && allocation_events == 0 &&
       deallocation_events == 0 && copy_events == 0 &&
       materialized_write_events == 0 && materialized_read_events == 0 &&
       transpose_events == 0 && pack_events == 0 && unpack_events == 0 &&
       parallel_events == 0)) {
    unlock_profile();
    return;
  }
  local_invocation_id = invocation_sequence;
  if (local_invocation_id == 0) {
    local_invocation_id = ++invocation_sequence;
  }
  local_complete = invocation_complete;
  if (atomic_load_explicit(&open_worker_spans, memory_order_acquire) != 0) {
    // A worker span is still open: its interval cannot be merged, so wall
    // coverage is incomplete and must not be reported as proven.
    interval_overflow = 1;
    local_complete = 0;
  }
  merge_worker_slots_locked();
  snapshot_count = record_count;
  for (index = 0; index < snapshot_count; ++index) {
    if (!records[index].worker_wall_union_known ||
        !records[index].worker_wall_attributed_known ||
        !records[index].wall_coverage_known) {
      local_complete = 0;
    }
    snapshot[index] = records[index];
  }
  local_allocation_events = allocation_events;
  local_allocation_bytes = allocation_bytes;
  local_allocation_bytes_known = allocation_bytes_known;
  local_deallocation_events = deallocation_events;
  local_deallocation_bytes = deallocation_bytes;
  local_deallocation_bytes_known = deallocation_bytes_known;
  local_copy_events = copy_events;
  local_copy_bytes = copy_bytes;
  local_copy_bytes_known = copy_bytes_known;
  local_materialized_write_events = materialized_write_events;
  local_materialized_write_bytes = materialized_write_bytes;
  local_materialized_write_bytes_known = materialized_write_bytes_known;
  local_materialized_read_events = materialized_read_events;
  local_materialized_read_bytes = materialized_read_bytes;
  local_materialized_read_bytes_known = materialized_read_bytes_known;
  local_materialized_expected_read_bytes = materialized_expected_read_bytes;
  local_materialized_expected_read_bytes_known =
    materialized_expected_read_bytes_known;
  local_transpose_events = transpose_events;
  local_transpose_write_bytes = transpose_write_bytes;
  local_transpose_write_bytes_known = transpose_write_bytes_known;
  local_pack_events = pack_events;
  local_pack_bytes = pack_bytes;
  local_pack_bytes_known = pack_bytes_known;
  local_unpack_events = unpack_events;
  local_unpack_bytes = unpack_bytes;
  local_unpack_bytes_known = unpack_bytes_known;
  local_parallel_events = parallel_events;
  local_peak_live_bytes = peak_live_bytes;
  local_live_bytes_known = live_bytes_known;
  local_mismatch_events = mismatch_events;
  local_top_level_time_ns = top_level_time_ns;
  local_top_level_time_known = top_level_time_known;
  local_record_overflow = record_overflow;
  local_allocation_overflow = allocation_overflow;
  if (local_mismatch_events != 0 || local_record_overflow ||
      local_allocation_overflow) {
    local_complete = 0;
  }

  for (index = 1; index < snapshot_count; ++index) {
    ncnn_profile_record value = snapshot[index];
    unsigned position = index;
    while (position != 0 && record_before(&value, &snapshot[position - 1])) {
      snapshot[position] = snapshot[position - 1];
      --position;
    }
    snapshot[position] = value;
  }

  char* json_buffer = NULL;
  size_t json_size = 0;
  FILE* file = open_memstream(&json_buffer, &json_size);
  if (!file) {
    unlock_profile();
    return;
  }
  const char* model =
    environment_or_default("NCNN_PROFILE_MODEL", NCNN_PROFILE_DEFAULT_MODEL);
  const char* plan = environment_or_default("NCNN_PROFILE_PLAN_HASH",
                                            NCNN_PROFILE_DEFAULT_PLAN_HASH);
  const char* input_hash = getenv("NCNN_PROFILE_INPUT_HASH");
  const char* build_identity = environment_or_default(
    "NCNN_PROFILE_BUILD_IDENTITY", NCNN_PROFILE_DEFAULT_BUILD_IDENTITY);
  const char* target =
    environment_or_default("NCNN_PROFILE_TARGET", NCNN_PROFILE_DEFAULT_TARGET);
  const char* mode = getenv("NCNN_PROFILE_MODE");
  const char* thread_text = getenv("NCNN_PROFILE_THREADS");
  if (!thread_text || !*thread_text) {
    thread_text = getenv("OMP_NUM_THREADS");
  }
  if (!thread_text || !*thread_text) {
    thread_text = NCNN_PROFILE_DEFAULT_THREADS;
  }
  int thread_known = 0;
  const unsigned thread_count = parse_unsigned(thread_text, &thread_known);
  const char* revision = environment_or_default(
    "NCNN_PROFILE_PLAN_REVISION", NCNN_PROFILE_DEFAULT_PLAN_REVISION);
  const char* attribution_revision =
    environment_or_default("NCNN_PROFILE_ATTRIBUTION_REVISION",
                           NCNN_PROFILE_DEFAULT_ATTRIBUTION_REVISION);
  fprintf(file, "{\n  \"schema_version\": %u,\n", profile_schema_version());
  fputs("  \"kind\": \"ncnn.model_execution_profile\",\n", file);
  fputs("  \"plan_revision\": ", file);
  if (revision && *revision) {
    write_json_string(file, revision);
  } else {
    fputs("\"static-v1\"", file);
  }
  fputs(",\n  \"attribution_revision\": ", file);
  write_json_string(file,
                    attribution_revision ? attribution_revision : "unknown");
  fputs(",\n  \"model\": ", file);
  write_json_string(file, model ? model : "unknown");
  fputs(",\n  \"plan_hash\": ", file);
  if (plan && *plan) {
    write_json_string(file, plan);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"input_hash\": ", file);
  if (input_hash && *input_hash) {
    write_json_string(file, input_hash);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"build_identity\": ", file);
  if (build_identity && *build_identity) {
    write_json_string(file, build_identity);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"target\": ", file);
  if (target && *target) {
    write_json_string(file, target);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"mode\": ", file);
  write_json_string(file, mode ? mode : "diagnostic");
  fputs(",\n  \"threads\": ", file);
  if (thread_known) {
    fprintf(file, "%u", thread_count);
  } else {
    fputs("null", file);
  }
  fprintf(file,
          ",\n  \"invocation_id\": %llu,\n"
          "  \"complete\": %s,\n"
          "  \"instrumentation\": {\"enabled\": true, \"coverage\": "
          "\"explicit-callbacks\", \"aggregation\": \"per-invocation\", "
          "\"invocation_count\": 1, \"record_overflow\": %s, "
          "\"allocation_table_overflow\": %s},\n",
          (unsigned long long)local_invocation_id,
          local_complete ? "true" : "false",
          local_record_overflow ? "true" : "false",
          local_allocation_overflow ? "true" : "false");
  fputs("  \"summary\": {\n", file);
  fprintf(file,
          "    \"allocation_count\": %llu,\n",
          (unsigned long long)local_allocation_events);
  if (local_allocation_bytes_known) {
    fprintf(file,
            "    \"allocation_bytes\": %llu,\n",
            (unsigned long long)local_allocation_bytes);
  } else {
    fputs("    \"allocation_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"allocation_bytes_known\": %s,\n",
          local_allocation_bytes_known ? "true" : "false");
  fprintf(file,
          "    \"deallocation_count\": %llu,\n",
          (unsigned long long)local_deallocation_events);
  if (local_deallocation_bytes_known) {
    fprintf(file,
            "    \"deallocation_bytes\": %llu,\n",
            (unsigned long long)local_deallocation_bytes);
  } else {
    fputs("    \"deallocation_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"deallocation_bytes_known\": %s,\n",
          local_deallocation_bytes_known ? "true" : "false");
  fprintf(
    file, "    \"copy_count\": %llu,\n", (unsigned long long)local_copy_events);
  if (local_copy_bytes_known) {
    fprintf(file,
            "    \"copy_bytes\": %llu,\n",
            (unsigned long long)local_copy_bytes);
  } else {
    fputs("    \"copy_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"copy_bytes_known\": %s,\n",
          local_copy_bytes_known ? "true" : "false");
  fprintf(file,
          "    \"materialized_write_count\": %llu,\n",
          (unsigned long long)local_materialized_write_events);
  if (local_materialized_write_events != 0 &&
      local_materialized_write_bytes_known) {
    fprintf(file,
            "    \"runtime_materialized_write_bytes\": %llu,\n",
            (unsigned long long)local_materialized_write_bytes);
  } else {
    fputs("    \"runtime_materialized_write_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"materialized_write_bytes_known\": %s,\n",
          (local_materialized_write_events != 0 &&
           local_materialized_write_bytes_known)
            ? "true"
            : "false");
  fprintf(file,
          "    \"materialized_read_count\": %llu,\n",
          (unsigned long long)local_materialized_read_events);
  if (local_materialized_read_events != 0 &&
      local_materialized_read_bytes_known) {
    fprintf(file,
            "    \"runtime_materialized_read_bytes\": %llu,\n",
            (unsigned long long)local_materialized_read_bytes);
  } else {
    fputs("    \"runtime_materialized_read_bytes\": null,\n", file);
  }
  fprintf(
    file,
    "    \"materialized_read_bytes_known\": %s,\n",
    (local_materialized_read_events != 0 && local_materialized_read_bytes_known)
      ? "true"
      : "false");
  if (local_materialized_write_events != 0 &&
      local_materialized_expected_read_bytes_known) {
    fprintf(file,
            "    \"expected_materialized_read_bytes\": %llu,\n",
            (unsigned long long)local_materialized_expected_read_bytes);
  } else {
    fputs("    \"expected_materialized_read_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"expected_materialized_read_bytes_known\": %s,\n",
          (local_materialized_write_events != 0 &&
           local_materialized_expected_read_bytes_known)
            ? "true"
            : "false");
  if (local_materialized_write_events == 0 ||
      !local_materialized_expected_read_bytes_known ||
      !local_materialized_read_bytes_known) {
    fputs("    \"materialized_read_complete\": null,\n", file);
  } else {
    fprintf(
      file,
      "    \"materialized_read_complete\": %s,\n",
      local_materialized_expected_read_bytes == local_materialized_read_bytes
        ? "true"
        : "false");
  }
  fprintf(file,
          "    \"transpose_count\": %llu,\n",
          (unsigned long long)local_transpose_events);
  if (local_transpose_write_bytes_known) {
    fprintf(file,
            "    \"runtime_transpose_write_bytes\": %llu,\n",
            (unsigned long long)local_transpose_write_bytes);
  } else {
    fputs("    \"runtime_transpose_write_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"runtime_transpose_write_bytes_known\": %s,\n",
          local_transpose_write_bytes_known ? "true" : "false");
  fprintf(
    file, "    \"pack_count\": %llu,\n", (unsigned long long)local_pack_events);
  if (local_pack_events != 0 && local_pack_bytes_known) {
    fprintf(file,
            "    \"pack_bytes\": %llu,\n",
            (unsigned long long)local_pack_bytes);
  } else {
    fputs("    \"pack_bytes\": null,\n", file);
  }
  fprintf(
    file,
    "    \"pack_bytes_known\": %s,\n",
    (local_pack_events != 0 && local_pack_bytes_known) ? "true" : "false");
  fprintf(file,
          "    \"unpack_count\": %llu,\n",
          (unsigned long long)local_unpack_events);
  if (local_unpack_events != 0 && local_unpack_bytes_known) {
    fprintf(file,
            "    \"unpack_bytes\": %llu,\n",
            (unsigned long long)local_unpack_bytes);
  } else {
    fputs("    \"unpack_bytes\": null,\n", file);
  }
  fprintf(
    file,
    "    \"unpack_bytes_known\": %s,\n",
    (local_unpack_events != 0 && local_unpack_bytes_known) ? "true" : "false");
  fprintf(file,
          "    \"parallel_region_count\": %llu,\n",
          (unsigned long long)local_parallel_events);
  if (local_live_bytes_known) {
    fprintf(file,
            "    \"peak_live_bytes\": %llu,\n",
            (unsigned long long)local_peak_live_bytes);
  } else {
    fputs("    \"peak_live_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"peak_live_proven\": %s,\n",
          local_live_bytes_known ? "true" : "false");
  if (local_top_level_time_known) {
    fprintf(file,
            "    \"top_level_time_ns\": %llu,\n",
            (unsigned long long)local_top_level_time_ns);
  } else {
    fputs("    \"top_level_time_ns\": null,\n", file);
  }
  fprintf(file,
          "    \"top_level_time_known\": %s,\n",
          local_top_level_time_known ? "true" : "false");
  fprintf(file,
          "    \"worker_sampling\": {\"duty\": %u, "
          "\"basis\": \"instance_counter_duty\", "
          "\"interval_overflow\": %s},\n",
          sampling_duty,
          interval_overflow ? "true" : "false");
  fputs(
    "    \"worker_wall_attribution\": {\"basis\": "
    "\"equal_split_across_concurrent_worker_spans\", \"additive\": true, "
    "\"time_domain\": \"wall\"},\n",
    file);
  fprintf(file,
          "    \"event_mismatch_count\": %llu\n",
          (unsigned long long)local_mismatch_events);
  fputs("  },\n  \"events\": [", file);
  for (index = 0; index < snapshot_count; ++index) {
    const ncnn_profile_record* record = &snapshot[index];
    if (index != 0) {
      fputs(",", file);
    }
    fprintf(file,
            "\n    {\"id\": %llu, \"category\": ",
            (unsigned long long)record->id);
    write_json_string(file, category_name(record->category));
    fputs(", \"time_domain\": ", file);
    write_json_string(file,
                      record->category == NCNN_PROFILE_WORKER_OPERATION
                        ? "worker_cpu"
                        : "wall");
    fprintf(file,
            ", \"calls\": %llu, \"inclusive_ns\": %llu, \"exclusive_ns\": ",
            (unsigned long long)record->calls,
            (unsigned long long)record->inclusive_ns);
    if (record->exclusive_known) {
      fprintf(file, "%llu", (unsigned long long)record->exclusive_ns);
    } else {
      fputs("null", file);
    }
    if (record->category == NCNN_PROFILE_WORKER_OPERATION) {
      fputs(", \"worker_wall_union_ns\": ", file);
      if (record->worker_wall_union_known) {
        fprintf(file, "%llu", (unsigned long long)record->worker_wall_union_ns);
      } else {
        fputs("null", file);
      }
      fprintf(file,
              ", \"worker_wall_union_known\": %s",
              record->worker_wall_union_known ? "true" : "false");
      fputs(", \"wall_attributed_ns\": ", file);
      if (record->worker_wall_attributed_known) {
        fprintf(
          file, "%llu", (unsigned long long)record->worker_wall_attributed_ns);
      } else {
        fputs("null", file);
      }
      fputs(", \"wall_attributed_estimated_ns\": ", file);
      if (record->worker_wall_attributed_known) {
        fprintf(file,
                "%llu",
                (unsigned long long)record->worker_wall_attributed_estimated_ns);
      } else {
        fputs("null", file);
      }
      fputs(", \"wall_attributed_share_of_sampled_window\": ", file);
      if (record->worker_wall_attributed_known &&
          attribution_window_total != 0) {
        fprintf(file,
                "%.9f",
                (double)record->worker_wall_attributed_ns /
                  (double)attribution_window_total);
      } else {
        fputs("null", file);
      }
      fprintf(file,
              ", \"wall_attributed_known\": %s, \"calls_estimated\": %llu",
              record->worker_wall_attributed_known ? "true" : "false",
              (unsigned long long)(sampling_duty <= 1
                                     ? record->calls
                                     : record->calls * sampling_duty));
    }
    if (record->category == NCNN_PROFILE_PARALLEL) {
      fprintf(file,
              ", \"worker_covered_wall_ns\": %llu, "
              "\"worker_covered_wall_sampled_ns\": %llu, "
              "\"sampled_window_wall_ns\": %llu, "
              "\"region_wall_ns\": %llu, "
              "\"region_worker_spans\": %llu, "
              "\"sample_scale\": %u, "
              "\"wall_projection_factor\": %.9f",
              (unsigned long long)record->worker_covered_wall_estimated_ns,
              (unsigned long long)record->worker_covered_wall_ns,
              (unsigned long long)record->sampled_window_wall_ns,
              (unsigned long long)record->region_wall_ns,
              (unsigned long long)record->region_worker_spans,
              sampling_duty,
              attribution_projection);
      fputs(", \"wall_coverage_share\": ", file);
      if (record->wall_coverage_known && record->sampled_window_wall_ns != 0) {
        fprintf(file,
                "%.9f",
                (double)record->worker_covered_wall_ns /
                  (double)record->sampled_window_wall_ns);
      } else {
        fputs("null", file);
      }
      fputs(", \"wall_exclusive_estimated_ns\": ", file);
      if (record->wall_coverage_known) {
        fprintf(file,
                "%llu",
                (unsigned long long)record->wall_exclusive_estimated_ns);
      } else {
        fputs("null", file);
      }
      fprintf(file,
              ", \"wall_exclusive_estimated_known\": %s, "
              "\"exclusive_semantics\": ",
              record->wall_coverage_known ? "true" : "false");
      write_json_string(file,
                        record->exclusive_known
                          ? "region_wall_minus_worker_span_union"
                          : "not_proven");
    }
    fputs(", \"bytes\": ", file);
    if (record->bytes_known) {
      fprintf(file, "%llu", (unsigned long long)record->bytes);
    } else {
      fputs("null", file);
    }
    fprintf(
      file, ", \"bytes_known\": %s}", record->bytes_known ? "true" : "false");
  }
  if (snapshot_count != 0) {
    fputs("\n  ", file);
  }
  fputs("]\n}\n", file);
  fflush(file);
  fclose(file);
  FILE* output = fopen(path, v2_output_initialized ? "a" : "w");
  if (!output) {
    free(json_buffer);
    unlock_profile();
    return;
  }
  for (size_t position = 0; position < json_size; ++position) {
    if (json_buffer[position] != '\n' && json_buffer[position] != '\r') {
      fputc(json_buffer[position], output);
    }
  }
  fputc('\n', output);
  fclose(output);
  free(json_buffer);
  v2_output_initialized = 1;
  reset_profile_state_locked();
  invocation_active = 0;
  invocation_complete = 1;
  unlock_profile();
}

void __ncnn_profile_flush(void) {
  const char* path = getenv("NCNN_PROFILE_PATH");
  if (!path || !*path) {
    return;
  }
  if (profile_v2_enabled()) {
    flush_profile_v2(path);
    return;
  }
  ncnn_profile_record snapshot[NCNN_PROFILE_MAX_RECORDS];
  unsigned snapshot_count;
  uint64_t local_allocation_events;
  uint64_t local_allocation_bytes;
  int local_allocation_bytes_known;
  uint64_t local_deallocation_events;
  uint64_t local_deallocation_bytes;
  int local_deallocation_bytes_known;
  uint64_t local_copy_events;
  uint64_t local_copy_bytes;
  int local_copy_bytes_known;
  uint64_t local_materialized_write_events;
  uint64_t local_materialized_write_bytes;
  int local_materialized_write_bytes_known;
  uint64_t local_materialized_read_events;
  uint64_t local_materialized_read_bytes;
  int local_materialized_read_bytes_known;
  uint64_t local_materialized_expected_read_bytes;
  int local_materialized_expected_read_bytes_known;
  uint64_t local_transpose_events;
  uint64_t local_transpose_write_bytes;
  int local_transpose_write_bytes_known;
  uint64_t local_pack_events;
  uint64_t local_pack_bytes;
  int local_pack_bytes_known;
  uint64_t local_unpack_events;
  uint64_t local_unpack_bytes;
  int local_unpack_bytes_known;
  uint64_t local_parallel_events;
  uint64_t local_peak_live_bytes;
  int local_live_bytes_known;
  uint64_t local_mismatch_events;
  uint64_t local_top_level_time_ns;
  int local_top_level_time_known;
  int local_record_overflow;
  int local_allocation_overflow;
  uint64_t local_flush_count;
  lock_profile();
  ++flush_count;
  local_flush_count = flush_count;
  merge_worker_slots_locked();
  snapshot_count = record_count;
  unsigned index;
  for (index = 0; index < snapshot_count; ++index) {
    snapshot[index] = records[index];
  }
  local_allocation_events = allocation_events;
  local_allocation_bytes = allocation_bytes;
  local_allocation_bytes_known = allocation_bytes_known;
  local_deallocation_events = deallocation_events;
  local_deallocation_bytes = deallocation_bytes;
  local_deallocation_bytes_known = deallocation_bytes_known;
  local_copy_events = copy_events;
  local_copy_bytes = copy_bytes;
  local_copy_bytes_known = copy_bytes_known;
  local_materialized_write_events = materialized_write_events;
  local_materialized_write_bytes = materialized_write_bytes;
  local_materialized_write_bytes_known = materialized_write_bytes_known;
  local_materialized_read_events = materialized_read_events;
  local_materialized_read_bytes = materialized_read_bytes;
  local_materialized_read_bytes_known = materialized_read_bytes_known;
  local_materialized_expected_read_bytes = materialized_expected_read_bytes;
  local_materialized_expected_read_bytes_known =
    materialized_expected_read_bytes_known;
  local_transpose_events = transpose_events;
  local_transpose_write_bytes = transpose_write_bytes;
  local_transpose_write_bytes_known = transpose_write_bytes_known;
  local_pack_events = pack_events;
  local_pack_bytes = pack_bytes;
  local_pack_bytes_known = pack_bytes_known;
  local_unpack_events = unpack_events;
  local_unpack_bytes = unpack_bytes;
  local_unpack_bytes_known = unpack_bytes_known;
  local_parallel_events = parallel_events;
  local_peak_live_bytes = peak_live_bytes;
  local_live_bytes_known = live_bytes_known;
  local_mismatch_events = mismatch_events;
  local_top_level_time_ns = top_level_time_ns;
  local_top_level_time_known = top_level_time_known;
  local_record_overflow = record_overflow;
  local_allocation_overflow = allocation_overflow;

  for (index = 1; index < snapshot_count; ++index) {
    ncnn_profile_record value = snapshot[index];
    unsigned position = index;
    while (position != 0 && record_before(&value, &snapshot[position - 1])) {
      snapshot[position] = snapshot[position - 1];
      --position;
    }
    snapshot[position] = value;
  }

  // Keep the lock until the complete snapshot is written.  Multiple generated
  // entry points can flush concurrently in one process.
  FILE* file = fopen(path, "w");
  if (!file) {
    unlock_profile();
    return;
  }
  const char* model =
    environment_or_default("NCNN_PROFILE_MODEL", NCNN_PROFILE_DEFAULT_MODEL);
  const char* plan = environment_or_default("NCNN_PROFILE_PLAN_HASH",
                                            NCNN_PROFILE_DEFAULT_PLAN_HASH);
  const char* input_hash = getenv("NCNN_PROFILE_INPUT_HASH");
  const char* build_identity = environment_or_default(
    "NCNN_PROFILE_BUILD_IDENTITY", NCNN_PROFILE_DEFAULT_BUILD_IDENTITY);
  const char* target =
    environment_or_default("NCNN_PROFILE_TARGET", NCNN_PROFILE_DEFAULT_TARGET);
  const char* mode = getenv("NCNN_PROFILE_MODE");
  const char* thread_text = getenv("NCNN_PROFILE_THREADS");
  if (!thread_text || !*thread_text) {
    thread_text = getenv("OMP_NUM_THREADS");
  }
  if (!thread_text || !*thread_text) {
    thread_text = NCNN_PROFILE_DEFAULT_THREADS;
  }
  int thread_known = 0;
  const unsigned thread_count = parse_unsigned(thread_text, &thread_known);
  // The revision must match the execution plan the sidecar claims to
  // describe; ncnn-compile passes the published plan revision at build time,
  // so the attribution join cannot be defeated by a stale hard-coded string.
  const char* revision = environment_or_default(
    "NCNN_PROFILE_PLAN_REVISION", NCNN_PROFILE_DEFAULT_PLAN_REVISION);
  const char* attribution_revision =
    environment_or_default("NCNN_PROFILE_ATTRIBUTION_REVISION",
                           NCNN_PROFILE_DEFAULT_ATTRIBUTION_REVISION);
  fputs("{\n  \"schema_version\": 1,\n", file);
  fputs("  \"kind\": \"ncnn.model_execution_profile\",\n", file);
  fputs("  \"plan_revision\": ", file);
  if (revision && *revision) {
    write_json_string(file, revision);
  } else {
    fputs("\"static-v1\"", file);
  }
  fputs(",\n  \"attribution_revision\": ", file);
  write_json_string(file,
                    attribution_revision ? attribution_revision : "unknown");
  fputs(",\n  \"model\": ", file);
  write_json_string(file, model ? model : "unknown");
  fputs(",\n  \"plan_hash\": ", file);
  if (plan && *plan) {
    write_json_string(file, plan);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"input_hash\": ", file);
  if (input_hash && *input_hash) {
    write_json_string(file, input_hash);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"build_identity\": ", file);
  if (build_identity && *build_identity) {
    write_json_string(file, build_identity);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"target\": ", file);
  if (target && *target) {
    write_json_string(file, target);
  } else {
    fputs("null", file);
  }
  fputs(",\n  \"mode\": ", file);
  write_json_string(file, mode ? mode : "diagnostic");
  fputs(",\n  \"threads\": ", file);
  if (thread_known) {
    fprintf(file, "%u", thread_count);
  } else {
    fputs("null", file);
  }
  fprintf(file,
          ",\n  \"instrumentation\": {\"enabled\": true, \"coverage\": "
          "\"explicit-callbacks\", \"aggregation\": \"process-cumulative\", "
          "\"invocation_count\": %llu, \"record_overflow\": %s, "
          "\"allocation_table_overflow\": %s},\n",
          (unsigned long long)local_flush_count,
          local_record_overflow ? "true" : "false",
          local_allocation_overflow ? "true" : "false");
  fputs("  \"summary\": {\n", file);
  fprintf(file,
          "    \"allocation_count\": %llu,\n",
          (unsigned long long)local_allocation_events);
  if (local_allocation_bytes_known) {
    fprintf(file,
            "    \"allocation_bytes\": %llu,\n",
            (unsigned long long)local_allocation_bytes);
  } else {
    fputs("    \"allocation_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"allocation_bytes_known\": %s,\n",
          local_allocation_bytes_known ? "true" : "false");
  fprintf(file,
          "    \"deallocation_count\": %llu,\n",
          (unsigned long long)local_deallocation_events);
  if (local_deallocation_bytes_known) {
    fprintf(file,
            "    \"deallocation_bytes\": %llu,\n",
            (unsigned long long)local_deallocation_bytes);
  } else {
    fputs("    \"deallocation_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"deallocation_bytes_known\": %s,\n",
          local_deallocation_bytes_known ? "true" : "false");
  fprintf(
    file, "    \"copy_count\": %llu,\n", (unsigned long long)local_copy_events);
  if (local_copy_bytes_known) {
    fprintf(file,
            "    \"copy_bytes\": %llu,\n",
            (unsigned long long)local_copy_bytes);
  } else {
    fputs("    \"copy_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"copy_bytes_known\": %s,\n",
          local_copy_bytes_known ? "true" : "false");
  fprintf(file,
          "    \"materialized_write_count\": %llu,\n",
          (unsigned long long)local_materialized_write_events);
  if (local_materialized_write_events != 0 &&
      local_materialized_write_bytes_known) {
    fprintf(file,
            "    \"runtime_materialized_write_bytes\": %llu,\n",
            (unsigned long long)local_materialized_write_bytes);
  } else {
    fputs("    \"runtime_materialized_write_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"materialized_write_bytes_known\": %s,\n",
          (local_materialized_write_events != 0 &&
           local_materialized_write_bytes_known)
            ? "true"
            : "false");
  fprintf(file,
          "    \"materialized_read_count\": %llu,\n",
          (unsigned long long)local_materialized_read_events);
  if (local_materialized_read_events != 0 &&
      local_materialized_read_bytes_known) {
    fprintf(file,
            "    \"runtime_materialized_read_bytes\": %llu,\n",
            (unsigned long long)local_materialized_read_bytes);
  } else {
    fputs("    \"runtime_materialized_read_bytes\": null,\n", file);
  }
  fprintf(
    file,
    "    \"materialized_read_bytes_known\": %s,\n",
    (local_materialized_read_events != 0 && local_materialized_read_bytes_known)
      ? "true"
      : "false");
  if (local_materialized_write_events != 0 &&
      local_materialized_expected_read_bytes_known) {
    fprintf(file,
            "    \"expected_materialized_read_bytes\": %llu,\n",
            (unsigned long long)local_materialized_expected_read_bytes);
  } else {
    fputs("    \"expected_materialized_read_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"expected_materialized_read_bytes_known\": %s,\n",
          (local_materialized_write_events != 0 &&
           local_materialized_expected_read_bytes_known)
            ? "true"
            : "false");
  if (local_materialized_write_events == 0 ||
      !local_materialized_expected_read_bytes_known ||
      !local_materialized_read_bytes_known) {
    fputs("    \"materialized_read_complete\": null,\n", file);
  } else {
    fprintf(
      file,
      "    \"materialized_read_complete\": %s,\n",
      local_materialized_expected_read_bytes == local_materialized_read_bytes
        ? "true"
        : "false");
  }
  fprintf(file,
          "    \"transpose_count\": %llu,\n",
          (unsigned long long)local_transpose_events);
  if (local_transpose_write_bytes_known) {
    fprintf(file,
            "    \"runtime_transpose_write_bytes\": %llu,\n",
            (unsigned long long)local_transpose_write_bytes);
  } else {
    fputs("    \"runtime_transpose_write_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"runtime_transpose_write_bytes_known\": %s,\n",
          local_transpose_write_bytes_known ? "true" : "false");
  fprintf(
    file, "    \"pack_count\": %llu,\n", (unsigned long long)local_pack_events);
  if (local_pack_events != 0 && local_pack_bytes_known) {
    fprintf(file,
            "    \"pack_bytes\": %llu,\n",
            (unsigned long long)local_pack_bytes);
  } else {
    fputs("    \"pack_bytes\": null,\n", file);
  }
  fprintf(
    file,
    "    \"pack_bytes_known\": %s,\n",
    (local_pack_events != 0 && local_pack_bytes_known) ? "true" : "false");
  fprintf(file,
          "    \"unpack_count\": %llu,\n",
          (unsigned long long)local_unpack_events);
  if (local_unpack_events != 0 && local_unpack_bytes_known) {
    fprintf(file,
            "    \"unpack_bytes\": %llu,\n",
            (unsigned long long)local_unpack_bytes);
  } else {
    fputs("    \"unpack_bytes\": null,\n", file);
  }
  fprintf(
    file,
    "    \"unpack_bytes_known\": %s,\n",
    (local_unpack_events != 0 && local_unpack_bytes_known) ? "true" : "false");
  fprintf(file,
          "    \"parallel_region_count\": %llu,\n",
          (unsigned long long)local_parallel_events);
  if (local_live_bytes_known) {
    fprintf(file,
            "    \"peak_live_bytes\": %llu,\n",
            (unsigned long long)local_peak_live_bytes);
  } else {
    fputs("    \"peak_live_bytes\": null,\n", file);
  }
  fprintf(file,
          "    \"peak_live_proven\": %s,\n",
          local_live_bytes_known ? "true" : "false");
  if (local_top_level_time_known) {
    fprintf(file,
            "    \"top_level_time_ns\": %llu,\n",
            (unsigned long long)local_top_level_time_ns);
  } else {
    fputs("    \"top_level_time_ns\": null,\n", file);
  }
  fprintf(file,
          "    \"top_level_time_known\": %s,\n",
          local_top_level_time_known ? "true" : "false");
  fprintf(file,
          "    \"worker_sampling\": {\"duty\": %u, "
          "\"basis\": \"instance_counter_duty\", "
          "\"interval_overflow\": %s},\n",
          sampling_duty,
          interval_overflow ? "true" : "false");
  fputs(
    "    \"worker_wall_attribution\": {\"basis\": "
    "\"equal_split_across_concurrent_worker_spans\", \"additive\": true, "
    "\"time_domain\": \"wall\"},\n",
    file);
  fprintf(file,
          "    \"event_mismatch_count\": %llu\n",
          (unsigned long long)local_mismatch_events);
  fputs("  },\n  \"events\": [", file);
  for (index = 0; index < snapshot_count; ++index) {
    const ncnn_profile_record* record = &snapshot[index];
    if (index != 0) {
      fputs(",", file);
    }
    fprintf(file,
            "\n    {\"id\": %llu, \"category\": ",
            (unsigned long long)record->id);
    write_json_string(file, category_name(record->category));
    fputs(", \"time_domain\": ", file);
    write_json_string(file,
                      record->category == NCNN_PROFILE_WORKER_OPERATION
                        ? "worker_cpu"
                        : "wall");
    fprintf(file,
            ", \"calls\": %llu, \"inclusive_ns\": %llu, \"exclusive_ns\": ",
            (unsigned long long)record->calls,
            (unsigned long long)record->inclusive_ns);
    if (record->exclusive_known) {
      fprintf(file, "%llu", (unsigned long long)record->exclusive_ns);
    } else {
      fputs("null", file);
    }
    if (record->category == NCNN_PROFILE_WORKER_OPERATION) {
      fputs(", \"worker_wall_union_ns\": ", file);
      if (record->worker_wall_union_known) {
        fprintf(file, "%llu", (unsigned long long)record->worker_wall_union_ns);
      } else {
        fputs("null", file);
      }
      fprintf(file,
              ", \"worker_wall_union_known\": %s",
              record->worker_wall_union_known ? "true" : "false");
      fputs(", \"wall_attributed_ns\": ", file);
      if (record->worker_wall_attributed_known) {
        fprintf(
          file, "%llu", (unsigned long long)record->worker_wall_attributed_ns);
      } else {
        fputs("null", file);
      }
      fputs(", \"wall_attributed_estimated_ns\": ", file);
      if (record->worker_wall_attributed_known) {
        fprintf(file,
                "%llu",
                (unsigned long long)record->worker_wall_attributed_estimated_ns);
      } else {
        fputs("null", file);
      }
      fputs(", \"wall_attributed_share_of_sampled_window\": ", file);
      if (record->worker_wall_attributed_known &&
          attribution_window_total != 0) {
        fprintf(file,
                "%.9f",
                (double)record->worker_wall_attributed_ns /
                  (double)attribution_window_total);
      } else {
        fputs("null", file);
      }
      fprintf(file,
              ", \"wall_attributed_known\": %s, \"calls_estimated\": %llu",
              record->worker_wall_attributed_known ? "true" : "false",
              (unsigned long long)(sampling_duty <= 1
                                     ? record->calls
                                     : record->calls * sampling_duty));
    }
    if (record->category == NCNN_PROFILE_PARALLEL) {
      fprintf(file,
              ", \"worker_covered_wall_ns\": %llu, "
              "\"worker_covered_wall_sampled_ns\": %llu, "
              "\"sampled_window_wall_ns\": %llu, "
              "\"region_wall_ns\": %llu, "
              "\"region_worker_spans\": %llu, "
              "\"sample_scale\": %u, "
              "\"wall_projection_factor\": %.9f",
              (unsigned long long)record->worker_covered_wall_estimated_ns,
              (unsigned long long)record->worker_covered_wall_ns,
              (unsigned long long)record->sampled_window_wall_ns,
              (unsigned long long)record->region_wall_ns,
              (unsigned long long)record->region_worker_spans,
              sampling_duty,
              attribution_projection);
      fputs(", \"wall_coverage_share\": ", file);
      if (record->wall_coverage_known && record->sampled_window_wall_ns != 0) {
        fprintf(file,
                "%.9f",
                (double)record->worker_covered_wall_ns /
                  (double)record->sampled_window_wall_ns);
      } else {
        fputs("null", file);
      }
      fputs(", \"wall_exclusive_estimated_ns\": ", file);
      if (record->wall_coverage_known) {
        fprintf(file,
                "%llu",
                (unsigned long long)record->wall_exclusive_estimated_ns);
      } else {
        fputs("null", file);
      }
      fprintf(file,
              ", \"wall_exclusive_estimated_known\": %s, "
              "\"exclusive_semantics\": ",
              record->wall_coverage_known ? "true" : "false");
      write_json_string(file,
                        record->exclusive_known
                          ? "region_wall_minus_worker_span_union"
                          : "not_proven");
    }
    fputs(", \"bytes\": ", file);
    if (record->bytes_known) {
      fprintf(file, "%llu", (unsigned long long)record->bytes);
    } else {
      fputs("null", file);
    }
    fprintf(
      file, ", \"bytes_known\": %s}", record->bytes_known ? "true" : "false");
  }
  if (snapshot_count != 0) {
    fputs("\n  ", file);
  }
  fputs("]\n}\n", file);
  fclose(file);
  unlock_profile();
}
