#define _POSIX_C_SOURCE 200809L

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

#define NCNN_PROFILE_MAX_RECORDS 4096
#define NCNN_PROFILE_MAX_STACK 128
#define NCNN_PROFILE_MAX_ALLOCS 4096

enum {
  NCNN_PROFILE_OPERATION = 0,
  NCNN_PROFILE_ALLOCATION = 1,
  NCNN_PROFILE_DEALLOCATION = 2,
  NCNN_PROFILE_COPY = 3,
  NCNN_PROFILE_PARALLEL = 4,
};

typedef struct {
  uint64_t id;
  uint64_t calls;
  uint64_t inclusive_ns;
  uint64_t exclusive_ns;
  int64_t category;
  int exclusive_known;
} ncnn_profile_record;

typedef struct {
  uint64_t id;
  uint64_t start_ns;
  uint64_t child_ns;
  int64_t category;
  int record;
} ncnn_profile_frame;

typedef struct {
  uint64_t id;
  int64_t bytes;
  uint64_t active;
} ncnn_profile_allocation;

static ncnn_profile_record records[NCNN_PROFILE_MAX_RECORDS];
static unsigned record_count;
static int record_overflow;
static ncnn_profile_allocation allocations[NCNN_PROFILE_MAX_ALLOCS];
static unsigned allocation_count;
static int allocation_overflow;
static atomic_flag profile_lock = ATOMIC_FLAG_INIT;
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
static uint64_t parallel_events;
static uint64_t live_bytes;
static uint64_t peak_live_bytes;
static int live_bytes_known = 1;
static uint64_t mismatch_events;
static uint64_t top_level_time_ns;
static int top_level_time_known = 1;
static uint64_t flush_count;

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

static int record_for(uint64_t id, int64_t category) {
  unsigned index;
  for (index = 0; index < record_count; ++index) {
    if (records[index].id == id && records[index].category == category) {
      return (int)index;
    }
  }
  if (record_count == NCNN_PROFILE_MAX_RECORDS) {
    record_overflow = 1;
    return -1;
  }
  records[record_count].id = id;
  records[record_count].calls = 0;
  records[record_count].inclusive_ns = 0;
  records[record_count].exclusive_ns = 0;
  records[record_count].category = category;
  records[record_count].exclusive_known = 1;
  return (int)record_count++;
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

void __ncnn_profile_event_begin(int64_t signed_id, int64_t category) {
  const uint64_t id = (uint64_t)signed_id;
  const uint64_t start = profile_now();
  lock_profile();
  const int record = record_for(id, category);
  if (record >= 0 && category != NCNN_PROFILE_ALLOCATION &&
      category != NCNN_PROFILE_COPY) {
    records[record].calls++;
  }
  if (category == NCNN_PROFILE_PARALLEL) {
    parallel_events++;
  }
  unlock_profile();
  if (stack_depth == NCNN_PROFILE_MAX_STACK) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  stack[stack_depth].id = id;
  stack[stack_depth].start_ns = start;
  stack[stack_depth].child_ns = 0;
  stack[stack_depth].category = category;
  stack[stack_depth].record = record;
  stack_depth++;
}

void __ncnn_profile_event_end(int64_t signed_id) {
  const uint64_t id = (uint64_t)signed_id;
  const uint64_t end = profile_now();
  if (stack_depth == 0 || stack[stack_depth - 1].id != id) {
    lock_profile();
    mismatch_events++;
    unlock_profile();
    return;
  }
  ncnn_profile_frame frame = stack[--stack_depth];
  const uint64_t elapsed = end >= frame.start_ns ? end - frame.start_ns : 0;
  const uint64_t exclusive =
    elapsed >= frame.child_ns ? elapsed - frame.child_ns : 0;
  lock_profile();
  if (frame.record >= 0) {
    records[frame.record].inclusive_ns += elapsed;
    if (frame.category == NCNN_PROFILE_PARALLEL) {
      // Worker callbacks run on different TLS stacks, so their time cannot be
      // subtracted from the parent frame without a runtime-wide span model.
      records[frame.record].exclusive_known = 0;
    } else if (records[frame.record].exclusive_known) {
      records[frame.record].exclusive_ns += exclusive;
    }
  }
  if (stack_depth != 0) {
    stack[stack_depth - 1].child_ns += elapsed;
  } else if (top_level_time_known) {
    if (top_level_time_ns <= UINT64_MAX - elapsed) {
      top_level_time_ns += elapsed;
    } else {
      top_level_time_known = 0;
    }
  }
  unlock_profile();
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
  }
  unlock_profile();
}

void __ncnn_profile_dealloc(int64_t signed_id) {
  const uint64_t id = (uint64_t)signed_id;
  lock_profile();
  deallocation_events++;
  const int slot = allocation_for(id);
  if (slot >= 0 && allocations[slot].active) {
    if (allocations[slot].bytes >= 0) {
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

void __ncnn_profile_flush(void) {
  const char* path = getenv("NCNN_PROFILE_PATH");
  if (!path || !*path) {
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
  fputs("{\n  \"schema_version\": 1,\n", file);
  fputs("  \"kind\": \"ncnn.model_execution_profile\",\n", file);
  fputs("  \"plan_revision\": ", file);
  if (revision && *revision) {
    write_json_string(file, revision);
  } else {
    fputs("\"static-v1\"", file);
  }
  fputs(",\n  \"model\": ", file);
  write_json_string(file, model ? model : "unknown");
  fputs(",\n  \"plan_hash\": ", file);
  if (plan && *plan) {
    write_json_string(file, plan);
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
    fprintf(file,
            ", \"calls\": %llu, \"inclusive_ns\": %llu, \"exclusive_ns\": ",
            (unsigned long long)record->calls,
            (unsigned long long)record->inclusive_ns);
    if (record->exclusive_known) {
      fprintf(file, "%llu", (unsigned long long)record->exclusive_ns);
    } else {
      fputs("null", file);
    }
    fputs("}", file);
  }
  if (snapshot_count != 0) {
    fputs("\n  ", file);
  }
  fputs("]\n}\n", file);
  fclose(file);
  unlock_profile();
}
