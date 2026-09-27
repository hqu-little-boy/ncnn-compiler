#!/usr/bin/env bash
# T-J1 前后字节对照采证：用固定 harness + 固定 env 跑 profile runtime，
# 收集生成的 profile 文本。改前/改后各跑一次，合法输入路径 diff 应为空。
#
# 覆盖两个 writer：schema 1（__ncnn_profile_flush）与 schema 3（flush_profile_v2）。
#
# 用法:
#   CC=/usr/bin/clang-21 RUNTIME=lib/ProfileRuntime/profile_runtime.c \
#     [WRITER=lib/ProfileRuntime/profile_json_writer.c] \
#     ./capture-profile-golden.sh <输出目录>
#
# 不把唯一证据留在 /tmp：本脚本与结果一并归档到
# docs/refactor/p5-2026-09-27/。
set -euo pipefail

CC="${CC:-/usr/bin/clang-21}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME="${RUNTIME:-${ROOT}/lib/ProfileRuntime/profile_runtime.c}"
WRITER="${WRITER:-${ROOT}/lib/ProfileRuntime/profile_json_writer.c}"
OUT="${1:?usage: capture-profile-golden.sh <output-directory>}"

mkdir -p "${OUT}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

SOURCES=("${RUNTIME}")
if [[ -f "${WRITER}" ]]; then
  SOURCES+=("${WRITER}")
fi

cat > "${WORK}/basic.c" <<'EOF'
#include <stdint.h>
extern void __ncnn_profile_event_begin(int64_t, int64_t);
extern void __ncnn_profile_event_end(int64_t);
extern void __ncnn_profile_alloc(int64_t, int64_t);
extern void __ncnn_profile_dealloc(int64_t);
extern void __ncnn_profile_copy(int64_t, int64_t);
extern void __ncnn_profile_movement(int64_t, int64_t, int64_t);
extern void __ncnn_profile_materialized(int64_t, int64_t, int64_t, int64_t);
extern void __ncnn_profile_flush(void);
int main(void) {
  __ncnn_profile_event_begin(7, 0);
  __ncnn_profile_event_begin(8, 4);
  __ncnn_profile_event_end(8);
  __ncnn_profile_event_end(7);
  __ncnn_profile_alloc(9, 64);
  __ncnn_profile_copy(10, 32);
  __ncnn_profile_materialized(14, 0, 256, 1);
  __ncnn_profile_materialized(14, 1, 256, 0);
  __ncnn_profile_movement(11, 0, 128);
  __ncnn_profile_movement(12, 1, 256);
  __ncnn_profile_movement(13, 2, 512);
  __ncnn_profile_dealloc(9);
  __ncnn_profile_flush();
  __ncnn_profile_flush();
  return 0;
}
EOF

"${CC}" -std=c11 -Wall -Wextra -Werror \
  -DNCNN_PROFILE_DEFAULT_PLAN_REVISION='"static-v1|int8-target-v1"' \
  "${WORK}/basic.c" "${SOURCES[@]}" -o "${WORK}/basic"

# run_case <name> <schema> <env...>  —— env 里必须含 NCNN_PROFILE_* 取值
run_case() {
  local name="$1" schema="$2"
  shift 2
  local out="${OUT}/${name}.json"
  # env -i 保证只有我们给的变量影响输出（尤其 NCNN_PROFILE_* 与 OMP_NUM_THREADS）
  env -i PATH="${PATH}" NCNN_PROFILE_SCHEMA="${schema}" \
    NCNN_PROFILE_PATH="${out}" "$@" "${WORK}/basic" >/dev/null
}

# 字符串字段共 6 个可注入：model / plan_hash / build_identity / target /
# mode / input_hash。转义路径都在 write_json_string/pj_string 里。
ESC_QUOTE='quote"back\slash'
ESC_CTRL="$(printf 'a\tb\nc\rd\be\ff\x01z')"
UTF8_OK='中文-é-🎯-日本語-я'
UTF8_ORPHAN="$(printf 'orphan-\x80-end')"
UTF8_TRUNC="$(printf 'trunc-\xe2\x82')"
UTF8_OVERLONG="$(printf 'overlong-\xc0\xaf')"
UTF8_SURROGATE="$(printf 'surrogate-\xed\xa0\x80')"
EDGE_EMPTY=''

for schema in 1 2 3; do
  s="s${schema}"
  run_case "${s}-01-ascii" "${schema}" \
    NCNN_PROFILE_MODEL=fixture NCNN_PROFILE_PLAN_HASH=plan-123 \
    NCNN_PROFILE_BUILD_IDENTITY=build-123 NCNN_PROFILE_TARGET=x86_64-linux \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE=diagnostic \
    NCNN_PROFILE_INPUT_HASH=hash-abc

  run_case "${s}-02-escape" "${schema}" \
    NCNN_PROFILE_MODEL="${ESC_QUOTE}" NCNN_PROFILE_PLAN_HASH="${ESC_CTRL}" \
    NCNN_PROFILE_BUILD_IDENTITY='tab\there' NCNN_PROFILE_TARGET='nl\nhere' \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE='cr\rhere' \
    NCNN_PROFILE_INPUT_HASH='bs\bff\f'

  run_case "${s}-03-utf8-legal" "${schema}" \
    NCNN_PROFILE_MODEL="${UTF8_OK}" NCNN_PROFILE_PLAN_HASH="${UTF8_OK}" \
    NCNN_PROFILE_BUILD_IDENTITY="${UTF8_OK}" NCNN_PROFILE_TARGET="${UTF8_OK}" \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE="${UTF8_OK}" \
    NCNN_PROFILE_INPUT_HASH="${UTF8_OK}"

  run_case "${s}-04-utf8-orphan" "${schema}" \
    NCNN_PROFILE_MODEL="${UTF8_ORPHAN}" NCNN_PROFILE_PLAN_HASH=plan \
    NCNN_PROFILE_BUILD_IDENTITY=build NCNN_PROFILE_TARGET=host \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE=mode NCNN_PROFILE_INPUT_HASH=hash

  run_case "${s}-05-utf8-truncated" "${schema}" \
    NCNN_PROFILE_MODEL="${UTF8_TRUNC}" NCNN_PROFILE_PLAN_HASH=plan \
    NCNN_PROFILE_BUILD_IDENTITY=build NCNN_PROFILE_TARGET=host \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE=mode NCNN_PROFILE_INPUT_HASH=hash

  run_case "${s}-06-utf8-overlong" "${schema}" \
    NCNN_PROFILE_MODEL="${UTF8_OVERLONG}" NCNN_PROFILE_PLAN_HASH=plan \
    NCNN_PROFILE_BUILD_IDENTITY=build NCNN_PROFILE_TARGET=host \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE=mode NCNN_PROFILE_INPUT_HASH=hash

  run_case "${s}-07-utf8-surrogate" "${schema}" \
    NCNN_PROFILE_MODEL="${UTF8_SURROGATE}" NCNN_PROFILE_PLAN_HASH=plan \
    NCNN_PROFILE_BUILD_IDENTITY=build NCNN_PROFILE_TARGET=host \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE=mode NCNN_PROFILE_INPUT_HASH=hash

  run_case "${s}-08-edge" "${schema}" \
    NCNN_PROFILE_MODEL="${EDGE_EMPTY}" NCNN_PROFILE_PLAN_HASH="${EDGE_EMPTY}" \
    NCNN_PROFILE_BUILD_IDENTITY='""' NCNN_PROFILE_TARGET='\\\\' \
    NCNN_PROFILE_THREADS=2 NCNN_PROFILE_MODE='\"' NCNN_PROFILE_INPUT_HASH=' '
done

echo "captured $(find "${OUT}" -name '*.json' | wc -l) golden profiles into ${OUT}"
