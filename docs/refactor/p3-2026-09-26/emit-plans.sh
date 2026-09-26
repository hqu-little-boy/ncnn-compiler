#!/usr/bin/env bash
# P3 逐字节对照采证：对固定的一组模型跑 ncnn-compile，只收集
# .plan.json / .h / manifest .json / IR dump —— 这些是「产物语义」的载体。
#
# 用法:
#   STAGE=<构建目录> ./emit-plans.sh <输出目录>
#
# 可选环境变量:
#   MODELS_ROOT  模型仓库根（默认取本脚本上两级目录）
#   MODELS       空格分隔的模型名子集（默认全部）
#
# 该脚本用于「改前 / 改后」两次采证后 diff -ru，是 T-A1/T-A3 零行为变化的证据。
# 不把唯一证据留在 /tmp：本脚本随 docs/refactor/p3-2026-09-26/ 一并归档。
set -euo pipefail

STAGE="${STAGE:?STAGE must point at a configured compiler build directory}"
OUT="${1:?usage: emit-plans.sh <output-directory>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 模型资源在 compiler 仓库之外（yolov5_v7.0_models/、ncnn_modelzoo/）与
# 仓库之内（test/third_party/）。统一按「compiler 仓库的父目录」为根，
# 避免硬编码开发者绝对路径。可用 MODELS_ROOT 覆盖。
if [[ -z "${MODELS_ROOT:-}" ]]; then
  repo_root="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null \
               || echo /mnt/ncnn-compiler/compiler)"
  MODELS_ROOT="$(dirname "${repo_root}")"
fi

NCNN_COMPILE="${STAGE}/tools/ncnn-compile"
DRIVER="${STAGE}/tools/ncnn-mlir-driver"
OPT="${STAGE}/bin/ncnn-mlir-opt"
TRANSLATE="${TRANSLATE:-/usr/bin/mlir-translate-21}"
CLANG="${CLANG:-/usr/bin/clang-21}"
LLVM_NM="${LLVM_NM:-/usr/bin/llvm-nm-21}"
LLVM_READELF="${LLVM_READELF:-/usr/bin/llvm-readelf-21}"
LLVM_AS="${LLVM_AS:-/usr/bin/llvm-as-21}"

for tool in "${NCNN_COMPILE}" "${DRIVER}" "${OPT}" "${TRANSLATE}" "${CLANG}"; do
  if [[ ! -x "${tool}" && ! -f "${tool}" ]]; then
    echo "missing tool: ${tool}" >&2
    exit 1
  fi
done

mkdir -p "${OUT}"
# ncnn-compile 会拒绝写入含「非自己产物」的输出目录，因此编译日志必须留在
# 对照树之外；diff -ru 也不需要 --exclude。
LOG_DIR="${OUT}.logs"
mkdir -p "${LOG_DIR}"

# ── 模型清单 ──────────────────────────────────────────────────────────────
# 沿用 P2 的 9 模型对照清单（docs/refactor/p2-2026-09-26/verification.log §6），
# 另加 resnet18_winograd 覆盖 --conv-strategy 分派路径（T-A1 改 parseStrategy）。
#
# 字段: name|param|bin|input-shape|extra-args...
read -r -d '' MODELS_SPEC <<'EOF' || true
squeezenet_v1_1|compiler/test/third_party/ncnn/examples/squeezenet_v1.1.param|compiler/test/third_party/ncnn/examples/squeezenet_v1.1.bin||
resnet18|yolov5_v7.0_models/resnet18.ncnn.param|yolov5_v7.0_models/resnet18.ncnn.bin|3x224x224|
resnet18_winograd|yolov5_v7.0_models/resnet18.ncnn.param|yolov5_v7.0_models/resnet18.ncnn.bin|3x224x224|--conv-strategy=winograd
chineseocr_lite_anglenet|ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.param|ncnn_modelzoo/liteocr/Chineseocr_Lite_AngleNet.bin|3x32x192|
yolov5n|yolov5_v7.0_models/yolov5n.ncnn.param|yolov5_v7.0_models/yolov5n.ncnn.bin|3x640x640|
yolov5x_seg|yolov5_v7.0_models/yolov5x_seg.ncnn.param|yolov5_v7.0_models/yolov5x_seg.ncnn.bin|3x640x640|
pp_ocrv6_tiny_rec|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.param|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.bin|3x48x320|
pp_ocrv6_tiny_det|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin|3x640x640|
pp_ocrv6_tiny_rec_dynamic|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.param|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_rec.bin|3x48x?|--input-dim-constraint=0:2:min=5,multiple=1
pp_ocrv6_tiny_det_dynamic|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.param|ncnn_modelzoo/liteocr/PP-OCRv6_tiny_det.bin|3x?x?|--input-dim-constraint=0:1:min=32,multiple=32 --input-dim-constraint=0:2:min=32,multiple=32
EOF

# 与 test/Numerical/CMakeLists.txt 的 ncnn_model_vector_args 保持一致：
# x86-64 固定 8-lane + AVX2/FMA，避免采证口径偏离 golden 口径。
VECTOR_ARGS=(--vector-mode fixed-width
             --target-feature=+avx2
             --target-feature=+fma
             --clang-arg=-march=x86-64-v3)

status=0
while IFS='|' read -r name param bin shape extra; do
  [[ -z "${name}" ]] && continue
  if [[ -n "${MODELS:-}" && " ${MODELS} " != *" ${name} "* ]]; then
    continue
  fi
  out="${OUT}/${name}"
  mkdir -p "${out}"

  args=(
    --driver "${DRIVER}"
    --opt "${OPT}"
    --translate "${TRANSLATE}"
    --clang "${CLANG}"
    --nm "${LLVM_NM}"
    --readelf "${LLVM_READELF}"
    --llvm-as "${LLVM_AS}"
    --param "${MODELS_ROOT}/${param}"
    --bin "${MODELS_ROOT}/${bin}"
    --model-name "${name}"
    --matmul-packing auto
    "${VECTOR_ARGS[@]}"
    --output-dir "${out}"
    --emit-manifest
    --emit-execution-plan
    --emit ncnn
    --emit tosa
    --emit linalg
  )
  if [[ -n "${shape}" ]]; then
    args+=(--input-shape "${shape}")
  fi
  if [[ -n "${extra}" ]]; then
    # shellcheck disable=SC2206
    extra_arr=(${extra})
    args+=("${extra_arr[@]}")
  fi

  echo "=== ${name} ==="
  start=${SECONDS}
  if ! "${NCNN_COMPILE}" "${args[@]}" > "${LOG_DIR}/${name}.log" 2>&1; then
    echo "FAILED: ${name} (see ${LOG_DIR}/${name}.log)" >&2
    status=1
  else
    echo "ok: ${name} ($((SECONDS - start))s)"
  fi
done <<< "${MODELS_SPEC}"

exit "${status}"
