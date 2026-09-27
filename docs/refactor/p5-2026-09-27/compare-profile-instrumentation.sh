#!/usr/bin/env bash
# T-M4 产物逐字节对照：--profile 插桩路径。
#
# T-M4 只动 InstrumentNCNNProfile 的文件组织，非 profile 编译根本不跑这些
# pass，因此 P3 的 emit-plans.sh（不带 --profile）测不到它。这里专测插桩：
# 同一组模型分别用「参考构建」与「本次构建」带 --profile 编译，逐字节 diff
# 全部阶段产物（含 model.memref.mlir —— 插桩就落在这一层）。
#
# 用法:
#   ./compare-profile-instrumentation.sh <参考构建目录> <本次构建目录> <输出目录>
#
# 可选环境变量:
#   MODELS_ROOT  模型仓库根（默认取本脚本上四级目录的父目录）
#   MODELS       空格分隔的模型名子集
set -euo pipefail

REF="${1:?usage: compare-profile-instrumentation.sh <ref-build> <new-build> <out>}"
NEW="${2:?usage: compare-profile-instrumentation.sh <ref-build> <new-build> <out>}"
OUT="${3:?usage: compare-profile-instrumentation.sh <ref-build> <new-build> <out>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${MODELS_ROOT:-}" ]]; then
  repo_root="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null \
               || echo /mnt/ncnn-compiler/compiler)"
  MODELS_ROOT="$(dirname "${repo_root}")"
fi

mkdir -p "${OUT}"
LOG_DIR="${OUT}.logs"
mkdir -p "${LOG_DIR}"

# 字段: name|param|bin|input-shape|extra-args
read -r -d '' MODELS_SPEC <<'EOF' || true
squeezenet_v1_1|compiler/test/third_party/ncnn/examples/squeezenet_v1.1.param|compiler/test/third_party/ncnn/examples/squeezenet_v1.1.bin||
resnet18|yolov5_v7.0_models/resnet18.ncnn.param|yolov5_v7.0_models/resnet18.ncnn.bin|3x224x224|
EOF

# 归一化 ncnn-compile 的随机 scratch 目录名，否则 diff 永远有差异。
normalize() {
  sed -E 's#/tmp/ncnn-compile-[0-9a-zA-Z]+#/tmp/ncnn-compile-SCRATCH#g'
}

overall=0
while IFS='|' read -r name param bin shape extra; do
  [[ -z "${name}" ]] && continue
  if [[ -n "${MODELS:-}" && " ${MODELS} " != *" ${name} "* ]]; then
    continue
  fi
  param_path="${MODELS_ROOT}/${param}"
  bin_path="${MODELS_ROOT}/${bin}"
  if [[ ! -f "${param_path}" ]]; then
    echo "skip ${name}: missing ${param_path}" >&2
    continue
  fi

  args=("${param_path}" "--bin=${bin_path}" "--model-name=${name}"
        "--output-dir=%OUT%" "--emit=all" "--profile"
        "--emit-manifest" "--threads=2")
  [[ -n "${shape}" ]] && args+=("--input-shape=${shape}")
  [[ -n "${extra}" ]] && args+=(${extra})

  for side in ref new; do
    build="${REF}"
    [[ "${side}" == "new" ]] && build="${NEW}"
    out="${OUT}/${side}/${name}"
    rm -rf "${out}"
    mkdir -p "${out}"
    cmd=("${build}/tools/ncnn-compile")
    for a in "${args[@]}"; do
      cmd+=("${a//%OUT%/${out}}")
    done
    if ! "${cmd[@]}" > "${LOG_DIR}/${name}.${side}.log" 2>&1; then
      echo "FAIL compile ${name} (${side}); see ${LOG_DIR}/${name}.${side}.log" >&2
      overall=1
    fi
  done

  # 归一化后逐字节 diff
  ref_tree="${OUT}/norm/ref/${name}"
  new_tree="${OUT}/norm/new/${name}"
  mkdir -p "${ref_tree}" "${new_tree}"
  ( cd "${OUT}/ref/${name}" && find . -type f | sort ) > "${OUT}/${name}.files.ref"
  ( cd "${OUT}/new/${name}" && find . -type f | sort ) > "${OUT}/${name}.files.new"
  if ! diff -u "${OUT}/${name}.files.ref" "${OUT}/${name}.files.new" \
       > "${OUT}/${name}.filelist.diff"; then
    echo "FAIL ${name}: 产物文件清单不一致" >&2
    overall=1
  fi
  # 只比「编译器产物」：插桩 IR / 头 / manifest / plan / 汇编。
  # .so / .o 含 profile_runtime 的机器码，那是 T-J1 的对照面（另证），
  # 且两侧 profile_runtime.c 源不同，比了没有意义。
  while IFS= read -r rel; do
    [[ -z "${rel}" ]] && continue
    case "${rel}" in
      *.so|*.o) continue ;;
    esac
    normalize < "${OUT}/ref/${name}/${rel}" > "${ref_tree}/${rel}" 2>/dev/null || true
    normalize < "${OUT}/new/${name}/${rel}" > "${new_tree}/${rel}" 2>/dev/null || true
  done < "${OUT}/${name}.files.ref"
  if diff -ru "${ref_tree}" "${new_tree}" > "${OUT}/${name}.diff"; then
    echo "PASS ${name}: 全部产物归一化后逐字节一致"
  else
    echo "FAIL ${name}: 产物有差异，见 ${OUT}/${name}.diff" >&2
    head -40 "${OUT}/${name}.diff" >&2
    overall=1
  fi
done <<< "${MODELS_SPEC}"

exit "${overall}"
