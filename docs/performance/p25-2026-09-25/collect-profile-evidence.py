#!/usr/bin/env python3
"""Build and profile current excess-leader models for P25 qualification."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


MODELS = {
    "yolov5x_seg": ("yolo", "yolov5x_seg", "3x640x640"),
    "yolov5x": ("yolo", "yolov5x", "3x640x640"),
    "yolov5l_seg": ("yolo", "yolov5l_seg", "3x640x640"),
    "yolov5l": ("yolo", "yolov5l", "3x640x640"),
    "yolov5m_seg": ("yolo", "yolov5m_seg", "3x640x640"),
    "pp_ocrv6_medium_det": ("ocr", "PP-OCRv6_medium_det", "3x640x640"),
    "efficientnet_b3": ("yolo", "efficientnet_b3", "3x300x300"),
}


def run(command: list[str], log: Path) -> None:
  with log.open("w", encoding="utf-8") as stream:
    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", type=Path, required=True)
  parser.add_argument("--compiler-root", type=Path, required=True)
  parser.add_argument("--yolo-root", type=Path, required=True)
  parser.add_argument("--ocr-root", type=Path, required=True)
  parser.add_argument("--evidence", type=Path, required=True)
  args = parser.parse_args()
  profile_root = args.evidence / "profile-on"
  commands_path = args.evidence / "profile-commands.txt"
  log_root = args.evidence / "profile-logs"
  if profile_root.exists() or commands_path.exists() or log_root.exists():
    parser.error("refusing to overwrite existing P25 profile evidence")
  profile_root.mkdir(parents=True)
  log_root.mkdir()

  compiler = args.stage / "tools/ncnn-compile"
  driver = args.stage / "tools/ncnn-mlir-driver"
  optimizer = args.stage / "bin/ncnn-mlir-opt"
  profiler = args.compiler_root / "tools/perf_attribution_report.py"
  runner = args.evidence / "run-profiled-model.py"
  with commands_path.open("w", encoding="utf-8") as commands_file:
    for model, (asset_group, asset_stem, input_shape) in MODELS.items():
      asset_root = args.ocr_root if asset_group == "ocr" else args.yolo_root
      if asset_group == "yolo":
        parameter = asset_root / f"{asset_stem}.ncnn.param"
        weights = asset_root / f"{asset_stem}.ncnn.bin"
      else:
        parameter = asset_root / f"{asset_stem}.param"
        weights = asset_root / f"{asset_stem}.bin"
      output = profile_root / model
      output.mkdir()
      command = [
          str(compiler),
          f"--driver={driver}",
          f"--opt={optimizer}",
          "--translate=/usr/bin/mlir-translate-21",
          "--clang=/usr/bin/clang-21",
          "--nm=/usr/bin/llvm-nm-21",
          "--readelf=/usr/bin/llvm-readelf-21",
          "--llvm-as=/usr/bin/llvm-as-21",
          f"--param={parameter}",
          f"--bin={weights}",
          f"--model-name={model}",
          f"--input-shape={input_shape}",
          "--matmul-packing=auto",
          "--march=x86-64-v3",
          "--target-feature=+avx2",
          "--target-feature=+fma",
          "--vector-mode=fixed-width",
          "--threads=6",
          "--profile",
          "--emit-manifest",
          "--emit-execution-plan",
          f"--output-dir={output}",
      ]
      commands_file.write(shlex.join(command) + "\n")
      commands_file.flush()
      run(command, log_root / f"{model}-compile.log")
      run([
          "python3", str(runner),
          f"--model-dir={output}",
          f"--profile={output / 'profile.ndjson'}",
          "--warmups=2",
          "--iterations=5",
      ], output / "profile-run.log")
      run([
          "python3", str(profiler),
          f"--plan={output / f'{model}.plan.json'}",
          f"--profile={output / 'profile.ndjson'}",
          f"--perf={output / 'diagnostic-perf.ndjson'}",
          "--mode=end_to_end",
          f"--output={output / 'attribution-report.json'}",
      ], output / "attribution.log")
      print(f"profiled {model}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
