import os

import lit.formats

config.name = "ncnn-mlir"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir"]
config.excludes = ["Runtime"]

# 路径变量由 lit.site.cfg.py（CMake 生成）注入。
config.substitutions.append(("%FileCheck", config.filecheck))
config.substitutions.append(("FileCheck", config.filecheck))
config.substitutions.append(
    ("ncnn-mlir-opt", os.path.join(config.ncnn_mlir_tools_dir, "ncnn-mlir-opt"))
)
config.substitutions.append(("%ncnn-mlir-driver", config.ncnn_mlir_driver))
config.substitutions.append(("%mlir-opt", config.mlir_opt))
config.substitutions.append(("%squeezenet-param", config.squeezenet_param))
config.substitutions.append(("%squeezenet-bin", config.squeezenet_bin))
