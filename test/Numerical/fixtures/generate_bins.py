import pathlib
import random
import struct
import sys


output = pathlib.Path(sys.argv[1])
case = sys.argv[2] if len(sys.argv) > 2 else "convolution"
sizes = {
    "convolution": (18, 2),
    "convolution_no_bias": (18, 0),
    "convolution_dilated": (8, 2),
    "convolution_asymmetric": (12, 0),
    "convolution_asymmetric_stride": (12, 0),
    "convolution_same_upper": (18, 0),
    "convolution_same_lower": (18, 0),
    "convolution_depthwise": (18, 2),
    "convolution_depthwise_asymmetric": (12, 2),
    "convolution_depthwise_same_upper": (18, 0),
    "convolution_depthwise_same_lower": (18, 0),
    "inner_product": (24, 3),
    "inner_product_fused_relu": (24, 4),
    "deconvolution": (16, 2),
}
if case == "batch_norm":
    slope = [0.0001, -0.75, 0.5]
    mean = [0.25, -0.5, 1.0]
    variance = [0.0, 0.25, 4.0]
    bias = [0.125, 0.5, -0.25]
    output.write_bytes(struct.pack("<12f", *(slope + mean + variance + bias)))
    raise SystemExit(0)
if case == "gemm":
    weight = [0.5, -0.25, 1.0, 0.75, -0.5, 0.125, 0.25, -1.0]
    bias = [0.2, -0.4, 0.6, -0.8]
    output.write_bytes(
        struct.pack("<I8f", 0, *weight) + struct.pack("<I4f", 0, *bias)
    )
    raise SystemExit(0)
if case not in sizes:
    output.write_bytes(b"")
    raise SystemExit(0)
kernel_count, bias_count = sizes[case]
if case == "deconvolution":
    # ncnn stores deconvolution weights as [output][input][kh][kw]. Distinct
    # values make an accidental input/output transpose visible in the golden.
    weights = [
        0.10, 0.20, 0.30, 0.40,
        -0.50, -0.60, -0.70, -0.80,
        -0.15, 0.25, -0.35, 0.45,
        0.55, -0.65, 0.75, -0.85,
    ]
    bias = [-0.05, 0.10]
elif case.startswith("convolution_depthwise") or case.startswith("inner_product"):
    rng = random.Random(0x4E434E4E if case.startswith("convolution_depthwise") else 0x49505052)
    weights = [rng.uniform(-0.25, 0.25) for _ in range(kernel_count)]
    bias = [rng.uniform(-0.1, 0.1) for _ in range(bias_count)]
else:
    weights = [
        0.25, -0.5, 0.75, -1.0, 0.5, 0.125, 0.375, -0.25, 1.0,
        -0.125, 0.625, -0.75, 0.5, 0.25, -0.375, 0.875, -0.625, 0.125,
    ][:kernel_count]
    bias = [0.125, -0.25][:bias_count]
payload = weights + bias
output.write_bytes(struct.pack("<I", 0) + struct.pack(f"<{len(payload)}f", *payload))
