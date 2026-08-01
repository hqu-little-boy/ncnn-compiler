import pathlib
import struct
import sys


output = pathlib.Path(sys.argv[1])
case = sys.argv[2] if len(sys.argv) > 2 else "convolution"
sizes = {
    "convolution": (18, 2),
    "convolution_no_bias": (18, 0),
    "convolution_dilated": (8, 2),
    "convolution_asymmetric": (12, 0),
    "convolution_same_upper": (18, 0),
    "convolution_same_lower": (18, 0),
}
kernel_count, bias_count = sizes[case]
weights = [
    0.25, -0.5, 0.75, -1.0, 0.5, 0.125, 0.375, -0.25, 1.0,
    -0.125, 0.625, -0.75, 0.5, 0.25, -0.375, 0.875, -0.625, 0.125,
][:kernel_count]
bias = [0.125, -0.25][:bias_count]
payload = weights + bias
output.write_bytes(struct.pack("<I", 0) + struct.pack(f"<{len(payload)}f", *payload))
