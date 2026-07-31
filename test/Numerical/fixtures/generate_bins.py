import pathlib
import struct
import sys


output = pathlib.Path(sys.argv[1])
weights = [
    0.25, -0.5, 0.75, -1.0, 0.5, 0.125, 0.375, -0.25, 1.0,
    -0.125, 0.625, -0.75, 0.5, 0.25, -0.375, 0.875, -0.625, 0.125,
]
bias = [0.125, -0.25]
output.write_bytes(struct.pack("<I", 0) + struct.pack("<20f", *(weights + bias)))
