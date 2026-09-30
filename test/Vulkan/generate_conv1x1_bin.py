#!/usr/bin/env python3
"""Generate a deterministic FP32 OIHW weight/bias stream for the V2 1x1 fixture."""

import pathlib
import struct
import sys


if len(sys.argv) != 2:
    raise SystemExit("usage: generate_conv1x1_bin.py OUTPUT")

output = pathlib.Path(sys.argv[1])
# OIHW, with two input channels and three output channels.
weights = [0.25, -0.5, 0.75, -1.0, 0.125, 0.375]
bias = [0.125, -0.25, 0.5]
output.write_bytes(struct.pack("<I9f", 0, *(weights + bias)))
