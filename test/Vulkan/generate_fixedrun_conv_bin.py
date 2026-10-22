#!/usr/bin/env python3
"""Write deterministic FP32 weights for the tiny VA4 Conv→ReLU graph."""

import pathlib
import struct
import sys


if len(sys.argv) != 2:
    raise SystemExit("usage: generate_fixedrun_conv_bin.py OUTPUT")

# OIHW: two input channels, two output channels, one-by-one kernel.
weights = [0.5, -0.25, -0.75, 1.25]
biases = [0.125, -0.5]
pathlib.Path(sys.argv[1]).write_bytes(struct.pack("<I6f", 0, *(weights + biases)))
