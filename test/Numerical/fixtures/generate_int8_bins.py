import pathlib
import struct
import sys


INT8_TAG = 0x000D4B38


def int8_tensor(values):
    payload = struct.pack(f"<{len(values)}b", *values)
    return struct.pack("<I", INT8_TAG) + payload + bytes((-len(payload)) % 4)


def floats(values):
    return struct.pack(f"<{len(values)}f", *values)


def convolution(term, inputs, outputs, bias=True):
    weights = [((index * 7 + 3) % 15) - 7 for index in range(inputs * outputs)]
    data = int8_tensor(weights)
    if bias:
        data += floats([0.125 * (index - 1) for index in range(outputs)])
    data += floats([5.0 + index * 1.5 for index in range(outputs)])
    data += floats([8.0])
    if term > 100:
        data += floats([6.0])
    return data


def depthwise(term, channels, bias=True):
    weights = [((index * 5 + 1) % 11) - 5 for index in range(channels)]
    data = int8_tensor(weights)
    if bias:
        data += floats([0.1 * (index - 1) for index in range(channels)])
    scale_count = channels if term in (1, 101) else 1
    data += floats([4.0 + index for index in range(scale_count)])
    data += floats([7.0])
    if term > 100:
        data += floats([5.0])
    return data


def inner_product(term):
    weights = [3, -2, 1, 4, -4, 2, 5, -1, 2, 3, -3, 1]
    return (
        int8_tensor(weights)
        + floats([0.25, -0.5, 0.125])
        + floats([4.0, 6.0, 8.0])
        + floats([9.0])
    )


def gemm():
    weights = [3, -2, 1, 4, -4, 2, 5, -1]
    return (
        int8_tensor(weights)
        + struct.pack("<I", 0)
        + floats([0.25, -0.5, 0.125, 0.75])
        + floats([6.0])
    )


case = sys.argv[2]
if case in ("convolution_int8_term1", "convolution_int8_term2"):
    term = 1 if case.endswith("term1") else 2
    payload = convolution(term, 2, 3)
elif case in ("convolution_int8_term101", "convolution_int8_term102"):
    term = 101 if case.endswith("term101") else 102
    payload = convolution(term, 2, 3) + convolution(1, 3, 2)
elif case in ("depthwise_int8_term1", "depthwise_int8_term2"):
    term = 1 if case.endswith("term1") else 2
    payload = depthwise(term, 3)
elif case in ("depthwise_int8_term101", "depthwise_int8_term102"):
    term = 101 if case.endswith("term101") else 102
    payload = depthwise(term, 3) + depthwise(1, 3)
elif case in ("inner_product_int8_term1", "inner_product_int8_term2"):
    payload = inner_product(1 if case.endswith("term1") else 2)
elif case == "int8_complete_chain":
    payload = convolution(101, 2, 3) + depthwise(101, 3) + convolution(1, 3, 2)
elif case == "gemm_int8_term2":
    payload = gemm()
elif case in {
    "pp_lcnet_doc_ori_int8_backbone",
    "pp_lcnet_textline_ori_int8_backbone",
}:
    source = pathlib.Path(sys.argv[3]).read_bytes()
    backbone_bytes = 466080
    if len(source) <= backbone_bytes:
        raise ValueError("PP-LCNet INT8 source bin does not contain tail weights")
    payload = source[:backbone_bytes]
else:
    raise ValueError(f"unknown INT8 fixture {case}")

pathlib.Path(sys.argv[1]).write_bytes(payload)
