"""P16 operator-level native INT8 oracles through the production MLIR lowerings.

No NumPy or C signed-overflow oracle: every expected result is a Python integer
reduced modulo 2**32. This deliberately starts at Linalg, not the ncnn importer,
so arbitrary i32 init, -128, and non-contiguous memrefs are representable.
Codegen always targets x86-64 Linux; execution requires a native Linux/x86-64
host and Clang's -march=native feature macros. Unsupported execution is reported
as SKIP, never confused with successful numerical validation.
"""

import argparse
import ctypes
import dataclasses
import math
import pathlib
import platform
import random
import re
import subprocess
import tempfile


INTRINSIC = "llvm.x86.avx512.vpdpbusd.256"
EDGES = (-128, -127, 0, 127)
MASK32 = (1 << 32) - 1
TARGET = ["--target=x86_64-unknown-linux-gnu", "-march=x86-64", "-mavx2",
          "-mno-avx512f", "-mno-avx512vl", "-mno-avx512bw"]


def run(command, **kwargs):
    result = subprocess.run([str(x) for x in command], capture_output=True,
                            text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): "
                           f"{' '.join(map(str, command))}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def i32(value):
    value &= MASK32
    return value if value < (1 << 31) else value - (1 << 32)


def values(rng, count, trial):
    if trial < 4:
        return [EDGES[(i + trial) % 4] for i in range(count)]
    return [rng.choice(EDGES) if i % 3 else rng.randint(-128, 127)
            for i in range(count)]


def initial(rng, count, trial):
    edges = (0, 1, -1, (1 << 31) - 1, -(1 << 31),
             (1 << 31) - 17, -(1 << 31) + 17)
    return [edges[(i + trial) % len(edges)] if i % 2 == 0
            else i32(rng.getrandbits(32)) for i in range(count)]


class Buffer:
    """Ranked memref descriptor with nonzero offset and sentinel-protected gaps."""

    def __init__(self, element, shape, data, strides=None):
        self.shape = shape
        dynamic_offset = strides is not None
        if strides is None:
            strides = tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))
        rank = len(shape)

        class Descriptor(ctypes.Structure):
            _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                        ("offset", ctypes.c_int64), ("sizes", ctypes.c_int64 * rank),
                        ("strides", ctypes.c_int64 * rank)]

        offsets = [3 + sum(index * stride for index, stride in zip(indices, strides))
                   for indices in self.indices(shape)]
        sentinel = 91 if element == ctypes.c_int8 else 0x13579BDF
        self.storage = (element * (max(offsets) + 4))()
        for i in range(len(self.storage)):
            self.storage[i] = sentinel
        self.offsets = offsets
        self.dynamic = dynamic_offset
        require(len(data) == len(offsets), "bad fixture data length")
        for offset, value in zip(offsets, data):
            self.storage[offset] = value
        self.before = list(self.storage)
        address = ctypes.addressof(self.storage)
        # Data always starts at storage[3]. Identity layouts must advertise
        # offset=0, so the +3 lives in the aligned pointer; a declared dynamic
        # offset keeps aligned at the raw allocation and carries it in the
        # offset field instead. Never both.
        aligned = address if dynamic_offset else address + 3 * ctypes.sizeof(element)
        self.descriptor = Descriptor(address, aligned,
                                     3 if dynamic_offset else 0,
                                     (ctypes.c_int64 * rank)(*shape),
                                     (ctypes.c_int64 * rank)(*strides))

    @staticmethod
    def indices(shape):
        import itertools
        return itertools.product(*(range(dim) for dim in shape))

    def read(self):
        return [self.storage[i] for i in self.offsets]

    def check(self, writable=False):
        written = set(self.offsets) if writable else set()
        require(all(value == self.before[i] for i, value in enumerate(self.storage)
                    if i not in written), "input, padding, or output guard modified")


def invoke(library, name, buffers):
    function = getattr(library, "_mlir_ciface_" + name)
    function.restype = None
    function.argtypes = [ctypes.c_void_p] * len(buffers)
    function(*(ctypes.byref(buffer.descriptor) for buffer in buffers))


@dataclasses.dataclass(frozen=True)
class Matmul:
    m: int
    n: int
    k: int
    layout: str = "contiguous"
    unsigned: bool = False

    @property
    def name(self):
        suffix = "_unsigned" if self.unsigned else ""
        return f"mm_{self.m}_{self.n}_{self.k}_{self.layout}{suffix}"

    @property
    def step(self):
        return 2 if self.layout != "contiguous" else 1

    def types(self):
        def operand(rows):
            layout = ""
            if self.layout == "strided":
                layout = f", strided<[{self.k * 2 + 3}, 2], offset: ?>"
            elif self.layout == "unknown":
                layout = ", strided<[?, ?], offset: ?>"
            return f"memref<{rows}x{self.k}xi8{layout}>"
        return operand(self.m), operand(self.n), f"memref<{self.m}x{self.n}xi32>"

    def source(self):
        a, b, c = self.types()
        full_a = a.replace(f"<{self.m}x", f"<{self.m * 2}x", 1)
        full_c = c.replace(f"<{self.m}x", f"<{self.m * 2}x", 1)
        row_stride = self.k if self.layout == "contiguous" else self.k * 2 + 3
        a_view = (f"memref<{self.m}x{self.k}xi8, strided<[?, ?], offset: ?>>"
                  if self.layout == "unknown" else
                  f"memref<{self.m}x{self.k}xi8, strided<[{row_stride}, {self.step}], offset: ?>>")
        c_view = f"memref<{self.m}x{self.n}xi32, strided<[{self.n}, 1], offset: ?>>"
        return f"""
  func.func @{self.name}(%a: {full_a}, %b: {b}, %c: {full_c}) attributes {{llvm.emit_c_interface}} {{
    %rows = arith.constant {self.m} : index
    scf.forall (%tile) in (2) {{
      %row = arith.muli %tile, %rows : index
      %av = memref.subview %a[%row, 0][{self.m}, {self.k}][1, 1] : {full_a} to {a_view}
      %cv = memref.subview %c[%row, 0][{self.m}, {self.n}][1, 1] : {full_c} to {c_view}
      linalg.matmul_transpose_b {'{cast = #linalg.type_fn<cast_unsigned>}' if self.unsigned else ''} ins(%av, %b : {a_view}, {b}) outs(%cv : {c_view})
    }}
    return
  }}
"""

    def execute(self, library):
        rng = random.Random(16000 + self.m * 100 + self.n * 10 + self.k)
        rows = self.m * 2  # Two disjoint tiles prevent canonical forall folding.
        for trial in range(8):
            a = values(rng, rows * self.k, trial)
            b = values(rng, self.n * self.k, (trial + 1) % 8)
            # Positive/negative extremes force overflow on both sides and make
            # saturating dpbusds or unsigned/sign-correction mistakes observable.
            if trial in (2, 3):
                a = [-128] * len(a)
                b = [(-128 if trial == 2 else 127)] * len(b)
            c = initial(rng, rows * self.n, trial)
            strides = ((self.k * 2 + 3, self.step)
                       if self.layout != "contiguous" else None)
            buffers = [Buffer(ctypes.c_int8, (rows, self.k), a, strides),
                       Buffer(ctypes.c_int8, (self.n, self.k), b, strides),
                       Buffer(ctypes.c_int32, (rows, self.n), c)]
            def widen(value):
                return value & 255 if self.unsigned else value

            sums = [sum(widen(a[m * self.k + k]) * widen(b[n * self.k + k])
                        for k in range(self.k))
                    for m in range(rows) for n in range(self.n)]
            for repeat in range(3):
                c = [i32(init + dot) for init, dot in zip(c, sums)]
                invoke(library, self.name, buffers)
                require(buffers[2].read() == c,
                        f"{self.name}: trial={trial}, repeat={repeat} != integer oracle")
                for index, buffer in enumerate(buffers):
                    buffer.check(writable=index == 2)


@dataclasses.dataclass(frozen=True)
class PackedMatmul:
    m: int
    n: int
    k: int

    @property
    def name(self):
        return f"packed_mm_{self.m}_{self.n}_{self.k}"

    @property
    def weight_name(self):
        return f"weights_{self.n}_{self.k}"

    @property
    def weights(self):
        return [EDGES[(n * 3 + k * 2) % len(EDGES)]
                for n in range(self.n) for k in range(self.k)]

    def source(self):
        rows = [self.weights[n * self.k:(n + 1) * self.k]
                for n in range(self.n)]
        initializer = "[" + ", ".join(
            "[" + ", ".join(map(str, row)) + "]" for row in rows) + "]"
        row_count = self.m * 2
        return f"""
  memref.global constant @{self.weight_name} : memref<{self.n}x{self.k}xi8> = dense<{initializer}>
  func.func @{self.name}(%a: memref<{row_count}x{self.k}xi8>,
                         %c: memref<{row_count}x{self.n}xi32>) attributes {{llvm.emit_c_interface}} {{
    %b = memref.get_global @{self.weight_name} : memref<{self.n}x{self.k}xi8>
    %tile_rows = arith.constant {self.m} : index
    scf.forall (%tile) in (2) {{
      %row = arith.muli %tile, %tile_rows : index
      %av = memref.subview %a[%row, 0][{self.m}, {self.k}][1, 1]
        : memref<{row_count}x{self.k}xi8> to memref<{self.m}x{self.k}xi8, strided<[{self.k}, 1], offset: ?>>
      %cv = memref.subview %c[%row, 0][{self.m}, {self.n}][1, 1]
        : memref<{row_count}x{self.n}xi32> to memref<{self.m}x{self.n}xi32, strided<[{self.n}, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%av, %b : memref<{self.m}x{self.k}xi8, strided<[{self.k}, 1], offset: ?>>,
                                                   memref<{self.n}x{self.k}xi8>)
                                outs(%cv : memref<{self.m}x{self.n}xi32, strided<[{self.n}, 1], offset: ?>>)
    }}
    return
  }}
"""

    def execute(self, library):
        row_count = self.m * 2
        weights = self.weights
        rng = random.Random(17000 + self.m * 100 + self.n * 10 + self.k)
        for trial in range(8):
            a = values(rng, row_count * self.k, trial)
            c = initial(rng, row_count * self.n, trial)
            if trial in (2, 3):
                a = [-128] * len(a)
            sums = [sum(a[row * self.k + k] *
                        weights[column * self.k + k]
                        for k in range(self.k))
                    for row in range(row_count) for column in range(self.n)]
            expected = list(c)
            buffers = [Buffer(ctypes.c_int8, (row_count, self.k), a),
                       Buffer(ctypes.c_int32, (row_count, self.n), c)]
            for repeat in range(3):
                expected = [i32(value + total)
                            for value, total in zip(expected, sums)]
                invoke(library, self.name, buffers)
                require(buffers[1].read() == expected,
                        f"{self.name}: trial={trial}, repeat={repeat} != integer oracle")
                for index, buffer in enumerate(buffers):
                    buffer.check(writable=index == 1)


@dataclasses.dataclass(frozen=True)
class Depthwise:
    c: int
    sh: int = 1
    sw: int = 1
    dh: int = 1
    dw: int = 1
    batch: int = 1

    @property
    def name(self):
        return f"dw_{self.c}_{self.sh}_{self.sw}_{self.dh}_{self.dw}_{self.batch}"

    @property
    def shapes(self):
        # KH=3, KW=2, OH=2, OW=3; last window exactly touches the input edge.
        h, w = self.sh + 2 * self.dh + 1, 2 * self.sw + self.dw + 1
        return ((self.batch, h, w, self.c), (3, 2, self.c, 1),
                (self.batch, 2, 3, self.c, 1), (self.batch, 2, 3, self.c))

    def source(self):
        a, b, c, out = ["tensor<" + "x".join(map(str, shape)) + "x" + element + ">"
                        for shape, element in zip(self.shapes, ("i8", "i8", "i32", "i32"))]
        return f"""
  func.func @{self.name}(%a: {a}, %b: {b}, %c: {c}) -> {out} attributes {{llvm.emit_c_interface}} {{
    %r = linalg.depthwise_conv_2d_nhwc_hwcm {{strides = dense<[{self.sh}, {self.sw}]> : tensor<2xi64>, dilations = dense<[{self.dh}, {self.dw}]> : tensor<2xi64>}} ins(%a, %b : {a}, {b}) outs(%c : {c}) -> {c}
    %out = tensor.collapse_shape %r [[0], [1], [2], [3, 4]] : {c} into {out}
    return %out : {out}
  }}
"""

    def execute(self, library):
        rng = random.Random(16100 + self.c)
        a_shape, b_shape, c_shape, out_shape = self.shapes
        _, h, w, channels = a_shape
        for trial in range(8):
            a = values(rng, math.prod(a_shape), trial)
            b = values(rng, math.prod(b_shape), (trial + 1) % 8)
            c = initial(rng, math.prod(c_shape), trial)
            if trial in (2, 3):
                a = [-128] * len(a)
                b = [(-128 if trial == 2 else 127)] * len(b)
            expected = []
            for batch, oh, ow, channel in Buffer.indices(out_shape):
                index = ((batch * 2 + oh) * 3 + ow) * channels + channel
                acc = c[index]
                for kh in range(3):
                    for kw in range(2):
                        ih, iw = oh * self.sh + kh * self.dh, ow * self.sw + kw * self.dw
                        acc += a[((batch * h + ih) * w + iw) * channels + channel] * \
                            b[(kh * 2 + kw) * channels + channel]
                expected.append(i32(acc))
            buffers = [Buffer(ctypes.c_int8, a_shape, a),
                       Buffer(ctypes.c_int8, b_shape, b),
                       Buffer(ctypes.c_int32, c_shape, c),
                       Buffer(ctypes.c_int32, out_shape, [17] * len(c))]
            for repeat in range(3):
                # Bufferization may consume the init tensor in-place. Restore
                # it for identical repeated calls; only A/B are read-only ABI
                # inputs. Check padding on init and result independently.
                for offset, value in zip(buffers[2].offsets, c):
                    buffers[2].storage[offset] = value
                invoke(library, self.name, buffers)
                require(buffers[3].read() == expected,
                        f"{self.name}: trial={trial}, repeat={repeat} != channel oracle")
                for index, buffer in enumerate(buffers):
                    buffer.check(writable=index >= 2)


def function_ir(text, name):
    marker = f"func.func @{name}("
    require(marker in text, f"missing function {name}")
    return text.split(marker, 1)[1].split("\n  func.func @", 1)[0]


def check_kernel_ir(text, cases, mode):
    for case in cases:
        body = function_ir(text, case.name)
        if isinstance(case, Matmul):
            selected = (mode == "vnni" and case.k >= 32
                        and case.layout == "contiguous" and not case.unsigned)
            require((f"llvm.call @{INTRINSIC}" in body) == selected,
                    f"{case.name}: incorrect VNNI selection for {mode}")
            if case.unsigned:
                # Unsigned casts keep the original named op: the generic signed
                # MAC emitter must never rewrite them.
                require("linalg.matmul_transpose_b" in body,
                        f"{case.name}: unsigned named op was rewritten")
                continue
            require("linalg.matmul_transpose_b" not in body,
                    f"{case.name}: row-dot kernel not selected")
            if selected:
                require("arith.xori" in body and "arith.subi" in body,
                        f"{case.name}: missing signed correction")
                require("llvm.call_intrinsic" not in body, "must exercise actual LLVM CallOp")
                if case.k % 32:
                    require("arith.extsi" in body, f"{case.name}: missing scalar K tail")
        elif isinstance(case, PackedMatmul):
            selected = mode == "vnni"
            schema = "p23-int8-panel-row-kpad64-v1"
            require((f'ncnn.pack_schema = "{schema}"' in body) == selected,
                    f"{case.name}: incorrect physical RHS packing for {mode}")
            if selected:
                require('ncnn.packing = "prepacked_B"' in body,
                        f"{case.name}: missing prepacked RHS contract")
                require("strided<[64, 1]" in body,
                        f"{case.name}: missing K-padded contiguous row view")
                require(f"llvm.call @{INTRINSIC}" in body,
                        f"{case.name}: missing VNNI consumer")
                require("ncnn.int8_k_alignment = 64 : i64" in body,
                        f"{case.name}: missing packed K alignment")
            else:
                require(INTRINSIC not in body,
                        f"{case.name}: portable packed RHS unexpectedly selected")
        else:
            selected = mode == "depthwise" and case.c >= 4
            require(('ncnn.implementation = "depthwise_simd"' in body) == selected,
                    f"{case.name}: incorrect depthwise SIMD selection")
            if selected:
                require("vector<" in body and "arith.extsi" in body,
                        f"{case.name}: no signed per-channel vector arithmetic")
                require("vector.reduction" not in body and INTRINSIC not in body,
                        f"{case.name}: channels must not be horizontally reduced")


def build(args, root, cases, mode):
    directory = root / mode
    directory.mkdir()
    source = directory / "input.mlir"
    memref = directory / "memref.mlir"
    llvm_mlir = directory / "llvm.mlir"
    llvm_ir = directory / "model.ll"
    obj = directory / "model.o"
    source.write_text("module {\n" + "".join(case.source() for case in cases) + "}\n")
    options = "vector-tail=true"
    if args.kind == "matmul":
        options += f" int8-kernel={mode} int8-target={'avx-vnni' if mode == 'vnni' else 'portable'}"
        if mode == "vnni":
            options += " tuning-profile=native-int8"
    else:
        options += f" vector-lanes=4 int8-depthwise={'true' if mode == 'depthwise' else 'false'}"
    run([args.opt, source, "--ncnn-linalg-to-memref-pipeline=" + options, "-o", memref])
    check_kernel_ir(memref.read_text(), cases, mode)
    run([args.opt, memref, "--ncnn-memref-to-llvm-pipeline=threads=1 vector-lowering=true",
         "-o", llvm_mlir])
    run([args.translate, "--mlir-to-llvmir", llvm_mlir, "-o", llvm_ir])
    text = llvm_ir.read_text()
    if mode == "vnni":
        require(re.search(r"call <8 x i32> @llvm\.x86\.avx512\.vpdpbusd\.256", text),
                "actual LLVM intrinsic call lost during lowering")
    else:
        require(INTRINSIC not in text, "portable/depthwise LLVM unexpectedly contains dot")
    flags = [*TARGET, "-mavxvnni" if mode == "vnni" else "-mno-avxvnni",
             "-O2", "-fPIC"]
    run([args.clang, *flags, "-c", llvm_ir, "-o", obj])
    disassembly = run([args.objdump, "-d", obj])
    (directory / "model.objdump").write_text(disassembly)
    require((bool(re.search(r"\bvpdpbusd\b", disassembly))) == (mode == "vnni"),
            f"{mode}: incorrect object VNNI instructions")
    require(not re.search(r"\bvpdpbusds\b|\bzmm\d+\b|\bk[0-7]\b", disassembly),
            f"{mode}: saturation or AVX512 register/mask found")
    # EVEX can use xmm/ymm without zmm or mask operands. Inspect real bytes,
    # not just mnemonic/register spelling: an EVEX instruction starts with 62.
    require(not re.search(r"^\s*[0-9a-f]+:\s+62\s", disassembly, re.MULTILINE),
            f"{mode}: unexpected EVEX (AVX512) encoding")
    for line in disassembly.splitlines():
        if re.search(r"\bvpdpbusd\b", line):
            require(re.search(r":\s+c4\s", line), "AVX-VNNI must use VEX encoding")
    return obj, flags, directory


def native_features(args):
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        return set(), "non-native x86-64 Linux target"
    macros = run([args.clang, "-march=native", "-dM", "-E", "-x", "c", "-"], input="")
    return set(re.findall(r"^#define (\w+) ", macros, re.MULTILINE)), ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("matmul", "depthwise"), required=True)
    for name in ("opt", "translate", "clang", "objdump", "work-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--codegen-only", action="store_true",
                        help="explicitly skip execution while retaining all lowering/codegen checks")
    args = parser.parse_args()
    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix=args.kind + "-", dir=work))
    print(f"Artifacts: {root}", flush=True)
    if args.kind == "matmul":
        cases = [Matmul(m, n, k) for k in (1, 6, 31, 32, 33, 72)
                 for m, n in ((1, 1), (3, 5), (5, 7))]
        cases += [Matmul(3, 5, 33, "strided"), Matmul(5, 7, 72, "unknown")]
        cases += [Matmul(3, 5, 33, unsigned=True)]
        cases += [PackedMatmul(3, 16, 33), PackedMatmul(5, 22, 64)]
        modes = ("portable", "vnni")
    else:
        cases = [Depthwise(3), Depthwise(4), Depthwise(5, 2, 1),
                 Depthwise(10, 1, 2, 2, 1), Depthwise(16, 2, 2, 1, 2),
                 Depthwise(17, 2, 1, 2, 2, 2)]
        modes = ("portable", "depthwise")
    features, reason = native_features(args)
    for mode in modes:
        obj, flags, directory = build(args, root, cases, mode)
        print(f"PASS {mode}: {len(cases)} actual lowering/object checks; no EVEX/AVX512", flush=True)
        required = {"__AVX2__"} | ({"__AVXVNNI__"} if mode == "vnni" else set())
        missing = required - features
        if args.codegen_only or reason or missing:
            why = "requested --codegen-only" if args.codegen_only else reason or \
                "native CPU macros missing " + ", ".join(sorted(missing))
            print(f"SKIP {mode} execution: {why}", flush=True)
            continue
        shared = directory / "model.so"
        run([args.clang, *flags, "-shared", obj, "-Wl,-z,defs", "-o", shared])
        library = ctypes.CDLL(str(shared))
        for case in cases:
            case.execute(library)
        print(f"PASS {mode}: {len(cases)} shapes x 8 trials x 3 calls; exact modulo32 oracle", flush=True)


if __name__ == "__main__":
    main()
