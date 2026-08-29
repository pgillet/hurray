"""
Converting an externally quantized tensor into a Hurray descriptor.

An external toolchain hands you quantized weights, per-block scales, and a
zero-point plane. The scales transfer as-is. The zero-point plane does not:
Hurray dequantizes as ``scale * (q - zero_point)``, subtracting the stored value
with no implicit bias, while toolchains in the GPTQ family have conventionally
stored ``zero_point - 1`` and removed the bias in the loader.

The plane is rebuilt during conversion regardless — the foreign side packs two
4-bit values per byte, Hurray's per-block affine scheme takes one int32 per
block — so the normalization is one line inside a transform you are already
writing, not a step to remember afterwards.

Run with:

    python hurray-python/examples/convert_external_quantized.py
"""

import struct

import hurray

ROWS, COLS, BLOCK = 2, 16, 8

# Blocks along the quantized axis per row, and across the whole tensor.
BLOCKS_PER_ROW = COLS // BLOCK
NUM_BLOCKS = BLOCKS_PER_ROW * ROWS

# Quantized weights, row-major, [ROWS, COLS].
WEIGHTS = [
    9, 8, 7, 10, 8, 6, 12, 8, 3, 9, 8, 11, 7, 8, 5, 10,
    8, 13, 8, 4, 9, 8, 7, 8, 10, 8, 6, 8, 9, 2, 8, 14,
]  # fmt: skip

# Per-block scales. These transfer unchanged — only the zero points carry a
# convention.
SCALES = [0.02, 0.015, 0.025, 0.01]

# The foreign zero-point plane: 4-bit values packed two per byte, low nibble
# first, each holding zero_point - 1. Decodes to 7, 6, 8, 7.
FOREIGN_ZERO_POINTS_PACKED = bytes([0x67, 0x78])


def normalize_zero_points(packed, count):
    """Rebuild the foreign 4-bit plane as Hurray's int32 plane.

    Two things happen here, and only one of them is visible in the output: the
    unpacking, which fails loudly if you get it wrong, and the ``+ 1``, which
    does not fail at all.
    """
    out = []
    for i in range(count):
        byte = packed[i // 2]
        nibble = byte & 0x0F if i % 2 == 0 else byte >> 4
        # The convention normalization. Hurray subtracts the stored zero point
        # as-is, so the toolchain's -1 bias is removed here or never.
        out.append(nibble + 1)
    return out


def block_of(row, col):
    """Block index for [row, col]: outer * blocks_per_row + col // BLOCK."""
    return row * BLOCKS_PER_ROW + col // BLOCK


def dequantize(weights, scales, zero_points):
    """Apply the spec's formula: x = scale * (q - zero_point).

    Hurray describes tensors, it does not compute on them — there is no
    dequantize in the bindings and there should not be. This is example code
    reading the formula out of the spec, not a library function.
    """
    return [
        scales[block_of(row, col)]
        * (weights[row * COLS + col] - zero_points[block_of(row, col)])
        for row in range(ROWS)
        for col in range(COLS)
    ]


# ── 1. Rebuild the zero-point plane ───────────────────────────────────────────

zero_points = normalize_zero_points(FOREIGN_ZERO_POINTS_PACKED, NUM_BLOCKS)
assert zero_points == [8, 7, 9, 8]

print("Converting an externally quantized tensor\n")
print(f"  foreign plane : {FOREIGN_ZERO_POINTS_PACKED.hex(' ').upper()} (4-bit, biased)")
print(f"  hurray plane  : {zero_points} (int32, as subtracted)\n")

# ── 2. Build the tensor over the rebuilt buffers ──────────────────────────────

# Buffer 0: the weights. Buffer 1: float32 scales. Buffer 2: int32 zero points.
weight_bytes = struct.pack(f"{ROWS * COLS}b", *WEIGHTS)
scale_bytes = struct.pack(f"{NUM_BLOCKS}f", *SCALES)
zero_point_bytes = struct.pack(f"{NUM_BLOCKS}i", *zero_points)

quant = hurray.PerBlockAffine.asymmetric(
    1,  # axis: blocks run along the columns
    BLOCK,  # block_size
    1,  # scale_buffer_index
    2,  # zero_point_buffer_index
    hurray.float32,  # scale_type
)

weights = hurray.Tensor(
    weight_bytes,
    hurray.int8,
    [ROWS, COLS],
    aux_buffers=[scale_bytes, zero_point_bytes],
    quantization=quant,
)

q = weights.quantization
print(f"  tensor        : {weights.shape} {weights.dtype}, {weights.buffer_count} buffers")
print(f"  quantization  : per-block affine, axis {q.axis}, block_size {q.block_size},")
print(f"                  scales in buffer {q.scale_buffer_index}, "
      f"zero points in buffer {q.zero_point_buffer_index}\n")

# ── 3. What skipping the normalization costs ──────────────────────────────────

correct = dequantize(WEIGHTS, SCALES, zero_points)

# The same conversion with the foreign values copied straight through.
unnormalized = [z - 1 for z in zero_points]
wrong = dequantize(WEIGHTS, SCALES, unnormalized)

# Every element is off by exactly one scale step of its own block — a valid
# descriptor, a clean decode, and the wrong numbers.
for row in range(ROWS):
    for col in range(COLS):
        i = row * COLS + col
        drift = wrong[i] - correct[i]
        step = SCALES[block_of(row, col)]
        assert abs(drift - step) < 1e-6, f"[{row}, {col}]: {drift} is not one step {step}"

print(f"  element [0, 0]: correct {correct[0]:.4f}, un-normalized {wrong[0]:.4f}")
print("  every element off by exactly one scale step of its block\n")
print("Nothing rejects the second descriptor. Only the arithmetic tells you.")
