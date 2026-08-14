"""
Authoring quantized tensor descriptors from Python.

Hurray describes quantization; it does not compute it. These classes package
scales and zero points you have already calculated — with your own quantizer, or
taken from a model you are converting — into a descriptor that any Hurray reader
understands.

Every scheme but per-tensor affine keeps its parameters in a separate buffer and
refers to it by index, so building one means supplying both the parameter bytes
and the index that points at them.

Run with:

    python hurray-python/examples/quantized_authoring.py
"""

import struct

import hurray

# ── Per-tensor affine: parameters inline ──────────────────────────────────────
#
# One scale and zero point for the whole tensor, stored in the descriptor itself.
# No extra buffer, so the tensor stays single-buffer.

per_tensor = hurray.PerTensorAffine(0.02, 128)
print("=== Per-tensor affine ===")
print(f"  {per_tensor!r}")

weights = hurray.Tensor(bytes(8), hurray.int8, [2, 4], quantization=per_tensor)
print(f"  tensor: shape={weights.shape} dtype={weights.dtype}")
print("  buffers: 1 (scale and zero point are inline)")

# ── Per-channel affine: scales in their own buffer ────────────────────────────
#
# One scale per slice along an axis. The scales live in buffer 1, which the
# descriptor references by index — so both must be supplied together.

scales = struct.pack("2f", 0.02, 0.017)  # one per row of a [2, 4] tensor

per_channel = hurray.PerChannelAffine.symmetric(axis=0, scale_buffer_index=1)
print("\n=== Per-channel affine ===")
print(f"  {per_channel!r}")

quantized = hurray.Tensor(
    bytes(8),                 # buffer 0: the int8 weights
    hurray.int8,
    [2, 4],
    aux_buffers=[scales],     # buffer 1: the scales it points at
    quantization=per_channel,
)
print(f"  tensor: shape={quantized.shape} dtype={quantized.dtype}")
print("  buffers: 2 (data + scales)")

# ── An index that points at nothing is refused ────────────────────────────────
#
# A descriptor claiming a scale buffer that was never supplied encodes and decodes
# perfectly well — the consumer would just find a dangling index. That is caught
# here, where the mistake was made.

print("\n=== Dangling buffer index ===")
try:
    hurray.Tensor(bytes(8), hurray.int8, [2, 4], quantization=per_channel)
except hurray.InvalidDescriptorError as exc:
    print(f"  refused: {exc}")

# ── Block schemes ─────────────────────────────────────────────────────────────

print("\n=== Block schemes ===")
print(f"  {hurray.PerBlockAffine.symmetric(1, 32, 1, hurray.float32)!r}")
print(f"  {hurray.NF4(axis=1, block_size=64, scale_buffer_index=1)!r}")
print(f"  {hurray.MXFP(axis=1, block_size=32, scale_buffer_index=1)!r}")

# ── Statistics: the mask follows what you supply ──────────────────────────────
#
# Each statistic has a validity bit saying whether it means anything. You pass
# values; the mask is derived, so a number can never be present with its bit
# unset. Grouped fields must be supplied together — the wire format gives them a
# single shared bit.

stats = hurray.Statistics(nnz=6, value_min=-1.0, value_max=1.0, value_abs_max=1.0)
print("\n=== Statistics ===")
print(f"  nnz={stats.nnz} value_abs_max={stats.value_abs_max}")
print(f"  not supplied, so not claimed: value_mean={stats.value_mean}")
print(f"  computed_mask=0x{stats.computed_mask:X}")

try:
    hurray.Statistics(value_min=-1.0)  # partial group
except hurray.InvalidDescriptorError as exc:
    print(f"  partial group refused: {exc}")

# ── Shard: this tensor's place in a bigger one ────────────────────────────────

shard = hurray.Shard(parent_shape=[4, 4], shard_offset=[2, 0])
print("\n=== Shard ===")
print(f"  {shard!r}")

piece = hurray.Tensor(bytes(32), hurray.float32, [2, 4], shard=shard, statistics=stats)
print(f"  tensor: shape={piece.shape} — the lower half of a [4, 4] parent")

# ── Persisting it ─────────────────────────────────────────────────────────────
#
# save() writes every buffer, so the scales travel with the weights and the
# scheme survives the round trip.

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "weights.hrry"
    hurray.save(str(path), {"w": quantized})
    back = hurray.load(str(path))["w"]
    print("\n=== Round trip ===")
    print(f"  loaded: shape={back.shape} dtype={back.dtype} buffers={back.buffer_count}")

    # The consumer side: ask the loaded tensor what it is holding. The getters
    # return the same classes the constructor accepts, so an inspected scheme can
    # be handed straight back to build another tensor.
    q = back.quantization
    print(f"  scheme:  {q!r}")
    print(f"  axis={q.axis} scale_buffer_index={q.scale_buffer_index}")
    print(f"  symmetric: {q.zero_point_buffer_index is None}")
    print("  run `hurray-inspect weights.hrry` to see the same thing byte by byte")

# ── Sections are None when absent ─────────────────────────────────────────────

plain = hurray.Tensor(bytes(16), hurray.float32, [4])
print("\n=== An ordinary tensor ===")
print(f"  quantization={plain.quantization} statistics={plain.statistics} shard={plain.shard}")
print(f"  buffer_count={plain.buffer_count}")
