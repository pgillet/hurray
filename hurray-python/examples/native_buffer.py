"""
Native Hurray buffer protocol: zero-copy tensor exchange via __hurray__.

Demonstrates in-process zero-copy transfer between hurray.Tensor objects using the
native protocol. Unlike DLPack, the protocol preserves the full Hurray descriptor
(device tag, memory class, sync mode, element type, shape, layout) without
flattening to DLPack's device enum — enabling exchange of tensors whose device or
memory-class combination has no DLPack representation.
"""

import struct

import hurray

# ── Basic round-trip ──────────────────────────────────────────────────────────

raw = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
source = hurray.Tensor(raw, hurray.float32, [2, 3])

print(f"Source: shape={source.shape}, dtype={source.dtype}")

# Transfer via native protocol — zero bytes copied.
target = hurray.from_hurray(source)

print(f"Target: shape={target.shape}, dtype={target.dtype}")
assert target.shape == source.shape
assert target.dtype == source.dtype

# ── Verify buffer content matches ─────────────────────────────────────────────

import numpy as np

src_arr = np.from_dlpack(source)
tgt_arr = np.from_dlpack(target)
assert (src_arr == tgt_arr).all(), "buffer content must match"
print(f"Buffer matches: {src_arr.flatten().tolist()}")

# ── Works for all dtypes, including Tier 2 ────────────────────────────────────

# Tier 2 types live on the hurray.dtype submodule — only the Tier 1 names are
# re-exported at the top level, since those are the ones NumPy also has.
int4_tensor = hurray.Tensor(bytes(8), hurray.dtype.int4, [16])
int4_copy = hurray.from_hurray(int4_tensor)
assert int4_copy.shape == int4_tensor.shape
assert int4_copy.dtype == int4_tensor.dtype
print(f"int4 round-trip: shape={int4_copy.shape}")

# ── Discovery via hasattr ────────────────────────────────────────────────────

assert hasattr(source, "__hurray__"), "all tensors expose __hurray__"
assert not hasattr(42, "__hurray__"), "non-tensors do not"

# ── Direct capsule inspection ─────────────────────────────────────────────────

capsule = source.__hurray__()
print(f"Capsule type: {type(capsule).__name__}")

# ── Error cases ───────────────────────────────────────────────────────────────

try:
    hurray.from_hurray(42)
except TypeError as e:
    print(f"TypeError (expected): {e}")

print("Native protocol example complete.")
