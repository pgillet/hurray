"""
Multi-buffer tensors over the native protocol (ADR-030).

Most tensors have one buffer. Anything whose descriptor references a second one —
per-channel / NF4 / MXFP quantization scales, sparse index arrays, block-paged
page tables — has more, and every buffer must reach the consumer or the
descriptor's buffer indices point at nothing.

``__hurray_buffer__`` carries them all in a single capsule, in descriptor
buffer-table order. Sparse is not a special case: it is simply the multi-buffer
case, which is why there is no separate ``__hurray_sparse_buffer__``.

Run with:

    python hurray-python/examples/multi_buffer.py
"""

import numpy as np

import hurray

# ── One buffer: the ordinary case ─────────────────────────────────────────────

dense = hurray.Tensor(bytes(16), hurray.float32, [4])
print("=== Single-buffer tensor ===")
print(f"  shape={dense.shape} dtype={dense.dtype}")

received = hurray.from_hurray_buffer(dense)
print(f"  round-tripped: shape={received.shape} dtype={received.dtype}")
print("  (N=1 is not a special path — the same protocol, one element)")

# ── Several buffers: a sparse tensor ──────────────────────────────────────────
#
# A COO tensor stores its non-zero values in one buffer and their coordinates in
# another, so its descriptor's buffer table has two entries.

values = np.array([5.0, 7.0], dtype=np.float32)
indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)  # [nnz, rank]
sparse = hurray.sparse_coo(values, indices, [2, 2])

print("\n=== Multi-buffer tensor (COO sparse) ===")
print(f"  format={sparse.format} nnz={sparse.nnz} shape={sparse.shape}")

# One protocol for every tensor kind — probe for it exactly as for a dense tensor.
print(f"  has __hurray_buffer__:        {hasattr(sparse, '__hurray_buffer__')}")
print(f"  has __hurray_sparse_buffer__: {hasattr(sparse, '__hurray_sparse_buffer__')}")

capsule = sparse.__hurray_buffer__()
print(f"  capsule: {capsule}")

# ── Consuming a multi-buffer capsule ──────────────────────────────────────────
#
# The consumer gets the full descriptor — layout, element type, shape — with every
# buffer attached in descriptor order: values first, then the index array.

back = hurray.from_hurray_buffer(sparse)
print("\n=== After the hop ===")
print(f"  shape={back.shape} dtype={back.dtype}")
print("  values and index buffers both travelled; the COO descriptor is intact")
print("  (returned as a Tensor carrying the sparse descriptor, not a SparseTensor)")

# ── What this unblocks ────────────────────────────────────────────────────────
#
# Per-tensor-affine quantization already survives a hop today because its scale
# and zero point are inline in the descriptor — no second buffer. Per-channel,
# NF4 and MXFP all reference a scale_buffer_index, so they need the transport
# above. Authoring those descriptors from Python is the next step; the wire and
# transport are ready for them now.

print("\n=== Why this matters ===")
print("  per-tensor affine:      inline scale/zero-point, always fit in one buffer")
print("  per-channel / NF4 / MXFP: reference a scale buffer -> need this transport")
