"""What a tensor's buffers declare, and why the declaration is worth trusting (ADR-037).

A Hurray descriptor states, for every buffer it carries, how many bytes it is, what
alignment its base address satisfies, when it may be read, and which device it lives on.
A consumer acts on those statements — it issues aligned SIMD loads because the descriptor
said 64, and it reads immediately because the descriptor said `producer_synced`.

So the interesting part of this page is not the accessor. It is that Hurray measures the
alignment instead of asserting it, and copies when the source cannot back the claim.

Run with:

    python hurray-python/examples/buffer_protocol.py
"""

import numpy as np

import hurray

# ── The buffer table ──────────────────────────────────────────────────────────

print("=== One handle per buffer ===")

dense = hurray.Tensor(bytes(4096), hurray.float32, [1024])
(handle,) = dense.buffer_handles

print(f"  {handle!r}")
print(f"  byte_size  {handle.byte_size}")
print(f"  alignment  {handle.alignment}")
print(f"  sync_mode  {handle.sync_mode}")
print(f"  device     {handle.device}  (is tensor.device: {handle.device is dense.device})")

# A sparse tensor carries several, and Device Colocation says they all share one device.
csr = hurray.Tensor(
    np.array([5.0, 7.0], dtype=np.float32).tobytes(),
    hurray.float32,
    [2, 2],
    aux_buffers=[
        np.array([0, 1], dtype=np.uint64).tobytes(),
        np.array([0, 1, 2], dtype=np.uint64).tobytes(),
    ],
    layout=hurray.CsrLayout(nnz=2),
)
print(f"\n  a CSR tensor has {len(csr.buffer_handles)} buffers:")
for index, row in enumerate(csr.buffer_handles):
    print(f"    [{index}] {row.byte_size:>3} bytes, {row.alignment}-byte aligned")

# ── A handle is a value, not a view ───────────────────────────────────────────

print("\n=== Metadata does not pin bytes ===")

print("  handle.byte_size answers without touching a byte, and the handle holds no")
print("  reference to the tensor — so collecting handles across a stream pins nothing.")
print("  That is also why t.buffer(i) is a separate call: on a CUDA tensor the handle")
print("  still answers where the byte view cannot.")

# ── Alignment is measured ─────────────────────────────────────────────────────

print("\n=== Measured, never asserted ===")

big = np.zeros(1 << 20, dtype=np.float32)  # 4 MiB
address = big.__array_interface__["data"][0]
print(f"  a 4 MiB NumPy array sits at ...{address % 4096} mod 4096")
print(f"  which leaves it {address % 64} bytes past a {hurray.MIN_BUFFER_ALIGNMENT}-byte boundary")
print("  (glibc puts a 16-byte chunk header before every mmap-served block, so a fresh")
print("   one never reaches 64; a recycled chunk can land anywhere. Unpredictable either")
print("   way, which is exactly why nothing here assumes it)")

# ── So ingest copies, unless you say otherwise ────────────────────────────────

print("\n=== copy: None | False | True ===")

t = hurray.from_numpy(big)
print(f"  copy=None (default): declares {t.buffer_handles[0].alignment}, copying if it had to")

try:
    hurray.from_numpy(big[1:], copy=False)      # an offset slice is always under-aligned
except hurray.CopyRequiredError as exc:
    print(f"  copy=False: {exc}")

print("\n  The cost is real and worth stating plainly: zero-copy NumPy ingest was never")
print("  conformant, and the honest version of it copies for most arrays. copy=False is")
print("  how you find out, instead of paying for a silent memcpy.")

# An array you allocated yourself can clear the bar — and then nothing is copied.
raw = np.zeros(1024 + 16, dtype=np.float32)
offset = (-raw.__array_interface__["data"][0] % 64) // 4
aligned = raw[offset : offset + 1024]

shared = hurray.from_numpy(aligned, copy=False)
aligned[0] = 42.0
print(f"\n  a 64-byte-aligned source is shared: {np.asarray(shared)[0] == 42.0}")
print("  (writes through the NumPy array are visible in the tensor — one allocation)")

# ── sync_mode is a promise, so it is read-only ────────────────────────────────

print("\n=== sync_mode ===")

print(f"  everything this binding builds: {dense.buffer_handles[0].sync_mode}")
print("  and there is no sync_mode= keyword, deliberately: 'event' means a device event")
print("  exists for a consumer to wait on, and no Python API supplies one. A settable")
print("  field could only author a contract nothing could honour.")
print()
print("  A tensor that arrives from a stream or a file reports what its producer")
print("  declared. If that is not producer_synced, the paths that hand out bytes —")
print("  buffer(), __array__, __dlpack__, to_torch — refuse, because a consumer owes")
print("  the producer a wait this binding cannot perform. Relaying it onward with")
print("  __hurray__ or StreamWriter.write still works: that reads no bytes.")

# ── The floors, from the format ───────────────────────────────────────────────

print("\n=== Constants ===")

print(f"  hurray.MIN_BUFFER_ALIGNMENT = {hurray.MIN_BUFFER_ALIGNMENT}  (SIMD)")
print(f"  hurray.PAGE_ALIGNMENT       = {hurray.PAGE_ALIGNMENT}  (GPU / IPC)")
