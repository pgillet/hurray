"""Demonstrates hurray.load() and hurray.save() for HRRYFILE round-trips."""

import os
import tempfile

import hurray

# ── Create tensors ────────────────────────────────────────────────────────────

weights = hurray.zeros((4, 4), dtype=hurray.float32)
bias = hurray.zeros((4,), dtype=hurray.float32)
labels = hurray.arange(4, dtype=hurray.int32)

print("Created tensors:")
print(f"  weights  shape={weights.shape} dtype={weights.dtype}")
print(f"  bias     shape={bias.shape}    dtype={bias.dtype}")
print(f"  labels   shape={labels.shape}  dtype={labels.dtype}")

# ── Save to file ──────────────────────────────────────────────────────────────

with tempfile.NamedTemporaryFile(suffix=".hrry", delete=False) as f:
    path = f.name

try:
    hurray.save(
        path,
        {"weights": weights, "bias": bias, "labels": labels},
        kv={"model": "example", "version": 1},
    )
    size = os.path.getsize(path)
    print(f"\nSaved to {path} ({size} bytes)")

    # ── Load all tensors ──────────────────────────────────────────────────────

    loaded = hurray.load(path)
    print(f"\nLoaded {len(loaded)} tensors: {list(loaded.keys())}")

    for name, t in loaded.items():
        print(f"  {name}: shape={t.shape} dtype={t.dtype} device={t.device}")

    # ── Load specific tensors by name ─────────────────────────────────────────

    subset = hurray.load(path, names=["bias", "labels"])
    print(f"\nLoaded subset: {list(subset.keys())}")

    # ── Error handling ────────────────────────────────────────────────────────

    try:
        hurray.load("/nonexistent/path.hrry")
    except hurray.FileError as e:
        print(f"\nFileError caught (expected): {e}")

finally:
    os.unlink(path)

# ── File metadata ─────────────────────────────────────────────────────────────

print("\n=== The KV section ===")

kv_path = os.path.join(tempfile.mkdtemp(), "annotated.hrry")

hurray.save(
    kv_path,
    {"layer0.weight": hurray.Tensor(bytes(64), hurray.float32, [4, 4])},
    kv={
        "model": "demo-v1",
        "layers": 12,
        "lr": 0.001,
        "quantized": False,
        "signature": b"\x01\x02\x03",
        "block_shape": [16, 16],
    },
)

for key, value in hurray.load_kv(kv_path).items():
    print(f"  {key:12} = {value!r}")

print("\n  load_kv is a separate call from load: it answers a different question")
print("  and returns a different thing, and a flag that changed load's return")
print("  type would make every caller unpack a tuple to ask about tensors.")
print("  It costs a footer read, not a scan — the tensors are never touched.")

print(f"\n  round trips exactly: {hurray.load_kv(kv_path)['block_shape'] == [16, 16]}")
print("  (one asymmetry: Python's int writes as int64, so a value written from")
print("   Python never uses the uint64 tag; one written by Rust reads back as int)")
