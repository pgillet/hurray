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
