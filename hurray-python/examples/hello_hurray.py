"""
Smoke test: import hurray, check the version, build a tensor, and round-trip it
through the file format.

Run after `maturin develop`:
    python examples/hello_hurray.py
"""

import os
import tempfile

import numpy as np

import hurray

print(f"hurray version: {hurray.__version__}")

# Build a float32 [2, 3] tensor, zero-copy from a NumPy array.
arr = np.arange(6, dtype=np.float32).reshape(2, 3)
t = hurray.from_numpy(arr)
print(f"tensor: shape={t.shape}, dtype={t.dtype}, device={t.device}")

# Hand it back to NumPy zero-copy via DLPack (dense Tier 1 tensors share the buffer).
view = np.from_dlpack(t)
assert np.array_equal(view, arr)
print("DLPack round-trip: values match")

# Round-trip through the Hurray file format.
path = os.path.join(tempfile.gettempdir(), "hello.hrry")
try:
    hurray.save(path, {"greeting": t})
    loaded = hurray.load(path)
    name = list(loaded)[0]
    print(f"loaded '{name}' from file: shape={loaded[name].shape}")
finally:
    if os.path.exists(path):
        os.unlink(path)
