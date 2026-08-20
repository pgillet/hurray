"""Zero-copy DLPack and NumPy interop example.

Demonstrates:
- Exporting a Tensor as a DLPack v1.0 capsule
- Importing a Tensor from a NumPy array via hurray.from_numpy
- Converting to NumPy via Tensor.__array__
- Converting to/from PyTorch via Tensor.to_torch / hurray.from_torch (requires torch)
"""

import struct

import numpy as np

import hurray

# ── Create a Tensor from raw bytes ───────────────────────────────────────────

buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
t = hurray.Tensor(buf, hurray.float32, [2, 3])
print(f"Created: {t}")

# ── DLPack v1.0 export ───────────────────────────────────────────────────────

# Emit a "dltensor_versioned" capsule (DLPack v1.0, D1).
capsule = t.__dlpack__()
print(f"DLPack capsule: {capsule}")

device_type, device_id = t.__dlpack_device__()
print(f"DLPack device: type={device_type} (kDLCPU=1), id={device_id}")

# Hand NumPy the *tensor*, not the capsule: a DLPack consumer takes an object
# exposing __dlpack__ and calls it itself, negotiating device and stream with the
# producer. (NumPy accepted a bare capsule until 2.1; it no longer does.)
arr_from_dlpack = np.from_dlpack(t)
print(f"NumPy from DLPack: shape={arr_from_dlpack.shape}, dtype={arr_from_dlpack.dtype}")
assert arr_from_dlpack.shape == (2, 3)
assert arr_from_dlpack.dtype == np.float32

# ── __array__ protocol ───────────────────────────────────────────────────────

arr = t.__array__()
print(f"__array__(): shape={arr.shape}, dtype={arr.dtype}")
assert arr.shape == (2, 3)
assert arr.dtype == np.float32

# __array__ with explicit dtype — performs a cast.
arr_f64 = t.__array__(dtype=np.float64)
print(f"__array__(dtype=float64): dtype={arr_f64.dtype}")
assert arr_f64.dtype == np.float64

# ── Zero-copy from NumPy ─────────────────────────────────────────────────────

src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t2 = hurray.from_numpy(src)
print(f"from_numpy: {t2}")
assert t2.shape == (2, 2)
assert t2.dtype == hurray.float32

# ── PyTorch bridge (optional) ─────────────────────────────────────────────────

try:
    import torch  # noqa: F401

    t3 = hurray.Tensor(bytes(16), hurray.float32, [4])
    torch_t = t3.to_torch()
    print(f"to_torch: {torch_t.shape}, dtype={torch_t.dtype}")

    t4 = hurray.from_torch(torch_t)
    print(f"from_torch: {t4}")

except ImportError:
    print("PyTorch not installed — skipping torch interop demo")

print("All assertions passed.")
