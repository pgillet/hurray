#!/usr/bin/env python3
"""Smoke test: hurray.Tensor — construction, properties, and error paths."""
import struct
import hurray

# Construct a 2x3 float32 tensor
buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
t = hurray.Tensor(buf, hurray.float32, [2, 3])

assert t.shape == (2, 3)
assert t.ndim == 2
assert t.size == 6
assert t.dtype == hurray.float32
assert t.device == hurray.device.cpu

# Default device is cpu
assert t.device.kind == "cpu"

# Explicit device
t_gpu = hurray.Tensor(buf, hurray.float32, [2, 3], hurray.Device("cuda", 0))
assert t_gpu.device.kind == "cuda"

# No Array API namespace: hurray-python is an interchange codec, not a compute
# library, so it does not claim Array API conformance (ADR-029).
assert not hasattr(t, "__array_namespace__")

# DLPack, however, is a standalone protocol and every tensor speaks it.
assert hasattr(t, "__dlpack__")
assert hasattr(t, "__dlpack_device__")

# T raises
try:
    _ = t.T
    assert False, "should have raised"
except NotImplementedError:
    pass

# Buffer too small raises BufferError
try:
    hurray.Tensor(buf[:8], hurray.float32, [2, 3])
    assert False, "should have raised"
except hurray.BufferError:
    pass

# int4 tensor (4 elements = 2 bytes packed)
buf_i4 = bytes([0xAB, 0xCD])
t_i4 = hurray.Tensor(buf_i4, hurray.dtype.int4, [4])
assert t_i4.size == 4
assert t_i4.dtype.bit_width == 4

print("04_tensor.py: all assertions passed")
