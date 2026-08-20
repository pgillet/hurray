# Native Interchange Protocol

`hurray.Tensor` exposes a `__hurray__()` method and a matching
`hurray.from_hurray()` constructor for in-process zero-copy tensor exchange
between Hurray-aware Python extensions.

Unlike DLPack, the native protocol preserves the full Hurray descriptor — device tag,
memory class, sync mode, element type, shape, and layout — without flattening to
DLPack's `DLDeviceType` enum. It is available on **all** dtypes (including Tier 2 /
quantized) and in both strict and relaxed modes.

## Quick start

```python
import struct, hurray

# Create a source tensor.
raw = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
source = hurray.Tensor(raw, hurray.float32, [2, 3])

# Zero-copy transfer via the native protocol.
target = hurray.from_hurray(source)

assert target.shape == source.shape   # (2, 3)
assert target.dtype == source.dtype   # hurray.float32
```

No data is copied. `target` borrows `source`'s buffer; `source` is kept alive for
as long as `target` exists.

## Discovery

Probe support with `hasattr` before calling:

```python
if hasattr(obj, "__hurray__"):
    tensor = hurray.from_hurray(obj)
else:
    # Fall back to DLPack or another protocol.
    tensor = hurray.from_dlpack(obj)
```

## Why not DLPack?

DLPack is the right tool for interoperating with external libraries (PyTorch, JAX,
NumPy). Use it when your consumers do not link `hurray-ffi`. The native protocol fills
three gaps that DLPack v1.0 cannot express:

| Situation | DLPack | Native protocol |
|---|---|---|
| ROCm `UNIFIED` memory | `UnsupportedError` | Supported |
| `PEER` memory (any device) | `UnsupportedError` | Supported |
| Private device tags (`0xF0`–`0xFE`) | `UnsupportedError` | Supported |
| Tier 2 / quantized dtypes | `BufferError` | Supported |

## Tier 2 and quantized tensors

`__hurray__` is available unconditionally — it is not gated on strict or relaxed
mode and does not require the dtype to be an Array API Tier 1 type:

```python
import hurray

q_tensor = hurray.Tensor(bytes(64), hurray.int4, [128])
q_copy = hurray.from_hurray(q_tensor)

assert q_copy.dtype == hurray.int4   # works in strict mode
```

## Capsule lifecycle

`__hurray__()` returns a `PyCapsule` named `"hurray_tensor"`. The capsule
holds a `HurrayBuffer` pointer (from `hurray-ffi`) and a strong Python reference to
the source `Tensor`.

`hurray.from_hurray()` renames the capsule to `"used_hurray_tensor"` before
taking ownership — preventing double-free if the capsule is later GC'd. Attempting to
consume the same capsule twice raises `hurray.BufferError`.

```python
t = hurray.Tensor(bytes(8), hurray.float32, [2])
cap = t.__hurray__()          # fresh capsule

t2 = hurray.from_hurray(t)    # OK: calls __hurray__() internally
cap2 = t.__hurray__()         # OK: each call produces a new capsule
```

## Error handling

```python
import hurray

try:
    hurray.from_hurray(42)
except TypeError:
    pass  # object does not expose __hurray__

try:
    # ABI version mismatch (cross-build scenario)
    hurray.from_hurray(some_other_build_tensor)
except hurray.UnsupportedError:
    pass  # producer and consumer ABI versions differ
```
