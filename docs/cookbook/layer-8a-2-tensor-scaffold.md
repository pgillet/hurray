# Layer 8a.2 — Python Bindings: Dtype, Device, and Tensor Scaffold

This entry covers three new items shipped in Phase 8a.2:

- `hurray.Dtype` — element type descriptor class + `hurray.dtype` submodule
- `hurray.Device` — device descriptor class + `hurray.device` submodule
- `hurray.Tensor` — tensor scaffold: constructor, properties, `__repr__`

## Overview

Phase 8a.2 builds the structural foundation for the Python API. The primary goal
is to expose Hurray's core type system to Python in a way that is consistent with
the Array API Standard naming conventions while leaving zero-copy DLPack interop
(`__dlpack__`) and full Array API compliance (`__array_namespace__`) for later
phases.

## Dtype system

### Tier 1 vs Tier 2

The Hurray element type system is split into two tiers, matching the spec:

| Tier | Criterion | Python location |
|------|-----------|-----------------|
| 1 | Array API-compatible | `hurray.<name>` **and** `hurray.dtype.<name>` |
| 2 | Extended / sub-byte | `hurray.dtype.<name>` only |

**Tier 1 types** (accessible at the top level):

| Name | Bit width | Kind |
|------|-----------|------|
| `bool` | 1 | boolean |
| `int8`, `uint8` | 8 | integer |
| `int16`, `uint16` | 16 | integer |
| `int32`, `uint32` | 32 | integer |
| `int64`, `uint64` | 64 | integer |
| `float16`, `bfloat16` | 16 | float |
| `float32` | 32 | float |
| `float64` | 64 | float |

**Tier 2 types** (only on `hurray.dtype.*`):

| Name | Bit width | Kind |
|------|-----------|------|
| `int4`, `uint4` | 4 | integer |
| `int2`, `uint2` | 2 | integer |
| `float8_e4m3`, `float8_e5m2`, `float8_e8m0` | 8 | float |
| `float4_e2m1` | 4 | float |
| `float6_e2m3`, `float6_e3m2` | 6 | float |
| `float128` | 128 | float |
| `complex64`, `complex128` | 64 / 128 | complex |

### Singleton identity

Tier 1 dtype constants are the **same Python object** on the top-level module
and on `hurray.dtype`:

```python
import hurray

assert hurray.float32 is hurray.dtype.float32   # same object, not just ==
assert hurray.int8 is hurray.dtype.int8
```

This ensures that set membership and dict key lookups work correctly regardless
of which path is used to import the constant.

### Using Dtype as a dict key

`Dtype` is `frozen` (immutable) and implements `__hash__` via the wire tag byte:

```python
import hurray

lookup = {hurray.float32: "fp32 weights", hurray.dtype.int4: "quantized"}
assert lookup[hurray.float32] == "fp32 weights"
assert lookup[hurray.dtype.int4] == "quantized"
```

### from_name round-trip

```python
import hurray

d = hurray.Dtype.from_name("float32")
assert d == hurray.float32
assert d.name == "float32"
assert d.bit_width == 32
assert d.is_float
assert d.is_array_api

# Unknown name raises hurray.InvalidDescriptorError
try:
    hurray.Dtype.from_name("not_a_type")
except hurray.InvalidDescriptorError as e:
    print(f"rejected: {e}")
```

### Submodule import

The `hurray.dtype` submodule is registered in `sys.modules`, so the following
import forms all work:

```python
import hurray.dtype
from hurray.dtype import int4, float8_e4m3
```

## Device system

A `Device` is a `(kind, device_id, memory_class)` triple. It is `frozen` and
hashable.

### Well-known constants

`hurray.device` exposes a constant for each supported device kind, all at
`device_id=0` and `memory_class="standard"`:

```python
import hurray

hurray.device.cpu         # CPU host memory
hurray.device.cuda        # CUDA device 0
hurray.device.rocm        # ROCm device 0
hurray.device.metal       # Metal (Apple Silicon)
hurray.device.vulkan
hurray.device.webgpu
hurray.device.hexagon
hurray.device.level_zero
hurray.device.opencl
```

### Constructor

```python
import hurray

# Default: device_id=0, memory_class="standard"
cpu = hurray.Device("cpu")

# Specific GPU
gpu1 = hurray.Device("cuda", 1)

# Unified memory (hardware-coherent CPU+GPU access)
gpu_um = hurray.Device("cuda", 0, "unified")

# Pinned host memory (CPU-accessible, device-mapped)
pinned = hurray.Device("cuda", 0, "host_pinned")
```

### Equality and hashing

```python
import hurray

assert hurray.Device("cpu") == hurray.device.cpu      # same triple
assert hurray.Device("cuda", 0) != hurray.Device("cuda", 1)  # different id

# Usable as dict key
device_names = {hurray.device.cpu: "host", hurray.device.cuda: "gpu0"}
assert device_names[hurray.Device("cpu")] == "host"
```

### Submodule import

```python
import hurray.device
from hurray.device import cpu, cuda
```

## Constructing a Tensor

```python
import struct
import hurray

# Pack six float32 values as raw bytes (little-endian)
buf = struct.pack("6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

# Construct a 2×3 float32 tensor on the CPU
t = hurray.Tensor(buf, hurray.float32, [2, 3])

assert t.shape == (2, 3)    # tuple[int | None, ...]
assert t.ndim == 2
assert t.size == 6          # total element count (None if any dim is dynamic)
assert t.dtype == hurray.float32
assert t.device == hurray.device.cpu
```

### Explicit device

```python
import hurray, struct

buf = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
t_gpu = hurray.Tensor(buf, hurray.float32, [4], hurray.Device("cuda", 0))
assert t_gpu.device.kind == "cuda"
```

### Sub-byte types

Buffer sizes for sub-byte types are computed with ceiling division over the bit
width. For `int4`, two elements pack into one byte:

```python
import hurray

buf = bytes([0xAB, 0xCD])           # 2 bytes = 4 nibbles = 4 int4 elements
t = hurray.Tensor(buf, hurray.dtype.int4, [4])
assert t.size == 4
assert t.dtype.is_sub_byte
```

### Dynamic dimensions

If the tensor shape contains a `DYNAMIC` dimension (wire value `u64::MAX`), that
dimension maps to `None` in the Python shape tuple and `size` returns `None`:

```python
# shape tuple with a dynamic dim:  (1, None, 768)
# t.size == None
```

## What is NOT in Phase 8a.2

The following items are intentionally absent and will be added in later phases:

| Feature | Phase |
|---------|-------|
| `Tensor.__array_namespace__` | 8a.3 — Array API compliance |
| `Tensor.__dlpack__` / `__dlpack_device__` | 8a.3 — zero-copy DLPack |
| `Tensor.__array__` | 8a.3 — NumPy interop |
| Zero-copy buffer (`__hurray_buffer__`) | 8c |
| `Tensor.T` (transpose) | 8a.4 — raises `NotImplementedError` for now |
| `hurray.asarray`, `hurray.zeros`, etc. | 8b — Array API creation functions |

`hasattr(tensor, '__array_namespace__')` returns `False` in Phase 8a.2. This is
intentional — the Tensor class does not yet claim Array API conformance.

## Running the examples

Build the wheel with maturin, then run:

```bash
cd hurray-python
maturin develop           # build + install in-place (requires a venv)

python examples/02_dtype.py
python examples/03_device.py
python examples/04_tensor.py
```

All three examples print a confirmation line and exit with code 0 on success.

## Spec references

- `docs/spec/element-types.md` — Tier 1 / Tier 2 partition and type properties
- `docs/spec/buffer-protocol.md` — `DeviceTag` and `MemoryClass` wire tables
- `docs/spec/data-model.md` — `DYNAMIC` dimension sentinel (`u64::MAX`)
- `docs/impl/python-bindings.md` — Python binding implementation guide
