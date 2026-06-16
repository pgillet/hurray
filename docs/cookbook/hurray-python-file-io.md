# File I/O — Saving and Loading Tensors

`hurray.save()` writes a collection of named tensors to an HRRYFILE container.
`hurray.load()` reads them back. Both functions are synchronous; the GIL is
released during I/O so other Python threads are not blocked.

## Saving tensors

```python
import hurray

weights = hurray.zeros((512, 512), dtype=hurray.float32)
bias    = hurray.zeros((512,),     dtype=hurray.float32)

hurray.save(
    "model.hrry",
    {"weights": weights, "bias": bias},
    kv={"arch": "linear", "version": 1},
)
```

`kv` is optional file-level metadata. Values may be `bool`, `int`, `float`,
`str`, `bytes`, or a homogeneous `list` of one of those scalar types.

## Loading tensors

```python
# Load all tensors
tensors = hurray.load("model.hrry")
w = tensors["weights"]   # hurray.Tensor
print(w.shape, w.dtype)  # (512, 512) float32

# Load a subset by name
subset = hurray.load("model.hrry", names=["bias"])
```

`hurray.load()` returns a `dict[str, hurray.Tensor]`. Tensors arrive with
an owned buffer (a copy of the bytes from disk). For zero-copy access via
memory-mapped files, use the native buffer protocol (Layer 8c).

## Round-trip example

```python
import os, tempfile, hurray

t = hurray.eye(3, dtype=hurray.float64)

with tempfile.NamedTemporaryFile(suffix=".hrry", delete=False) as f:
    path = f.name

try:
    hurray.save(path, {"identity": t})
    loaded = hurray.load(path)
    print(loaded["identity"].shape)  # (3, 3)
finally:
    os.unlink(path)
```

## Error handling

```python
try:
    hurray.load("missing.hrry")
except hurray.FileError as e:
    # subclass of OSError — also caught by `except OSError`
    print(f"file error: {e}")
```

| Exception | Raised by | Cause |
|---|---|---|
| `hurray.FileError` | `load()`, `save()` | File not found, corrupt container, CRC mismatch, unexpected EOF |
| `hurray.InvalidDescriptorError` | `load()` | Tensor descriptor failed to decode |
| `hurray.UnsupportedError` | `load()`, `save()` | Sparse (multi-buffer) tensors are not yet supported |

## Limitations in this release

- `hurray.save()` accepts only `hurray.Tensor` values. `hurray.SparseTensor`
  file I/O is not yet implemented.
- `hurray.load()` raises `hurray.UnsupportedError` for multi-buffer (sparse)
  tensor entries in the file.
- Loaded tensors hold an owned copy of the buffer data. Zero-copy
  memory-mapped loading will be added in a future release.
