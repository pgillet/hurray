# hurray-python Compatibility Matrix

This document records which [DLPack](https://dmlc.github.io/dlpack/latest/)
specification version, Hurray format/descriptor version, and CPython version range
each `hurray-python` release series supports.

Update this file whenever a release adds or drops support for any of these.

## Release matrix

| hurray-python | DLPack (producer) | DLPack (consumer) | Native buffer protocol | Min `HURRAY_C_ABI_VERSION` | CPython |
|---|---|---|---|---|---|
| 0.1.x | v1.0 (`dltensor_versioned`) | v0.8 + v1.0 | yes (Layer 8c) | 2 | ≥ 3.10 |

## Notes

### DLPack v1.0 producer

`hurray.Tensor.__dlpack__()` emits a `DLManagedTensorVersioned` capsule named
`"dltensor_versioned"`, conforming to DLPack v1.0.

### DLPack v0.8 + v1.0 consumer

`Tensor.from_dlpack()`, `hurray.from_torch()`, and `hurray.from_numpy()` accept both
the legacy `"dltensor"` (DLPack v0.8) and the versioned `"dltensor_versioned"`
(DLPack v1.0) capsule names, for compatibility with NumPy < 2.1 and PyTorch < 2.5.

### Native buffer protocol

The `__hurray_buffer__` / `hurray.from_hurray_buffer` protocol is implemented in
Layer 8c (shipped in 0.1.x). The minimum `HURRAY_C_ABI_VERSION` required by the
capsule payload format is **2** (the version at Layer 8c ship time). Consumers MUST
verify the version before dereferencing the handle; a mismatch raises
`hurray.UnsupportedError`. See ADR-023 for the full protocol specification.
