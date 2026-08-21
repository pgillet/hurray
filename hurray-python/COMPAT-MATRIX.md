# hurray-python Compatibility Matrix

This document records which [DLPack](https://dmlc.github.io/dlpack/latest/)
specification version, Hurray format/descriptor version, and CPython version range
each `hurray-python` release series supports.

Update this file whenever a release adds or drops support for any of these.

## Release matrix

| hurray-python | DLPack (producer) | DLPack (consumer) | Native protocol | Min `HURRAY_C_ABI_VERSION` | CPython |
|---|---|---|---|---|---|
| 0.1.x | v1.0 (`dltensor_versioned`) | v0.8 + v1.0 | yes, multi-buffer | 4 | ≥ 3.10 |

## Notes

### DLPack v1.0 producer

`hurray.Tensor.__dlpack__()` emits a `DLManagedTensorVersioned` capsule named
`"dltensor_versioned"`, conforming to DLPack v1.0.

### DLPack v0.8 + v1.0 consumer

`Tensor.from_dlpack()`, `hurray.from_torch()`, and `hurray.from_numpy()` accept both
the legacy `"dltensor"` (DLPack v0.8) and the versioned `"dltensor_versioned"`
(DLPack v1.0) capsule names, for compatibility with NumPy < 2.1 and PyTorch < 2.5.

### Native protocol

The `__hurray__` / `hurray.from_hurray` protocol is shipped in 0.1.x.
The `HURRAY_C_ABI_VERSION` required by the capsule payload format is **4**:

- ADR-030 raised it to 3, changing the capsule pointer from a single `HurrayBuffer`
  to a `HurrayBufferList` carrying every buffer of the tensor, so a version-2 consumer
  is told rather than misreading a list as a buffer.
- ADR-034 raised it to 4, making the capsule context a `HurrayTensorContext` so that a
  consumer in any language can read the descriptor and the version — before, only
  `hurray-python` could.

Consumers MUST verify the version before dereferencing the pointer; a mismatch
raises `hurray.UnsupportedError`. The check is exact equality, so version 4 is both
the minimum and the maximum this release accepts. See ADR-023 for the protocol,
ADR-030 for the multi-buffer change, and ADR-034 for the readable context.
