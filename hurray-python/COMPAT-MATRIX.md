# hurray-python Compatibility Matrix

This document records which [DLPack](https://dmlc.github.io/dlpack/latest/)
specification version, Hurray format/descriptor version, and CPython version range
each `hurray-python` release series supports.

Update this file whenever a release adds or drops support for any of these.

## Release matrix

| hurray-python | DLPack (producer) | DLPack (consumer) | Native protocol | `HURRAY_C_ABI_VERSION` | CPython |
|---|---|---|---|---|---|
| 0.1.x | v1.0 (`dltensor_versioned`) | v0.8 + v1.0 | yes, multi-buffer | 5 | ≥ 3.10 |
| 0.2.x | v1.0 (`dltensor_versioned`) | v0.8 + v1.0 | yes, multi-buffer | 10 | ≥ 3.10 |

The ABI column is both the minimum and the maximum: the check is exact equality, so a
capsule from any other version is refused (see [Native protocol](#native-protocol)).

## Notes

### DLPack v1.0 producer

`hurray.Tensor.__dlpack__()` emits a `DLManagedTensorVersioned` capsule named
`"dltensor_versioned"`, conforming to DLPack v1.0.

### DLPack v0.8 + v1.0 consumer

`Tensor.from_dlpack()`, `hurray.from_torch()`, and `hurray.from_numpy()` accept both
the legacy `"dltensor"` (DLPack v0.8) and the versioned `"dltensor_versioned"`
(DLPack v1.0) capsule names, for compatibility with NumPy < 2.1 and PyTorch < 2.5.

### Native protocol

The `__hurray__` / `hurray.from_hurray` protocol is shipped since 0.1.x. The capsule
payload is whatever `HURRAY_C_ABI_VERSION` the build exposes, and the check is **exact
equality** — every bump is a capsule-compatibility boundary whether or not the payload
itself changed, because the C surface a consumer links against changed.

What moved the number, for the capsule specifically:

- ADR-030 raised it to 3, changing the capsule pointer from a single `HurrayBuffer`
  to a `HurrayBufferList` carrying every buffer of the tensor, so a version-2 consumer
  is told rather than misreading a list as a buffer.
- ADR-034 raised it to 4, making the capsule context a `HurrayTensorContext` so that a
  consumer in any language can read the descriptor and the version — before, only
  `hurray-python` could.
- ADR-039 raised it to 5, adding layout-payload field reads. The capsule payload was
  unchanged; 0.1.x shipped against this version.
- ADR-040 (6) and ADR-041 (7, 8, 9) grew the field identifier space and added the C
  stream, file and composite surfaces. The capsule payload was unchanged throughout.
- **ADR-043 raised it to 10**, and this one is a capsule break: the pointer is now a
  `HurrayTensor *` owning the descriptor and the buffer list, the context is gone with
  the `HurrayTensorContext` type that filled it, and the capsule is renamed
  `"hurray_tensor_handle"` so that a stale consumer gets a `TypeError` rather than
  dereferencing a pointer whose type changed under it. Teardown is one
  `hurray_tensor_destroy`.

Consumers MUST verify the version before dereferencing the pointer — with
`hurray_tensor_abi_version`, the one accessor that works on a handle from any version;
a mismatch raises `hurray.UnsupportedError`. See ADR-023 for the protocol, ADR-030 for
the multi-buffer change, ADR-034 for the readable context, ADR-039/ADR-040 for the field
reads, and ADR-043 for the single handle.

### Platform differences

One capability differs by platform: `hurray.StreamReader` and
`hurray.StreamWriter` accept an object with `fileno()` — a socket, a pipe, an open file —
on **Unix only**. On Windows that argument raises `hurray.UnsupportedError`; paths,
`bytes` and the in-memory writer are unaffected, as is everything outside streaming.
Tracked in [issue #215](https://github.com/pgillet/hurray/issues/215).
