# hurray-python Compatibility Matrix

This document records which `hurray-python` release series supports which
[Python Array API Standard](https://data-apis.org/array-api/) version,
[DLPack](https://dmlc.github.io/dlpack/latest/) specification version, and
CPython version range.

Update this file whenever a release adds or drops support for a standard version.
The conformance test suite must be run against every Array API version listed here.

## Release matrix

| hurray-python | Array API versions | DLPack (producer) | DLPack (consumer) | CPython |
|---|---|---|---|---|
| 0.1.x | 2023.12 | v1.0 (`dltensor_versioned`) | v0.8 + v1.0 | ≥ 3.10 |

## Notes

### Array API 2023.12

This is the minimum supported version. `hurray-python` exposes `bfloat16` (added in
Array API 2023.12) and implements `size` as `Optional[int]` (semantics clarified in
2023.12). Array API 2022.12 is not supported.

### DLPack v1.0 producer

`hurray.Tensor.__dlpack__()` emits a `DLManagedTensorVersioned` capsule named
`"dltensor_versioned"`, conforming to DLPack v1.0.

### DLPack v0.8 + v1.0 consumer

`Tensor.from_dlpack()`, `hurray.from_torch()`, and `hurray.from_numpy()` accept both
the legacy `"dltensor"` (DLPack v0.8) and the versioned `"dltensor_versioned"`
(DLPack v1.0) capsule names, for compatibility with NumPy < 2.1 and PyTorch < 2.5.
