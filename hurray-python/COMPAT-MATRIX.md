# hurray-python Compatibility Matrix

This document records which `hurray-python` release series supports which
[Python Array API Standard](https://data-apis.org/array-api/) version,
[DLPack](https://dmlc.github.io/dlpack/latest/) specification version, and
CPython version range.

Update this file whenever a release adds or drops support for a standard version.
The conformance test suite must be run against every Array API version listed here.

## Release matrix

| hurray-python | Array API versions | DLPack (producer) | DLPack (consumer) | Native buffer protocol | Min `HURRAY_C_ABI_VERSION` | CPython |
|---|---|---|---|---|---|---|
| 0.1.x | 2025.12 | v1.0 (`dltensor_versioned`) | v0.8 + v1.0 | not yet (Layer 8c) | — | ≥ 3.10 |

## Notes

### Array API 2025.12

This is the minimum and current supported version. `hurray-python` targets 2025.12
from day one to avoid retrofitting structural decisions (e.g. multi-return functions
MUST return `tuple`, not `list`; `permute_dims` accepts negative axes; `expand_dims`
accepts a tuple of axes).

2025.12 is a superset of 2023.12 and 2024.12. Key features inherited:
- `bfloat16` and `size: Optional[int]` (2023.12)
- Integer-array (fancy) indexing; scalar arguments on ~35 binary ops (2024.12)
- `broadcast_shapes`, `isin`, `linalg.eig`/`linalg.eigvals` (2025.12)

Array API 2022.12 and 2023.12 are not declared; `__array_namespace__(api_version=...)`
calls for older versions will raise `ValueError`.

### DLPack v1.0 producer

`hurray.Tensor.__dlpack__()` emits a `DLManagedTensorVersioned` capsule named
`"dltensor_versioned"`, conforming to DLPack v1.0.

### DLPack v0.8 + v1.0 consumer

`Tensor.from_dlpack()`, `hurray.from_torch()`, and `hurray.from_numpy()` accept both
the legacy `"dltensor"` (DLPack v0.8) and the versioned `"dltensor_versioned"`
(DLPack v1.0) capsule names, for compatibility with NumPy < 2.1 and PyTorch < 2.5.

### Native buffer protocol

The `__hurray_buffer__` / `hurray.from_hurray_buffer` protocol is implemented in
Layer 8c. Until then, `hasattr(tensor, '__hurray_buffer__')` returns `False` and
the "Native buffer protocol" column reads "not yet". When Layer 8c ships, this
column will record `"yes"` and the minimum `HURRAY_C_ABI_VERSION` required by
the capsule payload format.
