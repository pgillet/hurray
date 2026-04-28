# Compliance Requirements — Hurray Implementation Requirements

## Overview

A **conforming Hurray implementation** is one that correctly reads and writes the
binary format as defined in `docs/spec/`. This document defines conformance levels,
the mandatory and optional feature surface, and the test requirements that determine
whether an implementation is conforming.

## Conformance Levels

### Level 1 — Reader

A Level 1 conforming implementation can **read** any valid Hurray tensor descriptor
and associated data buffer for all Tier 1 element types and all Tier 1 named layouts.
It does not need to produce output.

Mandatory:
- Parse all fixed-header fields (magic, version, flags, type_tag, layout_tag, rank).
- Reject descriptors with invalid magic, unsupported major version, or set reserved flag bits.
- Correctly interpret shape, byte_offset, and layout-specific fields for all Tier 1 layouts (`0x01`–`0x09`).
- Read and validate the buffer table (count, byte_size, alignment, device_tag).
- Skip optional sections using `descriptor_length` when flags are not understood.
- Return an error for unrecognised Tier 1 layout or type tags (unless in permissive mode).

Optional:
- Tier 2 element types (`float8_e4m3`, `float8_e5m2`, `float8_e8m0`, `complex64`, `complex128`, sub-byte types).
- Tier 2 layouts (`hilbert`, tag `0x40`).
- Quantization section (`HAS_QUANTIZATION`).
- Statistics section (`HAS_STATISTICS`).
- Extension type and layout tags.
- Permissive mode.

### Level 2 — Writer

A Level 2 conforming implementation can **write** valid Hurray tensor descriptors and
data buffers. Level 2 implies Level 1.

Mandatory (in addition to Level 1):
- Emit a correctly structured fixed header (magic `0x48 0x52 0x52 0x59`, current version `0x01 0x00`).
- Compute and emit a correct `descriptor_length`.
- Emit all mandatory fields for the chosen layout tag.
- Emit a buffer table with correct `byte_size` and `alignment` (minimum 64 bytes).
- Set `byte_offset = 0` for sparse layouts (COO, CSR, CSC).
- Set reserved flag bits to `0`.

### Level 3 — Network Transport

A Level 3 conforming implementation supports the Hurray network transport protocol
as defined in `docs/spec/interchange.md`. Level 3 implies Level 2.

Mandatory:
- Implement `CLIENT_HELLO` / `SERVER_HELLO` session establishment.
- Implement `TENSOR_REQUEST` / `TENSOR_DESCRIPTOR` / `TENSOR_DATA` / `TENSOR_DATA_END`.
- Implement `ERROR`, `PING`, `PONG`.
- Correctly handle `descriptor_length` to delimit descriptors in the stream.

Optional:
- Layout negotiation (`supported_layouts`, `ALLOW_TRANSCODE`).
- On-the-fly transcoding (`TRANSCODING` capability flag).
- Parallel shard transfers (`PARALLEL_STREAMS`, `PARALLEL_OK`).
- RDMA data plane (`RDMA_DATA_PLANE`, `RDMA_REGISTER`, `RDMA_READY`).
- `TENSOR_PUT` / `TENSOR_PUT_ACK`.

## Mandatory Element Type Support

All conforming implementations MUST support **Tier 1** element types:

| Type | Tag |
|------|-----|
| `float16` | `0x01` |
| `bfloat16` | `0x02` |
| `float32` | `0x03` |
| `float64` | `0x04` |
| `int8` | `0x10` |
| `uint8` | `0x11` |
| `int16` | `0x12` |
| `uint16` | `0x13` |
| `int32` | `0x14` |
| `uint32` | `0x15` |
| `int64` | `0x16` |
| `uint64` | `0x17` |
| `bool` | `0x20` |

"Support" means: correctly read the descriptor, compute element addresses and buffer
sizes, and preserve element bit patterns exactly during zero-copy interchange. An
implementation is not required to perform arithmetic on every supported type.

## Mandatory Layout Support

All conforming implementations MUST support **Tier 1** layouts for reading descriptors:

| Layout | Tag |
|--------|-----|
| Row-major | `0x01` |
| Column-major | `0x02` |
| Strided | `0x03` |
| Tiled / Blocked | `0x04` |
| Morton | `0x05` |
| General Subpaving | `0x06` |
| COO | `0x07` |
| CSR | `0x08` |
| CSC | `0x09` |

## Test Requirements

A conforming implementation MUST pass a test suite that covers:

- **Descriptor parsing**: valid descriptors for all mandatory type × layout combinations.
- **Rejection cases**: invalid magic, unsupported major version, reserved flag bits set, out-of-bounds `byte_offset`, invalid shard offset, **`rank = 65`** (descriptor MUST be rejected per `data-model.md` § Rank, ADR-008).
- **Round-trip**: write a tensor descriptor, read it back, verify all fields are identical.
- **Buffer size**: verify computed buffer sizes match expected values for all mandatory types and layouts.
- **Zero-copy invariant**: verify that bit patterns are preserved exactly after a round-trip (no NaN canonicalization, no subnormal flushing).
- **Sparse invariant validation**: for COO, CSR, CSC — verify that constraint violations are rejected.

### Empty Tensor Round-Trip Vectors

Per ADR-007 (permit empty tensors), a conforming implementation MUST round-trip the following empty-tensor descriptors without loss of information:

- A rank-1 tensor with shape `[0]` (empty vector).
- A rank-3 tensor with shape `[3, 0, 5]` (zero-size middle dimension; `element_count = 0`).
- A rank-2 CSR sparse matrix with `shape = [4, 5]` and `nnz = 0` (no stored non-zeros; `row_ptr = [0, 0, 0, 0, 0]`, `values` and `col_indices` buffers have `byte_size = 0`).

For each empty-tensor vector: writer emits the descriptor with a complete buffer table (zero-size buffers permitted), reader accepts the descriptor without error, all descriptor fields match exactly after round-trip.

The reference Rust implementation provides the canonical test suite. Language binding
test suites SHOULD mirror the reference suite and additionally test language-specific
interoperability (see `python-bindings.md`, `c-ffi.md`).
