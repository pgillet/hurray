# ADR-020: Memory Class Field in the Buffer Handle

## Status
Draft

## Context

The buffer handle (defined in `docs/spec/buffer-protocol.md`) carries a `device_tag`
field that identifies *where* a buffer resides (CPU, CUDA, ROCm, Metal, etc.). This
encodes a single dimension of buffer identity: the allocator or hardware domain.

A second, orthogonal dimension is left unrepresented: *how* a buffer is accessible —
specifically, whether it can be read without copying by more than one compute unit
simultaneously. Modern hardware offers at least three meaningfully distinct access
classes beyond device-exclusive memory:

1. **Host-pinned** — CPU RAM page-locked for GPU DMA. The CPU can read it at native
   speed; the GPU can access it over PCIe/interconnect without a copy, but at reduced
   bandwidth (no device-local caching). Example: `cudaMallocHost`, `hipHostMalloc`,
   `CL_MEM_ALLOC_HOST_PTR`.

2. **Unified / managed** — A single allocation coherently accessible by both the CPU
   and one or more accelerators, with hardware-managed page migration or physical
   sharing. Examples: `cudaMallocManaged`, ROCm HMM (`hipMallocManaged` on supported
   hardware), Metal `MTLStorageModeShared` on Apple Silicon.

3. **Peer-to-peer (P2P)** — A device-local buffer made directly accessible to a
   specific set of peer accelerators (not the CPU) via NVLink, xGMI, or PCIe BAR
   mapping. Examples: CUDA P2P (`cudaDeviceEnablePeerAccess`), ROCm P2P.

Under the current model, a consumer receiving a CUDA buffer handle cannot distinguish
`cudaMalloc` (GPU-exclusive VRAM) from `cudaMallocManaged` (CPU+GPU) from
`cudaMallocHost` (CPU-accessible, GPU-mapped). All three carry `device_tag = 0x01`.
The consumer must therefore either copy unconditionally or agree out-of-band — which
defeats the purpose of a self-describing handle.

ADR-016 explicitly deferred "Memory-class sub-distinctions (CUDA Managed, CUDA Host,
ROCm Host)" as a follow-up design question. This ADR resolves it.

DLPack already encodes this as separate device type values: `kDLCUDAManaged` (13),
`kDLCUDAHost` (3), `kDLROCMHost` (11). Hurray previously chose not to mirror DLPack's
integers (ADR-016 rationale), but the translation cost is low and bounded to the
Python bindings layer. The question is whether to follow DLPack's flat-enum approach
(Option 2) or to factor the concept into a separate field (this ADR, Option 3).

### Why not Option 2 (new device tags per access mode)?

Adding `CudaManaged`, `CudaHost`, `RocmHost`, `RocmManaged`, etc. as separate tags
encodes the same information but hides the structure. Every new accelerator needs
2–4 tag variants; the tag space grows quadratically with the number of device types
and access modes. It also obscures that the distinction is conceptually orthogonal to
device identity: a consumer that doesn't care about access mode must enumerate all
variants for each device type it supports.

### Why not Option 1 (a single "unified" flag bit)?

A single boolean cannot model the three-way distinction above. Host-pinned and
Unified have different performance and coherency semantics: pinned memory has no
hardware-managed coherency; unified/managed memory does. A consumer choosing between
a GPU kernel and a CPU-path fast route needs to distinguish them.

## Decision

### Wire format change

Repurpose byte 14 of the 16-byte buffer handle — currently `_reserved[0]` — as a new
`memory_class` field. Byte 15 remains `_reserved` (MUST be `0x00`; readers MUST
reject non-zero values).

Updated buffer handle layout:

| Offset | Field          | Type       | Description |
|--------|----------------|------------|-------------|
| 0      | `byte_size`    | `uint64`   | Size of the buffer in bytes (little-endian). |
| 8      | `alignment`    | `uint32`   | Minimum alignment in bytes (little-endian). |
| 12     | `device_tag`   | `uint8`    | Device where this buffer resides. |
| 13     | `sync_mode`    | `uint8`    | Producer-side synchronisation mechanism. |
| 14     | `memory_class` | `uint8`    | Memory access class. See § Memory Class Values. |
| 15     | `_reserved`    | `uint8`    | MUST be `0x00`. |

The total handle size remains 16 bytes. No existing field is displaced.

### Memory class values

| Value | Name        | Semantics |
|-------|-------------|-----------|
| `0x00` | `STANDARD`  | Device-exclusive memory. Only the primary compute unit of the tagged device can access this buffer without a copy. Default for all device types. CPU buffers (`device_tag = 0x00`) with this class are standard heap allocations. |
| `0x01` | `HOST_PINNED` | CPU-accessible, device-mapped. The CPU can read and write at native cache speed. The device can access it over its interconnect (PCIe, NVLink) without an explicit copy, but with reduced bandwidth compared to device-local memory. No hardware-managed coherency between CPU and device caches. |
| `0x02` | `UNIFIED`   | Hardware-managed unified/coherent memory. Both CPU and device can access this buffer at any time; the hardware (driver or MMU) ensures coherency. Physical pages may migrate. |
| `0x03` | `PEER`      | Peer-to-peer device memory. Directly accessible by a specific set of peer accelerators agreed out-of-band (NVLink, xGMI, PCIe BAR mapping). Not accessible from the CPU without a copy. The set of peers is communicated via the interchange protocol, not this field. |
| `0x04`–`0xEF` | (reserved) | Reserved for future specification versions. Readers MUST reject a buffer handle with a `memory_class` in this range. |
| `0xF0`–`0xFE` | (private)  | Implementation-private memory classes, valid only when paired with a private `device_tag` (`0xF0`–`0xFE`). Semantics are agreed out-of-band. Readers that do not recognise the private class MUST reject the handle unless they have agreed out-of-band. |
| `0xFF` | (invalid)   | Reserved. Readers MUST reject a buffer handle whose `memory_class` is `0xFF`. |

### Per-device validity table

Not all `(device_tag, memory_class)` pairs are meaningful. The following table
defines the valid combinations. Readers MUST reject handles with combinations not
listed as valid for the declared device. Private device tags (`0xF0`–`0xFE`) MAY
use any private memory class; semantics are out-of-band.

| Device              | `STANDARD` | `HOST_PINNED` | `UNIFIED` | `PEER` |
|---------------------|------------|---------------|-----------|--------|
| CPU (`0x00`)        | ✓          | ✓ (pinned for GPU DMA) | ✓ (unified addr space, device = CPU side) | ✗ |
| CUDA (`0x01`)       | ✓ `cudaMalloc` | ✓ `cudaMallocHost` | ✓ `cudaMallocManaged` | ✓ P2P |
| ROCm (`0x02`)       | ✓ | ✓ `hipHostMalloc` | ✓ `hipMallocManaged` (hw-dependent) | ✓ xGMI/PCIe |
| Metal (`0x03`)      | ✓ `StoragePrivate` | ✓ `StorageManaged` (discrete GPU only) | ✓ `StorageShared` (Apple Silicon) | ✗ |
| Vulkan (`0x04`)     | ✓ `DEVICE_LOCAL` | ✓ `HOST_VISIBLE` | ✓ `DEVICE_LOCAL\|HOST_VISIBLE` (integrated) | ✓ via external memory ext |
| WebGPU (`0x05`)     | ✓ | ✗ | ✗ | ✗ |
| Hexagon (`0x06`)    | ✓ VTCM/DDR | ✓ FastRPC shared | ✓ FastRPC coherent | ✗ |
| Level Zero (`0x07`) | ✓ `zeMemAllocDevice` | ✓ `zeMemAllocHost` | ✓ `zeMemAllocShared` | ✓ |
| OpenCL (`0x08`)     | ✓ | ✓ `CL_MEM_ALLOC_HOST_PTR` | ✓ SVM (OpenCL 2.0+) | ✗ |

> **Note (non-normative):** Metal `HOST_PINNED` (`StorageManaged`) is deprecated and
> unavailable on Apple Silicon. Producers targeting Apple Silicon MUST use `UNIFIED`
> (`StorageShared`) instead. The `HOST_PINNED` value remains defined for discrete
> Metal GPU configurations.

> **Note (non-normative):** ROCm `UNIFIED` requires hardware support for Heterogeneous
> Memory Management (HMM). Producers MUST verify hardware support before tagging a
> buffer `UNIFIED`; consumers MAY fall back to a copy-based path if `UNIFIED` is
> declared but the consumer's runtime does not support HMM on the current device.

### Backward compatibility

Existing descriptors that set `_reserved[0]` to `0x00` are implicitly `STANDARD`
(`memory_class = 0x00`), which is the correct interpretation for all allocations that
predate this field. The field defaults to the most conservative semantics; no existing
consumer is broken.

Readers compiled before this amendment will reject descriptors with `memory_class !=
0x00` at the `_reserved` byte check. This is the correct fail-safe: a consumer that
doesn't understand the memory class should not silently treat a `UNIFIED` buffer as
`STANDARD`, as it may issue incorrect synchronisation.

The `supported_memory_classes` field SHOULD be added to `CLIENT_HELLO`/`SERVER_HELLO`
in the interchange protocol to allow peers to advertise which memory classes they
support, analogous to `supported_devices`. This is a follow-up editorial change routed
to `format-spec-writer`.

### DLPack mapping

The Python bindings layer MUST translate `(device_tag, memory_class)` pairs to DLPack
`DLDeviceType` values. The mapping is maintained in `docs/impl/python-bindings.md`.
Representative entries:

| Hurray `device_tag` | Hurray `memory_class` | DLPack `DLDeviceType` |
|---------------------|-----------------------|-----------------------|
| `0x00` CPU          | `STANDARD`            | `kDLCPU` (1)          |
| `0x01` CUDA         | `STANDARD`            | `kDLCUDA` (2)         |
| `0x01` CUDA         | `HOST_PINNED`         | `kDLCUDAHost` (3)     |
| `0x01` CUDA         | `UNIFIED`             | `kDLCUDAManaged` (13) |
| `0x02` ROCm         | `STANDARD`            | `kDLROCM` (10)        |
| `0x02` ROCm         | `HOST_PINNED`         | `kDLROCMHost` (11)    |
| `0x03` Metal        | `STANDARD` / `UNIFIED`| `kDLMetal` (8)        |

Metal `STANDARD` and `UNIFIED` both map to `kDLMetal` (8) because DLPack does not
distinguish Metal storage modes. Consumers that need to distinguish storage modes
MUST use the Hurray `memory_class` field directly.

## Alternatives Considered

**Option 2 — new named device tags per access mode** (e.g., `0x0A CudaUnified`,
`0x0B CudaHostPinned`). Rejected: encodes the same information but hides the
factored structure, requiring O(devices × access_modes) tag assignments. Each new
device would need multiple tag variants, growing the tag table quadratically.

**Option 1 — a single `is_unified` flag bit** in a reserved byte. Rejected: a boolean
cannot distinguish `HOST_PINNED` from `UNIFIED`, which have different coherency and
performance semantics. Consumers choosing between GPU and CPU execution paths need the
three-way distinction.

**Widen device_tag to uint16 and encode access mode in the high byte.** Rejected:
breaking wire layout change with no benefit over a separate byte; the handle is already
16 bytes with a free byte available.

## Consequences

- `docs/spec/buffer-protocol.md`:
  - Buffer handle table: byte 14 renamed from `_reserved[0]` to `memory_class`;
    byte 15 remains `_reserved` (now a single byte, not a two-byte array).
  - New § Memory Class Values section.
  - New § Per-Device Validity Table section.
  - Existing per-device alignment subsections amended to note valid memory classes.
- `docs/spec/interchange.md`: `supported_memory_classes` advertisement added to
  `CLIENT_HELLO`/`SERVER_HELLO` (editorial follow-up → `format-spec-writer`).
- `docs/impl/python-bindings.md`: DLPack mapping table extended with
  `(device_tag, memory_class)` → `DLDeviceType` entries.
- `hurray-core/src/buffer.rs`: `BufferHandle` struct gains a `memory_class: MemoryClass`
  field at byte offset 14; `MemoryClass` enum defined with the values above. The
  existing `_reserved: [u8; 2]` field is replaced by `memory_class: MemoryClass` +
  `_reserved: u8`. Routed to `rust-developer` as part of the next buffer-touching pass.
- ADR-016 § Consequences note "Memory-class sub-distinctions deferred": resolved by
  this ADR.
- Backward compatibility: descriptors with `memory_class = 0x00` (`STANDARD`) are
  identical to pre-ADR-020 descriptors with `_reserved[0] = 0x00`. No existing
  producer or consumer is broken if they write `0x00` (the common case).
