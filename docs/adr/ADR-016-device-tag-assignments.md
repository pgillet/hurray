# ADR-016: Device Tag Assignments

## Status
Accepted

## Context

`docs/spec/buffer-protocol.md` § Device Tags currently assigns four named device
tags (`0x00` CPU, `0x01` CUDA, `0x02` ROCm, `0x03` Metal), reserves `0x04`–`0xEF`
for future specification versions, reserves `0xF0`–`0xFE` for implementation-private
device types, and treats `0xFF` as permanently invalid.

The same tag space is reproduced in `docs/spec/interchange.md` § Device Tags (for
transport convenience), in `CLIENT_HELLO`/`SERVER_HELLO` `supported_devices` lists,
and in `TENSOR_REQUEST.preferred_device`.

Once a tag value is published, it is frozen: changing it is a breaking wire-format
change. The reserved range provides 236 unallocated slots, so there is no slot
pressure, but gratuitous proliferation creates a maintenance and conformance burden —
every named tag is a device that compliant readers MUST be able to identify, even if
they cannot execute on it.

DLPack — the dominant zero-copy tensor protocol that the Python binding layer
(`hurray-python`) will bridge to via `__dlpack__` / `__dlpack_device__` — currently
enumerates these additional device types beyond the four Hurray already names:
`kDLOpenCL`, `kDLVulkan`, `kDLVPI`, `kDLOneAPI`, `kDLWebGPU`, `kDLHexagon`,
`kDLMAIA`, `kDLTrn`. Hurray does not need to match DLPack's integers (the binding
layer translates), but DLPack's coverage is a strong signal of which devices are
used for cross-runtime tensor exchange today.

## Decision

### Named device tags (v1 amendment)

The following named tags are added to `buffer-protocol.md` § Device Tags. All other
reserved values remain reserved.

| Value | Device |
|-------|--------|
| `0x00` | CPU host memory |
| `0x01` | CUDA device memory |
| `0x02` | ROCm device memory |
| `0x03` | Metal device memory (Apple Silicon unified memory) |
| `0x04` | Vulkan device memory |
| `0x05` | WebGPU device memory |
| `0x06` | Qualcomm Hexagon (HVX/HMX) memory |
| `0x07` | Intel Level Zero / oneAPI device memory |
| `0x08` | OpenCL device memory |
| `0x09`–`0xEF` | Reserved for future specification versions |
| `0xF0`–`0xFE` | Implementation-private device types |
| `0xFF` | Reserved (invalid) |

Rationale for each addition:

- **`0x04` Vulkan** — Cross-platform GPU compute on Android, desktop Linux/Windows.
  Used by llama.cpp (Vulkan backend), MNN, ncnn. Fully standardised memory model.
- **`0x05` WebGPU** — Browser-based inference (ONNX Runtime Web, Transformers.js,
  WebLLM). W3C-standardised memory model. Growing rapidly; already in DLPack.
- **`0x06` Hexagon** — Qualcomm DSP is the dominant on-device accelerator for
  Android inference. QNN SDK uses it heavily. Mobile inference is in scope for Hurray.
- **`0x07` Intel Level Zero / oneAPI** — Intel Arc GPUs and Gaudi accelerators are
  in production inference deployments today. DLPack added `kDLOneAPI` in 2022. Without
  a named tag every implementation would converge on a private tag; promoting it
  later is more disruptive.
- **`0x08` OpenCL** — Declining, but still present on embedded Arm/Intel hardware
  and in long-tail ML runtimes. Cheap to name now; promoting after implementations
  have settled on private tags is costly.

The numeric ordering is contiguous from the existing assignments and reflects rough
priority (Vulkan, WebGPU first; Hexagon, Level Zero next; OpenCL last among named
tags). The values intentionally do **not** match DLPack's integers. Translation
occurs at the Python binding layer; binding implementors MUST consult
`docs/impl/python-bindings.md` for the authoritative Hurray ↔ DLPack mapping table.

The exact memory model for each tag (allocation API, alignment requirements beyond
the 64-byte minimum, synchronisation requirements, and DLPack mapping) MUST be
documented in `buffer-protocol.md` before this ADR is considered fully realised.
This ADR assigns the integers; the per-device prose subsection is an editorial
follow-up routed to `format-spec-writer`.

### Generic NPU tag — deferred

A "generic NPU" tag is **not** added. The label covers vastly different architectures
(Apple ANE, Hexagon NPU subsystem, Intel NPU, Google Edge TPU, Rockchip NPU, Huawei
Ascend, AWS Trainium, Microsoft Maia, etc.) with different memory models and
synchronisation primitives. A reader that sees a generic NPU tag cannot dereference,
transcode, or verify alignment without out-of-band knowledge — which is exactly what
the `0xF0`–`0xFE` private range already handles. Specific NPUs with broad cross-runtime
adoption are promoted as named tags individually.

### Future device tag policy (private-first promotion)

1. **Private slot first.** A new device starts in the `0xF0`–`0xFE` range. Implementations
   document their chosen private tag and exchange it only with peers that have agreed on
   the semantics out of band.
2. **Promotion criteria.** A private tag MAY be proposed for promotion when:
   - At least two independent implementations (not just two bindings of the same library)
     use the device for tensor interchange.
   - The device's memory model is sufficiently specified that a non-Rust reader can
     implement allocation, alignment validation, and DLPack translation from the spec alone.
   - The device is in active production use, not a discontinued or research-only platform.
3. **Promotion process.** A new ADR records the tag assignment and rationale;
   `buffer-protocol.md` and `interchange.md` are amended in lockstep; `python-bindings.md`
   is updated with the DLPack mapping.
4. **Tag values are forever.** Once published, a tag MUST NOT be reassigned.
5. **No silent expansion.** Adding a named tag is a minor-revision change. Readers compiled
   against a prior revision will reject the new tag; the `supported_devices` advertisement
   in `CLIENT_HELLO`/`SERVER_HELLO` ensures this is detected at session establishment, not
   mid-stream.

## Alternatives Considered

**Match DLPack's integers exactly.** Rejected. Hurray's tag space is `uint8` with a private
range at `0xF0`+; DLPack's enum is open-ended with non-contiguous integers. Forcing alignment
would either waste Hurray slots or constrain future assignments. Translation at the Python
binding layer is the correct boundary.

**Add only WebGPU and Hexagon; defer everything else.** Rejected as too conservative. The cost
of naming Vulkan, OpenCL, and Intel Level Zero today is one byte each. Deferring forces
implementations to use private tags that then need to be migrated.

**Add a generic NPU tag.** Rejected. The label is too coarse to be actionable without
out-of-band information, which is exactly what the private range handles.

**Widen `device_tag` from `uint8` to `uint16`.** Rejected. The buffer handle is a fixed
16-byte structure; widening requires a breaking re-layout. The 236-slot reserved range
will not be exhausted within the v1 spec lifetime.

## Consequences

- `buffer-protocol.md` § Device Tags: table gains rows `0x04`–`0x08`; reserved-future range
  narrows to `0x09`–`0xEF`; a per-device prose subsection is authored by `format-spec-writer`.
- `interchange.md` § Device Tags: reproduced table updated in lockstep.
- `docs/impl/python-bindings.md`: Hurray ↔ DLPack mapping table added for all eight named
  tags (CPU, CUDA, ROCm, Metal, Vulkan, WebGPU, Hexagon, Level Zero, OpenCL).
- `hurray-core/src/buffer.rs`: `DeviceTag` enum extended with five new variants at the
  assigned values; reserved-range validation narrows to `0x09`–`0xEF`. Routed to
  `rust-developer` as part of the next implementation pass that touches buffer types.
- Backward compatibility: producers using only `0x00`–`0x03` continue to work unchanged.
  Readers compiled before this amendment reject the new tags; `supported_devices`
  advertisement surfaces this at session establishment.
- Devices explicitly deferred: NVIDIA VPI, Google TPU, AWS Trainium/Inferentia,
  Microsoft Maia, generic NPU. All eligible for private tags; revisit when cross-runtime
  adoption evidence emerges.
- Memory-class sub-distinctions (CUDA Managed, CUDA Host, ROCm Host) are a separate
  design question and are deferred. Implementations MAY use private tags in the interim.
