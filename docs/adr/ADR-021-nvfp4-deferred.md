# ADR-021: NVFP4 Quantization Scheme — Deferred

## Status
Deferred

## Context

NVFP4 is NVIDIA's 4-bit floating-point quantization format introduced with the
Blackwell GPU architecture (B100, B200, GB200). It is implemented in hardware via
Blackwell Tensor Cores and used in production by NVIDIA TensorRT-LLM, vLLM
(Blackwell path), and — for model-weight compatibility on non-NVIDIA hardware —
Apple's MLX framework.

### Encoding summary

- **Element type:** 4-bit float, E2M1 (2-bit biased exponent, 1-bit mantissa,
  1-bit sign), packed LSB-first into byte sequences.
- **Scale format:** E4M3 (4-bit) per-group scale value. One scale per group of 16
  elements. No zero-point / bias term.
- **Group size:** Fixed at 16 elements.

### How it differs from existing Hurray schemes

NVFP4 cannot be expressed by any scheme currently defined in `quantization.md`:

| Property | MXFP (0x05) | NVFP4 |
|----------|-------------|-------|
| Scale format | E8M0 (8-bit, OCP MX) | E4M3 (4-bit, NVIDIA) |
| Default group size | 32 | 16 |
| Standardisation body | OCP (Open Compute Project) | NVIDIA (proprietary) |

A different scale type and group size means the MXFP scheme tag cannot
accommodate NVFP4 even with a different `block_size` parameter.

### Arguments for adding NVFP4 as Tier 2

1. **Hardware support is real and shipping.** Blackwell Tensor Cores execute NVFP4
   natively. Tier 2 precedent (MXFP) was set on the same basis.
2. **Cross-ecosystem adoption is observable.** Apple MLX added NVFP4 support on
   Apple Silicon solely for model-weight portability — not native hardware support.
   This is a concrete instance of the interchange scenario Hurray targets.
3. **Existing scheme machinery applies.** Tier 2 schemes are OPTIONAL; adding one
   imposes no conformance burden on implementations that don't support it.

### Arguments for deferring

1. **Vendor-proprietary, not an open standard.** MXFP is specified by OCP; NVFP4
   is documented in NVIDIA CUDA/CUTLASS SDK documentation. If NVIDIA alters the
   encoding, the Hurray spec would require a `scheme_version` bump — coupling the
   spec lifecycle to a vendor's SDK.
2. **Published NVFP4 model weights remain scarce.** As of May 2026, the dominant
   open-weight formats are GGUF (Q4–Q8), safetensors (bf16/fp16), and AWQ int4.
   NVFP4 weights exist but are not yet widely distributed.
3. **Specification source is not stable enough to write a normative spec section.**
   The canonical NVFP4 encoding details are spread across NVIDIA CUTLASS headers and
   TensorRT-LLM documentation rather than a versioned, citable specification document.
   Writing a normative spec section against a moving target risks drift.
4. **Private extension range already covers the interim need.** Implementations that
   need NVFP4 interchange today MAY use a private scheme tag (`0xF0`–`0xFE`) and
   agree on semantics out of band. This is the intended path for formats that are not
   yet ready for standardisation.

## Decision

**Deferred.** NVFP4 is not assigned a Tier 2 (or Tier 1) scheme tag at this time.

Implementations that need to exchange NVFP4 tensors MAY use a private scheme tag
in the `0xF0`–`0xFE` range with the encoding derived from NVIDIA CUTLASS / TRT-LLM
documentation. The recommended private encoding mirrors the Tier 2 MXFP scheme
structure (same 4-byte header, same buffer table placement rules) with:
- `scale_format = E4M3` (to be defined if/when the scheme is promoted)
- `block_size = 16`
- No zero-point field

### Promotion criteria

This decision SHOULD be revisited when **two or more** of the following conditions
are met:

1. NVFP4 weights are distributed in at least two major open-weight model releases
   (e.g., on Hugging Face Hub) in a format that requires cross-runtime interchange
   (not just single-runtime inference).
2. A second hardware vendor implements native NVFP4 Tensor Core support, or a
   standards body (OCP, ISO, JEDEC) adopts a compatible specification.
3. NVIDIA publishes a versioned, citable NVFP4 specification document (not just
   SDK headers) that can be referenced normatively.
4. At least two independent Hurray implementations have shipped private-tag NVFP4
   support and request promotion.

## Alternatives Considered

**Add NVFP4 as Tier 2 immediately.** Rejected at this stage: the normative source
is not stable enough to write a durable spec section. `scheme_version` provides a
forward migration path, but repeated version bumps due to upstream churn would erode
spec credibility.

**Add NVFP4 as Tier 1.** Not considered: Tier 1 requires implementation by all
conforming implementations that advertise quantization support. A vendor-specific
4-bit format with no open standard is not an appropriate Tier 1 candidate.

**Do nothing / no ADR.** Rejected: the question has been raised and evaluated; an
explicit deferral with promotion criteria is more useful than silence. Future
reviewers will otherwise re-investigate the same question from scratch.

## Consequences

- No changes to `quantization.md` or the scheme tag table at this time.
- Implementations that need NVFP4 exchange today SHOULD use private scheme tag
  `0xF0` and document their encoding internally, pending promotion.
- This ADR is the tracking record for the NVFP4 promotion decision. It SHOULD be
  revisited when the promotion criteria above are met, at which point it is
  superseded by a new ADR assigning a Tier 2 tag.
- `TODO.md`: add a note to re-evaluate NVFP4 when Blackwell model-weight distribution
  becomes mainstream (estimated: late 2026 based on hardware ramp).
