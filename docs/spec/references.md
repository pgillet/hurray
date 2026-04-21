# References — Hurray Format Specification

> **Status:** Draft

## Normative References

The following documents are referenced normatively by one or more sections of
the Hurray format specification. Conforming implementations MUST comply with
the relevant portions of these documents as cited.

---

### [RFC2119]

Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels",
BCP 14, RFC 2119, March 1997.

<https://www.rfc-editor.org/rfc/rfc2119>

*Used by:* All sections. Normative keywords (`MUST`, `SHOULD`, `MAY`, etc.)
are interpreted as defined in this document.

---

### [IEEE754]

IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019, July 2019.

<https://ieeexplore.ieee.org/document/8766229>

*Used by:* `element-types.md` — `float16` (binary16), `float32` (binary32),
`float64` (binary64) bit-pattern definitions and NaN/infinity handling rules.

---

### [OFP8]

Open Compute Project, "OCP 8-bit Floating Point Specification (OFP8)",
Version 1.0, September 2023.

<https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>

*Used by:* `element-types.md` — `float8_e4m3` and `float8_e5m2` bit-pattern
definitions. `quantization/mxfp.md` — `float8_e8m0` scale factor encoding and
the MXFP8 element type definitions.

---

### [OCPMX]

Open Compute Project, "OCP Microscaling Formats (MX) Specification",
Version 1.0, 2023.

<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

*Used by:* `quantization/mxfp.md` — MXFP block size (`32`), `float8_e8m0`
shared exponent scale encoding, and valid MXFP element types.

---

### [QLoRA]

Dettmers, T., Pagnoni, A., Rodola, G., and Zettlemoyer, L., "QLoRA: Efficient
Finetuning of Quantized LLMs", arXiv:2305.14314, May 2023.

<https://arxiv.org/abs/2305.14314>

*Used by:* `quantization/nf4.md` — the NF4 lookup table values and the NF4
dequantization formula are taken from the reference implementation accompanying
this paper.

---

### [SKILLING2004]

Skilling, J., "Programming the Hilbert Curve", AIP Conference Proceedings,
Volume 707, pp. 381–387, 2004.

<https://doi.org/10.1063/1.1751381>

*Used by:* `layouts/hilbert.md` — the normative `CoordsToHilbert` and
`HilbertToCoords` algorithms are the algorithms defined in this paper.

---

## Informative References

The following documents are referenced for context or as prior art. They are
not normative; conforming implementations need not comply with them.

---

### [DLPACK]

DLPack: Open In-Memory Tensor Structure.

<https://github.com/dmlc/dlpack>

*Referenced by:* `buffer-protocol.md` — the release callback (single `deleter`)
model is inspired by `DLManagedTensor.deleter`. `docs/impl/python-bindings.md` —
the `__dlpack__` / `__dlpack_device__` protocol for Python zero-copy interop.

---

### [ARROW]

Apache Arrow: A cross-language development platform for in-memory data.

<https://arrow.apache.org>

*Referenced by:* `buffer-protocol.md` and `interchange.md` — IPC framing and
buffer protocol design reference. `docs/prior-art.md` § 2.2.

---

### [ARROWFLIGHT]

Apache Arrow Flight: A framework for high-performance data services.

<https://arrow.apache.org/docs/format/Flight.html>

*Referenced by:* `interchange.md` — streaming RPC model reference.
`docs/prior-art.md` § 2.3.

---

### [SAFETENSORS]

Hugging Face SafeTensors: Safe serialization for tensors.

<https://github.com/huggingface/safetensors>

*Referenced by:* `docs/prior-art.md` § 2.4.

---

### [GGUF]

GGUF: GPT-Generated Unified Format (llama.cpp).

<https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

*Referenced by:* `quantization/per-block-affine.md` — per-block affine
quantization covers the GGUF block quantization family (`Q4_0`, `Q4_1`, `Q8_0`).
`docs/prior-art.md` § 2.5.

---

### [ONNX]

Open Neural Network Exchange (ONNX): An open standard for machine learning
interoperability.

<https://onnx.ai>

*Referenced by:* `element-types.md` — type system breadth reference.
`data-model.md` — zero-size dimension policy comparison.
`docs/prior-art.md` § 2.6.

---

### [ZARR]

Zarr: Chunked, compressed, N-dimensional arrays.

<https://zarr.dev>

*Referenced by:* `docs/prior-art.md` § 2.7 — chunk/shard layout reference.

---

### [ARRAYAPI]

Consortium for Python Data API Standards, "Python Array API Standard".

<https://data-apis.org/array-api>

*Referenced by:* `element-types.md` — Tier 1 type vocabulary alignment with
the Python Array API Standard.

---

### [UCX]

Unified Communication X (UCX): An optimized communication layer.

<https://openucx.org>

*Referenced by:* `interchange.md` — RDMA data plane, UCX packed rkey blob as
one possible RDMA key encoding.

---

### [BITSANDBYTES]

Dettmers, T. et al., bitsandbytes: 8-bit optimizers and quantization.

<https://github.com/TimDettmers/bitsandbytes>

*Referenced by:* `quantization/nf4.md` — `block_size = 64` is the bitsandbytes
default for NF4 quantization.
