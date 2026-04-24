# Spec Section Checklist

This checklist is applied to every new or modified section of the Hurray format specification. Items that do not apply to a given section should be noted as N/A with a brief justification.

The 12 categories correspond directly to the 10 Core Properties defined in `README.md`, plus two cross-cutting quality checks. Core Property 3 (Streamable by Design) applies differently to the streaming format and the file format; the checklist items below are labelled accordingly.

---

## 1. Zero-Copy Compatibility
*(Core Property 1)*

- [ ] Does the section introduce any field or requirement that forces a data copy? If so, is the copy justified and bounded?
- [ ] Are buffer alignment requirements stated explicitly (64-byte minimum for SIMD, page-aligned for GPU/IPC)?
- [ ] Are buffer handles described in a way compatible with the zero-copy semantics in `buffer-protocol.md`?
- [ ] If new buffers are introduced (e.g., quantization parameter buffers), are device-colocation rules stated?

## 2. Two Formats, One Descriptor
*(Core Property 2)*

- [ ] Does the section apply equally to both the streaming format and the file format, or is it specific to one? If specific, is this stated explicitly?
- [ ] Does the section reuse the tensor descriptor encoding from `metadata.md` without modification?
- [ ] If the section introduces container-level structure (headers, indexes, trailers), is it clearly separated from descriptor-level structure?

## 3. Streamability
*(Core Property 3)*

**Streaming format:**
- [ ] Does the section preserve the invariant that a reader can determine section length from fields appearing before variable-length content?
- [ ] Is there any forward reference or back-reference to data that appears later in the stream? (MUST NOT exist in the streaming format.)
- [ ] Is the section self-delimiting? Can a reader skip it using only the length prefix or a fixed field at a known offset?
- [ ] Does the section require buffering the full tensor or descriptor before processing? (MUST NOT be required.)

**File format:**
- [ ] Can the section be written in a single forward pass without seeking backward?
- [ ] Does any field require knowing the final file size or tensor count before the writer begins? (MUST NOT be required; use `0xFFFFFFFFFFFFFFFF` sentinel if count is unknown.)
- [ ] Does the file format section introduce a footer, trailer, or index? If so, is it located by a fixed-size trailer at end-of-file?

## 4. Memory Layout Consistency
*(Core Property 4)*

- [ ] Are strides expressed in logical elements (not bytes)?
- [ ] Is the layout tag used consistently with the tag table in `memory-layout.md`?
- [ ] Does the section correctly describe how quantization parameter buffers extend the buffer table beyond the layout baseline?
- [ ] Are negative and zero strides explicitly permitted or excluded as appropriate?

## 5. Quantization Consistency
*(Core Property 5)*

- [ ] If the section references quantized tensors, does it use the scheme_tag model from `quantization.md` (not ad-hoc fields)?
- [ ] Are storage types (`type_tag`) kept orthogonal from quantization scheme (`scheme_tag`)?
- [ ] Does the section describe the correct buffer table structure (data buffer + separate parameter buffers)?
- [ ] Are dequantization formulas consistent with `quantization.md`?

## 6. Language-Agnosticism
*(Core Property 6)*

- [ ] Are all field types expressed using language-agnostic names (`int32`, `uint64`, `float32` — never `i32`, `usize`, `String`)?
- [ ] Does the section avoid Rust-specific idioms, lifetimes, or type names?
- [ ] Is the binary layout specified precisely enough for a C, Python, or Go implementor without Rust knowledge?

## 7. Self-Describing and Self-Delimiting
*(Core Property 7)*

- [ ] Does every variable-length field or section begin with a length prefix (`uint32` or equivalent)?
- [ ] Are magic bytes or tag fields present where needed to allow format identification without context?
- [ ] Is the total byte length of the section determinable from a fixed, early-offset field?
- [ ] Are reserved fields explicitly required to be `0x00`, and must readers reject non-zero values?

## 8. Type System Compliance
*(Core Property 8)*

- [ ] Are all byte values expressed as hex literals (`0x00`, `0xFF` — never decimal)?
- [ ] Are all multi-byte fields declared as little-endian explicitly?
- [ ] Are type tags consistent with the type tag space defined in `element-types.md`?
- [ ] Are reserved tag ranges respected (`0x80`–`0xEF` must not be assigned)?

## 9. Interchange and File Format Compatibility
*(Core Property 9)*

- [ ] If the section defines new streaming message types or fields, are they consistent with the framing defined in `interchange.md`?
- [ ] If the section defines new file format fields or sections, are they consistent with the container defined in `file-format.md`?
- [ ] Does the section break any existing invariant in either protocol (e.g., descriptor-before-data ordering, trailer-at-end-of-file)?
- [ ] If new flags are introduced, are flag bit positions documented and reserved bits required to be `0`?

## 10. Array Database Compatibility
*(Core Property 10)*

- [ ] Does the section introduce any requirement that would prevent chunk-based or tile-based storage and retrieval? If so, is the restriction justified?
- [ ] If the section modifies the tiled, Morton, or Hilbert layout definitions, does it preserve spatial locality properties required for range-query cache efficiency?
- [ ] If the section modifies the file format footer index, does the change remain compatible with future extension by a spatial or dimension-range index?
- [ ] Does the section preclude or complicate sub-array queries (reading a contiguous region of a tensor without loading the full buffer)? If so, flag for architect review.
- [ ] If new metadata fields are introduced, could they carry dimension domain or coordinate information relevant to an array database (e.g., axis labels, tile extents, dimension ranges)?

## 11. RFC 2119 Correctness

- [ ] Does the section include the RFC 2119 notice near the top?
- [ ] Are normative keywords (`MUST`, `SHOULD`, `MAY`, etc.) in uppercase?
- [ ] Are normative keywords absent from non-normative blocks (prefixed `> **Note (non-normative):**`)?
- [ ] Are open questions marked with `> **[OQ-N]:**` and sequentially numbered within the file?

## 12. Cross-Section Consistency

- [ ] Does any term, field, or formula in this section contradict another spec section?
- [ ] Are all cross-references to other sections accurate (correct section names and field names)?
- [ ] Are open questions from other sections that this section resolves explicitly acknowledged?
- [ ] Does this section duplicate normative content that belongs in a canonical section? (Duplication MUST NOT exist; use a cross-reference instead.)
