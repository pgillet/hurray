# Element Types

> **Status:** Draft

## Scope

This section defines the complete numeric element type system for the Hurray tensor format, including type identifiers, bit widths, encoding rules, and sub-byte packing conventions.

> **Note (non-normative):** Hurray is a strictly numeric tensor format. String types, structured types, and arbitrary user-defined element types are out of scope. The type system is designed to cover the full range of numeric precisions encountered in modern AI/ML inference, from float64 down to 2-bit quantized integers.

## Normative Requirements

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Byte Order

All multi-byte fields in the Hurray format — including element data, descriptor fields, and metadata — MUST be encoded in little-endian byte order (least significant byte at the lowest address). The format does not define a big-endian encoding and contains no endianness indicator field.

Implementations running on big-endian host architectures MUST convert to little-endian on write and from little-endian on read. The wire format is always little-endian regardless of host byte order.

> **Note (non-normative):** This matches the choice made by Apache Arrow, DLPack, and SafeTensors. All hardware platforms targeted by AI/ML inference workloads (x86, ARM, RISC-V, NVIDIA/AMD GPUs, Apple Silicon) are little-endian. A fixed byte order eliminates the need for byte-swap detection and preserves zero-copy semantics between any two conforming implementations.

## Type Identifier Encoding

Every element type is identified by a **type tag**, encoded as a `uint8` value in the binary tensor descriptor. The type tag space is partitioned as follows:

| Range | Allocation |
|-------|------------|
| `0x00` | Reserved (invalid) |
| `0x01` -- `0x3F` | Tier 1 core types |
| `0x40` -- `0x7F` | Tier 2 extended types |
| `0x80` -- `0xEF` | Reserved for future specification versions |
| `0xF0` -- `0xFE` | Reserved for implementation-private extensions |
| `0xFF` | Reserved (invalid) |

A conforming reader MUST reject a tensor descriptor containing a type tag of `0x00` or `0xFF`.

A conforming reader MUST reject a tensor descriptor containing a type tag it does not recognize, unless the reader is operating in an explicitly configured permissive mode. In permissive mode, the reader MAY accept the descriptor but MUST NOT attempt to interpret the tensor data buffer.

Implementations MUST NOT assign semantics to type tags in the range `0x80` -- `0xEF`; these are reserved for future versions of this specification.

Implementations MAY use type tags in the range `0xF0` -- `0xFE` for private extensions. Tensors using private extension type tags MUST NOT be exchanged between independent implementations unless both parties have agreed on the semantics out of band.

## Tier 1 -- Core Types

All conforming implementations MUST support every Tier 1 type. A conforming implementation MUST be able to read a tensor descriptor for any Tier 1 type and correctly interpret its metadata (shape, strides, buffer layout). Whether the implementation can perform computation on every Tier 1 type is outside the scope of this specification.

### Floating-Point Types

| Type | Tag | Bit Width | Description |
|------|-----|-----------|-------------|
| `float16` | `0x01` | 16 | IEEE 754 binary16 |
| `bfloat16` | `0x02` | 16 | Brain floating point |
| `float32` | `0x03` | 32 | IEEE 754 binary32 |
| `float64` | `0x04` | 64 | IEEE 754 binary64 |

#### float16 (`0x01`)

`float16` is the IEEE 754 binary16 format: 1 sign bit, 5 exponent bits, 10 significand (mantissa) bits. Total width is 16 bits (2 bytes). The two bytes MUST be stored in little-endian order (least significant byte first).

All IEEE 754 binary16 bit patterns are valid, including positive and negative zero, infinities, and NaN values. Implementations MUST preserve the bit pattern exactly during zero-copy interchange; they MUST NOT canonicalize NaN payloads or flush subnormals to zero.

#### bfloat16 (`0x02`)

`bfloat16` uses 1 sign bit, 8 exponent bits, and 7 significand (mantissa) bits. Total width is 16 bits (2 bytes). The two bytes MUST be stored in little-endian order.

> **Note (non-normative):** `bfloat16` has the same exponent range as `float32` but with reduced mantissa precision. It is widely used in machine learning training and inference.

All 16-bit patterns are valid `bfloat16` values, including infinities and NaN values. Implementations MUST preserve the bit pattern exactly during zero-copy interchange.

#### float32 (`0x03`)

`float32` is the IEEE 754 binary32 format: 1 sign bit, 8 exponent bits, 23 significand bits. Total width is 32 bits (4 bytes). The four bytes MUST be stored in little-endian order.

All IEEE 754 binary32 bit patterns are valid. Implementations MUST preserve the bit pattern exactly during zero-copy interchange.

#### float64 (`0x04`)

`float64` is the IEEE 754 binary64 format: 1 sign bit, 11 exponent bits, 52 significand bits. Total width is 64 bits (8 bytes). The eight bytes MUST be stored in little-endian order.

All IEEE 754 binary64 bit patterns are valid. Implementations MUST preserve the bit pattern exactly during zero-copy interchange.

### Integer Types

| Type | Tag | Bit Width | Signed | Range |
|------|-----|-----------|--------|-------|
| `int8` | `0x10` | 8 | yes | -128 to 127 |
| `uint8` | `0x11` | 8 | no | 0 to 255 |
| `int16` | `0x12` | 16 | yes | -32768 to 32767 |
| `uint16` | `0x13` | 16 | no | 0 to 65535 |
| `int32` | `0x14` | 32 | yes | -2147483648 to 2147483647 |
| `uint32` | `0x15` | 32 | no | 0 to 4294967295 |
| `int64` | `0x16` | 64 | yes | -2^63 to 2^63 - 1 |
| `uint64` | `0x17` | 64 | no | 0 to 2^64 - 1 |

All signed integer types use two's complement representation.

All multi-byte integer types MUST be stored in little-endian byte order.

`int8` and `uint8` occupy exactly one byte; byte order is not applicable.

All bit patterns within the specified width are valid for both signed and unsigned integer types. There are no trap representations.

### Boolean Type

| Type | Tag | Bit Width | Description |
|------|-----|-----------|-------------|
| `bool` | `0x20` | 1 | Boolean, packed 8 per byte |

A `bool` element represents a logical true or false value. Each boolean occupies a single bit. Booleans are packed 8 per byte using **LSB-first** (least significant bit first) order.

Packing rule: logical element at index `i` within a group of 8 is stored in bit `(i % 8)` of byte `floor(i / 8)`. Bit 0 is the least significant bit of the byte.

- A bit value of `0x1` represents **true**.
- A bit value of `0x0` represents **false**.

When the total number of boolean elements is not a multiple of 8, the remaining high-order bits in the final byte MUST be set to `0x0`.

**Example:** A 1-D boolean tensor with shape `[5]` and values `[true, false, true, true, false]` is stored as a single byte: `0x0D` (binary `00001101`). Bits 5, 6, and 7 are padding and MUST be zero.

> **Note (non-normative):** This packing convention is identical to the one used by Apache Arrow for boolean arrays.

## Tier 2 -- Extended Types

Tier 2 types are OPTIONAL. Conforming implementations MAY support any subset of Tier 2 types, including none. Implementations that do not support a given Tier 2 type MUST still reject (or, in permissive mode, skip) descriptors using that type tag according to the rules in the Type Identifier Encoding section.

### Float8 Variants

| Type | Tag | Bit Width | Format | Description |
|------|-----|-----------|--------|-------------|
| `float8_e4m3` | `0x40` | 8 | 1-4-3 | IEEE-style: 1 sign, 4 exponent, 3 mantissa |
| `float8_e5m2` | `0x41` | 8 | 1-5-2 | IEEE-style: 1 sign, 5 exponent, 2 mantissa |
| `float8_e8m0` | `0x42` | 8 | 0-8-0 | Exponent-only: 8 exponent bits, no sign, no mantissa |

`float8_e4m3` and `float8_e5m2` follow the OCP (Open Compute Project) 8-bit Floating Point Specification (OFP8). Each occupies exactly 1 byte; byte order is not applicable.

**float8_e4m3 (`0x40`)**: 1 sign bit, 4 exponent bits, 3 mantissa bits. Exponent bias is 7. NaN is represented by the bit patterns `0x7F` and `0xFF` (all exponent and mantissa bits set). There are no infinity representations.

**float8_e5m2 (`0x41`)**: 1 sign bit, 5 exponent bits, 2 mantissa bits. Exponent bias is 15. This format supports infinities (`0x7C` and `0xFC`) and NaN values (exponent all ones, mantissa non-zero).

**float8_e8m0 (`0x42`)**: 8 exponent bits, no sign bit, no mantissa bits. This is a power-of-two scale factor format. The value is `2^(bits - 127)` where `bits` is the unsigned 8-bit integer in the range `[0x01, 0xFE]`. The bit patterns `0x00` and `0xFF` are reserved (NaN per OCP MX v1.0 § 5.6) and MUST NOT be used as data values. A reader encountering `0x00` or `0xFF` in a `float8_e8m0` buffer MUST treat the containing descriptor as invalid.

> **Note (non-normative):** `float8_e8m0` is primarily used as a scale factor type in microscaling (MX) quantization formats, not as a general-purpose element type.

Implementations MUST preserve all float8 bit patterns exactly during zero-copy interchange.

### Sub-Byte Floating-Point Types

| Type | Tag | Bit Width | Format | Description |
|------|-----|-----------|--------|-------------|
| `float4_e2m1` | `0x43` | 4 | 1-2-1 | OCP MX: 1 sign, 2 exponent, 1 mantissa |
| `float6_e2m3` | `0x44` | 6 | 1-2-3 | OCP MX: 1 sign, 2 exponent, 3 mantissa |
| `float6_e3m2` | `0x45` | 6 | 1-3-2 | OCP MX: 1 sign, 3 exponent, 2 mantissa |

**float4_e2m1 (`0x43`)**: 1 sign bit, 2 exponent bits, 1 mantissa bit. Exponent bias is 1. This is the MXFP4 format defined in the OCP Microscaling (MX) Specification. Each element occupies 4 bits.

`float4_e2m1` uses the same LSB-first 4-bit packing as `int4` (see § Sub-Byte Integer Types § 4-bit packing). When the total element count is odd, the high nibble of the final byte MUST be set to `0x0`.

The complete value map, per OCP MX v1.0 § 5.2, is:

| Bit pattern | Value |
|-------------|-------|
| `0x0` (`0b0000`) | +0.0 |
| `0x8` (`0b1000`) | -0.0 |
| `0x1`–`0x3` | positive subnormals: `(mantissa / 2) * 2^(1-bias)` = `0.5`, `1.0` |
| `0x9`–`0xB` | negative subnormals |
| `0x4`–`0x7` | positive normals: `(1 + mantissa / 2) * 2^(exponent - bias)` |
| `0xC`–`0xF` | negative normals |

Maximum representable value: `1.5 * 2^2 = 6.0`. There are no infinity or NaN representations; values outside `[-6.0, 6.0]` MUST be clamped by hardware.

Implementations MUST preserve all `float4_e2m1` bit patterns exactly during zero-copy interchange.

> **Note (non-normative):** `float4_e2m1` has native Tensor Core support on NVIDIA Blackwell (B100/B200) and is used in production quantized LLM inference. It is typically paired with `float8_e8m0` block scales under the MX quantization scheme (see `quantization.md`).

**float6_e2m3 (`0x44`) and float6_e3m2 (`0x45`)**: Both are OCP MX Specification 6-bit floating-point formats. Each element occupies 6 bits.

`float6_e2m3`: 1 sign bit, 2 exponent bits, 3 mantissa bits. Exponent bias is 1. Maximum representable value: `(1 + 7/8) * 2^(3-1) = 3.75`. Zero is represented by all-zero exponent and mantissa (sign-preserving). No infinity or NaN representations; out-of-range values MUST be clamped. Subnormals: exponent all-zero, mantissa non-zero → value = `(mantissa/8) * 2^(1-bias)`. Normative bit-pattern → value mapping per OCP MX v1.0 § 5.3.

`float6_e3m2`: 1 sign bit, 3 exponent bits, 2 mantissa bits. Exponent bias is 3. Maximum representable value: `(1 + 3/4) * 2^(7-3) = 28.0`. Zero is represented by all-zero exponent and mantissa (sign-preserving). No infinity or NaN representations; out-of-range values MUST be clamped. Subnormals: exponent all-zero, mantissa non-zero → value = `(mantissa/4) * 2^(1-bias)`. Normative bit-pattern → value mapping per OCP MX v1.0 § 5.4.

#### 6-bit packing

Four 6-bit elements are packed into 3 bytes using **LSB-first** order across byte boundaries. Given four elements at logical indices `4k`, `4k+1`, `4k+2`, `4k+3` stored in bytes `B0`, `B1`, `B2`:

| Element | Bits occupied |
|---------|--------------|
| `4k+0` | bits [5:0] of `B0` |
| `4k+1` | bits [7:6] of `B0`, bits [3:0] of `B1` |
| `4k+2` | bits [7:4] of `B1`, bits [1:0] of `B2` |
| `4k+3` | bits [7:2] of `B2` |

When the total number of elements is not a multiple of 4, the unused high-order bits in the final group of 3 bytes MUST be set to `0x0`.

Buffer size in bytes for `N` elements: `ceil(N * 6 / 8)` = `ceil(N / 4) * 3`.

**Worked example.** Four 6-bit elements with bit patterns `0b000001`, `0b000010`, `0b000100`, `0b001000` (logical indices `0`, `1`, `2`, `3`) pack into bytes `B0`, `B1`, `B2` as follows.

| Source | Bits taken | Destination | Resulting byte bits |
|--------|-----------|-------------|---------------------|
| element 0 = `0b000001` | all 6 bits | `B0[5:0]` | `B0[5:0] = 000001` |
| element 1 = `0b000010` | bits [1:0] = `0b10` | `B0[7:6]` | `B0[7:6] = 10` |
| element 1 = `0b000010` | bits [5:2] = `0b0000` | `B1[3:0]` | `B1[3:0] = 0000` |
| element 2 = `0b000100` | bits [3:0] = `0b0100` | `B1[7:4]` | `B1[7:4] = 0100` |
| element 2 = `0b000100` | bits [5:4] = `0b00` | `B2[1:0]` | `B2[1:0] = 00` |
| element 3 = `0b001000` | all 6 bits | `B2[7:2]` | `B2[7:2] = 001000` |

Assembling each byte (MSB on the left):

```
B0 = 10_000001 = 0x81
B1 = 0100_0000 = 0x40
B2 = 001000_00 = 0x20
```

The 3-byte packed group on the wire is therefore `0x81 0x40 0x20`.

Implementations MUST preserve all `float6_e2m3` and `float6_e3m2` bit patterns exactly during zero-copy interchange.

> **Note (non-normative):** `float6_e2m3` and `float6_e3m2` are defined in the OCP MX specification and intended for use with MX block quantization (see `quantization.md`). Hardware adoption is currently limited compared to MXFP4.

### Extended Floating-Point Types

| Type | Tag | Bit Width | Description |
|------|-----|-----------|-------------|
| `float128` | `0x46` | 128 | IEEE 754 binary128 (quad precision) |

**float128 (`0x46`)**: 1 sign bit, 15 exponent bits, 112 significand bits. Exponent bias is 16383. Total width is 128 bits (16 bytes). The sixteen bytes MUST be stored in little-endian order.

All IEEE 754 binary128 bit patterns are valid, including positive and negative zero, infinities, and NaN values. Implementations MUST preserve the bit pattern exactly during zero-copy interchange.

> **Note (non-normative):** `float128` is rarely used in ML inference. It is included for high-precision scientific computing workloads (e.g., physics simulations, climate modelling) that may share tensor data with inference pipelines via the array database use case (Core Property 10).

### Sub-Byte Integer Types

| Type | Tag | Bit Width | Signed | Range |
|------|-----|-----------|--------|-------|
| `int4` | `0x48` | 4 | yes | -8 to 7 |
| `uint4` | `0x49` | 4 | no | 0 to 15 |
| `int2` | `0x4A` | 2 | yes | -2 to 1 |
| `uint2` | `0x4B` | 2 | no | 0 to 3 |

Sub-byte integer types are packed into bytes using **LSB-first** order, analogous to the boolean packing rule.

#### 4-bit packing

Two 4-bit elements are packed per byte. The element at even logical index `2k` occupies bits [3:0] (the low nibble) of byte `k`. The element at odd logical index `2k+1` occupies bits [7:4] (the high nibble) of byte `k`.

**`int4`** values use two's complement representation within 4 bits. The valid bit patterns are `0x0` through `0xF`, representing values -8 (`0x8`) through 7 (`0x7`).

**`uint4`** values are unsigned. The valid bit patterns are `0x0` through `0xF`, representing values 0 through 15.

When the total number of 4-bit elements is odd, the high nibble of the final byte MUST be set to `0x0`.

**Example:** A 1-D `uint4` tensor with shape `[3]` and values `[5, 12, 3]` is stored as two bytes. Byte 0: element 0 in low nibble, element 1 in high nibble = `0xC5`. Byte 1: element 2 in low nibble, padding in high nibble = `0x03`.

#### 2-bit packing

Four 2-bit elements are packed per byte. The element at logical index `4k + j` (where `0 <= j < 4`) occupies bits `[2j+1 : 2j]` of byte `k`.

| Position in byte | Logical index offset | Bits |
|-----------------|---------------------|------|
| 0 | `4k + 0` | [1:0] |
| 1 | `4k + 1` | [3:2] |
| 2 | `4k + 2` | [5:4] |
| 3 | `4k + 3` | [7:6] |

**`int2`** values use two's complement representation within 2 bits. The valid bit patterns are `0b00` (0), `0b01` (1), `0b10` (-2), `0b11` (-1).

**`uint2`** values are unsigned. The valid bit patterns are `0b00` (0), `0b01` (1), `0b10` (2), `0b11` (3).

When the total number of 2-bit elements is not a multiple of 4, the unused high-order bits in the final byte MUST be set to `0x0`.

**Example:** A 1-D `uint2` tensor with shape `[3]` and values `[3, 1, 2]` is stored as one byte. Element 0 in bits [1:0] = `0b11`, element 1 in bits [3:2] = `0b01`, element 2 in bits [5:4] = `0b10`, padding in bits [7:6] = `0b00`. Result: `0x27` (binary `00100111`).

### Complex Types

| Type | Tag | Bit Width | Description |
|------|-----|-----------|-------------|
| `complex64` | `0x50` | 64 | Two `float32` values (real, imaginary) |
| `complex128` | `0x51` | 128 | Two `float64` values (real, imaginary) |

A `complex64` element consists of two consecutive `float32` values: the real part followed by the imaginary part. Total width is 64 bits (8 bytes). Each constituent `float32` MUST be stored in little-endian order.

A `complex128` element consists of two consecutive `float64` values: the real part followed by the imaginary part. Total width is 128 bits (16 bytes). Each constituent `float64` MUST be stored in little-endian order.

> **Note (non-normative):** Complex types are included for signal processing and scientific computing workloads that may share tensor data with ML inference pipelines. They are not commonly needed in LLM inference.

## Type Properties Summary

The following table summarizes all defined types and their properties.

| Type | Tag | Tier | Bit Width | Byte Width | Sub-Byte | Alignment (bytes) |
|------|-----|------|-----------|------------|----------|-------------------|
| `float16` | `0x01` | 1 | 16 | 2 | no | 2 |
| `bfloat16` | `0x02` | 1 | 16 | 2 | no | 2 |
| `float32` | `0x03` | 1 | 32 | 4 | no | 4 |
| `float64` | `0x04` | 1 | 64 | 8 | no | 8 |
| `int8` | `0x10` | 1 | 8 | 1 | no | 1 |
| `uint8` | `0x11` | 1 | 8 | 1 | no | 1 |
| `int16` | `0x12` | 1 | 16 | 2 | no | 2 |
| `uint16` | `0x13` | 1 | 16 | 2 | no | 2 |
| `int32` | `0x14` | 1 | 32 | 4 | no | 4 |
| `uint32` | `0x15` | 1 | 32 | 4 | no | 4 |
| `int64` | `0x16` | 1 | 64 | 8 | no | 8 |
| `uint64` | `0x17` | 1 | 64 | 8 | no | 8 |
| `bool` | `0x20` | 1 | 1 | n/a | yes | 1 |
| `float8_e4m3` | `0x40` | 2 | 8 | 1 | no | 1 |
| `float8_e5m2` | `0x41` | 2 | 8 | 1 | no | 1 |
| `float8_e8m0` | `0x42` | 2 | 8 | 1 | no | 1 |
| `float4_e2m1` | `0x43` | 2 | 4 | n/a | yes | 1 |
| `float6_e2m3` | `0x44` | 2 | 6 | n/a | yes | 1 |
| `float6_e3m2` | `0x45` | 2 | 6 | n/a | yes | 1 |
| `float128` | `0x46` | 2 | 128 | 16 | no | 16 |
| `int4` | `0x48` | 2 | 4 | n/a | yes | 1 |
| `uint4` | `0x49` | 2 | 4 | n/a | yes | 1 |
| `int2` | `0x4A` | 2 | 2 | n/a | yes | 1 |
| `uint2` | `0x4B` | 2 | 2 | n/a | yes | 1 |
| `complex64` | `0x50` | 2 | 64 | 8 | no | 4 |
| `complex128` | `0x51` | 2 | 128 | 16 | no | 8 |

The **Alignment** column specifies the minimum alignment requirement of the natural element type. This is the minimum alignment of an individual element within a contiguous buffer; the buffer itself has a separate, stricter alignment requirement (see `buffer-protocol.md`).

For sub-byte types, the alignment refers to the packed byte granularity: data for sub-byte types MUST start at a byte boundary within the buffer.

Tag `0x47` is reserved for future assignment by this specification. Implementations MUST NOT use tag `0x47`. Private extensions MUST NOT assign tag `0x47`.

> **Note (non-normative):** Tag `0x47` does not appear in the table because it is intentionally reserved. It is held for a future Tier 2 type whose assignment will be defined in a later revision.

> **Note (non-normative):** The alignment column for complex types reflects the natural alignment of the constituent floating-point element: `complex64` lists alignment `4` (per `float32` half) and `complex128` lists alignment `8` (per `float64` half). This matches the natural alignment of the constituent element type. Consumers loading a full complex value as a single 128-bit SIMD register must account for the fact that only the buffer-level alignment (64 bytes minimum, see `buffer-protocol.md`) guarantees register-width alignment, not the element-level alignment.

## Buffer Size Calculation

For whole-byte types with bit width `W >= 8`, the minimum buffer size in bytes for a contiguous tensor with `N` total elements is:

```
buffer_size = N * (W / 8)
```

For sub-byte types with bit width `B < 8`, the general minimum buffer size in bytes for a contiguous tensor with `N` total elements is:

```
buffer_size = ceil(N * B / 8)
```

where `ceil` denotes the ceiling function (rounding up to the next integer).

For sub-byte bit widths `B ∈ {1, 2, 4}` (i.e. `bool`, `int2`/`uint2`, `int4`/`uint4`/`float4_e2m1`), `8 / B` is an integer packing factor `P` and the formula reduces to the equivalent expression `ceil(N / P)`. For `B = 6` (`float6_e2m3` and `float6_e3m2`), the general form `ceil(N * 6 / 8) = ceil(N / 4) * 3` MUST be used, because four 6-bit elements pack into three bytes rather than into a whole number of elements per byte; see § Sub-Byte Floating-Point Types § 6-bit packing.

These formulas apply to contiguous (dense) layouts. For strided layouts, the buffer size depends on the strides; see `memory-layout.md`.

## Interaction with Other Sections

- **Quantization (`quantization.md`)**: Quantized tensors use an element type from this section as their **storage type** and attach a quantization descriptor that defines the dequantization mapping. The storage type tag in the tensor descriptor always refers to a type defined here.
- **Memory Layout (`memory-layout.md`)**: Stride semantics for sub-byte types require special treatment. Strides are expressed in logical elements; the packing rules in this section define how logical element indices map to bit positions within the buffer.
- **Metadata (`metadata.md`)**: The type tag defined here is stored as a `uint8` field in the binary tensor descriptor.

## Relationship to the Python Array API

> **Note (non-normative):** The Tier 1 numeric types — `bool`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`, `float16`, `bfloat16`, `float32`, `float64`, `complex64`, `complex128` — overlap with the dtype vocabulary defined by the Python Array API Standard (data-apis.org/array-api). This alignment is intentional: Hurray tensors carrying Tier 1 element types can be exposed to Python Array API consumers without dtype translation.

> **Note (non-normative):** Tier 2 types (`float8_e4m3`, `float8_e5m2`, `float8_e8m0`, `float4_e2m1`, `float6_e2m3`, `float6_e3m2`, `float128`, `int4`, `uint4`, `int2`, `uint2`) have no counterpart in the Python Array API Standard and are exposed as `hurray`-namespaced dtype objects in the Python bindings. Quantized tensor types (see `quantization.md`) are similarly out of scope for the Array API. Requirements for the Python bindings are defined in [`docs/impl/python-bindings`](../impl/python-bindings.md).

## Open Questions

> **[OQ-1]:** ~~Should `float8_e4m3` follow the OCP OFP8 convention (no infinities, two NaN values) or the IEEE 754 draft for binary8 (which may differ)?~~ **Resolved:** `float8_e4m3` normatively follows OCP OFP8 (no infinities, two NaN bit patterns, exponent bias 7), matching production hardware (NVIDIA H100/H200, AMD MI300). If IEEE 754 binary8 diverges when finalized, a separate type tag will be assigned rather than redefining this one.

> **[OQ-2]:** ~~Should the specification define a `float128` (IEEE 754 binary128) type?~~ **Resolved:** `float128` is added as a Tier 2 type with tag `0x46`. Note: the originally proposed tag `0x05` falls in the Tier 1 range (`0x01`–`0x3F`) and was corrected to `0x46` (Tier 2 range `0x40`–`0x7F`). Rationale: high-precision scientific computing workloads sharing tensor data with inference pipelines via the array database use case (Core Property 10).

> **[OQ-3]:** ~~Should the private extension range (`0xF0` -- `0xFE`) require implementations to include a type descriptor (name, bit width) in the tensor metadata so that readers can at least compute buffer sizes for unknown types?~~ **Resolved by ADR-001:** Private extension type tags MUST carry an inline descriptor encoding at minimum the bit width, packing, and floating-point parameters (sign/exponent/mantissa widths, exponent bias, NaN/Inf flags). See `docs/adr/ADR-001-private-extension-type-descriptors.md`. The descriptor binary encoding will be defined in `metadata.md`.

> **[OQ-4]:** ~~Should `float4_e2m1` (MXFP4) be added as a Tier 2 type?~~ **Resolved:** `float4_e2m1` is added as Tier 2 with tag `0x43`. Packing follows the LSB-first `int4` convention (two elements per byte). Rationale: native Tensor Core support on NVIDIA Blackwell and production use in quantized LLM inference.

> **[OQ-5]:** ~~Should `float6_e2m3` and `float6_e3m2` be added as Tier 2 types?~~ **Resolved:** Both are added as Tier 2 with tags `0x44` and `0x45` respectively. Packing: 4 elements per 3 bytes, LSB-first across byte boundaries. Buffer size: `ceil(N / 4) * 3` bytes.
