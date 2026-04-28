# Per-Tensor Affine Quantization — Hurray Format Specification

**Scheme tag:** `0x01` | **Tier:** 1

> This section uses RFC 2119 key words: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT,
> SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

## Description

A single `scale` and `zero_point` pair applies to every element of the tensor.
This scheme covers both asymmetric quantization (arbitrary `zero_point`) and
symmetric quantization (`zero_point = 0`).

The scale and zero-point are stored inline in the quantization descriptor; no
additional buffer table entries are required beyond the tensor data buffer.

## Binary Encoding

Total descriptor length: **16 bytes**.

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0 | `scheme_tag` | `uint8` | MUST be `0x01`. |
| 1 | `scheme_version` | `uint8` | MUST be `0x01`. For the version compatibility policy, see `quantization.md` § Version Compatibility. |
| 2 | `flags` | `uint16` | MUST be `0x0000`. No flags are defined for this scheme. |
| 4 | `scale` | `float32` | Dequantization scale. MUST be a finite, non-zero value. |
| 8 | `zero_point` | `int32` | Quantization zero point. For symmetric quantization, MUST be `0x00000000`. |
| 12 | `_reserved` | `uint8[4]` | MUST be `0x00`. |

All multi-byte fields MUST be encoded in little-endian byte order.

## Dequantization Formula

For each storage element `q`:

```
x_real = scale * (q - zero_point)
```

The subtraction is performed in signed 32-bit integer arithmetic. The result
MUST then be converted to `float32` and multiplied by `scale`. The real-valued
element type produced by dequantization is `float32`. A consumer MAY further
convert to `float64` or to a lower-precision float type; such conversion is out
of scope for this specification.

## Validity Constraints

- `scale` MUST NOT be zero, NaN, or infinity. A reader MUST reject a descriptor
  that violates this constraint.
- `zero_point` MUST lie within the representable range of the storage type. For
  example, for a `uint8` storage type, `zero_point` MUST be in `[0, 255]`. A
  reader MUST reject a descriptor that violates this constraint.
- The `_reserved` bytes MUST be `0x00`. A reader MUST reject a descriptor that
  violates this constraint.

## Valid Storage Types

The storage type (`type_tag` in the tensor descriptor) MUST be one of:

- `int8` (`0x10`), `uint8` (`0x11`)
- `int16` (`0x12`), `uint16` (`0x13`)
- `int32` (`0x14`), `uint32` (`0x15`)
- `int4` (`0x48`), `uint4` (`0x49`)
- `int2` (`0x4A`), `uint2` (`0x4B`)

A reader MUST reject a descriptor whose storage type is not in this list.
