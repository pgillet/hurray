# Layer 0: Element Types and Shape

## Purpose

`ElementType` defines the numeric storage format for tensor elements (float32, int4, bool, etc.). `Shape` describes the dimensions of a tensor. Together, they form the foundation of the Hurray data model: *what* is stored (element type) and *how many* (shape). The `buffer_size_bytes` function computes how much memory a tensor requires.

## ML Model with Mixed Precision

Suppose you're building an inference runtime for an LLM with quantization. Different layers use different types:

<div class="lang-tabs">

```rust
use hurray_core::{ElementType, Shape, buffer_size_bytes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Weight tensor: float16 (half precision)
    let weights_shape = Shape::new(vec![768u64, 3072])?;
    let weights_type = ElementType::Float16;
    let weights_elements = weights_shape.element_count().expect("no dynamic dims");
    let weights_bytes = buffer_size_bytes(weights_type, weights_elements);
    println!("Weights [768, 3072] as float16: {} bytes", weights_bytes);

    // Activation tensor: float32 (full precision for numerical stability)
    let activation_shape = Shape::new(vec![32u64, 768])?;
    let activation_type = ElementType::Float32;
    let activation_elements = activation_shape.element_count().expect("no dynamic dims");
    let activation_bytes = buffer_size_bytes(activation_type, activation_elements);
    println!("Activations [32, 768] as float32: {} bytes", activation_bytes);

    // Quantized layer: int4 (4-bit integers)
    let quantized_shape = Shape::new(vec![768u64, 1024])?;
    let quantized_type = ElementType::Int4;
    let quantized_elements = quantized_shape.element_count().expect("no dynamic dims");
    let quantized_bytes = buffer_size_bytes(quantized_type, quantized_elements);
    println!("Quantized layer [768, 1024] as int4: {} bytes", quantized_bytes);

    Ok(())
}
```

```python
import hurray

# Weight tensor: float16 (half precision)
weights = hurray.buffer_size_bytes(hurray.float16, 768 * 3072)
print(f"Weights [768, 3072] as float16: {weights} bytes")

# Activation tensor: float32 (full precision for numerical stability)
activations = hurray.buffer_size_bytes(hurray.float32, 32 * 768)
print(f"Activations [32, 768] as float32: {activations} bytes")

# Quantized layer: int4 (4-bit integers, two per byte)
quantized = hurray.buffer_size_bytes(hurray.dtype.int4, 768 * 1024)
print(f"Quantized layer [768, 1024] as int4: {quantized} bytes")
```

</div>

Output:
```
Weights [768, 3072] as float16: 4718592 bytes
Activations [32, 768] as float32: 98304 bytes
Quantized layer [768, 1024] as int4: 393216 bytes
```

## Dynamic Dimensions (Batch Size Unknown)

When the batch dimension is not known at model load time, mark it `DYNAMIC`:

<div class="lang-tabs">

```rust
use hurray_core::{Shape, DYNAMIC};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Batch size unknown; sequence length fixed at 512
    let shape = Shape::new(vec![DYNAMIC, 512u64, 768])?;
    
    println!("Shape: {}", shape);                    // [?, 512, 768]
    println!("Has dynamic: {}", shape.has_dynamic()); // true
    println!("Element count: {:?}", shape.element_count()); // None
    
    // At runtime, after batch size is resolved to (say) 8:
    let resolved_shape = Shape::new(vec![8u64, 512, 768])?;
    let elements = resolved_shape.element_count().expect("now static");
    println!("Resolved element count: {}", elements); // 3145728

    Ok(())
}
```

```python
import hurray

# Batch size unknown; sequence length fixed at 512.
signature = hurray.Tensor(b"", hurray.float32, [None, 512, 768])

assert signature.shape == (None, 512, 768)
assert signature.size is None          # an unknown extent, so an unknown count

# At runtime, once the batch size is resolved to (say) 8:
resolved = hurray.Tensor(bytes(4 * 8 * 512 * 768), hurray.float32, [8, 512, 768])
assert resolved.size == 3145728
```

Python spells a dynamic dimension `None` rather than exposing the wire sentinel, because
`Tensor.shape` already *returns* `None` for one — so a shape read off a tensor can be
handed straight back to the constructor. Anything that allocates (`zeros`, `ones`,
`empty`, `full`) refuses it: you cannot allocate an unknown number of bytes.

</div>

## Type Properties and Alignment

Query element type metadata:

<div class="lang-tabs">

```rust
use hurray_core::ElementType;

fn main() {
    let ty = ElementType::Float32;
    
    println!("Type: {}", ty);                         // float32
    println!("Wire tag: 0x{:02X}", ty.tag());         // 0x03
    println!("Bit width: {}", ty.bit_width());        // 32
    println!("Bytes per element: {}", ty.element_alignment()); // 4
    println!("Is float: {}", ty.is_float());          // true
    println!("Is integer: {}", ty.is_integer());      // false
    println!("Is signed: {}", ty.is_signed());        // true
    println!("Tier: {}", ty.tier());                  // 1 (core type)

    // Sub-byte types require special handling
    let int4 = ElementType::Int4;
    println!("\nType: {}", int4);                     // int4
    println!("Bit width: {}", int4.bit_width());      // 4
    println!("Is sub-byte: {}", int4.is_sub_byte());  // true
}
```

```python
import hurray

ty = hurray.float32

print(f"Type: {ty.name}")                          # float32
print(f"Wire tag: 0x{ty.tag:02X}")                 # 0x03
print(f"Bit width: {ty.bit_width}")                # 32
print(f"Element alignment: {ty.element_alignment}")  # 4
print(f"Is float: {ty.is_float}")                  # True
print(f"Is integer: {ty.is_integer}")              # False
print(f"Is signed: {ty.is_signed}")                # True
print(f"Tier: {ty.tier}")                          # 1 (core type)

# Sub-byte types require special handling
int4 = hurray.dtype.int4
print(f"Bit width: {int4.bit_width}")              # 4
print(f"Is sub-byte: {int4.is_sub_byte}")          # True
print(f"Element alignment: {int4.element_alignment}")  # 1 — two share a byte
```

`element_alignment` is the element's own alignment, not the buffer's: a `float32` buffer
starts on a `hurray.MIN_BUFFER_ALIGNMENT` boundary, but its elements are 4-aligned within
it. A packed element reports `1`, having no address of its own.

</div>

## Buffer Size Calculations

For different element types, the buffer size formula varies. `buffer_size_bytes` handles all cases:

<div class="lang-tabs">

```rust
use hurray_core::{ElementType, buffer_size_bytes};

fn main() {
    // Whole-byte types: element_count × byte_width
    println!("float32 × 100 elements: {} bytes",
        buffer_size_bytes(ElementType::Float32, 100)); // 400

    // 6-bit types: ceil(N/4) × 3
    println!("float6_e2m3 × 100 elements: {} bytes",
        buffer_size_bytes(ElementType::Float6E2M3, 100)); // 75

    // 4-bit types: ceil(N×4/8) = ceil(N/2)
    println!("int4 × 7 elements: {} bytes",
        buffer_size_bytes(ElementType::Int4, 7)); // 4

    // Boolean (1-bit): ceil(N/8)
    println!("bool × 9 elements: {} bytes",
        buffer_size_bytes(ElementType::Bool, 9)); // 2

    // Sub-byte types are packed; 0 elements always yields 0 bytes
    println!("any type × 0 elements: {} bytes",
        buffer_size_bytes(ElementType::Float32, 0)); // 0
}
```

```python
import hurray

# Whole-byte types: element_count x byte_width
assert hurray.buffer_size_bytes(hurray.float32, 100) == 400

# 6-bit types: ceil(N/4) x 3
assert hurray.buffer_size_bytes(hurray.dtype.float6_e2m3, 100) == 75

# 4-bit types: ceil(N x 4 / 8) = ceil(N/2)
assert hurray.buffer_size_bytes(hurray.dtype.int4, 7) == 4

# Boolean (1-bit): ceil(N/8)
assert hurray.buffer_size_bytes(hurray.bool, 9) == 2

# Sub-byte types are packed; 0 elements always yields 0 bytes
assert hurray.buffer_size_bytes(hurray.float32, 0) == 0
```

</div>

## Type Tag Round-Trip

Serialize and deserialize element types using the wire tag:

<div class="lang-tabs">

```rust
use hurray_core::ElementType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serialize: type to tag
    let original_type = ElementType::Float8E4M3;
    let tag = original_type.tag();
    println!("Serialized {} to tag 0x{:02X}", original_type, tag);

    // Deserialize: tag to type
    let recovered_type = ElementType::from_tag(tag)?;
    assert_eq!(original_type, recovered_type);
    println!("Deserialized tag 0x{:02X} back to {}", tag, recovered_type);

    Ok(())
}
```

```python
import hurray

# Serialize: type to tag
tag = hurray.dtype.float8_e4m3.tag
assert tag == 0x40

# Deserialize: tag to type — and back to the same object, not a copy
assert hurray.Dtype.from_tag(tag) is hurray.dtype.float8_e4m3
```

</div>

## Invalid Tags

Tags in reserved ranges are rejected:

<div class="lang-tabs">

```rust
use hurray_core::{ElementType, Error};

fn main() {
    // Permanently invalid sentinels
    assert!(matches!(ElementType::from_tag(0x00), Err(Error::InvalidTypeTag(0x00))));
    assert!(matches!(ElementType::from_tag(0xFF), Err(Error::InvalidTypeTag(0xFF))));

    // Reserved for future spec versions
    assert!(matches!(ElementType::from_tag(0x47), Err(Error::ReservedTypeTag(0x47))));
    assert!(matches!(ElementType::from_tag(0x80), Err(Error::ReservedTypeTag(0x80))));

    // The private-extension range is NOT an error: 0xF0-0xFE resolve to
    // ElementType::Extension(tag), whose semantics travel out of band.
    assert!(matches!(ElementType::from_tag(0xF0), Ok(ElementType::Extension(0xF0))));

    println!("All invalid tags correctly rejected");
}
```

```python
import hurray

for tag in (
    0x00, 0xFF,   # permanently invalid sentinels
    0x47, 0x80,   # reserved for future spec versions
):
    try:
        hurray.Dtype.from_tag(tag)
        raise AssertionError(f"0x{tag:02X} should not resolve")
    except hurray.InvalidDescriptorError as exc:
        print(exc)

# The private-extension range is not an error: it resolves to an extension type
# whose semantics travel out of band. The tag is what identifies it.
private = hurray.Dtype.from_tag(0xF0)
assert private.name == "extension"
assert private.tag == 0xF0
assert private != hurray.Dtype.from_tag(0xF5)
```

Python collapses the Rust error variants into `InvalidDescriptorError`, but the message
still says which case it was — and none of them is guessed at, since inventing a type for
a reserved tag turns "the producer is newer than this reader" into silent wrong data.

</div>

## Empty and Scalar Tensors

Hurray supports edge cases:

<div class="lang-tabs">

```rust
use hurray_core::{Shape, ElementType, buffer_size_bytes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Scalar tensor (rank 0): one element, no shape
    let scalar = Shape::scalar();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.dims(), &[]);
    assert_eq!(scalar.element_count(), Some(1));
    let scalar_bytes = buffer_size_bytes(ElementType::Float32, 1);
    println!("Scalar float32: {} bytes", scalar_bytes); // 4

    // Empty tensor: any zero dimension
    let empty = Shape::new(vec![5u64, 0, 10])?;
    assert!(empty.is_empty_tensor());
    assert_eq!(empty.element_count(), Some(0));
    let empty_bytes = buffer_size_bytes(ElementType::Float32, 0);
    println!("Empty tensor: {} bytes", empty_bytes); // 0

    Ok(())
}
```

```python
import hurray

# Scalar tensor (rank 0): one element, no shape
scalar = hurray.Tensor(bytes(4), hurray.float32, [])
assert scalar.ndim == 0
assert scalar.shape == ()
assert scalar.size == 1

# Empty tensor: any zero dimension
empty = hurray.Tensor(b"", hurray.float32, [5, 0, 10])
assert empty.size == 0
assert empty.buffer_handles[0].byte_size == 0
assert empty.buffer_handles[0].alignment == 1     # nothing to align
```

</div>

## Key Takeaways

- **ElementType** — an enum with 26 numeric types from Tier 1 (core) and Tier 2 (extended)
- **Shape** — a vector of `u64` dimension sizes, supporting dynamic (`DYNAMIC`) and zero-size dimensions
- **buffer_size_bytes()** — handles all packing rules (1-bit, 2-bit, 4-bit, 6-bit, and whole-byte types)
- Tags are serialized as `u8` in descriptors; use `from_tag()` / `tag()` for round-trip conversion
- **In Python**, a dynamic dimension is spelled `None` — the same thing `Tensor.shape` returns — and `hurray.buffer_size_bytes(dtype, count)` applies the same packing rules
- Scalar tensors have rank 0; empty tensors have 0 total elements but valid descriptors

See `docs/spec/element-types.md` and `docs/spec/data-model.md` for the normative specification.
