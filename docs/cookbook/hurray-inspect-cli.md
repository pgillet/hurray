# hurray-inspect CLI

## Purpose

`hurray-inspect` is a diagnostic CLI tool that decodes a Hurray binary tensor descriptor and
prints a 3-column hex table showing the byte offset, raw hex value, and field name of every
field in the descriptor.

It is useful for:

- Verifying that a hand-crafted or encoded descriptor matches the spec.
- Debugging format mismatches between implementations.
- Learning the wire format interactively.

Parsing is delegated entirely to `hurray-core`'s `TensorDescriptor::decode()` — the tool
never implements its own format logic.

## Building

```bash
cargo build -p hurray-inspect
# binary lands at target/debug/hurray-inspect
```

For a release build:

```bash
cargo build --release -p hurray-inspect
```

## Usage

```text
hurray-inspect <file>      # inspect a file on disk
hurray-inspect -           # read from stdin
```

## Inspecting a file

Write a small Rust program (or use the worked example in `hurray-core`) to produce a binary
descriptor, save it to disk, then pass it to `hurray-inspect`:

```rust
// src/bin/write_example.rs (or any scratch binary)
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Shape, SyncMode, MIN_BUFFER_ALIGNMENT,
    descriptor::TensorDescriptor,
    layout::LayoutDescriptor,
};
use std::fs;

fn main() {
    let shape  = Shape::new(vec![3u64, 4]).unwrap();
    let buffer = BufferHandle::new(
        192, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
    ).unwrap();
    let desc = TensorDescriptor::new(
        1, 0, ElementType::Float32, shape, 0,
        LayoutDescriptor::RowMajor, vec![buffer],
        None, None, None, None,
    ).unwrap();

    fs::write("example.hrry", desc.encode().unwrap()).unwrap();
    println!("wrote example.hrry ({} bytes)", desc.encode().unwrap().len());
}
```

Then inspect it:

```bash
hurray-inspect example.hrry
```

Output:

```
Offset  Value (hex)                     Field
------  ------------------------------  -----
     0  48 52 52 59                     magic = "HRRY"
     4  01                              version_major = 1
     5  00                              version_minor = 0
     6  3D 00 00 00                     descriptor_length = 61
    10  00 00 00 00                     flags = 0x00000000
    14  03                              type_tag = 0x03 (float32)
    15  01                              layout_tag = 0x01 (row-major)
    16  02 00 00 00                     rank = 2
    20  03 00 00 00 00 00 00 00         shape[0] = 3
    28  04 00 00 00 00 00 00 00         shape[1] = 4
    36  00 00 00 00 00 00 00 00         byte_offset = 0
    44  01                              buffer_count = 1
    45  C0 00 00 00 00 00 00 00         buffer[0].byte_size = 192
    53  40 00 00 00                     buffer[0].alignment = 64
    57  00                              buffer[0].device_tag = 0x00 (cpu)
    58  00                              buffer[0].sync_mode = 0x00 (producer_synced)
    59  00 00                           buffer[0]._reserved
```

## Reading from stdin

Pipe raw bytes directly — useful for scripting or combining with other tools:

```bash
# Pipe the spec's 61-byte worked example (little-endian hex literals)
printf '\x48\x52\x52\x59\x01\x00\x3D\x00\x00\x00\x00\x00\x00\x00\x03\x01' \
       '\x02\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00' \
       '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\xC0\x00\x00' \
       '\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00' \
  | hurray-inspect -
```

## Optional sections

When a descriptor carries optional sections (quantization, shard, statistics, or extension
type), those fields appear after the buffer table in spec-mandated order.

Example with statistics attached (flags bit 3 set):

```
    ...
    44  01                              buffer_count = 1
    45  C0 00 00 00 00 00 00 00         buffer[0].byte_size = 192
    53  40 00 00 00                     buffer[0].alignment = 64
    57  00                              buffer[0].device_tag = 0x00 (cpu)
    58  00                              buffer[0].sync_mode = 0x00 (producer_synced)
    59  00 00                           buffer[0]._reserved
    61  08 00 00 00                     flags = 0x00000008     ← HAS_STATISTICS
    ...
    XX  04 00 00 00                     stats.computed_mask = 0x00000004
    XX  00 00 00 00                     stats._reserved
    XX  ...                             stats.nnz, sparsity_ratio, value_min/max, ...
```

## Error output

When a descriptor is malformed, `hurray-inspect` prints whatever it could read before the
failure, then an error row and a message on stderr:

```bash
echo -n "BAAD" | hurray-inspect -
```

```
Offset  Value (hex)                     Field
------  ------------------------------  -----
     0  42 41 41 44                     magic = "BAAD"
                                        ERROR: invalid magic bytes
error: invalid magic bytes
```

Exit code is `1` on any parse or I/O error, `0` on success.

## Relationship to hurray-core

`hurray-inspect` contains no format parsing logic of its own. It calls
`TensorDescriptor::decode()` from `hurray-core` and then walks the original byte slice a
second time — guided by the decoded struct's field values — to annotate each byte range for
display. This means:

- Any format change in `hurray-core` is automatically reflected in `hurray-inspect`.
- An unrecognised layout tag produces an `Unknown` variant; its raw bytes are shown as a
  single opaque hex block.
- Future minor-version additions (bytes beyond the fields this version understands) appear
  as a `(unknown / padding bytes)` row at the end.
