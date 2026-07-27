# Composite Tensors in Files

## Purpose

A [composite tensor](composite-tensors.md) — a data-less head plus an ordered set of member
tensors — is stored in a Hurray file as its head followed by each member's descriptor and
data, written **contiguously and in order**. Every tensor — the head and each member — gets
its own footer-index entry, so all are individually addressable by name. Membership is
recovered from the head's `member_count` plus **file-offset adjacency**: the members are the
tensors written immediately after the head (ADR-027 § Binding; `docs/spec/file-format.md`).

`hurray-io` provides a matched pair:

- `FileWriter::write_composite` — validates the whole group up front (member count,
  partition exact-cover, overlay ordering) via `hurray-core`'s `CompositeValidator`, then
  writes the head and members. Nested composites are written recursively.
- `FileReader::read_composite` — reassembles a head with its members (recursively for
  nesting) and validates the group, returning a `FileComposite`.

## Writing a composite to a file

Each node carries a **name** because every tensor gets its own index entry:

```rust,no_run
use hurray_core::{
    layout::{CompositeLayout, CompositionRule},
    ElementType, LayoutDescriptor, Shape, TensorDescriptor,
};
use hurray_io::file::{FileCompositeNode, FileWriter};

# async fn run(left: TensorDescriptor, right: TensorDescriptor,
#              left_data: Vec<u8>, right_data: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
let head = TensorDescriptor::new(
    1, 0, ElementType::Float32, Shape::new(vec![8u64, 8])?, 0,
    LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2)?),
    vec![], None, None, None, None,
)?;

let left_buffers: [&[u8]; 1] = [left_data.as_slice()];
let right_buffers: [&[u8]; 1] = [right_data.as_slice()];
let members = vec![
    FileCompositeNode::Tensor { name: "weight.left",  descriptor: &left,  buffers: &left_buffers },
    FileCompositeNode::Tensor { name: "weight.right", descriptor: &right, buffers: &right_buffers },
];

let file = tokio::fs::File::create("model.hrry").await?;
let mut writer = FileWriter::new(file).await?;
writer.write_composite("weight", &head, &members).await?; // validated before any tensor is written
writer.finish(vec![]).await?;
# Ok(())
# }
```

## Reading a composite from a file

`read_composite` takes the head's name and returns the reassembled group. Members remain
individually readable by name with `read_tensor`:

```rust,no_run
use hurray_io::file::{FileItem, FileReader};

# async fn run(file: tokio::fs::File) -> Result<(), Box<dyn std::error::Error>> {
let mut reader = FileReader::open(file).await?;

// Whole composite, grouped and validated:
let composite = reader.read_composite("weight").await?;
println!("head {:?}, {} member(s)", composite.head.shape.dims(), composite.members.len());
for member in &composite.members {
    match member {
        FileItem::Tensor(t) => println!("  member {}: {} buffer(s)", t.name, t.buffers.len()),
        FileItem::Composite(c) => println!("  nested composite {}", c.name), // recurses
    }
}

// Or just one member, by name:
let left = reader.read_tensor("weight.left").await?;
# Ok(())
# }
```

## Recovery is independent of index sort order

The file writer's `sorted_index` option sorts the *index array* by name for binary search,
but the tensors' positions in the file are unchanged. `read_composite` recovers membership
by **descriptor offset**, so it returns the members in file (write) order regardless of how
the index is sorted.

## Nested composites and the depth guard

A member may itself be a composite; `read_composite` reassembles the tree recursively. The
recursion is bounded by `FileReader::with_max_composite_depth` (default 64) to guard against
a maliciously deep composite.

## Errors

- `Error::NotAComposite` — the named tensor exists but its head is not a composite.
- `Error::TornComposite` — fewer tensors follow the head than its `member_count` declares.
- `Error::CompositeNestingTooDeep` — nesting exceeded the configured maximum.
- `Error::Core` — composite validation failed (member-count mismatch, partition coverage,
  overlay ordering).

## Runnable example

```text
cargo run --example composite_file --features tokio -p hurray-io
```

See `hurray-io/examples/composite_file.rs` for the full program.
