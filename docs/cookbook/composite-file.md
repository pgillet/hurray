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

<div class="lang-tabs">

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

```python
import hurray

# A composite is one named entry, like any other tensor.
hurray.save("model.hrry", {"weight": weight, "bias": bias})
```

```c
/* The same preorder array the stream surface takes, with the names filled in: a file
   gives every tensor its own index entry, so each node needs one. */
HurrayCompositeNode nodes[3] = {
    { .kind = HURRAY_ITEM_COMPOSITE, .name = "weight", .name_len = 6,
      .descriptor = head, .member_count = 2 },
    { .kind = HURRAY_ITEM_TENSOR, .name = "weight.left", .name_len = 11,
      .descriptor = left,  .payloads = &left_slice,  .payload_count = 1 },
    { .kind = HURRAY_ITEM_TENSOR, .name = "weight.right", .name_len = 12,
      .descriptor = right, .payloads = &right_slice, .payload_count = 1 },
};

HurrayStatus s = hurray_file_writer_write_composite(writer, nodes, 3);
/* A group spans several index entries, so a failure part-way through leaves a head with
   fewer members after it than it declares — not a file any reader may open. The writer is
   poisoned rather than left to write a clean index over it. */
if (s < 0) return -1;
hurray_file_writer_finish(writer, NULL, 0);
```

</div>

Every tensor in a file gets an index entry, head and member alike, so a composite's
members are named `"{head}.{index}"` — `weight.0`, `weight.1`. Those names are an artifact
of the container rather than of the composite, so Python generates them for you.

## Reading a composite from a file

`read_composite` takes the head's name and returns the reassembled group. Members remain
individually readable by name with `read_tensor`:

<div class="lang-tabs">

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

```python
import hurray

loaded = hurray.load("model.hrry")

loaded["weight"]            # a hurray.Composite
loaded["weight"].members    # its tiles, in write order

# The head's name is the only top-level entry: on the wire the members belong
# to it, so they do not also come back on their own. Asking still works:
tile = hurray.load("model.hrry", names=["weight.0"])["weight.0"]
```

```c
HurrayItem *item = NULL;
HurrayStatus s = hurray_file_reader_read_composite(reader, "weight", 6, &item);
if (s == HURRAY_ERR_NOT_COMPOSITE) { /* the name is in the index, but it is a tensor */ }
if (s == HURRAY_ERR_NOT_FOUND)     { /* no such name */ }
if (s < 0) return -1;

uint64_t count = 0;
hurray_item_node_count(item, &count);
for (uint64_t i = 0; i < count; i++) {
    HurrayItemNode node = {0};
    hurray_item_node(item, i, &node);
    /* Named, unlike on a stream: every tensor of the group has its own index entry.
       The name is UTF-8 and NOT NUL-terminated, so name_len is not optional. */
    printf("%.*s: %u member(s)\n", (int)node.name_len, (const char *)node.name,
           node.member_count);
}

hurray_item_destroy(&item);   /* one call for the whole subtree */

/* Or just one member, by name — a composite's members stay individually addressable. */
HurrayTensor *left = NULL;
hurray_file_reader_read_tensor(reader, "weight.left", 11, &left);
```

</div>

The C walk is the same loop the [stream page](composite-streaming.md) shows: nodes in
preorder, nesting in the `member_count` column. One shape for both surfaces, because a
composite read out of a file is the same tree as one read off a wire.

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

- `Error::NotAComposite` — the named tensor exists but its head is not a composite. In C
  this is `HURRAY_ERR_NOT_COMPOSITE`, its own status for the reason `HURRAY_ERR_NOT_FOUND`
  has one: it is a question the caller asked wrongly, not a defect in the file.
- `Error::TornComposite` — fewer tensors follow the head than its `member_count` declares.
- `Error::CompositeNestingTooDeep` — nesting exceeded the configured maximum.
- `Error::Core` — composite validation failed (member-count mismatch, partition coverage,
  overlay ordering).

## Runnable example

<div class="lang-tabs">

```text
cargo run --example composite_file --features tokio -p hurray-io
```

```text
python hurray-python/examples/composites.py
```

```text
cargo run --example composite_vtable -p hurray-ffi
```

</div>

See `hurray-io/examples/composite_file.rs`, `hurray-python/examples/composites.py`, and
`hurray-ffi/examples/composite_vtable.rs` for the full programs. The last one writes the
same composite to a stream and to a file through the C entry points, which is the shortest
way to see what the two surfaces share and where they differ.
