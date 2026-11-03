# Streaming Composite Tensors

## Purpose

A [composite tensor](composite-tensors.md) — a data-less head plus an ordered set of
member tensors — is streamed as its head followed by each member's descriptor and data, in
order. This "head precedes its members precede their data" rule is a *forward* promise: no
back-references and no end-of-file index, so a composite stays streamable exactly like an
ordinary tensor (ADR-027 § Binding; `docs/spec/interchange.md`).

`hurray-io` gives you a matched pair:

- `StreamWriter::write_composite` — validates the whole group up front (member count,
  partition exact-cover, overlay ordering), then writes the head and members. A torn or
  invalid composite never reaches the wire.
- `StreamReader::next_item` — reads the next item and, when it is a composite head,
  reassembles the head with its declared members (validating as it goes), returning a
  `StreamItem::Composite`. Members that are themselves composites are assembled recursively.

## Writing a composite

Build the head (a descriptor with a `Composite` layout and no buffers) and the members
(ordinary descriptors), then hand them to `write_composite` as `CompositeNode`s:

<div class="lang-tabs">

```rust,no_run
use hurray_core::{
    buffer_size_bytes,
    layout::{CompositeLayout, CompositionRule},
    ElementType, LayoutDescriptor, Shape, ShardDescriptor, TensorDescriptor,
};
use hurray_io::stream::{CompositeNode, StreamWriter};

# async fn run(left: TensorDescriptor, right: TensorDescriptor,
#              left_data: Vec<u8>, right_data: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
// A partition head presenting one logical [8, 8] float32 view over two [8, 4] tiles.
let head = TensorDescriptor::new(
    1, 0, ElementType::Float32, Shape::new(vec![8u64, 8])?, 0,
    LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2)?),
    vec![], None, None, None, None,
)?;

let left_buffers: [&[u8]; 1] = [left_data.as_slice()];
let right_buffers: [&[u8]; 1] = [right_data.as_slice()];
let members = vec![
    CompositeNode::Tensor { descriptor: &left,  buffers: &left_buffers },
    CompositeNode::Tensor { descriptor: &right, buffers: &right_buffers },
];

let mut wire = Vec::<u8>::new();
let mut writer = StreamWriter::new(&mut wire);
writer.write_composite(&head, &members).await?; // validated before any byte is written
writer.finish().await?;
# Ok(())
# }
```

```python
import hurray

def tile(offset):
    return hurray.Tensor(
        bytes(128), hurray.float32, [8, 4], shard=hurray.Shard([8, 8], [0, offset])
    )

weight = hurray.Composite(
    "partition", shape=[8, 8], dtype=hurray.float32, members=[tile(0), tile(4)]
)
plain = hurray.Tensor(bytes(16), hurray.float32, [4])

with hurray.StreamWriter(destination) as writer:
    writer.write(plain)
    writer.write(weight)     # takes a composite exactly as it takes a tensor
```

```c
/* C has no sum type, so the tree crosses as the format's own shape: the head, then its
   members, in preorder, each node saying how many of the nodes after it are its own. */
HurrayCompositeNode nodes[3] = {
    { .kind = HURRAY_ITEM_COMPOSITE, .descriptor = head, .member_count = 2 },
    { .kind = HURRAY_ITEM_TENSOR, .descriptor = left,
      .payloads = &left_slice,  .payload_count = 1 },
    { .kind = HURRAY_ITEM_TENSOR, .descriptor = right,
      .payloads = &right_slice, .payload_count = 1 },
};

/* Validated whole before a byte goes out: a torn or invalid composite never reaches the
   wire. Names are ignored here — the stream format carries none. */
HurrayStatus s = hurray_writer_write_composite(writer, nodes, 3);
```

</div>

## Reading a composite

`next_item` returns a `StreamItem` — either a plain `Tensor` or a `Composite` with its
members already grouped and validated:

<div class="lang-tabs">

```rust,no_run
use hurray_io::stream::{StreamItem, StreamReader};

# async fn run(wire: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
let mut reader = StreamReader::new(wire);
while let Some(item) = reader.next_item().await? {
    match item {
        StreamItem::Tensor(t) => {
            println!("tensor: {} buffer(s)", t.buffers.len());
        }
        StreamItem::Composite(c) => {
            println!("composite: head {:?}, {} member(s)", c.head.shape.dims(), c.members.len());
            for member in &c.members {
                // Each member is itself a StreamItem — a nested composite recurses here.
                println!("  member governed by {:?}", member.descriptor().layout);
            }
        }
    }
}
# Ok(())
# }
```

```python
import hurray

for item in hurray.StreamReader(source):
    if isinstance(item, hurray.Composite):
        print(item.layout.composition_rule, item.member_count)
    else:
        print(item.shape)
```

```c
HurrayItem *item = NULL;
HurrayStatus s = hurray_reader_next_item(reader, &item);
if (s == HURRAY_END_OF_STREAM) return 0;   /* positive: not an error */
if (s < 0) return -1;

/* One item, however many tensors it took on the wire. Walking it is a loop with a stack
   of outstanding member counts — `remaining` below is the whole algorithm. */
uint64_t count = 0;
hurray_item_node_count(item, &count);

uint32_t remaining[64];
size_t depth = 0;
for (uint64_t i = 0; i < count; i++) {
    HurrayItemNode node = {0};
    hurray_item_node(item, i, &node);

    if (depth > 0) { remaining[depth - 1]--; }
    size_t level = depth;
    while (depth > 0 && remaining[depth - 1] == 0) { depth--; }
    if (node.member_count > 0) { remaining[depth++] = node.member_count; }

    /* node.tensor is borrowed: its descriptor and buffers are the item's, and a head's
       buffer list is empty. node.name is NULL here — a stream carries no names. */
    printf("%*s%s\n", (int)(2 * level), "",
           node.kind == HURRAY_ITEM_COMPOSITE ? "composite head" : "tensor");
}

hurray_item_destroy(&item);   /* one call for the whole subtree */
```

</div>

A stream carrying one composite of two members yields **one** item, not three. The Python
reader uses `next_item` for exactly the reason this page gives: `next_tensor` would hand
back the head as an empty tensor and its members as top-level ones, losing the composition
without raising.

## Nested composites

A member may itself be a composite. On the wire this is just more heads and members in
forward order; `next_item` assembles the tree recursively. Reads are bounded by
`StreamReaderOptions::max_composite_depth` (default 64) so a maliciously deep composite on
an untrusted stream cannot exhaust the stack.

## Choosing the API

| You want… | Use |
|-----------|-----|
| Composites grouped and validated for you | `next_item` → `StreamItem` (C: `hurray_reader_next_item` → `HurrayItem`) |
| Every descriptor flat, composition-agnostic | `next_tensor` → `StreamTensor` (C: `hurray_reader_next_tensor` → `HurrayTensor`) |

`next_tensor` is unchanged: it still yields the head (with no buffers) and then each member
as individual tensors, leaving composition to the caller. `next_item` is the
composite-aware layer on top. Both views come from one core and the assembly happens above
it, so mixing them on one reader is allowed but rarely meant: a flat read taken while a
head is open consumes the member that head was waiting for, and the group stays open.

## Errors

- `Error::TornComposite` — the stream ended before the head's declared `member_count`
  members were read.
- `Error::CompositeNestingTooDeep` — nesting exceeded `max_composite_depth`.
- `Error::Core` — composite validation failed (member-count mismatch, partition does not
  cover the index space, overlay ordering).

## Runnable example

```text
cargo run --example composite_stream --features tokio -p hurray-io
python hurray-python/examples/composites.py
cargo run --example composite_vtable -p hurray-ffi
```

See `hurray-io/examples/composite_stream.rs`, `hurray-python/examples/composites.py`, and
`hurray-ffi/examples/composite_vtable.rs` for the full programs. The last one drives the C
entry points, over a transport vtable and with no runtime linked
([Streaming from C](streaming-from-c.md)).
