# Random Access Without a Runtime

## Purpose

A Hurray file is read by **range**, not by feeding bytes forward. The trailer sits at
`file_size - 40`, the footer index is located from it, and tensors are found by name — so
a reader seeks. `hurray-io` keeps that logic in a sans-io core, `FileReaderCore`, which
names the bytes it wants and is told how many arrived. Nothing inside it reads, waits, or
owns a file (ADR-041 §§ 1, 7).

That matters for three reasons:

- **No runtime is required.** The core compiles with `--no-default-features`, so a
  consumer that has no executor — a C program, a synchronous Python call — pays for none.
- **The payload is copied once.** The core allocates each buffer at the alignment its
  handle *declares* and the transport writes straight into it; what comes back out is that
  same memory, not a copy of it (§§ 2, 5).
- **An interrupted read costs nothing.** The read position advances only when a byte count
  is reported, and every request is absolute, so a driver that abandons a transfer is
  handed the same range again.

`FileReader` and `FileWriter` are thin drivers over the cores and remain the ordinary way
to read and write files from async Rust; reach for the cores when you need the bytes to
come from somewhere they do not know about. From C, that driver is already written:
`HurrayFileReader` and `HurrayFileWriter` drive the same cores over the transport vtable,
which is what the C tabs below use.

## The protocol

Three calls, in a loop:

| Call | Meaning |
|---|---|
| `step()` | What the core needs next: a `Range { offset, len }`, or `Ready`, `Descriptor`, `Tensor`, `Composite` |
| `input_dst()` | The memory that range must be read into — exactly `len` bytes, at a stable address |
| `report_filled(n)` | How many bytes actually arrived. A short read is normal; `0` means the file is truncated |

`step()` never blocks, which is what lets the same core sit behind a blocking C callback
and an event loop alike (§ 4).

A C caller never writes that loop. It supplies two callbacks — `read_at(ctx, offset, dst,
len, &filled)` and `size(ctx, &size)` — and `hurray_file_reader_*` runs the protocol
inside them. The callbacks are *allowed* to block, which is the point of them; nothing
inside Hurray does.

## Reading a file by range

The driver below is the whole bridge between the core and `std::io` — a seek, a read, and
a count:

<div class="lang-tabs">

```rust
use std::io::{Read, Seek, SeekFrom};

use hurray_io::file::{FileReaderCore, FileStep};

/// Answers every range the core asks for, and returns the step that ended the job.
fn drive<R: Read + Seek>(
    core: &mut FileReaderCore,
    source: &mut R,
) -> hurray_io::Result<FileStep> {
    loop {
        match core.step()? {
            FileStep::Range { offset, len } => {
                source.seek(SeekFrom::Start(offset))?;
                // `read`, not `read_exact`: a bulk helper that fails cannot say how many
                // bytes it moved, and that count is the read position.
                let n = source.read(&mut core.input_dst()[..len])?;
                core.report_filled(n)?;
            }
            other => return Ok(other),
        }
    }
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use hurray_core::{
#     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
#     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
# };
# use hurray_io::file::{FileWriteOp, FileWriterCore};
# use hurray_io::stream::WriteStep;
# fn drain(mut op: FileWriteOp<'_>, sink: &mut Vec<u8>) -> hurray_io::Result<()> {
#     while let WriteStep::Output { .. } = op.step()? {
#         let src = op.output_src();
#         sink.extend_from_slice(src);
#         let n = src.len();
#         op.report_written(n)?;
#     }
#     Ok(())
# }
# let handle = BufferHandle::new(
#     256, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
# )?;
# let descriptor = TensorDescriptor::new(
#     1, 0, ElementType::Float32, Shape::new(vec![8u64, 8])?, 0,
#     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
# )?;
# let data: Vec<u8> = (0..=255u8).collect();
# let buffers: [&[u8]; 1] = [&data];
# let mut bytes = Vec::new();
# let mut writer = FileWriterCore::new();
# drain(writer.header(), &mut bytes)?;
# drain(writer.tensor("embeddings", &descriptor, &buffers)?, &mut bytes)?;
# drain(writer.finish(vec![])?, &mut bytes)?;
# let mut source = std::io::Cursor::new(bytes);
let size = source.seek(SeekFrom::End(0))?;
let mut core = FileReaderCore::new(size);

// Opening is four ranges at most: header, trailer, index, KV section.
assert_eq!(drive(&mut core, &mut source)?, FileStep::Ready);
println!("{} tensor(s)", core.index().len());

// A descriptor on its own — one range, and not a byte of payload.
core.begin_descriptor("embeddings")?;
assert_eq!(drive(&mut core, &mut source)?, FileStep::Descriptor);
let descriptor = core.take_descriptor().expect("a descriptor is ready");
println!("{:?} {:?}", descriptor.element_type, descriptor.shape.dims());

// The whole tensor — the descriptor, then one range per buffer.
core.begin_tensor("embeddings")?;
assert_eq!(drive(&mut core, &mut source)?, FileStep::Tensor);
let tensor = core.take_tensor().expect("a tensor is ready");

// The buffer carries the alignment its handle declared, so an aligned SIMD load on it is
// sound. This is the guarantee, not a property of the allocator that happened to hold.
let declared = tensor.descriptor.buffers[0].alignment() as usize;
assert_eq!(tensor.buffers[0].as_ptr() as usize % declared, 0);
# assert_eq!(tensor.buffers[0].as_ref(), &data[..]);
# Ok(())
# }
```

```c
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hurray.h"

/* One absolute range. A positional read keeps no cursor, which is why re-issuing a
   failed request reads the same bytes rather than resuming somewhere else. */
static HurrayStatus file_read_at(void *ctx, uint64_t offset, uint8_t *dst, size_t len,
                                 size_t *out_filled) {
    ssize_t n = pread(*(int *)ctx, dst, len, (off_t)offset);
    if (n < 0) { *out_filled = 0; return HURRAY_ERR_TRANSPORT; }  /* moved nothing */
    *out_filled = (size_t)n;  /* short is normal; 0 means the file is truncated */
    return HURRAY_OK;
}

/* Asked once, at open: a container is located from its end. */
static HurrayStatus file_size(void *ctx, uint64_t *out_size) {
    struct stat st;
    if (fstat(*(int *)ctx, &st) != 0) { return HURRAY_ERR_TRANSPORT; }
    *out_size = (uint64_t)st.st_size;
    return HURRAY_OK;
}

int fd = open("model.hrry", O_RDONLY);
HurrayTransport transport = { .ctx = &fd, .read_at = file_read_at, .size = file_size };

HurrayFileReader *reader = NULL;
if (hurray_file_reader_new(&transport, &reader) != HURRAY_OK) return -1;

/* The names come from the footer index, so listing them reads no tensor data. */
uint64_t count = 0;
hurray_file_reader_tensor_count(reader, &count);
for (uint64_t i = 0; i < count; i++) {
    const uint8_t *name = NULL;
    size_t name_len = 0;
    hurray_file_reader_tensor_name(reader, i, &name, &name_len);  /* borrowed, no NUL */
    printf("%.*s\n", (int)name_len, (const char *)name);
}

/* One tensor by name; the bytes of every other tensor stay on disk. */
const char *want = "embeddings";
HurrayTensor *tensor = NULL;
HurrayStatus s = hurray_file_reader_read_tensor(
    reader, (const uint8_t *)want, strlen(want), &tensor);
if (s == HURRAY_ERR_NOT_FOUND) {
    /* The file is fine — that name is simply not in it, and the reader stays usable. */
} else if (s == HURRAY_OK) {
    HurrayBufferList *buffers = NULL;   /* borrowed — do not destroy */
    hurray_tensor_buffers(tensor, &buffers);
    /* ... hand the buffers to your kernels ... */
    hurray_tensor_destroy(&tensor);     /* one call frees descriptor and buffers */
}

hurray_file_reader_destroy(&reader);
close(fd);
```

```python
import numpy as np
import hurray

tensor = hurray.from_numpy(np.arange(64, dtype=np.float32))
hurray.save("model.hrry", {"embeddings": tensor}, kv={"producer": "cookbook"})

# The synchronous Python surface drives the same core, with the runtime detached for the
# duration of the call.
loaded = hurray.load("model.hrry")
assert np.array_equal(np.from_dlpack(loaded["embeddings"]), np.arange(64, dtype=np.float32))
assert hurray.load_kv("model.hrry")["producer"] == "cookbook"
```

</div>

## Writing a file in one forward pass

The write side needs no ranges at all: a file is written forwards, and the footer index is
accumulated as tensors go out and emitted last. What the core will not do is assemble your
payloads — it offers each buffer **in place**, one stretch at a time, and holds no pointer
into it once the item is written (§ 3):

<div class="lang-tabs">

```rust
use hurray_io::file::{FileWriteOp, FileWriterCore};
use hurray_io::stream::WriteStep;

/// Writes every byte of one submitted item. The sink may take fewer bytes than offered.
fn drain<W: std::io::Write>(
    mut op: FileWriteOp<'_>,
    sink: &mut W,
) -> hurray_io::Result<()> {
    loop {
        match op.step()? {
            WriteStep::Output { .. } => match sink.write(op.output_src())? {
                // A zero-length write is never progress: a closed sink is its own signal.
                0 => return Err(op.report_sink_closed()),
                n => op.report_written(n)?,
            },
            WriteStep::ItemWritten => return Ok(()),
        }
    }
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# use hurray_core::{
#     BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
#     TensorDescriptor, MIN_BUFFER_ALIGNMENT,
# };
use hurray_io::file::KvValue;

# let handle = BufferHandle::new(
#     256, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced,
# )?;
# let descriptor = TensorDescriptor::new(
#     1, 0, ElementType::Float32, Shape::new(vec![8u64, 8])?, 0,
#     LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
# )?;
let payload: Vec<u8> = (0..=255u8).collect();
// Bound, not a temporary: the operation borrows the payloads until the item is written.
let buffers: [&[u8]; 1] = [&payload];

let mut file = Vec::new();
let mut core = FileWriterCore::new();
drain(core.header(), &mut file)?;
drain(core.tensor("embeddings", &descriptor, &buffers)?, &mut file)?;
drain(
    core.finish(vec![("producer".into(), KvValue::String("cookbook".into()))])?,
    &mut file,
)?;

assert_eq!(&file[0..8], b"HRRYFILE");
assert_eq!(&file[file.len() - 4..], b"HRRY");
# Ok(())
# }
```

```c
#include "hurray.h"

static HurrayStatus file_write(void *ctx, const uint8_t *src, size_t len,
                               size_t *out_written) {
    ssize_t n = write(*(int *)ctx, src, len);
    if (n < 0) { *out_written = 0; return HURRAY_ERR_TRANSPORT; }
    *out_written = (size_t)n;  /* never 0 on success: that means the sink is closed */
    return HURRAY_OK;
}

int fd = open("model.hrry", O_WRONLY | O_CREAT | O_TRUNC, 0644);
/* A writer needs no range reads: it emits one forward pass. */
HurrayTransport transport = { .ctx = &fd, .write = file_write };

HurrayFileWriter *writer = NULL;
/* NULL options: 4096-byte data alignment, index in write order. The 64-byte header
   goes out during this call. */
if (hurray_file_writer_new(&transport, NULL, &writer) != HURRAY_OK) return -1;

const char *name = "embeddings";
HurrayIoSlice payload = { .ptr = data, .len = data_len };
if (hurray_file_writer_write_tensor(writer, (const uint8_t *)name, strlen(name),
                                    descriptor, &payload, 1) != HURRAY_OK) {
    /* A failed file write ends the file: the writer is poisoned, and the recovery is
       to write the file again from the start, not to retry this tensor. */
    hurray_file_writer_destroy(&writer);
    return -1;
}

/* KV values are borrowed, and the same struct a reader hands back — so copying one
   file's metadata into another needs no transcription. */
const char *producer = "cookbook";
HurrayKvEntry kv[1] = {{
    .key = (const uint8_t *)"producer",
    .key_len = 8,
    .value = { .kind = HURRAY_KV_STRING,
               .data = (const uint8_t *)producer,
               .data_len = 8 },
}};

/* Until this returns, the bytes on disk are not a container: the index is written here. */
hurray_file_writer_finish(writer, kv, 1);
hurray_file_writer_destroy(&writer);
close(fd);
```

```python
import numpy as np
import hurray

# `save` is the same forward pass: header, tensors, KV, index, trailer.
hurray.save(
    "two.hrry",
    {
        "embeddings": hurray.from_numpy(np.arange(64, dtype=np.float32)),
        "bias": hurray.from_numpy(np.zeros(16, dtype=np.float32)),
    },
    kv={"producer": "cookbook", "layers": 32},
)

names = list(hurray.load("two.hrry"))
assert names == ["embeddings", "bias"]
```

</div>

## Composites: validated first, then flattened

A composite is written as its head followed by each member, contiguously, and recovered on
read from the head's `member_count` plus file-offset adjacency. `flatten_composite`
validates the whole tree — member counts, partition exact-cover, overlay ordering, at every
level — and returns the order its tensors go on disk, so a torn group never reaches the
sink. The nesting is walked with an explicit stack rather than by recursing, which is what
lets a state machine do it at all (§ 10):

```rust,no_run
use hurray_io::file::{flatten_composite, FileCompositeNode, FileWriterCore};

# fn run(
#     head: hurray_core::TensorDescriptor,
#     members: &[FileCompositeNode<'_>],
#     core: &mut FileWriterCore,
#     file: &mut Vec<u8>,
# ) -> hurray_io::Result<()> {
# fn drain(_: hurray_io::file::FileWriteOp<'_>, _: &mut Vec<u8>) -> hurray_io::Result<()> { Ok(()) }
for node in flatten_composite("weight", &head, members)? {
    // head first, then "weight.left", "weight.right", … depth first
    drain(core.tensor(node.name, node.descriptor, node.buffers)?, file)?;
}
# Ok(())
# }
```

Reading one back is a single job: the core walks the index in file order and rebuilds the
tree, member counts validated as each level closes.

```rust,no_run
use hurray_io::file::{FileItem, FileReaderCore, FileStep};

# fn run(core: &mut FileReaderCore, source: &mut std::io::Cursor<Vec<u8>>) -> hurray_io::Result<()> {
# fn drive(_: &mut FileReaderCore, _: &mut std::io::Cursor<Vec<u8>>) -> hurray_io::Result<FileStep> {
#     Ok(FileStep::Composite)
# }
core.begin_composite("weight")?;
assert_eq!(drive(core, source)?, FileStep::Composite);
let composite = core.take_composite().expect("a composite is ready");

for member in &composite.members {
    match member {
        FileItem::Tensor(t) => println!("{} — {} buffer(s)", t.name, t.buffers.len()),
        FileItem::Composite(c) => println!("{} — nested, {} member(s)", c.name, c.members.len()),
    }
}
# Ok(())
# }
```

## What a failure costs

The file core distinguishes two kinds of failure, and the difference is the whole point of
addressing bytes rather than consuming them:

- **A failed job leaves the core usable.** A missing name, a descriptor that will not
  decode, a torn composite — the next `begin_descriptor` / `begin_tensor` /
  `begin_composite` starts from the index again, not from wherever the file pointer
  stopped.
- **A failed *open* is terminal.** Bad magic, an unsupported container version, a trailer
  whose index overruns it, a CRC mismatch: there is no index to start from, so `step()`
  reports `CoreFailed` from then on.

There is no "unusable" state for a lost byte count here, unlike the stream core. A count
that never arrives simply means the next `step()` re-offers the same absolute range, and
re-filling the destination from the start of that range is idempotent.

The write side has no such symmetry, and it is worth knowing before you build a retry
around it: **a failed file write ends the file.** By the time bytes move, the tensor has
consumed its name in the index and staged its descriptor, so a second attempt under that
name would describe a file other than the one on the sink. Writing the file again from the
beginning is the recovery — which, for the storage failure that caused it, is the only one
worth having.

## Runnable example

`hurray-io/examples/sans_io_file.rs` writes a file with two tensors, a partition composite
and KV metadata, then reads it back through the core — all over `std::io`, with no runtime
linked:

```text
cargo run --example sans_io_file
```

`hurray-ffi/examples/file_vtable.rs` does the same thing through the C entry points, over
a real file on disk reached only through callbacks:

```text
cargo run --example file_vtable -p hurray-ffi
```

## See also

- [Layer 6: File Format](layer-6-file-format.md) — the container's bytes, and the async
  `FileReader`/`FileWriter`
- [Composite Tensors in Files](composite-file.md) — membership, naming, and validation
- [Streaming from C](streaming-from-c.md) — the same core/driver split over the C
  transport vtable, for streams
- [Python: File I/O](hurray-python-file-io.md) — `save`, `load`, `load_kv`
