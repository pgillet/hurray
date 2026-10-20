# Streaming from C

A C consumer never hands Hurray a path or a file descriptor. It hands over a context
pointer and two callbacks, and Hurray asks them for bytes. That is the **transport
vtable**, and it is what lets the same stream format be read from a socket, a pipe, a
memory span, or a caller's own object — without a file-descriptor conversion that does not
port, and without an async runtime in the library.

```c
typedef struct HurrayTransport {
    void*                    ctx;
    HurrayTransportReadFn    read;     /* required by a reader */
    HurrayTransportWriteFn   write;    /* required by a writer */
    HurrayTransportReleaseFn release;  /* optional; frees ctx */
} HurrayTransport;
```

Every call is synchronous and returns a `HurrayStatus`. The callbacks are allowed to block
— that is the point of them — and nothing inside Hurray does.

## Reading a stream

```c
#include "hurray.h"

static HurrayStatus sock_read(void *ctx, uint8_t *dst, size_t len, size_t *out_filled) {
    ssize_t n = recv(*(int *)ctx, dst, len, 0);
    if (n < 0) { *out_filled = 0; return HURRAY_ERR_TRANSPORT; }  /* moved nothing */
    *out_filled = (size_t)n;                                      /* 0 == end of input */
    return HURRAY_OK;
}

int fd = /* a connected socket */;
HurrayTransport transport = { .ctx = &fd, .read = sock_read };

HurrayReader *reader = NULL;
if (hurray_reader_new(&transport, NULL, &reader) != HURRAY_OK) return -1;

for (;;) {
    HurrayTensor *tensor = NULL;
    HurrayStatus s = hurray_reader_next_tensor(reader, &tensor);
    if (s == HURRAY_END_OF_STREAM) break;   /* positive: not an error */
    if (s < 0) break;                       /* every error is negative */

    HurrayDescriptor *descriptor = NULL;    /* borrowed — do not destroy */
    HurrayBufferList *buffers = NULL;       /* borrowed — do not destroy */
    hurray_tensor_descriptor(tensor, &descriptor);
    hurray_tensor_buffers(tensor, &buffers);

    /* ... read fields off the descriptor, hand the buffers to your kernels ... */

    hurray_tensor_destroy(&tensor);         /* one call frees both halves */
}
hurray_reader_destroy(&reader);
```

`HurrayTensor` exists because a tensor *is* a descriptor plus its buffers. Keeping them in
one handle means a consumer cannot end up holding bytes it cannot interpret, or a
description of bytes it does not have, and the teardown order is not its problem.

## Writing a stream

A tensor's payloads are passed as borrowed slices, one per buffer in the descriptor's
buffer table:

```c
static HurrayStatus sock_write(void *ctx, const uint8_t *src, size_t len, size_t *out) {
    ssize_t n = send(*(int *)ctx, src, len, 0);
    if (n < 0) { *out = 0; return HURRAY_ERR_TRANSPORT; }
    *out = (size_t)n;                        /* a short write is normal */
    return HURRAY_OK;
}

HurrayTransport transport = { .ctx = &fd, .write = sock_write };
HurrayWriter *writer = NULL;
hurray_writer_new(&transport, NULL, &writer);

HurrayIoSlice payload = { .ptr = data, .len = data_len };
HurrayStatus s = hurray_writer_write_tensor(writer, descriptor, &payload, 1);

hurray_writer_destroy(&writer);
```

Nothing is retained after the call returns: Hurray holds no pointer into `data` once the
tensor is written. There is also nothing to flush — a tensor is either entirely on the
wire or the call that wrote it failed.

## The four rules a transport has to keep

| Rule | Why it is not obvious |
|---|---|
| Report the count you actually moved | Fewer bytes than asked for is normal, not an error, and not end of input. |
| `0` from `read` is end of input; `0` from `write` is a closed sink | They are not symmetrical. A sink accepting nothing is a distinct signal, and an error if an item is in flight. |
| A **failing** callback must have moved **no** bytes | This is what makes a failure recoverable. A transport that partly succeeds then fails must report the partial count with `HURRAY_OK` and fail on the *next* call — what POSIX `read` and `write` already do. |
| Never call back into the handle that invoked you | The alternative is a reentrancy contract on every handle, bought for a use case nobody has. |

Keep those and a failed call is retryable: the handle's position still matches the
transport's, so calling `hurray_reader_next_tensor` again **resumes** mid-tensor rather
than restarting it. The one exception is a write that fails after its first byte — a
truncated item is on the wire, a stream cannot be un-sent, and the writer is poisoned
rather than left pretending.

One handle, one thread at a time. Hurray runs no threads of its own, so that is the whole
threading model.

## What a buffer points at

The reader allocates each buffer at the alignment the buffer handle *declares* — 64 bytes
by default, 4096 for GPU and IPC memory — and the transport writes straight into that
allocation. So the pointer a consumer gets is aligned for the SIMD loads the format
entitles it to, and no copy of the payload happened after the transport's write.

## The same stream, in Rust and in Python

The Rust tab drives the C entry points, because that is what a binding does. The Python
tab is `hurray-python`'s own streaming API, which is blocking today and gets an
`async`/`await` surface of its own.

<div class="lang-tabs">

```rust
use std::ffi::c_void;

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    buffer::{hurray_buffer_byte_size, hurray_buffer_data_ptr},
    buffer_list::hurray_buffer_list_get,
    hurray_descriptor_decode, hurray_reader_destroy, hurray_reader_new,
    hurray_reader_next_tensor, hurray_tensor_buffers, hurray_tensor_destroy,
    hurray_writer_destroy, hurray_writer_new, hurray_writer_write_tensor, HurrayBuffer,
    HurrayBufferList, HurrayDescriptor, HurrayIoSlice, HurrayReader, HurrayStatus, HurrayTensor,
    HurrayTransport, HurrayWriter, HURRAY_END_OF_STREAM, HURRAY_OK,
};

/// Stands in for a socket: bytes, a read cursor, and a small transfer size so the short
/// read and short write paths are the ones exercised.
struct Pipe {
    bytes: Vec<u8>,
    read_pos: usize,
    chunk: usize,
}

unsafe extern "C" fn pipe_read(
    ctx: *mut c_void,
    dst: *mut u8,
    len: usize,
    out_filled: *mut usize,
) -> HurrayStatus {
    let pipe = &mut *(ctx as *mut Pipe);
    let n = len.min(pipe.chunk).min(pipe.bytes.len() - pipe.read_pos);
    std::ptr::copy_nonoverlapping(pipe.bytes.as_ptr().add(pipe.read_pos), dst, n);
    pipe.read_pos += n;
    *out_filled = n; // 0 means end of input; fewer than `len` does not
    HURRAY_OK
}

unsafe extern "C" fn pipe_write(
    ctx: *mut c_void,
    src: *const u8,
    len: usize,
    out_written: *mut usize,
) -> HurrayStatus {
    let pipe = &mut *(ctx as *mut Pipe);
    let n = len.min(pipe.chunk);
    pipe.bytes.extend_from_slice(std::slice::from_raw_parts(src, n));
    *out_written = n; // never 0: that would mean the sink is closed
    HURRAY_OK
}

fn vtable(pipe: &mut Pipe) -> HurrayTransport {
    HurrayTransport {
        ctx: pipe as *mut Pipe as *mut c_void,
        read: Some(pipe_read),
        write: Some(pipe_write),
        release: None, // the context is ours; Hurray never frees it
    }
}

let handle =
    BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
let descriptor = TensorDescriptor::new(
    1, 0, ElementType::Uint8, Shape::new(vec![64u64]).unwrap(), 0,
    LayoutDescriptor::RowMajor, vec![handle], None, None, None, None,
)
.unwrap();
let payload: Vec<u8> = (0u8..64).collect();

let mut pipe = Pipe { bytes: Vec::new(), read_pos: 0, chunk: 24 };

// A C caller names a descriptor by decoding its bytes.
let encoded = descriptor.encode().unwrap();
let mut descriptor_handle: *mut HurrayDescriptor = std::ptr::null_mut();
unsafe {
    assert_eq!(
        hurray_descriptor_decode(encoded.as_ptr(), encoded.len(), &mut descriptor_handle),
        HURRAY_OK
    );

    // Write.
    let transport = vtable(&mut pipe);
    let mut writer: *mut HurrayWriter = std::ptr::null_mut();
    assert_eq!(hurray_writer_new(&transport, std::ptr::null(), &mut writer), HURRAY_OK);
    let slice = HurrayIoSlice { ptr: payload.as_ptr(), len: payload.len() };
    assert_eq!(
        hurray_writer_write_tensor(writer, descriptor_handle, &slice, 1),
        HURRAY_OK
    );
    hurray_writer_destroy(&mut writer);

    // Read it back.
    let transport = vtable(&mut pipe);
    let mut reader: *mut HurrayReader = std::ptr::null_mut();
    assert_eq!(hurray_reader_new(&transport, std::ptr::null(), &mut reader), HURRAY_OK);

    let mut tensor: *mut HurrayTensor = std::ptr::null_mut();
    assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_OK);

    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    hurray_tensor_buffers(tensor, &mut list);
    let mut buffer: *mut HurrayBuffer = std::ptr::null_mut();
    hurray_buffer_list_get(list, 0, &mut buffer);
    let mut data: *mut c_void = std::ptr::null_mut();
    let mut size: u64 = 0;
    hurray_buffer_data_ptr(buffer, &mut data);
    hurray_buffer_byte_size(buffer, &mut size);
    assert_eq!(
        std::slice::from_raw_parts(data as *const u8, size as usize),
        &payload[..]
    );
    // The declared alignment survives the trip into C.
    assert_eq!(data as usize % MIN_BUFFER_ALIGNMENT as usize, 0);

    hurray_tensor_destroy(&mut tensor);
    assert_eq!(hurray_reader_next_tensor(reader, &mut tensor), HURRAY_END_OF_STREAM);
    hurray_reader_destroy(&mut reader);
    hurray_ffi::descriptor::hurray_descriptor_destroy(descriptor_handle);
}
```

```python
import numpy as np
import hurray

tensor = hurray.from_numpy(np.arange(64, dtype=np.uint8))

# Nowhere to put it: the writer keeps the bytes.
writer = hurray.StreamWriter()
writer.write(tensor)
writer.finish()
wire = writer.getvalue()

# The reader is an iterator; each tensor is available as its bytes arrive.
for received in hurray.StreamReader(wire):
    assert received.shape == (64,)
    assert np.array_equal(np.from_dlpack(received), np.arange(64, dtype=np.uint8))
```

</div>

## Not here yet

- **Composite tensors.** A composite head reads back as the buffer-less tensor it is on
  the wire; walking its members from C, and writing one, come with the C sum-type for a
  composite.
- **File containers from C.** `HRRYFILE` needs random access — the trailer sits at
  `file_size - 40` — so it needs a range-request protocol and two more vtable entries
  (`seek`, `size`), not a byte feed. The range-request core itself exists: see
  [Random Access Without a Runtime](random-access-without-a-runtime.md), which drives it
  from blocking `std::io` today. Only the C surface over it is still to come.
- **Building a descriptor in C.** A writer can emit descriptors it decoded; originating
  one from C needs an encode surface of its own.
