# Layer 7 — C FFI Cookbook

The `hurray-ffi` crate exposes a stable C ABI over `hurray-core` types.
All functions return a `HurrayStatus` integer (`0` = OK, negative = error).
All handles are opaque — never inspect their internals.

> **Note (non-normative):** this page has no Python tabs, and should not. The C ABI is the
> layer *underneath* a language binding — `hurray-python` is one of its consumers, not a
> way to call it. A Python program that wants a Hurray tensor uses `hurray.Tensor` and the
> `__hurray__` protocol; the function table here is for whoever is writing the next
> binding.

## ABI version check

Always verify the ABI version at startup so mismatched builds are caught early.

**C:**
```c
#include "hurray.h"
#include <assert.h>

void startup_check(void) {
    uint32_t v = hurray_c_abi_version();
    /* Compare against the header you compiled against, never a literal — the
       ABI version moves, and a stale literal fails on the wrong builds. */
    assert(v == HURRAY_C_ABI_VERSION && "unexpected Hurray C ABI version");
}
```

**Rust (via FFI):**
```rust
use hurray_ffi::{hurray_c_abi_version, HURRAY_C_ABI_VERSION};

assert_eq!(unsafe { hurray_c_abi_version() }, HURRAY_C_ABI_VERSION);
```


## Creating a buffer with a release callback

The release callback is called exactly once by `hurray_buffer_destroy`.
Use it to free or unmap the underlying memory.

**C:**
```c
#include "hurray.h"
#include <stdlib.h>
#include <stdio.h>

static void my_release(void *data, void *ctx) {
    (void)ctx;
    free(data);
    printf("buffer freed\n");
}

HurrayBuffer *create_cpu_buffer(size_t n_bytes) {
    void *data = aligned_alloc(64, n_bytes);
    if (!data) return NULL;

    HurrayBuffer *handle = NULL;
    HurrayStatus s = hurray_buffer_from_ptr(
        data, (uint64_t)n_bytes,
        /*alignment=*/64,
        /*device_tag=*/0x00,    /* CPU */
        /*sync_mode=*/0x00,     /* ProducerSynced */
        /*memory_class=*/0x00,  /* Standard */
        my_release, /*release_context=*/NULL,
        &handle
    );
    if (s != HURRAY_OK) { free(data); return NULL; }
    return handle;
}
```

**Rust:**
```rust
use hurray_ffi::{hurray_buffer_from_ptr, HurrayBuffer, HurrayReleaseCallback, HURRAY_OK};
use hurray_core::{DeviceTag, MemoryClass, SyncMode, MIN_BUFFER_ALIGNMENT};
use std::ffi::c_void;

unsafe extern "C" fn release(data: *mut c_void, _ctx: *mut c_void) {
    drop(Vec::from_raw_parts(data as *mut u8, 4096, 4096));
}

let mut storage = vec![0u8; 4096];
let ptr = storage.as_mut_ptr() as *mut c_void;
std::mem::forget(storage);

let mut handle: *mut HurrayBuffer = std::ptr::null_mut();
let status = unsafe {
    hurray_buffer_from_ptr(
        ptr, 4096, MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu.to_byte(),
        SyncMode::ProducerSynced.to_byte(),
        MemoryClass::Standard.to_byte(),
        Some(release),
        std::ptr::null_mut(),
        &mut handle,
    )
};
assert_eq!(status, HURRAY_OK);
```

## Destroying a buffer

```c
HurrayStatus s = hurray_buffer_destroy(handle);
assert(s == HURRAY_OK);
/* handle is invalid after this point — do not dereference */
```

In debug builds, a second call to `hurray_buffer_destroy` on the same handle
returns `HURRAY_ERR_INTERNAL` (sentinel-based double-free detection).

## Decoding a tensor descriptor

```c
#include "hurray.h"

HurrayDescriptor *decode_descriptor(const uint8_t *bytes, size_t len) {
    HurrayDescriptor *desc = NULL;
    HurrayStatus s = hurray_descriptor_decode(bytes, len, &desc);
    if (s != HURRAY_OK) return NULL; /* inspect s for the specific error */
    return desc;
}

void inspect(HurrayDescriptor *desc) {
    uint32_t rank;
    hurray_descriptor_rank(desc, &rank);

    uint64_t dims[64];
    size_t capacity = rank;
    hurray_descriptor_shape(desc, dims, &capacity);
    /* capacity now holds the true rank; dims[0..rank] are the dimension sizes */

    hurray_descriptor_destroy(desc);
}
```

### Shape capacity/query pattern

`hurray_descriptor_shape` uses an in/out `out_rank` parameter:

1. Set `*out_rank` to the number of `uint64_t` slots in `out_dims`.
2. If the function returns `HURRAY_ERR_BUFFER_TOO_SMALL`, `*out_rank` has been
   updated to the true rank — allocate that many slots and retry.

```c
size_t cap = 0;
/* Query-only call: out_dims=NULL forces BUFFER_TOO_SMALL, writes true rank */
hurray_descriptor_shape(desc, NULL, &cap);

uint64_t *dims = malloc(cap * sizeof(uint64_t));
hurray_descriptor_shape(desc, dims, &cap);
```

## Sync mode handoff cross-check

Before consuming a GPU buffer, call the matching handoff function to verify
that the producer's declared sync mode matches your payload.

```c
/* Event mode: producer recorded a CUDA event */
HurraySyncEventPayload payload = {
    .sync_handle           = cuda_event,
    .sync_handle_device_tag = 0x01, /* CUDA */
    .event_release_fn      = my_event_release,
    .event_release_context = NULL,
};
HurrayStatus s = hurray_buffer_handoff_event(buffer, &payload);
if (s == HURRAY_ERR_SYNC_MODE_MISMATCH) { /* handle disagreement */ }
```

```c
/* ConsumerStream mode: consumer declares its target stream */
HurraySyncConsumerStreamPayload sp = {
    .consumer_stream            = my_cuda_stream,
    .consumer_stream_device_tag = 0x01, /* CUDA */
};
HurrayStatus s = hurray_buffer_handoff_consumer_stream(buffer, &sp);
```

```c
/* ProducerSynced mode: producer issued a host-side wait; no payload needed */
HurrayStatus s = hurray_buffer_handoff_producer_synced(buffer);
```

## Reading a capsule from outside Python

A native-protocol capsule carries a `HurrayBufferList` as its pointer and a
`HurrayTensorContext` as its context. The list holds the bytes; the context holds the
descriptor that says what those bytes *are* (ADR-034). Without it a consumer gets
element data with no element type, shape, or layout — which is what every non-Python
consumer got before ADR-034.

**Check the version first.** It is the one accessor guaranteed to work across ABI
versions, and every other accessor assumes a caller that has already checked:

```c
uint32_t abi_version = 0;
if (hurray_tensor_context_abi_version(ctx, &abi_version) != HURRAY_OK) return -1;
if (abi_version != HURRAY_C_ABI_VERSION) {
    /* Producer and consumer disagree — refuse rather than dereference. */
    return -1;
}
```

Then borrow the descriptor and decode it:

```c
const uint8_t *bytes = NULL;
uint64_t len = 0;
if (hurray_tensor_context_descriptor(ctx, &bytes, &len) != HURRAY_OK) return -1;

HurrayDescriptor *descriptor = NULL;
if (hurray_descriptor_decode(bytes, (uintptr_t)len, &descriptor) != HURRAY_OK) return -1;

uint8_t type_tag = 0;
uint32_t rank = 0;
hurray_descriptor_element_type_tag(descriptor, &type_tag);
hurray_descriptor_rank(descriptor, &rank);
```

The pointer `bytes` is **borrowed** — owned by the context, valid until the context is
destroyed. Copy it if you need it longer. An empty descriptor reports a null pointer
and a zero length.

Now the buffers mean something, because the descriptor said what they hold:

```c
uint64_t count = 0;
hurray_buffer_list_len(list, &count);
for (uint64_t i = 0; i < count; i++) {
    HurrayBuffer *borrowed = NULL;          /* owned by the list — do not destroy */
    hurray_buffer_list_get(list, i, &borrowed);

    void *ptr = NULL;
    uint64_t size = 0;
    hurray_buffer_data_ptr(borrowed, &ptr);
    hurray_buffer_byte_size(borrowed, &size);
}
```

Destroying the context runs its `owner_release` callback, which is how the producer
learns it can let the tensor go:

```c
hurray_descriptor_destroy(descriptor);
hurray_tensor_context_destroy(&ctx);       /* runs owner_release exactly once */
hurray_buffer_list_destroy(&list);         /* destroys every handle it owns */
```

Runnable version, in Rust because that is what this repository builds — but the
sequence is the one any language follows:

```bash
cargo run -p hurray-ffi --example tensor_context
```

## Reading a layout's payload fields

Decoding tells you the layout *tag*. It does not, on its own, tell you anything
inside the layout — and for a sparse or paged tensor that is most of what you
need. A consumer handed a CSR descriptor reads `layout_tag == 0x07` and still
cannot address an element until it knows `nnz`.

Payload fields are named by a `HURRAY_LAYOUT_FIELD_*` identifier and read through
one of three functions — one for scalars, one for unsigned arrays, one for signed
arrays:

```c
#include "hurray.h"

/* A scalar: 137 stored non-zeros in this CSR matrix. */
uint64_t nnz;
HurrayStatus s = hurray_descriptor_field_u64(
    desc, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0, &nnz);
if (s == HURRAY_ERR_FIELD_NOT_APPLICABLE) {
    /* This descriptor is not CSR. Check layout_tag before asking. */
}
```

Arrays follow the same capacity/query pattern as
[`hurray_descriptor_shape`](#shape-capacityquery-pattern):

```c
size_t len = 0;
/* Query-only: out_values=NULL forces BUFFER_TOO_SMALL and writes the true length */
hurray_descriptor_field_i64_array(
    desc, HURRAY_LAYOUT_FIELD_STRIDED_STRIDES, 0, NULL, &len);

int64_t *strides = malloc(len * sizeof(int64_t));
hurray_descriptor_field_i64_array(
    desc, HURRAY_LAYOUT_FIELD_STRIDED_STRIDES, 0, strides, &len);
```

Strides are read through the **signed** accessor because their sign carries
meaning: a negative stride reverses a dimension and a zero stride broadcasts one.
An unsigned read would report `-4` as `18446744073709551612`.

**Three statuses are specific to these reads:**

| Status | Meaning |
|---|---|
| `HURRAY_ERR_FIELD_NOT_APPLICABLE` | The field belongs to a different layout than this descriptor's — or is not a field of the type you asked for (a scalar read of an array field). |
| `HURRAY_ERR_FIELD_NOT_PRESENT` | The field is valid for this layout but this descriptor does not carry it. Optional fields only: `layer_index`, `outer_strides`, `inner_strides`. |
| `HURRAY_ERR_INDEX_OUT_OF_BOUNDS` | `level` exceeds the tile nesting depth, or is non-zero for a layout that does not nest. |

Absence is never reported as a zero. `HURRAY_ERR_FIELD_NOT_PRESENT` leaves
`*out_value` untouched, because a cleared optional field is a state in this
format rather than a value — see
[ADR-039](../adr/ADR-039-c-abi-structured-descriptor-fields.md).

### The `level` parameter

Every accessor takes a `level`. It selects which tile to read in a **nested**
tiled layout, where `0` is the outermost;
`HURRAY_LAYOUT_FIELD_TILED_DEPTH` reports how many levels remain from there
down. For every other layout `level` MUST be `0`, and a non-zero value is
rejected rather than ignored — silently ignoring it would let a caller believe it
had read a nested level that does not exist.

**Rust:**
```rust
use hurray_core::layout::CsrLayout;
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    descriptor::hurray_descriptor_destroy,
    hurray_descriptor_decode,
    field::{hurray_descriptor_field_u64, HURRAY_LAYOUT_FIELD_CSR_NNZ},
    HurrayDescriptor, HURRAY_OK,
};

let buffers = (0..3)
    .map(|_| {
        BufferHandle::new(
            256,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap()
    })
    .collect();
let bytes = TensorDescriptor::new(
    1,
    0,
    ElementType::Float32,
    Shape::new(vec![8, 8]).unwrap(),
    0,
    LayoutDescriptor::Csr(CsrLayout::new(137)),
    buffers,
    None,
    None,
    None,
    None,
)
.unwrap()
.encode()
.unwrap();

let mut desc: *mut HurrayDescriptor = std::ptr::null_mut();
assert_eq!(
    unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut desc) },
    HURRAY_OK
);

let mut nnz: u64 = 0;
assert_eq!(
    unsafe {
        hurray_descriptor_field_u64(desc, HURRAY_LAYOUT_FIELD_CSR_NNZ, 0, &mut nnz)
    },
    HURRAY_OK
);
assert_eq!(nnz, 137);

assert_eq!(unsafe { hurray_descriptor_destroy(desc) }, HURRAY_OK);
```

A runnable walk through a sparse, a strided and a nested tiled descriptor:

```bash
cargo run -p hurray-ffi --example layout_fields
```

## Reading the optional descriptor sections

A descriptor may carry five sections beyond its fixed header — quantization, shard,
statistics, extension type, composite member — each gated by a flag bit. They use the
same identifiers and the same accessors as the layout payload; only the range in the
high half-word differs:

| Range | Section | Constants |
|---|---|---|
| `0x0002` | quantization | `HURRAY_QUANT_FIELD_*` |
| `0x0003` | statistics | `HURRAY_STATS_FIELD_*` |
| `0x0004` | shard | `HURRAY_SHARD_FIELD_*` |
| `0x0005` | extension type | `HURRAY_EXT_TYPE_FIELD_*` |
| `0x0006` | composite member | `HURRAY_COMPOSITE_MEMBER_FIELD_*` |

**Read the quantization scheme tag first.** It plays the part `layout_tag` plays for
the payload: it decides which scheme-specific identifiers apply. Asking for the wrong
scheme's field is diagnosable (`HURRAY_ERR_FIELD_NOT_APPLICABLE`), but the tag is what
tells you which to ask for.

```c
#include "hurray.h"

uint64_t scheme;
HurrayStatus s = hurray_descriptor_field_u64(
    desc, HURRAY_QUANT_FIELD_SCHEME_TAG, 0, &scheme);
if (s == HURRAY_ERR_FIELD_NOT_PRESENT) {
    /* Not a quantized tensor — it stores its values directly. */
} else if (scheme == 0x03) {
    uint64_t block_size, scale_buffer;
    hurray_descriptor_field_u64(
        desc, HURRAY_QUANT_FIELD_PER_BLOCK_BLOCK_SIZE, 0, &block_size);
    hurray_descriptor_field_u64(
        desc, HURRAY_QUANT_FIELD_PER_BLOCK_SCALE_BUFFER, 0, &scale_buffer);

    /* A symmetric scheme has no zero point at all — absence, not zero. */
    uint64_t zp_buffer;
    if (hurray_descriptor_field_u64(
            desc, HURRAY_QUANT_FIELD_PER_BLOCK_ZERO_POINT_BUFFER, 0, &zp_buffer)
        == HURRAY_ERR_FIELD_NOT_PRESENT) {
        /* symmetric */
    }
}
```

Two accessors exist only for the sections. A per-tensor affine `zero_point` is signed
and its `scale` is real, so neither can go through the `u64` accessor:

```c
/* value = (stored - zero_point) * scale */
double  scale;
int64_t zero_point;
hurray_descriptor_field_f64(desc, HURRAY_QUANT_FIELD_PER_TENSOR_SCALE, 0, &scale);
hurray_descriptor_field_i64(desc, HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT, 0, &zero_point);
```

A zero point is routinely negative. Reading one through the unsigned accessor is not
merely wrong but *plausibly* wrong — `-128` would arrive as `18446744073709551488` —
so the unsigned accessor rejects the field outright rather than widening it.

**Statistics are mask-guarded, and the accessor enforces the guard.** Every statistics
field is physically present whatever `computed_mask` says, holding whatever the writer
left there. Rather than hand you those bytes and trust you to check the bit, a field
whose guard bit is clear reads as `HURRAY_ERR_FIELD_NOT_PRESENT`:

```c
uint64_t nnz;
if (hurray_descriptor_field_u64(desc, HURRAY_STATS_FIELD_NNZ, 0, &nnz) == HURRAY_OK) {
    /* nnz was actually computed by the writer */
}

/* Or branch on the mask directly — it is always readable. */
uint64_t mask;
hurray_descriptor_field_u64(desc, HURRAY_STATS_FIELD_COMPUTED_MASK, 0, &mask);
```

Shard bounds are arrays, so they use the same capacity/query pattern as the strides
above:

```c
size_t len = 0;
hurray_descriptor_field_u64_array(desc, HURRAY_SHARD_FIELD_PARENT_SHAPE, 0, NULL, &len);

uint64_t *parent = malloc(len * sizeof(uint64_t));
hurray_descriptor_field_u64_array(desc, HURRAY_SHARD_FIELD_PARENT_SHAPE, 0, parent, &len);
```

**Rust:**
```rust
use hurray_core::descriptor::{Statistics, StatisticsMask};
use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerTensorAffine,
    QuantizationDescriptor, Shape, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_ffi::{
    descriptor::hurray_descriptor_destroy,
    field::{
        hurray_descriptor_field_f64, hurray_descriptor_field_i64, hurray_descriptor_field_u64,
        HURRAY_QUANT_FIELD_PER_TENSOR_SCALE, HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT,
        HURRAY_QUANT_FIELD_SCHEME_TAG, HURRAY_STATS_FIELD_NNZ, HURRAY_STATS_FIELD_VALUE_MEAN,
    },
    hurray_descriptor_decode, HurrayDescriptor, HURRAY_ERR_FIELD_NOT_PRESENT, HURRAY_OK,
};

let buffers = (0..1)
    .map(|_| {
        BufferHandle::new(
            256,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .unwrap()
    })
    .collect();

// int8 weights: value = (stored - zero_point) * scale, with only nnz computed.
let quantization = QuantizationDescriptor::PerTensorAffine(
    PerTensorAffine::new(0.0078125, -128).unwrap(),
);
let statistics = Statistics {
    computed_mask: StatisticsMask(StatisticsMask::NNZ_VALID),
    nnz: 4096,
    sparsity_ratio: 0.0,
    value_min: 0.0,
    value_max: 0.0,
    value_abs_max: 0.0,
    value_mean: 0.0,
    value_stddev: 0.0,
    nm_n: 0,
    nm_m: 0,
    has_nan: false,
    has_inf: false,
};
let bytes = TensorDescriptor::new(
    1,
    0,
    ElementType::Int8,
    Shape::new(vec![64, 64]).unwrap(),
    0,
    LayoutDescriptor::RowMajor,
    buffers,
    Some(quantization.encode_to_vec()),
    None,
    Some(statistics),
    None,
)
.unwrap()
.encode()
.unwrap();

let mut desc: *mut HurrayDescriptor = std::ptr::null_mut();
assert_eq!(
    unsafe { hurray_descriptor_decode(bytes.as_ptr(), bytes.len(), &mut desc) },
    HURRAY_OK
);

// The scheme tag first: it says which identifiers apply.
let mut scheme: u64 = 0;
unsafe { hurray_descriptor_field_u64(desc, HURRAY_QUANT_FIELD_SCHEME_TAG, 0, &mut scheme) };
assert_eq!(scheme, 0x01);

let mut scale: f64 = 0.0;
let mut zero_point: i64 = 0;
unsafe {
    hurray_descriptor_field_f64(desc, HURRAY_QUANT_FIELD_PER_TENSOR_SCALE, 0, &mut scale);
    hurray_descriptor_field_i64(
        desc,
        HURRAY_QUANT_FIELD_PER_TENSOR_ZERO_POINT,
        0,
        &mut zero_point,
    );
}
assert_eq!(scale, 0.0078125);
assert_eq!(zero_point, -128);

// nnz was computed; the mean was not, and reads as absent rather than as 0.0.
let mut nnz: u64 = 0;
assert_eq!(
    unsafe { hurray_descriptor_field_u64(desc, HURRAY_STATS_FIELD_NNZ, 0, &mut nnz) },
    HURRAY_OK
);
assert_eq!(nnz, 4096);

let mut mean: f64 = f64::NAN;
assert_eq!(
    unsafe { hurray_descriptor_field_f64(desc, HURRAY_STATS_FIELD_VALUE_MEAN, 0, &mut mean) },
    HURRAY_ERR_FIELD_NOT_PRESENT
);
assert!(mean.is_nan(), "a failed read leaves the output untouched");

assert_eq!(unsafe { hurray_descriptor_destroy(desc) }, HURRAY_OK);
```

A runnable walk through a per-tensor quantized tensor, a per-block quantized shard with
statistics, and a plain tensor carrying no sections at all:

```bash
cargo run -p hurray-ffi --example section_fields
```

## Key takeaways

- **No panics cross the boundary.** Every function returns `HURRAY_OK` or a
  negative error code. `HURRAY_ERR_INTERNAL_PANIC` means the library panicked
  internally — the handle is in an undefined state and MUST NOT be reused.
- **Opaque handles.** `HurrayBuffer`, `HurrayDescriptor`, `HurrayReader`, and
  `HurrayWriter` are opaque; never dereference or cast their pointers.
- **Check `layout_tag` before reading payload fields, and the quantization
  scheme tag before reading quantization fields.** A field identifier names the
  sub-object it belongs to, so asking the wrong descriptor is diagnosable
  (`HURRAY_ERR_FIELD_NOT_APPLICABLE`) rather than silent — but the tag is what
  tells you which identifier to use in the first place.
- **Absence is a status, never a value.** An absent section, an unset optional
  field, a symmetric scheme's zero point, and an uncomputed statistic all read as
  `HURRAY_ERR_FIELD_NOT_PRESENT` with the output left untouched.
- **`HURRAY_ERR_NULL_POINTER` for null required arguments.** Every function
  checks its required pointer arguments and returns this code immediately if
  any is null. Optional context pointers (e.g., `release_context`) MAY be null.
- **Exactly one destroy per create.** Each handle created by a `*_from_ptr`,
  `*_decode`, or `*_new` function MUST be destroyed exactly once.
- **Read the ABI version before anything else.** For a `HurrayTensorContext` this is
  normative, not advisory: it is what lets later versions add accessors without
  breaking consumers compiled against an earlier header.
