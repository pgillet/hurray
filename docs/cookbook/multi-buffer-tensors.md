# Multi-Buffer Tensors

Most tensors have one buffer. Anything whose descriptor references a second one has
more:

| Feature | Extra buffers |
|---|---|
| Per-channel affine quantization | scale buffer, optional zero-point buffer |
| NF4 / MXFP quantization | scale buffer |
| COO sparse | index array |
| CSR / CSC sparse | primary and secondary index arrays |
| Block-paged | page table |

Per-tensor affine quantization is the exception: its `scale` and `zero_point` are
inline in the descriptor, so a per-tensor-affine tensor still has exactly one buffer.

The rule that makes this work is positional: **element `i` of the transport is buffer
index `i` of the descriptor's buffer table**. A `scale_buffer_index` of `1` means the
second buffer handed over, whatever the transport. Drop a buffer along the way and the
descriptor still decodes cleanly — it just points at something that was never
delivered.

## Carrying every buffer in process

`__hurray__` puts all of a tensor's buffers in **one** capsule wrapping a
`HurrayBufferList` (ADR-030). One capsule, one lifetime, one destroy.

Sparse is not a special case here — it is simply the multi-buffer case, which is why
there is no separate `__hurray_sparse_buffer__` to probe for.

<div class="lang-tabs">

```rust
use hurray_ffi::buffer::{hurray_buffer_byte_size, hurray_buffer_from_ptr};
use hurray_ffi::buffer_list::{
    hurray_buffer_list_destroy, hurray_buffer_list_get, hurray_buffer_list_len,
    hurray_buffer_list_new, hurray_buffer_list_push,
};
use hurray_ffi::{HurrayBuffer, HurrayBufferList, HURRAY_OK};

#[repr(align(64))]
struct Aligned([u8; 64]);

fn main() {
    let mut weights = Aligned([0xAB; 64]);
    let mut scales = Aligned([0x01; 64]);

    let mut list: *mut HurrayBufferList = std::ptr::null_mut();
    // SAFETY: out-pointer is a valid stack variable.
    unsafe { hurray_buffer_list_new(2, &mut list) };

    // Push order is descriptor buffer-table order: weights are index 0, the
    // per-channel scales index 1 — what scale_buffer_index refers to.
    for data in [&mut weights, &mut scales] {
        let mut handle: *mut HurrayBuffer = std::ptr::null_mut();
        // SAFETY: data is 64-byte aligned and 64 bytes long.
        unsafe {
            hurray_buffer_from_ptr(
                data.0.as_mut_ptr().cast(),
                64,
                64,
                0x00, // CPU
                0x00, // ProducerSynced
                0x00, // Standard
                None,
                std::ptr::null_mut(),
                &mut handle,
            );
            // Push transfers ownership of the handle to the list.
            hurray_buffer_list_push(list, handle);
        }
    }

    let mut len: u64 = 0;
    // SAFETY: list is live.
    unsafe { hurray_buffer_list_len(list, &mut len) };
    assert_eq!(len, 2);

    for index in 0..len {
        let mut borrowed: *mut HurrayBuffer = std::ptr::null_mut();
        // SAFETY: list is live and index < len. The handle is BORROWED — the list
        // owns it, so it must not be destroyed here.
        unsafe { hurray_buffer_list_get(list, index, &mut borrowed) };
        let mut byte_size: u64 = 0;
        // SAFETY: borrowed is a live handle owned by the list.
        unsafe { hurray_buffer_byte_size(borrowed, &mut byte_size) };
        println!("buffer[{index}]: {byte_size} bytes");
    }

    // Destroys the list and every handle it owns, exactly once, then nulls the
    // caller's pointer — so a second destroy is a safe no-op.
    // SAFETY: first and only destroy.
    unsafe { hurray_buffer_list_destroy(&mut list) };
    assert!(list.is_null());
}
```

```python
import numpy as np
import hurray

# A COO tensor keeps values in one buffer and coordinates in another.
values = np.array([5.0, 7.0], dtype=np.float32)
indices = np.array([[0, 0], [1, 1]], dtype=np.uint64)  # [nnz, rank]
sparse = hurray.sparse_coo(values, indices, [2, 2])

# One protocol for every tensor kind — probe exactly as for a dense tensor.
assert hasattr(sparse, "__hurray__")
assert not hasattr(sparse, "__hurray_sparse_buffer__")

capsule = sparse.__hurray__()

# The consumer receives the full descriptor with every buffer attached, in
# descriptor order: values first, then the index array.
back = hurray.from_hurray(sparse)
assert back.shape == (2, 2)
assert back.dtype == hurray.float32
```

</div>

## Ownership: one owner, everything else borrowed

The list **owns** every handle in it. `hurray_buffer_list_get` returns a *borrowed*
pointer: the caller reads it and must not destroy it. Destroying the list destroys
every handle exactly once.

This is the same discipline as Arrow's C Data Interface, where a consumer releases the
base structure but never its children. Getting it wrong in the other direction — a
consumer destroying a borrowed handle — double-frees when the list is destroyed.

`hurray_buffer_list_destroy` takes a **pointer to your pointer** and writes null
through it:

```rust
# use hurray_ffi::buffer_list::{hurray_buffer_list_destroy, hurray_buffer_list_new};
# use hurray_ffi::HurrayBufferList;
# let mut list: *mut HurrayBufferList = std::ptr::null_mut();
# unsafe { hurray_buffer_list_new(0, &mut list) };
// SAFETY: list is live; first and only destroy.
unsafe { hurray_buffer_list_destroy(&mut list) };
assert!(list.is_null());

// Idempotent: destroying an already-nulled pointer does nothing.
// SAFETY: *list is null, treated as a no-op.
unsafe { hurray_buffer_list_destroy(&mut list) };
```

Nulling the caller's variable is the sound half of Arrow's "release marks the structure
released" trick. Arrow can leave a marker *inside* the struct because the consumer owns
that memory and it outlives the call; here the list allocation is freed, so the only
memory that can safely be marked is the caller's own pointer.

## Files

`hurray.save()` writes every buffer of a tensor, and `hurray.load()` reads them back in
descriptor order. A multi-buffer tensor round-trips through a `.hrry` file byte for
byte.

A tensor whose buffer count disagrees with its descriptor's buffer table is rejected on
both paths, rather than producing a tensor whose buffer indices do not resolve.

## Version check

The capsule shape changed in C ABI version 3. A consumer built against version 2 that
receives a version 3 capsule raises `hurray.UnsupportedError` instead of misreading a
`HurrayBufferList` as a `HurrayBuffer`:

```python
import hurray

# Producer and consumer must agree on the ABI version; the check happens before
# the capsule pointer is ever dereferenced.
tensor = hurray.Tensor(bytes(16), hurray.float32, [4])
received = hurray.from_hurray(tensor)  # UnsupportedError on mismatch
```

## What this does not yet do

The transport carries any number of buffers, but Python cannot yet *author* a
quantization descriptor, so a per-channel-quantized tensor cannot be built from Python
even though it now travels correctly once built. Descriptor-authoring classes are the
next step.

## See also

- [Native Interchange Protocol](hurray-python-native-buffer.md) — the
  single-buffer basics and how the protocol compares to DLPack
- [Quantized Inference](quantized-inference.md) — which schemes need a scale buffer
- [Sparse Tensors with SciPy](hurray-python-sparse-scipy.md) — building COO/CSR/CSC
  tensors
- `cargo run --example buffer_list -p hurray-ffi`
- `python hurray-python/examples/multi_buffer.py`
