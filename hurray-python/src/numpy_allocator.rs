//! A 64-byte-aligned NumPy data-memory allocator (NEP 49; ADR-037 § 6a).
//!
//! ```python
//! with hurray.aligned_allocator():
//!     weights = numpy.zeros(shape, dtype=numpy.float32)
//! tensor = hurray.from_numpy(weights, copy=False)     # genuinely zero-copy
//! ```
//!
//! ## Why this exists
//!
//! `buffer-protocol.md` § Alignment requires every non-empty buffer to start on a
//! 64-byte boundary. NumPy does not promise one, and a large array served by a fresh
//! `mmap` never has one — glibc puts a 16-byte chunk header before the pointer — so
//! `from_numpy` copies most arrays (ADR-037 § 6).
//!
//! NEP 49 is the sanctioned answer, and alignment is the first motivation it lists.
//! NumPy considered guaranteeing alignment, declined, and shipped this hook instead. So
//! this is not a workaround: it turns "Hurray always copies NumPy arrays" into "arrays
//! allocated for Hurray are not copied", which is the bargain that matters to a producer
//! writing its own checkpoints.
//!
//! ## Reaching the two functions
//!
//! `PyDataMem_SetHandler` and `PyDataMem_GetHandler` are C-API only — NumPy exposes
//! `get_handler_name` to Python but no setter — and the `numpy` crate this binding
//! depends on declares them at API slots 304/305 but leaves them **commented out**. So
//! the table is fetched from NumPy's `_ARRAY_API` capsule here and indexed directly.
//!
//! Both MUST be called with the GIL held: they read and write a `ContextVar`, and
//! calling them without it segfaults immediately. The allocator callbacks are the
//! opposite case — NumPy calls them with the GIL released for large allocations, so they
//! touch no Python state at all.

use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
use std::ffi::{c_char, c_void};
use std::ptr;

use pyo3::ffi;
use pyo3::prelude::*;

use hurray_core::MIN_BUFFER_ALIGNMENT;

use crate::errors::UnsupportedError;

// ── The allocator ─────────────────────────────────────────────────────────────

/// Bytes reserved before every payload to record its size.
///
/// A whole alignment's worth rather than a bare `usize`: the payload has to land on a
/// 64-byte boundary too, so anything smaller would be padding either way.
///
/// The header exists because NumPy's `realloc` hook is handed the *new* size only, and
/// Rust's deallocator needs the old layout. Keeping the size next to the block also
/// means every `alloc`/`dealloc` pair is matched inside Rust, which is what NEP 49's
/// implementation notes warn to preserve.
const HEADER: usize = MIN_BUFFER_ALIGNMENT as usize;

/// The layout backing a payload of `payload` bytes, or `None` if the size overflows.
fn block_layout(payload: usize) -> Option<Layout> {
    Layout::from_size_align(HEADER.checked_add(payload)?, HEADER).ok()
}

/// Allocate `payload` bytes at 64-byte alignment, returning a pointer past the header.
///
/// # Safety
///
/// The returned pointer must be released through [`hurray_free`], which reads the header
/// this writes.
unsafe fn allocate(payload: usize, zeroed: bool) -> *mut c_void {
    let Some(layout) = block_layout(payload) else {
        return ptr::null_mut();
    };
    // SAFETY: layout has non-zero size — HEADER is 64 even when payload is 0, which also
    // means a zero-byte request never reaches the allocator as a zero-size layout.
    let base = unsafe {
        if zeroed {
            alloc_zeroed(layout)
        } else {
            alloc(layout)
        }
    };
    if base.is_null() {
        // Returning null is the contract: NumPy raises MemoryError. Aborting the process
        // the way Rust's own OOM handler does would be wrong on the far side of a C ABI.
        return ptr::null_mut();
    }
    // SAFETY: base is a fresh allocation of at least HEADER bytes, aligned to 64 and so
    // to align_of::<usize>().
    unsafe { (base as *mut usize).write(payload) };
    // SAFETY: the allocation covers HEADER + payload bytes.
    unsafe { base.add(HEADER) as *mut c_void }
}

/// The size recorded in the header of a live block, and the block's base pointer.
///
/// # Safety
///
/// `ptr` must be non-null and must have come from [`allocate`].
unsafe fn header_of(ptr: *mut c_void) -> (*mut u8, usize) {
    // SAFETY: allocate returned base + HEADER, so this recovers base.
    let base = unsafe { (ptr as *mut u8).sub(HEADER) };
    // SAFETY: allocate wrote the payload size there.
    (base, unsafe { (base as *mut usize).read() })
}

unsafe extern "C" fn hurray_malloc(_ctx: *mut c_void, size: usize) -> *mut c_void {
    // SAFETY: the result is only ever freed through hurray_free, which NumPy pairs with
    // this handler for the lifetime of every array allocated under it.
    unsafe { allocate(size, false) }
}

unsafe extern "C" fn hurray_calloc(_ctx: *mut c_void, nelem: usize, elsize: usize) -> *mut c_void {
    let Some(total) = nelem.checked_mul(elsize) else {
        return ptr::null_mut();
    };
    // SAFETY: as above.
    unsafe { allocate(total, true) }
}

unsafe extern "C" fn hurray_realloc(
    _ctx: *mut c_void,
    ptr: *mut c_void,
    new_size: usize,
) -> *mut c_void {
    if ptr.is_null() {
        // SAFETY: as above.
        return unsafe { allocate(new_size, false) };
    }
    // SAFETY: a non-null pointer here came from this handler's malloc or calloc.
    let (base, old_size) = unsafe { header_of(ptr) };
    let (Some(old_layout), Some(new_layout)) = (block_layout(old_size), block_layout(new_size))
    else {
        return ptr::null_mut();
    };
    // SAFETY: base and old_layout are the pointer and layout the block was allocated
    // with, and new_layout.size() is non-zero. `realloc` preserves layout.align(), so the
    // result is still 64-byte aligned — which is the entire point of this handler.
    let new_base = unsafe { realloc(base, old_layout, new_layout.size()) };
    if new_base.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the new block is at least HEADER bytes and correctly aligned.
    unsafe { (new_base as *mut usize).write(new_size) };
    // SAFETY: the new block covers HEADER + new_size bytes.
    unsafe { new_base.add(HEADER) as *mut c_void }
}

unsafe extern "C" fn hurray_free(_ctx: *mut c_void, ptr: *mut c_void, _size: usize) {
    if ptr.is_null() {
        return;
    }
    // The size NumPy passes is ignored in favour of the header: the header is what
    // `alloc` was actually called with, so freeing from it cannot disagree with the
    // allocation even if the caller's bookkeeping does.
    // SAFETY: a non-null pointer here came from this handler.
    let (base, size) = unsafe { header_of(ptr) };
    if let Some(layout) = block_layout(size) {
        // SAFETY: base and layout are the pointer and layout from the allocation.
        unsafe { dealloc(base, layout) };
    }
}

// ── NumPy's handler structs ───────────────────────────────────────────────────

/// `PyDataMemAllocator` from `numpy/ndarraytypes.h`.
#[repr(C)]
struct DataMemAllocator {
    ctx: *mut c_void,
    malloc: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    calloc: unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void,
    realloc: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void, *mut c_void, usize),
}

/// `PyDataMem_Handler` from `numpy/ndarraytypes.h`.
#[repr(C)]
struct DataMemHandler {
    name: [c_char; 127],
    version: u8,
    allocator: DataMemAllocator,
}

// SAFETY: HANDLER is immutable for the life of the process and holds only function
// pointers and a null ctx. NumPy reads it from any thread and never writes to it.
unsafe impl Sync for DataMemHandler {}

/// Pad a name into the fixed 127-byte field NumPy declares.
const fn handler_name(bytes: &[u8]) -> [c_char; 127] {
    let mut name = [0 as c_char; 127];
    let mut index = 0;
    while index < bytes.len() {
        name[index] = bytes[index] as c_char;
        index += 1;
    }
    name
}

/// The handler NumPy is pointed at. `static`, deliberately: an array outlives the `with`
/// block that allocated it, and is freed through the handler it was born under — so
/// anything with a destructor would be a use-after-free waiting for interpreter shutdown.
static HANDLER: DataMemHandler = DataMemHandler {
    name: handler_name(b"hurray_aligned"),
    version: 1,
    allocator: DataMemAllocator {
        ctx: ptr::null_mut(),
        malloc: hurray_malloc,
        calloc: hurray_calloc,
        realloc: hurray_realloc,
        free: hurray_free,
    },
};

/// The capsule name NEP 49 requires of a handler.
const HANDLER_CAPSULE_NAME: &[u8] = b"mem_handler\0";

// ── Reaching PyDataMem_SetHandler ─────────────────────────────────────────────

/// Index of `PyDataMem_SetHandler` in `PyArray_API`, per `numpy/__multiarray_api.h`.
const SLOT_SET_HANDLER: usize = 304;

type SetHandlerFn = unsafe extern "C" fn(*mut ffi::PyObject) -> *mut ffi::PyObject;

/// NumPy's multiarray module, once it is known to be new enough for NEP 49.
fn multiarray_module(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let numpy = py.import("numpy")?;
    let version: String = numpy.getattr("__version__")?.extract()?;
    let parsed = py
        .import("numpy.lib")?
        .getattr("NumpyVersion")?
        .call1((version.as_str(),))?;
    let major: u8 = parsed.getattr("major")?.extract()?;
    let minor: u8 = parsed.getattr("minor")?.extract()?;

    if (major, minor) < (1, 22) {
        return Err(UnsupportedError::new_err(format!(
            "hurray.aligned_allocator needs NumPy >= 1.22 for the pluggable data-memory \
             allocator (NEP 49); this is NumPy {version}"
        )));
    }

    // numpy 2 renamed the private core module.
    let name = if major >= 2 {
        "numpy._core.multiarray"
    } else {
        "numpy.core.multiarray"
    };
    Ok(py.import(name)?.into_any())
}

/// NumPy's C function table.
///
/// The table lives inside a capsule held by a module in `sys.modules`, so the pointer
/// stays valid for the life of the interpreter — the same assumption the `numpy` crate
/// makes when it caches this.
fn api_table(py: Python<'_>) -> PyResult<*const *const c_void> {
    let capsule = multiarray_module(py)?.getattr("_ARRAY_API")?;
    // SAFETY: `_ARRAY_API` is an unnamed capsule whose pointer is the API table.
    let table = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), ptr::null()) };
    if table.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(table as *const *const c_void)
}

/// Install `handler` and return the previous one.
///
/// `handler` may be a `mem_handler` capsule or a handler NumPy handed back earlier.
fn set_handler(py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let table = api_table(py)?;
    // SAFETY: the table has at least 306 entries for NumPy >= 1.22, checked above.
    let slot = unsafe { *table.add(SLOT_SET_HANDLER) };
    if slot.is_null() {
        return Err(UnsupportedError::new_err(
            "this NumPy build exposes no PyDataMem_SetHandler; the pluggable allocator \
             (NEP 49) is unavailable",
        ));
    }
    // SAFETY: slot 304 holds PyDataMem_SetHandler, whose signature is
    // `PyObject *(PyObject *)`. It is non-null, so the transmute to a fn pointer is
    // valid.
    let set: SetHandlerFn = unsafe { std::mem::transmute::<*const c_void, SetHandlerFn>(slot) };

    // SAFETY: the GIL is held (we have `py`), which SetHandler requires — it reads and
    // writes a ContextVar. It borrows `handler` and returns a new reference to the old.
    let previous = unsafe { set(handler.as_ptr()) };
    if previous.is_null() {
        return Err(PyErr::fetch(py));
    }
    // SAFETY: SetHandler returns a new reference, which this takes ownership of.
    Ok(unsafe { Bound::from_owned_ptr(py, previous) }.unbind())
}

/// A capsule pointing at [`HANDLER`], in the form NEP 49 expects.
fn handler_capsule(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // SAFETY: HANDLER is a `static`, so the capsule may outlive every caller; the name is
    // a NUL-terminated literal with static lifetime, as PyCapsule_New requires; and no
    // destructor is given because there is nothing to destroy.
    let capsule = unsafe {
        ffi::PyCapsule_New(
            &HANDLER as *const DataMemHandler as *mut c_void,
            HANDLER_CAPSULE_NAME.as_ptr() as *const c_char,
            None,
        )
    };
    // SAFETY: PyCapsule_New returns a new reference or null with an exception set.
    unsafe { Bound::from_owned_ptr_or_err(py, capsule) }.map(Bound::unbind)
}

// ── The context manager ───────────────────────────────────────────────────────

/// The context manager returned by `aligned_allocator`.
///
/// ## Examples
///
/// ```python
/// import numpy as np, hurray
///
/// with hurray.aligned_allocator():
///     arr = np.zeros(1 << 20, dtype=np.float32)
///
/// assert arr.__array_interface__["data"][0] % hurray.MIN_BUFFER_ALIGNMENT == 0
/// t = hurray.from_numpy(arr, copy=False)      # no copy, and none was needed
/// ```
#[pyclass(name = "AlignedAllocatorCtx")]
pub struct AlignedAllocatorCtx {
    /// The handler displaced on entry, restored on exit. `None` while not active.
    previous: Option<Py<PyAny>>,
}

#[pymethods]
impl AlignedAllocatorCtx {
    #[new]
    pub fn new() -> Self {
        AlignedAllocatorCtx { previous: None }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<()> {
        if slf.previous.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this hurray.aligned_allocator() is already active; call it again to \
                 nest, rather than re-entering the same object",
            ));
        }
        let capsule = handler_capsule(py)?;
        slf.previous = Some(set_handler(py, capsule.bind(py))?);
        Ok(())
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_val: &Bound<'_, PyAny>,
        _exc_tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(previous) = slf.previous.take() {
            set_handler(py, previous.bind(py))?;
        }
        Ok(false)
    }
}

impl Default for AlignedAllocatorCtx {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocate NumPy arrays 64-byte aligned for the duration of a `with` block.
///
/// Arrays allocated inside the block satisfy the format's alignment floor, so
/// `hurray.from_numpy(arr, copy=False)` shares their buffer instead of copying it. Arrays
/// allocated outside are untouched.
///
/// The handler is stored **per array**, so an array allocated inside the block is freed
/// through the matching deallocator long after the block exits. It is also thread- and
/// context-local, so installing it cannot leak into unrelated code — with one consequence
/// worth knowing: **a thread started inside the block does not inherit it**, and arrays
/// that thread allocates get NumPy's default allocator and will be copied on ingest like
/// any other.
///
/// Blocks nest: each restores the handler that was in place when it was entered.
///
/// ## Errors
///
/// - `hurray.UnsupportedError` — NumPy is older than 1.22, which predates NEP 49.
///
/// ## Examples
///
/// ```python
/// import numpy as np, hurray
///
/// with hurray.aligned_allocator():
///     weights = np.zeros((512, 512), dtype=np.float32)
///
/// tensor = hurray.from_numpy(weights, copy=False)
/// assert tensor.buffer_handles[0].alignment >= hurray.MIN_BUFFER_ALIGNMENT
/// ```
#[pyfunction]
pub fn aligned_allocator() -> AlignedAllocatorCtx {
    AlignedAllocatorCtx::new()
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aligned_allocator, m)?)?;
    m.add_class::<AlignedAllocatorCtx>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header must not disturb the alignment it exists to guarantee.
    #[test]
    fn every_allocation_is_aligned_and_remembers_its_size() {
        for payload in [0usize, 1, 63, 64, 4096, 1 << 20] {
            // SAFETY: freed below through the same handler.
            let ptr = unsafe { allocate(payload, false) };
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize % HEADER, 0, "payload {payload}");

            // SAFETY: ptr came from allocate.
            let (_, recorded) = unsafe { header_of(ptr) };
            assert_eq!(recorded, payload);

            // SAFETY: ptr came from allocate and is freed exactly once.
            unsafe { hurray_free(ptr::null_mut(), ptr, payload) };
        }
    }

    #[test]
    fn calloc_zeroes_the_payload() {
        // SAFETY: freed below.
        let ptr = unsafe { hurray_calloc(ptr::null_mut(), 128, 4) };
        assert!(!ptr.is_null());
        // SAFETY: the block covers 512 bytes.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, 512) };
        assert!(bytes.iter().all(|&b| b == 0));
        // SAFETY: allocated just above.
        unsafe { hurray_free(ptr::null_mut(), ptr, 512) };
    }

    #[test]
    fn calloc_refuses_a_size_that_overflows() {
        // SAFETY: no allocation happens on the overflow path.
        let ptr = unsafe { hurray_calloc(ptr::null_mut(), usize::MAX, 2) };
        assert!(ptr.is_null());
    }

    /// `realloc` is the reason the header exists: NumPy passes the new size only.
    #[test]
    fn realloc_keeps_the_bytes_the_alignment_and_the_new_size() {
        // SAFETY: freed below.
        let ptr = unsafe { hurray_calloc(ptr::null_mut(), 64, 1) };
        // SAFETY: the block covers 64 bytes.
        unsafe { (ptr as *mut u8).write_bytes(0xAB, 64) };

        // SAFETY: ptr came from this handler.
        let grown = unsafe { hurray_realloc(ptr::null_mut(), ptr, 8192) };
        assert!(!grown.is_null());
        assert_eq!(grown as usize % HEADER, 0);

        // SAFETY: grown came from hurray_realloc.
        let (_, recorded) = unsafe { header_of(grown) };
        assert_eq!(recorded, 8192);

        // SAFETY: the first 64 bytes were copied by realloc.
        let bytes = unsafe { std::slice::from_raw_parts(grown as *const u8, 64) };
        assert!(bytes.iter().all(|&b| b == 0xAB));

        // SAFETY: freed exactly once.
        unsafe { hurray_free(ptr::null_mut(), grown, 8192) };
    }

    #[test]
    fn realloc_of_null_allocates() {
        // SAFETY: null is the documented "allocate fresh" case.
        let ptr = unsafe { hurray_realloc(ptr::null_mut(), ptr::null_mut(), 256) };
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % HEADER, 0);
        // SAFETY: allocated just above.
        unsafe { hurray_free(ptr::null_mut(), ptr, 256) };
    }

    #[test]
    fn freeing_null_is_a_no_op() {
        // SAFETY: the null case returns immediately.
        unsafe { hurray_free(ptr::null_mut(), ptr::null_mut(), 0) };
    }

    /// NumPy reads the name out of the struct, so it has to survive the padding.
    #[test]
    fn the_handler_declares_its_name_and_version() {
        assert_eq!(HANDLER.version, 1);
        let name: Vec<u8> = HANDLER
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        assert_eq!(name, b"hurray_aligned");
    }
}
