//! The streaming interchange format from Python (ADR-035).
//!
//! ```python
//! with hurray.StreamWriter(sock) as writer:
//!     for tensor in tensors:
//!         writer.write(tensor)
//!
//! for tensor in hurray.StreamReader(sock):
//!     ...                                    # each tensor as it arrives
//! ```
//!
//! ## Blocking, with an owned runtime
//!
//! Each stream object owns a current-thread `tokio` runtime for its lifetime and
//! releases the GIL around every call into it. `load` and `save` already do this per
//! call; a stream is long-lived, so the runtime moves from the call to the object
//! (ADR-035 § 1). An `asyncio` surface is deferred, not rejected — see the ADR.
//!
//! ## Why `next_item`, not `next_tensor`
//!
//! `next_tensor` would hand back a composite *head* as though it were an ordinary
//! tensor — it owns no buffers — and then read the head's members as separate
//! top-level tensors. The caller would get a stream that decoded "successfully" while
//! silently losing the composition. `next_item` recognises the head, and this module
//! refuses it by name (ADR-035 § 4).

use std::os::fd::RawFd;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::runtime::Runtime;

use hurray_io::stream::{StreamItem, StreamReader as IoReader, StreamWriter as IoWriter};

use crate::buffer::BufferStore;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::errors::{FileError, InvalidDescriptorError, StreamError, UnsupportedError};
use crate::tensor::Tensor;

/// Any source `hurray-io` can read a stream from.
type BoxedSource = Box<dyn AsyncRead + Unpin + Send>;

/// Any sink `hurray-io` can write a stream to.
type BoxedSink = Box<dyn AsyncWrite + Unpin + Send>;

// ── Error mapping ─────────────────────────────────────────────────────────────

/// Maps a `hurray-io` error to the Python exception that describes it.
///
/// Three kinds, kept distinguishable so a caller can tell a broken peer from a broken
/// descriptor from a broken socket:
///
/// - framing — a truncated frame, a bad header, an oversized frame → `hurray.StreamError`
/// - descriptor — anything core rejected → `hurray.InvalidDescriptorError`
/// - transport — the underlying read or write failed → `hurray.FileError` (an `OSError`)
fn stream_err(e: hurray_io::Error) -> PyErr {
    match e {
        hurray_io::Error::Io(io) => FileError::new_err(io.to_string()),
        hurray_io::Error::Core(core) => InvalidDescriptorError::new_err(core.to_string()),
        other => StreamError::new_err(other.to_string()),
    }
}

/// Builds the current-thread runtime a stream object owns for its lifetime.
fn build_runtime() -> PyResult<Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| FileError::new_err(format!("failed to start the stream runtime: {e}")))
}

// ── Transport resolution ──────────────────────────────────────────────────────

/// Duplicates `fd` so the stream owns a descriptor of its own.
///
/// Without the `dup` the stream would close the caller's descriptor out from under
/// them on `finish` — a socket handed to a writer would stop working the moment the
/// writer was done with it (ADR-035 § 3).
fn dup_fd(fd: RawFd) -> PyResult<std::os::fd::OwnedFd> {
    // SAFETY: borrow_raw only asserts the descriptor is valid for this call; the clone
    // below is what takes ownership, and the borrow is dropped immediately after.
    let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
    borrowed
        .try_clone_to_owned()
        .map_err(|e| FileError::new_err(format!("could not duplicate file descriptor {fd}: {e}")))
}

/// Opens the `std::fs::File` a duplicated descriptor becomes.
///
/// `tokio::fs::File` reads and writes on the blocking pool, which works for a socket
/// or a pipe as well as a regular file. A dedicated reactor type per descriptor kind
/// would be faster, but it would require knowing which kind this is, and the caller
/// gave us an integer.
fn file_from_fd(fd: std::os::fd::OwnedFd) -> tokio::fs::File {
    tokio::fs::File::from_std(std::fs::File::from(fd))
}

/// Resolves the `source` argument to something readable: a path, an object with
/// `fileno()`, or bytes.
fn resolve_source(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    rt: &Runtime,
) -> PyResult<BoxedSource> {
    if let Ok(data) = source.extract::<Vec<u8>>() {
        return Ok(Box::new(std::io::Cursor::new(data)));
    }
    if let Ok(fd) = fileno_of(source) {
        return Ok(Box::new(file_from_fd(dup_fd(fd)?)));
    }
    let path: String = source.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a path, bytes, or an object with fileno(); \
             an object without a file descriptor (io.BytesIO) must be passed as .getvalue()",
        )
    })?;
    let file = py
        .detach(|| rt.block_on(tokio::fs::File::open(&path)))
        .map_err(|e| FileError::new_err(format!("could not open {path}: {e}")))?;
    Ok(Box::new(file))
}

/// The file descriptor behind an object, if it has one.
fn fileno_of(obj: &Bound<'_, PyAny>) -> PyResult<RawFd> {
    let fd = obj.call_method0("fileno")?.extract::<i32>()?;
    if fd < 0 {
        return Err(FileError::new_err(format!("fileno() returned {fd}")));
    }
    Ok(fd)
}

// ── In-memory sink ────────────────────────────────────────────────────────────

/// The sink behind `StreamWriter()` with no destination: an in-memory buffer the
/// writer hands back from `getvalue()`.
#[derive(Clone, Default)]
struct SharedBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl SharedBuffer {
    fn take(&self) -> Vec<u8> {
        match self.0.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            // A poisoned lock means a writer thread panicked mid-write; the bytes are
            // not trustworthy, so report empty rather than a torn stream.
            Err(_) => Vec::new(),
        }
    }
}

impl AsyncWrite for SharedBuffer {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        match self.0.lock() {
            Ok(mut guard) => {
                guard.extend_from_slice(buf);
                std::task::Poll::Ready(Ok(buf.len()))
            }
            Err(_) => std::task::Poll::Ready(Err(std::io::Error::other("stream buffer poisoned"))),
        }
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(Ok(()))
    }
}

/// A buffer's raw parts, carried across a GIL release.
///
/// `Python::detach` requires a `Send` closure, and neither `PyRef` nor a raw pointer
/// is `Send`. The bytes themselves are stable for the duration of the call: the
/// caller holds a strong reference to the tensor that owns them, so releasing the GIL
/// cannot collect it, and no other thread has a path to the stores.
struct SendSlice(*const u8, usize);

// SAFETY: see the type's documentation — the pointee outlives the call, and only the
// detached thread reads it.
unsafe impl Send for SendSlice {}

// ── StreamReader ──────────────────────────────────────────────────────────────

/// Reads tensors from a Hurray stream, one at a time.
///
/// Iterating yields a `hurray.Tensor` per tensor on the wire and stops at a clean end
/// of stream. Nothing buffers the whole input: a tensor is available as soon as its
/// descriptor and buffers have arrived, which is the property the streaming format
/// exists for.
///
/// ## Sources
///
/// A filesystem path, an object with `fileno()` (a socket, a pipe, an open file), or
/// `bytes`. An object with no descriptor — `io.BytesIO` — should be passed as
/// `.getvalue()`.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// for tensor in hurray.StreamReader("tensors.hrry"):
///     print(tensor.shape, tensor.dtype)
/// ```
#[pyclass(name = "StreamReader", unsendable)]
pub struct StreamReader {
    inner: Option<IoReader<BoxedSource>>,
    runtime: Runtime,
}

#[pymethods]
impl StreamReader {
    /// Open a stream for reading.
    ///
    /// ## Errors
    ///
    /// - `TypeError` — `source` is not a path, bytes, or an object with `fileno()`.
    /// - `hurray.FileError` — the path could not be opened, or `fileno()` failed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// reader = hurray.StreamReader(b"")     # an empty stream yields nothing
    /// assert list(reader) == []
    /// ```
    #[new]
    pub fn new(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let runtime = build_runtime()?;
        let boxed = resolve_source(py, source, &runtime)?;
        Ok(Self {
            inner: Some(IoReader::new(boxed)),
            runtime,
        })
    }

    /// Readers are iterators over the tensors in the stream.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// reader = hurray.StreamReader(b"")
    /// assert iter(reader) is reader
    /// ```
    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next tensor, or `StopIteration` at a clean end of stream.
    ///
    /// ## Errors
    ///
    /// - `hurray.StreamError` — a truncated or malformed frame.
    /// - `hurray.UnsupportedError` — the stream contains a composite, which
    ///   `hurray.Tensor` cannot represent.
    /// - `hurray.FileError` — the transport failed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// reader = hurray.StreamReader(b"")
    /// assert next(reader, "done") == "done"
    /// ```
    pub fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(reader) = self.inner.as_mut() else {
            return Ok(None); // closed: iteration is over, not an error
        };

        // next_item, never next_tensor: a composite head owns no buffers, so
        // next_tensor would return it as an empty tensor and then surface its members
        // as top-level ones — a stream that "read fine" having lost the composition.
        let item = py
            .detach(|| self.runtime.block_on(reader.next_item()))
            .map_err(stream_err)?;

        match item {
            None => Ok(None),
            Some(StreamItem::Tensor(t)) => {
                Ok(Some(stream_tensor_to_py(py, t.descriptor, t.buffers)?))
            }
            Some(StreamItem::Composite(c)) => Err(UnsupportedError::new_err(format!(
                "the stream contains a composite ({} members); hurray.Tensor cannot \
                 represent a composite head, which owns no buffers. Read this stream \
                 with the Rust API until composite support lands in Python.",
                c.members.len()
            ))),
        }
    }

    /// Readers are context managers, so the transport can be released promptly.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// with hurray.StreamReader(b"") as reader:
    ///     assert list(reader) == []
    /// ```
    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Closes the stream, releasing the transport.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// reader = hurray.StreamReader(b"")
    /// reader.close()
    /// assert list(reader) == []      # a closed reader is simply exhausted
    /// ```
    #[pyo3(signature = (*_args))]
    pub fn __exit__(&mut self, _args: &Bound<'_, PyTuple>) -> bool {
        self.close();
        false // never suppress an exception
    }

    /// Release the transport now rather than at collection.
    pub fn close(&mut self) {
        self.inner = None;
    }
}

// ── StreamWriter ──────────────────────────────────────────────────────────────

/// Writes tensors to a Hurray stream, one at a time.
///
/// `finish` flushes the transport, and a caller who forgets it loses whatever was
/// still buffered — so the writer is a context manager and closing is automatic on the
/// happy path. `finish()` is also available explicitly, and is idempotent.
///
/// ## Destinations
///
/// A filesystem path, an object with `fileno()`, or nothing at all — in which case the
/// stream is built in memory and returned by `getvalue()`.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// with hurray.StreamWriter() as writer:
///     writer.write(hurray.Tensor(bytes(16), hurray.float32, [4]))
/// # writer.getvalue() holds the encoded stream
/// ```
#[pyclass(name = "StreamWriter", unsendable)]
pub struct StreamWriter {
    inner: Option<IoWriter<BoxedSink>>,
    runtime: Runtime,
    /// Present only for an in-memory stream; holds the bytes `getvalue()` returns.
    memory: Option<SharedBuffer>,
    finished: bool,
}

#[pymethods]
impl StreamWriter {
    /// Open a stream for writing.
    ///
    /// With no destination the stream is built in memory; read it back with
    /// `getvalue()` after finishing.
    ///
    /// ## Errors
    ///
    /// - `TypeError` — `destination` is not a path or an object with `fileno()`.
    /// - `hurray.FileError` — the path could not be created, or `fileno()` failed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// writer = hurray.StreamWriter()        # in memory
    /// writer.finish()
    /// assert writer.getvalue() == b""       # an empty stream is empty
    /// ```
    #[new]
    #[pyo3(signature = (destination = None))]
    pub fn new(py: Python<'_>, destination: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let runtime = build_runtime()?;
        let (sink, memory): (BoxedSink, Option<SharedBuffer>) = match destination {
            None => {
                let buffer = SharedBuffer::default();
                (Box::new(buffer.clone()), Some(buffer))
            }
            Some(dest) => {
                if let Ok(fd) = fileno_of(dest) {
                    (Box::new(file_from_fd(dup_fd(fd)?)), None)
                } else {
                    let path: String = dest.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "expected a path or an object with fileno(); \
                             omit the argument to build the stream in memory",
                        )
                    })?;
                    let file = py
                        .detach(|| runtime.block_on(tokio::fs::File::create(&path)))
                        .map_err(|e| FileError::new_err(format!("could not create {path}: {e}")))?;
                    (Box::new(file), None)
                }
            }
        };
        Ok(Self {
            inner: Some(IoWriter::new(sink)),
            runtime,
            memory,
            finished: false,
        })
    }

    /// Write one tensor, with every buffer its descriptor references.
    ///
    /// ## Errors
    ///
    /// - `hurray.StreamError` — the writer has already been finished.
    /// - `hurray.FileError` — the transport failed.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// with hurray.StreamWriter() as writer:
    ///     writer.write(hurray.Tensor(bytes(16), hurray.float32, [4]))
    /// ```
    pub fn write(&mut self, py: Python<'_>, tensor: &Bound<'_, Tensor>) -> PyResult<()> {
        let Some(writer) = self.inner.as_mut() else {
            return Err(StreamError::new_err(
                "this stream has been finished; create a new StreamWriter to write more",
            ));
        };

        let borrowed = tensor.borrow();
        let descriptor = borrowed.descriptor.clone();
        // The slices are handed across the GIL release as raw parts: a PyRef is not
        // Send, but the bytes it points at are stable for the whole call. See SendSlice.
        let raw: Vec<SendSlice> = borrowed
            .buffers()
            .map(|store| {
                // SAFETY: the GIL is held here; the store is alive and so is its base.
                let slice = unsafe { store.as_slice() };
                SendSlice(slice.as_ptr(), slice.len())
            })
            .collect();
        drop(borrowed);

        let runtime = &self.runtime;
        py.detach(move || {
            // SAFETY: `tensor` is a strong reference held for the whole call, so the
            // buffers cannot be freed while this runs — releasing the GIL does not
            // drop it, and nothing else can reach it to mutate the stores.
            let buffers: Vec<&[u8]> = raw
                .iter()
                .map(|s| unsafe { std::slice::from_raw_parts(s.0, s.1) })
                .collect();
            runtime.block_on(writer.write_tensor(&descriptor, &buffers))
        })
        .map_err(stream_err)
    }

    /// Finish the stream, writing its terminator.
    ///
    /// Idempotent: finishing an already-finished stream is a no-op, so the explicit
    /// call and the context manager compose.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// writer = hurray.StreamWriter()
    /// writer.finish()
    /// writer.finish()                       # no-op, not an error
    /// ```
    pub fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(writer) = self.inner.take() else {
            return Ok(());
        };
        self.finished = true;
        py.detach(|| self.runtime.block_on(writer.finish()))
            .map(|_sink| ())
            .map_err(stream_err)
    }

    /// The encoded stream, for a writer with no destination.
    ///
    /// ## Errors
    ///
    /// - `hurray.StreamError` — this writer has a destination, so there is nothing to
    ///   return; the bytes went there.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// with hurray.StreamWriter() as writer:
    ///     writer.write(hurray.Tensor(bytes(16), hurray.float32, [4]))
    /// assert len(writer.getvalue()) > 0
    /// ```
    pub fn getvalue(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let Some(memory) = self.memory.as_ref() else {
            return Err(StreamError::new_err(
                "this writer has a destination; its bytes were written there, \
                 not buffered in memory",
            ));
        };
        Ok(PyBytes::new(py, &memory.take()).unbind())
    }

    /// Writers are context managers; the stream is finished on exit.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// with hurray.StreamWriter() as writer:
    ///     pass
    /// ```
    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Finishes the stream on exit, so a caller cannot forget the terminator.
    #[pyo3(signature = (*_args))]
    pub fn __exit__(&mut self, py: Python<'_>, _args: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.finish(py)?;
        Ok(false) // never suppress an exception
    }
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Builds a `hurray.Tensor` from a decoded stream tensor.
///
/// The buffers are **copied** out of the `Bytes` the reader produced. Borrowing would
/// need a Python object owning the `Bytes` for the tensor to hold, which is machinery
/// in exchange for skipping a memcpy that a caller has already paid for once by
/// reading the bytes off a transport. If a profile ever says otherwise, the borrow can
/// be added behind the same API.
fn stream_tensor_to_py(
    py: Python<'_>,
    descriptor: hurray_core::TensorDescriptor,
    buffers: Vec<impl AsRef<[u8]>>,
) -> PyResult<Py<PyAny>> {
    if buffers.len() != descriptor.buffers.len() {
        return Err(InvalidDescriptorError::new_err(format!(
            "stream carried {} buffers but the descriptor declares {}",
            buffers.len(),
            descriptor.buffers.len()
        )));
    }

    let (device_tag, memory_class) = descriptor
        .buffers
        .first()
        .map(|b| (b.device_tag(), b.memory_class()))
        .unwrap_or((
            hurray_core::DeviceTag::Cpu,
            hurray_core::MemoryClass::Standard,
        ));

    let dtype_py = Py::new(
        py,
        Dtype {
            inner: descriptor.element_type,
        },
    )?;
    let device_py = Py::new(
        py,
        Device {
            tag: device_tag,
            memory_class,
            device_id: 0,
        },
    )?;

    let mut stores = buffers.iter().map(|b| BufferStore::from_slice(b.as_ref()));
    let buffer = match stores.next() {
        Some(b) => b,
        None => return Err(InvalidDescriptorError::new_err("tensor carries no buffers")),
    };
    let aux_buffers: Vec<BufferStore> = stores.collect();

    Ok(Py::new(
        py,
        Tensor {
            descriptor,
            buffer,
            aux_buffers,
            dtype_py,
            device_py,
        },
    )?
    .into_any())
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StreamReader>()?;
    m.add_class::<StreamWriter>()?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use hurray_core::{
        buffer_size_bytes, layout::CompositeLayout, layout::CompositionRule, BufferHandle,
        DeviceTag, ElementType, LayoutDescriptor, Shape, ShardDescriptor, SyncMode,
        TensorDescriptor, MIN_BUFFER_ALIGNMENT,
    };
    use hurray_io::stream::CompositeNode;

    fn init() {
        pyo3::Python::initialize();
    }

    fn buf(byte_size: u64) -> BufferHandle {
        BufferHandle::new(
            byte_size,
            MIN_BUFFER_ALIGNMENT,
            DeviceTag::Cpu,
            SyncMode::ProducerSynced,
        )
        .expect("valid buffer handle")
    }

    fn composite_head(member_count: u32) -> TensorDescriptor {
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![8u64, 8]).expect("valid shape"),
            0,
            LayoutDescriptor::Composite(
                CompositeLayout::new(CompositionRule::Partition, member_count)
                    .expect("valid composite"),
            ),
            vec![],
            None,
            None,
            None,
            None,
        )
        .expect("valid head")
    }

    fn partition_member(offset: u64) -> TensorDescriptor {
        TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![8u64, 4]).expect("valid shape"),
            0,
            LayoutDescriptor::RowMajor,
            vec![buf(buffer_size_bytes(ElementType::Float32, 8 * 4))],
            None,
            Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).expect("valid shard")),
            None,
            None,
        )
        .expect("valid member")
    }

    /// Encodes a stream holding one valid partition composite.
    ///
    /// Built in Rust because Python cannot write one: `hurray.Tensor` has no way to
    /// express a buffer-less composite head, which is the same gap that makes the
    /// reader refuse composites in the first place.
    fn composite_stream() -> Vec<u8> {
        let head = composite_head(2);
        let (m0, m1) = (partition_member(0), partition_member(4));
        let d0 = vec![0xAAu8; 128];
        let d1 = vec![0xBBu8; 128];
        let b0: [&[u8]; 1] = [d0.as_slice()];
        let b1: [&[u8]; 1] = [d1.as_slice()];
        let members = vec![
            CompositeNode::Tensor {
                descriptor: &m0,
                buffers: &b0,
            },
            CompositeNode::Tensor {
                descriptor: &m1,
                buffers: &b1,
            },
        ];

        let runtime = build_runtime().expect("runtime");
        runtime.block_on(async {
            let mut writer = IoWriter::new(Vec::new());
            writer
                .write_composite(&head, &members)
                .await
                .expect("composite written");
            writer.finish().await.expect("stream finished")
        })
    }

    #[test]
    fn a_composite_is_refused_by_name_rather_than_skipped() {
        init();
        let wire = composite_stream();
        Python::attach(|py| {
            let bytes = pyo3::types::PyBytes::new(py, &wire);
            let mut reader = StreamReader::new(py, bytes.as_any()).expect("reader opens");

            let err = reader
                .__next__(py)
                .expect_err("a composite must not decode as a tensor");
            assert!(
                err.is_instance_of::<crate::errors::UnsupportedError>(py),
                "expected UnsupportedError, got {err}"
            );
            let message = err.to_string();
            assert!(
                message.contains("composite"),
                "the error must name what it refused: {message}"
            );
        });
    }

    /// The failure this guards against: `next_tensor` would have returned the head as
    /// an empty tensor and then surfaced its two members as top-level tensors, so a
    /// caller would see three tensors and no error at all.
    #[test]
    fn a_composite_stream_yields_no_tensors_at_all() {
        init();
        let wire = composite_stream();
        Python::attach(|py| {
            let bytes = pyo3::types::PyBytes::new(py, &wire);
            let mut reader = StreamReader::new(py, bytes.as_any()).expect("reader opens");
            assert!(
                reader.__next__(py).is_err(),
                "the very first item must fail, not the third"
            );
        });
    }
}
