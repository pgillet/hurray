//! `hurray.load` and `hurray.save` — Python file I/O bridge (Layer 8b).
//!
//! Both functions are synchronous from the Python side. They drive the async
//! `hurray-io` file reader/writer via a disposable single-threaded Tokio runtime
//! created per call. The GIL is released while the runtime runs, so other Python
//! threads are not blocked during I/O.
//!
//! ## Error mapping
//!
//! `hurray_io::Error::Core` → `hurray.InvalidDescriptorError`
//! All other `hurray_io::Error` variants → `hurray.FileError`

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};
use pyo3::IntoPyObjectExt;

use hurray_io::file::{FileItem, FileReader, FileTensor, FileWriter, KvValue};

use crate::{
    buffer::BufferStore,
    errors::{FileError, InvalidDescriptorError, UnsupportedError},
    tensor::Tensor,
};

/// Map `hurray_io::Error` to a Python exception (GIL must be held).
fn io_err_to_py(e: hurray_io::Error) -> PyErr {
    match e {
        hurray_io::Error::Core(ce) => InvalidDescriptorError::new_err(ce.to_string()),
        other => FileError::new_err(other.to_string()),
    }
}

/// Build a `hurray.Tensor` from a `FileTensor` returned by `FileReader`.
///
/// Multi-buffer tensors — per-channel / NF4 / MXFP quantization, sparse layouts,
/// block-paged — are carried in descriptor buffer-table order (ADR-030 § 3).
fn file_tensor_to_tensor(py: Python<'_>, ft: FileTensor) -> PyResult<Py<Tensor>> {
    if ft.buffers.is_empty() {
        return Err(InvalidDescriptorError::new_err(format!(
            "tensor {:?} carries no buffers",
            ft.name
        )));
    }

    // The reader yields one byte range per buffer handle, so a mismatch means the
    // file's buffer table and its payload disagree — reject rather than hand back
    // a descriptor whose buffer indices do not resolve.
    if ft.buffers.len() != ft.descriptor.buffers.len() {
        return Err(InvalidDescriptorError::new_err(format!(
            "tensor {:?}: descriptor declares {} buffers but {} were read",
            ft.name,
            ft.descriptor.buffers.len(),
            ft.buffers.len()
        )));
    }

    let desc = ft.descriptor;
    let element_type = desc.element_type;

    let (device_tag, memory_class) = desc
        .buffers
        .first()
        .map(|bh| (bh.device_tag(), bh.memory_class()))
        .unwrap_or((
            hurray_core::DeviceTag::Cpu,
            hurray_core::MemoryClass::Standard,
        ));

    let mut read_buffers = ft.buffers.iter();
    // The emptiness check above guarantees a first element.
    let buffer = match read_buffers.next() {
        Some(b) => BufferStore::from_slice(b),
        None => return Err(InvalidDescriptorError::new_err("tensor carries no buffers")),
    };
    let aux_buffers: Vec<BufferStore> = read_buffers.map(|b| BufferStore::from_slice(b)).collect();

    let dtype_py = Py::new(
        py,
        crate::dtype::Dtype {
            inner: element_type,
        },
    )?;
    let device_py = Py::new(
        py,
        crate::device::Device {
            tag: device_tag,
            memory_class,
            device_id: 0,
        },
    )?;

    Py::new(
        py,
        Tensor {
            descriptor: desc,
            buffer,
            aux_buffers,
            dtype_py,
            device_py,
        },
    )
}

/// Convert a Python scalar/list to a `KvValue`.
///
/// Type mapping:
/// - `bool` → `KvValue::Bool` (checked before `int`)
/// - `int` → `KvValue::Int64`
/// - `float` → `KvValue::Float64`
/// - `str` → `KvValue::String`
/// - `bytes` → `KvValue::Bytes`
/// - `list[T]` → `KvValue::Array` (homogeneous; validation delegated to `hurray_io`)
fn py_to_kv_value(val: &Bound<'_, PyAny>) -> PyResult<KvValue> {
    // bool must be checked before int: Python's bool is a subclass of int.
    if val.is_instance_of::<PyBool>() {
        return Ok(KvValue::Bool(val.extract::<bool>()?));
    }
    if val.is_instance_of::<PyInt>() {
        return Ok(KvValue::Int64(val.extract::<i64>()?));
    }
    if val.is_instance_of::<PyFloat>() {
        return Ok(KvValue::Float64(val.extract::<f64>()?));
    }
    if val.is_instance_of::<PyString>() {
        return Ok(KvValue::String(val.extract::<String>()?));
    }
    if val.is_instance_of::<PyBytes>() {
        return Ok(KvValue::Bytes(val.extract::<Vec<u8>>()?));
    }
    if val.is_instance_of::<PyList>() {
        let list = val.cast::<PyList>()?;
        if list.is_empty() {
            return Err(FileError::new_err("KV array must not be empty"));
        }
        let elements: PyResult<Vec<KvValue>> = list.iter().map(|v| py_to_kv_value(&v)).collect();
        return Ok(KvValue::Array(elements?));
    }
    Err(FileError::new_err(format!(
        "unsupported KV value type: {} (expected bool, int, float, str, bytes, or list)",
        val.get_type().name()?
    )))
}

/// Convert a `KvValue` back to the Python object it came from.
///
/// The exact inverse of [`py_to_kv_value`], so a dict written with `save(kv=...)` comes
/// back equal — `Uint64` is the one wire type Python cannot produce (its `int` maps to
/// `Int64`), and it decodes to `int` as well.
fn kv_value_to_py(py: Python<'_>, value: &KvValue) -> PyResult<Py<PyAny>> {
    match value {
        KvValue::Bool(v) => v.into_py_any(py),
        KvValue::Int64(v) => v.into_py_any(py),
        KvValue::Uint64(v) => v.into_py_any(py),
        KvValue::Float64(v) => v.into_py_any(py),
        KvValue::String(v) => v.into_py_any(py),
        KvValue::Bytes(v) => Ok(PyBytes::new(py, v).into_any().unbind()),
        KvValue::Array(elements) => {
            let items: PyResult<Vec<Py<PyAny>>> =
                elements.iter().map(|e| kv_value_to_py(py, e)).collect();
            Ok(PyList::new(py, items?)?.into_any().unbind())
        }
        // KvValue is non_exhaustive: a value type added to the format after this build
        // reaches here. Refusing names the situation; returning None for it would put a
        // hole in a metadata dict and call it success.
        other => Err(FileError::new_err(format!(
            "KV value type not supported by this build of hurray: {other:?}"
        ))),
    }
}

fn py_dict_to_kv(kv_dict: &Bound<'_, PyDict>) -> PyResult<Vec<(String, KvValue)>> {
    kv_dict
        .iter()
        .map(|(k, v)| {
            let key: String = k.extract()?;
            let value = py_to_kv_value(&v)?;
            Ok((key, value))
        })
        .collect()
}

/// Read a Hurray file's key-value metadata section.
///
/// The other half of `save(path, tensors, kv=...)`: what that writes, this reads back.
/// Returns an empty dict for a file with no KV section.
///
/// A separate call rather than an argument to `load` because it answers a different
/// question and returns a different thing — a flag that changed `load`'s return type
/// would make every caller unpack a tuple to ask about tensors. Reading it costs a footer
/// seek, not a scan of the file.
///
/// ## Value types
///
/// The seven wire types map back to the Python objects that produced them: `bool`, `int`
/// (from both `int64` and `uint64`), `float`, `str`, `bytes`, and `list` of any of those.
///
/// ## Errors
///
/// - `hurray.FileError` — the file is missing, truncated, or its KV section is malformed.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// hurray.save(
///     "model.hrry",
///     {"w": hurray.Tensor(bytes(16), hurray.float32, [4])},
///     kv={"model": "demo", "layers": 12, "quantized": False},
/// )
///
/// meta = hurray.load_kv("model.hrry")
/// assert meta == {"model": "demo", "layers": 12, "quantized": False}
/// ```
#[pyfunction]
pub fn load_kv(py: Python<'_>, path: String) -> PyResult<Bound<'_, PyDict>> {
    // Release the GIL for the open and the footer read, as `load` does.
    let pairs = py
        .detach(|| -> hurray_io::Result<Vec<(String, KvValue)>> {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(hurray_io::Error::Io)?
                .block_on(async {
                    let file = tokio::fs::File::open(&path).await?;
                    let reader = FileReader::open(file).await?;
                    Ok(reader.kv().to_vec())
                })
        })
        .map_err(io_err_to_py)?;

    let out = PyDict::new(py);
    for (key, value) in &pairs {
        out.set_item(key, kv_value_to_py(py, value)?)?;
    }
    Ok(out)
}

/// Load tensors from a Hurray file.
///
/// Opens the HRRYFILE at `path` and returns a `dict` mapping tensor names to
/// `hurray.Tensor` objects. If `names` is given, only those tensors are loaded;
/// otherwise every tensor in the file is returned.
///
/// Multi-buffer tensors load as a `hurray.Tensor` carrying every buffer in descriptor
/// order (ADR-030). A sparse tensor round-trips its values and index arrays and comes
/// back with its layout intact — there is one tensor class, so nothing is reconstructed
/// into a different type. A composite comes back as a `hurray.Composite`, and its members
/// do not also appear under their own names.
///
/// The GIL is released during file I/O so other Python threads are not blocked.
///
/// ## Errors
///
/// - `hurray.FileError` — file not found, corrupt HRRYFILE, unexpected EOF, invalid
///   magic, CRC mismatch, or a name in `names` that the file does not carry.
/// - `hurray.InvalidDescriptorError` — a tensor descriptor failed to decode.
///
/// ## Examples
///
/// ```python
/// import hurray
///
/// tensors = hurray.load("model.hrry")
/// embeddings = tensors["embeddings"]   # hurray.Tensor
/// print(embeddings.shape, embeddings.dtype)
///
/// # Load only specific tensors
/// subset = hurray.load("model.hrry", names=["embeddings", "bias"])
/// ```
#[pyfunction]
#[pyo3(signature = (path, *, names = None))]
pub fn load(
    py: Python<'_>,
    path: String,
    names: Option<Vec<String>>,
) -> PyResult<Bound<'_, PyDict>> {
    // Release GIL while doing async file I/O.
    let file_tensors = py
        .detach(|| -> hurray_io::Result<Vec<(String, FileItem)>> {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(hurray_io::Error::Io)?
                .block_on(async {
                    let file = tokio::fs::File::open(&path).await?;
                    let mut reader = FileReader::open(file).await?;

                    let names_to_load: Vec<String> = match names {
                        Some(ref ns) => ns.clone(),
                        None => reader.tensor_names().map(|s| s.to_string()).collect(),
                    };

                    // A composite's members have index entries of their own, so a name
                    // already claimed by a composite must not also come back alone.
                    // Membership is the reader's to decide, recovered from write order.
                    let mut claimed: std::collections::HashSet<String> =
                        std::collections::HashSet::new();

                    let mut out = Vec::with_capacity(names_to_load.len());
                    for name in names_to_load {
                        if claimed.contains(&name) {
                            continue;
                        }
                        // Read the descriptor alone first: read_tensor rejects a
                        // composite head, which owns no buffers, so the kind has to be
                        // known before choosing how to read it.
                        let descriptor = reader.read_descriptor(&name).await?;
                        let item = if matches!(
                            descriptor.layout,
                            hurray_core::LayoutDescriptor::Composite(_)
                        ) {
                            FileItem::Composite(reader.read_composite(&name).await?)
                        } else {
                            FileItem::Tensor(reader.read_tensor(&name).await?)
                        };
                        consumed_names(&item, &mut claimed);
                        out.push((name, item));
                    }
                    Ok(out)
                })
        })
        .map_err(io_err_to_py)?;

    // GIL re-acquired: build Python dict.
    let dict = PyDict::new(py);
    for (name, item) in file_tensors {
        dict.set_item(name, file_item_to_py(py, item)?)?;
    }
    Ok(dict)
}

/// Builds the Python object for one entry read from a file.
///
/// A composite becomes a `hurray.Composite`, recursively (ADR-036). The tree came
/// through core's `CompositeValidator` on read, so this does not revalidate it.
fn file_item_to_py(py: Python<'_>, item: FileItem) -> PyResult<Py<PyAny>> {
    match item {
        FileItem::Tensor(ft) => Ok(file_tensor_to_tensor(py, ft)?.into_any()),
        FileItem::Composite(fc) => {
            let members = fc
                .members
                .into_iter()
                .map(|m| file_item_to_py(py, m))
                .collect::<PyResult<Vec<_>>>()?;
            crate::composite::composite_from_parts(py, fc.head, members)
        }
    }
}

/// Every name a composite tree occupies: its head and, recursively, its members.
///
/// A file gives every tensor an index entry, head and member alike, so without this
/// the members would come back twice — once inside their composite and once as
/// top-level entries of their own.
fn consumed_names(item: &FileItem, into: &mut std::collections::HashSet<String>) {
    match item {
        FileItem::Tensor(t) => {
            into.insert(t.name.clone());
        }
        FileItem::Composite(c) => {
            into.insert(c.name.clone());
            for member in &c.members {
                consumed_names(member, into);
            }
        }
    }
}

/// One entry queued for `save`: a tensor, or a composite tree with names attached.
enum SaveEntry {
    Tensor {
        name: String,
        descriptor: hurray_core::TensorDescriptor,
        buffers: Vec<Vec<u8>>,
    },
    Composite {
        name: String,
        head: hurray_core::TensorDescriptor,
        members: Vec<crate::composite::NamedNode>,
    },
}

/// Save tensors to a Hurray file.
///
/// Writes all entries in `tensors` (a `dict` mapping `str` names to
/// `hurray.Tensor` objects) to the HRRYFILE at `path`. The optional `kv`
/// argument stores file-level metadata as key-value pairs.
///
/// The GIL is released during file I/O so other Python threads are not blocked.
///
/// # KV value types
///
/// `kv` values may be `bool`, `int`, `float`, `str`, `bytes`, or a homogeneous
/// `list` of one of those scalar types (not nested lists).
///
/// # Errors
///
/// - `hurray.FileError` — path not writable, I/O failure, duplicate tensor name,
///   invalid KV keys, etc.
/// - `hurray.UnsupportedError` — a value in `tensors` is not a `hurray.Tensor`.
///
/// # Examples
///
/// ```python
/// import hurray
///
/// t = hurray.zeros((4, 4), dtype=hurray.float32)
/// hurray.save("model.hrry", {"weights": t}, kv={"version": "1.0"})
///
/// # Round-trip
/// loaded = hurray.load("model.hrry")
/// assert loaded["weights"].shape == (4, 4)
/// ```
#[pyfunction]
#[pyo3(signature = (path, tensors, *, kv = None))]
pub fn save(
    py: Python<'_>,
    path: String,
    tensors: &Bound<'_, PyDict>,
    kv: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    // Extract tensor data while holding GIL: descriptor + raw bytes per buffer.
    let mut entries: Vec<SaveEntry> = Vec::with_capacity(tensors.len());
    for (key, val) in tensors {
        let name: String = key.extract()?;
        if let Ok(composite) = val.extract::<PyRef<crate::composite::Composite>>() {
            let (head, members) = crate::composite::named_tree(py, &composite, &name);
            entries.push(SaveEntry::Composite {
                name,
                head,
                members,
            });
            continue;
        }
        // Every layout goes down this path: since ADR-031 a sparse tensor is a
        // Tensor whose layout is COO/CSR/CSC, and its component buffers are just
        // its buffer table (#156).
        let tensor = val.extract::<PyRef<Tensor>>().map_err(|_| {
            UnsupportedError::new_err(
                "hurray.save() only accepts hurray.Tensor and hurray.Composite values",
            )
        })?;
        // Every buffer in descriptor order (ADR-030 § 3), so quantization scale
        // and sparse index buffers reach the file alongside the data.
        // SAFETY: GIL is held; buffers are valid for the lifetime of `tensor`.
        let buffers: Vec<Vec<u8>> = tensor
            .buffers()
            .map(|b| unsafe { b.as_slice() }.to_vec())
            .collect();
        entries.push(SaveEntry::Tensor {
            name,
            descriptor: tensor.descriptor.clone(),
            buffers,
        });
    }
    let kv_pairs = if let Some(d) = kv {
        py_dict_to_kv(d)?
    } else {
        Vec::new()
    };

    // Release GIL while doing async file I/O.
    py.detach(|| -> hurray_io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(hurray_io::Error::Io)?;

        // One block_on per step rather than one around the whole loop: a composite's
        // borrowed node tree is built by a *synchronous* continuation (see
        // with_file_nodes), and block_on cannot be called from inside another.
        let file = runtime.block_on(tokio::fs::File::create(&path))?;
        let mut writer = runtime.block_on(FileWriter::new(file))?;

        for entry in &entries {
            match entry {
                SaveEntry::Tensor {
                    name,
                    descriptor,
                    buffers,
                } => {
                    let slices: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
                    runtime.block_on(writer.write_tensor(name, descriptor, &slices))?;
                }
                SaveEntry::Composite {
                    name,
                    head,
                    members,
                } => {
                    let mut result = Ok(());
                    crate::composite::with_file_nodes(members, &[], &mut |nodes| {
                        result = runtime.block_on(writer.write_composite(name, head, nodes));
                    });
                    result?;
                }
            }
        }

        runtime.block_on(writer.finish(kv_pairs))?;
        Ok(())
    })
    .map_err(io_err_to_py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_kv, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_python() {
        pyo3::Python::initialize();
    }

    #[test]
    fn load_nonexistent_file_raises_file_error() {
        init_python();
        Python::attach(|py| {
            let result = load(
                py,
                "/nonexistent/path/does_not_exist.hrry".to_string(),
                None,
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<FileError>(py),
                "expected FileError, got: {err}"
            );
        });
    }

    #[test]
    fn save_to_invalid_path_raises_file_error() {
        init_python();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            let result = save(py, "/nonexistent/dir/out.hrry".to_string(), &dict, None);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<FileError>(py),
                "expected FileError, got: {err}"
            );
        });
    }

    #[test]
    fn py_to_kv_value_bool() {
        init_python();
        Python::attach(|py| {
            let val = pyo3::types::PyBool::new(py, true);
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert_eq!(kv, KvValue::Bool(true));
        });
    }

    #[test]
    fn py_to_kv_value_int() {
        init_python();
        Python::attach(|py| {
            // Create a Python int by converting from Rust
            let obj = 42_i64.into_py_any(py).unwrap();
            let val = obj.bind(py);
            let kv = py_to_kv_value(val).unwrap();
            assert_eq!(kv, KvValue::Int64(42));
        });
    }

    #[test]
    fn py_to_kv_value_float() {
        init_python();
        Python::attach(|py| {
            let val = pyo3::types::PyFloat::new(py, 2.5);
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert!(matches!(kv, KvValue::Float64(v) if (v - 2.5).abs() < 1e-10));
        });
    }

    #[test]
    fn py_to_kv_value_str() {
        init_python();
        Python::attach(|py| {
            let val = pyo3::types::PyString::new(py, "hello");
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert_eq!(kv, KvValue::String("hello".to_string()));
        });
    }

    #[test]
    fn py_to_kv_value_bytes() {
        init_python();
        Python::attach(|py| {
            let val = pyo3::types::PyBytes::new(py, b"raw");
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert_eq!(kv, KvValue::Bytes(b"raw".to_vec()));
        });
    }

    #[test]
    fn py_to_kv_value_list_of_ints() {
        init_python();
        Python::attach(|py| {
            let list = PyList::new(py, [1i64, 2, 3]).unwrap();
            let kv = py_to_kv_value(list.as_any()).unwrap();
            assert_eq!(
                kv,
                KvValue::Array(vec![
                    KvValue::Int64(1),
                    KvValue::Int64(2),
                    KvValue::Int64(3)
                ])
            );
        });
    }

    #[test]
    fn py_to_kv_value_empty_list_is_error() {
        init_python();
        Python::attach(|py| {
            let list: &[i64] = &[];
            let list = PyList::new(py, list).unwrap();
            let result = py_to_kv_value(list.as_any());
            assert!(result.is_err());
        });
    }

    #[test]
    fn multi_buffer_tensor_round_trips_through_save_and_load() {
        use hurray_core::{
            BufferHandle, DeviceTag, ElementType, LayoutDescriptor, MemoryClass, Shape, SyncMode,
            TensorDescriptor, DESCRIPTOR_VERSION_MAJOR, DESCRIPTOR_VERSION_MINOR,
            MIN_BUFFER_ALIGNMENT,
        };
        use pyo3::types::PyDict;

        init_python();
        Python::attach(|py| {
            let data = vec![9u8; 8];
            let scales = vec![3u8; 16];

            let bh = |len: u64| {
                BufferHandle::with_memory_class(
                    len,
                    MIN_BUFFER_ALIGNMENT,
                    DeviceTag::Cpu,
                    SyncMode::ProducerSynced,
                    MemoryClass::Standard,
                )
                .unwrap()
            };
            let descriptor = TensorDescriptor::new(
                DESCRIPTOR_VERSION_MAJOR,
                DESCRIPTOR_VERSION_MINOR,
                ElementType::Int8,
                Shape::new(vec![2u64, 4]).unwrap(),
                0,
                LayoutDescriptor::RowMajor,
                vec![bh(8), bh(16)],
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let tensor = Tensor {
                descriptor,
                buffer: BufferStore::from_slice(&data),
                aux_buffers: vec![BufferStore::from_slice(&scales)],
                dtype_py: Py::new(
                    py,
                    crate::dtype::Dtype {
                        inner: ElementType::Int8,
                    },
                )
                .unwrap(),
                device_py: Py::new(
                    py,
                    crate::device::Device {
                        tag: DeviceTag::Cpu,
                        memory_class: MemoryClass::Standard,
                        device_id: 0,
                    },
                )
                .unwrap(),
            };

            let dir = std::env::temp_dir().join("hurray_multi_buffer_roundtrip");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("two_buffers.hrry");

            let tensors = PyDict::new(py);
            tensors.set_item("w", Py::new(py, tensor).unwrap()).unwrap();
            save(py, path.to_string_lossy().into_owned(), &tensors, None).unwrap();

            let loaded = load(py, path.to_string_lossy().into_owned(), None).unwrap();
            let got: Py<Tensor> = loaded
                .get_item("w")
                .unwrap()
                .expect("tensor 'w' must be present")
                .extract()
                .unwrap();
            let got = got.borrow(py);

            // Both buffers survive, in descriptor order, byte for byte.
            assert_eq!(got.buffer_count(), 2);
            let bytes: Vec<Vec<u8>> = got
                .buffers()
                .map(|b| unsafe { b.as_slice() }.to_vec())
                .collect();
            assert_eq!(bytes[0], data);
            assert_eq!(bytes[1], scales);
            assert_eq!(got.descriptor.buffers.len(), 2);

            std::fs::remove_file(&path).ok();
        });
    }

    #[test]
    fn python_authored_quantized_tensor_round_trips_through_a_file() {
        use pyo3::types::{PyBytes, PyDict};

        init_python();
        Python::attach(|py| {
            // #146 acceptance: author a per-channel INT8 tensor with its scale
            // buffer from Python, save it, and read the scheme back off disk.
            let m = pyo3::types::PyModule::new(py, "hurray").unwrap();
            crate::errors::register(&m).unwrap();
            crate::dtype::register(&m).unwrap();
            crate::device::register(&m).unwrap();
            let sys = py.import("sys").unwrap();
            sys.getattr("modules")
                .unwrap()
                .set_item("hurray", &m)
                .unwrap();

            let data = PyBytes::new(py, &[7u8; 8]);
            let scales = PyBytes::new(py, &[1u8; 8]);
            let dtype = Py::new(
                py,
                crate::dtype::Dtype {
                    inner: hurray_core::ElementType::Int8,
                },
            )
            .unwrap();
            let quant = Py::new(
                py,
                crate::quantization::PerChannelAffine::symmetric(0, 1).unwrap(),
            )
            .unwrap();

            let tensor = Tensor::new(
                py,
                data.as_any(),
                dtype.bind(py),
                vec![Some(2), Some(4)],
                None,
                Some(vec![scales.into_any().unbind()]),
                None,
                Some(quant.bind(py).as_any()),
                None,
                None,
            )
            .unwrap();

            let dir = std::env::temp_dir().join("hurray_quantized_roundtrip");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("weights.hrry");

            let tensors = PyDict::new(py);
            tensors.set_item("w", Py::new(py, tensor).unwrap()).unwrap();
            save(py, path.to_string_lossy().into_owned(), &tensors, None).unwrap();

            let loaded = load(py, path.to_string_lossy().into_owned(), None).unwrap();
            let got: Py<Tensor> = loaded
                .get_item("w")
                .unwrap()
                .expect("tensor 'w' must be present")
                .extract()
                .unwrap();
            let got = got.borrow(py);

            // Data and scales both survived, and the scheme came back intact.
            assert_eq!(got.buffer_count(), 2);
            let encoded = got.descriptor.quantization.as_ref().unwrap();
            let (decoded, _) = hurray_core::QuantizationDescriptor::decode(encoded).unwrap();
            match decoded {
                hurray_core::QuantizationDescriptor::PerChannelAffine(q) => {
                    assert_eq!(q.axis(), 0);
                    assert_eq!(q.scale_buffer_index(), 1);
                }
                other => panic!("expected per-channel affine, got {other:?}"),
            }

            std::fs::remove_file(&path).ok();
        });
    }

    #[test]
    fn sparse_tensor_round_trips_through_save_and_load() {
        use pyo3::types::PyDict;

        init_python();
        Python::attach(|py| {
            // #156: save() used to reject sparse outright. With one tensor class
            // (ADR-031) its component buffers are just its buffer table.
            let tensor = crate::sparse::tests::make_csr(py);
            let nnz = tensor.nnz().unwrap();
            let layout = crate::layout::layout_name(&tensor.descriptor.layout);

            let dir = std::env::temp_dir().join("hurray_sparse_roundtrip");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("m.hrry");

            let tensors = PyDict::new(py);
            tensors.set_item("m", Py::new(py, tensor).unwrap()).unwrap();
            save(py, path.to_string_lossy().into_owned(), &tensors, None).unwrap();

            let loaded = load(py, path.to_string_lossy().into_owned(), None).unwrap();
            let got: Py<Tensor> = loaded
                .get_item("m")
                .unwrap()
                .expect("tensor 'm' must be present")
                .extract()
                .unwrap();
            let got = got.borrow(py);

            // Layout, nnz and all three component buffers survive.
            assert_eq!(crate::layout::layout_name(&got.descriptor.layout), layout);
            assert_eq!(got.nnz().unwrap(), nnz);
            assert_eq!(got.buffer_count(), 3);

            std::fs::remove_file(&path).ok();
        });
    }

    #[test]
    fn file_and_stream_error_are_os_errors() {
        init_python();
        Python::attach(|py| {
            assert!(FileError::new_err("x").is_instance_of::<pyo3::exceptions::PyOSError>(py));
            assert!(crate::errors::StreamError::new_err("x")
                .is_instance_of::<pyo3::exceptions::PyOSError>(py));
        });
    }
}
