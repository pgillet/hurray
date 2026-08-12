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
#[cfg(test)]
use pyo3::IntoPyObjectExt;

use hurray_io::file::{FileReader, FileTensor, FileWriter, KvValue};

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

/// Load tensors from a Hurray file.
///
/// Opens the HRRYFILE at `path` and returns a `dict` mapping tensor names to
/// `hurray.Tensor` objects. If `names` is given, only those tensors are loaded;
/// otherwise every tensor in the file is returned.
///
/// The GIL is released during file I/O so other Python threads are not blocked.
///
/// # Errors
///
/// - `hurray.FileError` — file not found, corrupt HRRYFILE, unexpected EOF,
///   invalid magic, CRC mismatch, etc.
/// - `hurray.InvalidDescriptorError` — a tensor descriptor failed to decode.
/// Multi-buffer tensors load as a `hurray.Tensor` carrying every buffer in
/// descriptor order (ADR-030). A sparse tensor therefore round-trips its values
/// and index arrays, though it is returned as a `Tensor`, not a `SparseTensor`.
///
/// # Examples
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
        .detach(|| -> hurray_io::Result<Vec<(String, FileTensor)>> {
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

                    let mut out = Vec::with_capacity(names_to_load.len());
                    for name in names_to_load {
                        let ft = reader.read_tensor(&name).await?;
                        out.push((ft.name.clone(), ft));
                    }
                    Ok(out)
                })
        })
        .map_err(io_err_to_py)?;

    // GIL re-acquired: build Python dict.
    let dict = PyDict::new(py);
    for (name, ft) in file_tensors {
        let tensor = file_tensor_to_tensor(py, ft)?;
        dict.set_item(name, tensor)?;
    }
    Ok(dict)
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
/// - `hurray.UnsupportedError` — a value in `tensors` is not a `hurray.Tensor`
///   (e.g. `hurray.SparseTensor`).
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
    let mut entries: Vec<(String, hurray_core::TensorDescriptor, Vec<Vec<u8>>)> =
        Vec::with_capacity(tensors.len());
    for (key, val) in tensors {
        let name: String = key.extract()?;
        let tensor = val.extract::<PyRef<Tensor>>().map_err(|_| {
            UnsupportedError::new_err(
                "hurray.save() only accepts hurray.Tensor values; \
                 SparseTensor file I/O is not yet supported",
            )
        })?;
        // Every buffer in descriptor order (ADR-030 § 3), so quantization scale
        // and sparse index buffers reach the file alongside the data.
        // SAFETY: GIL is held; buffers are valid for the lifetime of `tensor`.
        let buffers: Vec<Vec<u8>> = tensor
            .buffers()
            .map(|b| unsafe { b.as_slice() }.to_vec())
            .collect();
        entries.push((name, tensor.descriptor.clone(), buffers));
    }
    let kv_pairs = if let Some(d) = kv {
        py_dict_to_kv(d)?
    } else {
        Vec::new()
    };

    // Release GIL while doing async file I/O.
    py.detach(|| -> hurray_io::Result<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(hurray_io::Error::Io)?
            .block_on(async {
                let file = tokio::fs::File::create(&path).await?;
                let mut writer = FileWriter::new(file).await?;
                for (name, desc, buffers) in &entries {
                    let slices: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
                    writer.write_tensor(name, desc, &slices).await?;
                }
                writer.finish(kv_pairs).await?;
                Ok(())
            })
    })
    .map_err(io_err_to_py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
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
    fn file_and_stream_error_are_os_errors() {
        init_python();
        Python::attach(|py| {
            assert!(FileError::new_err("x").is_instance_of::<pyo3::exceptions::PyOSError>(py));
            assert!(crate::errors::StreamError::new_err("x")
                .is_instance_of::<pyo3::exceptions::PyOSError>(py));
        });
    }
}
