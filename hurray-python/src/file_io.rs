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

// PyO3 0.22 macro expansion emits a redundant .into() — false positive.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};

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
/// Only single-buffer (dense) tensors are supported. Multi-buffer descriptors
/// (COO / CSR / CSC sparse layouts) raise `hurray.UnsupportedError`.
fn file_tensor_to_tensor(py: Python<'_>, ft: FileTensor) -> PyResult<Py<Tensor>> {
    if ft.buffers.len() != 1 {
        return Err(UnsupportedError::new_err(format!(
            "tensor {:?} has {} buffers; sparse tensors are not yet supported by \
             hurray.load() — use the SparseTensor API directly",
            ft.name,
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

    let buffer = BufferStore::from_slice(&ft.buffers[0]);

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
        let list = val.downcast::<PyList>()?;
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
/// - `hurray.UnsupportedError` — the file contains a multi-buffer (sparse) tensor.
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
        .allow_threads(|| -> hurray_io::Result<Vec<(String, FileTensor)>> {
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
    let dict = PyDict::new_bound(py);
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
    // Extract tensor data while holding GIL: descriptor + raw bytes.
    let mut entries: Vec<(String, hurray_core::TensorDescriptor, Vec<u8>)> =
        Vec::with_capacity(tensors.len());
    for (key, val) in tensors {
        let name: String = key.extract()?;
        let tensor = val.extract::<PyRef<Tensor>>().map_err(|_| {
            UnsupportedError::new_err(
                "hurray.save() only accepts hurray.Tensor values; \
                 SparseTensor file I/O is not yet supported",
            )
        })?;
        // SAFETY: GIL is held; buffer is valid for the lifetime of `tensor`.
        let bytes = unsafe { tensor.buffer.as_slice() }.to_vec();
        entries.push((name, tensor.descriptor.clone(), bytes));
    }
    let kv_pairs = if let Some(d) = kv {
        py_dict_to_kv(d)?
    } else {
        Vec::new()
    };

    // Release GIL while doing async file I/O.
    py.allow_threads(|| -> hurray_io::Result<()> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(hurray_io::Error::Io)?
            .block_on(async {
                let file = tokio::fs::File::create(&path).await?;
                let mut writer = FileWriter::new(file).await?;
                for (name, desc, buf) in &entries {
                    writer.write_tensor(name, desc, &[buf.as_slice()]).await?;
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
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn load_nonexistent_file_raises_file_error() {
        init_python();
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
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
        Python::with_gil(|py| {
            let val = pyo3::types::PyBool::new_bound(py, true);
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert_eq!(kv, KvValue::Bool(true));
        });
    }

    #[test]
    fn py_to_kv_value_int() {
        init_python();
        Python::with_gil(|py| {
            // PyO3 0.22: create a Python int by converting from Rust
            let obj = 42_i64.into_py(py);
            let val = obj.bind(py);
            let kv = py_to_kv_value(val).unwrap();
            assert_eq!(kv, KvValue::Int64(42));
        });
    }

    #[test]
    fn py_to_kv_value_float() {
        init_python();
        Python::with_gil(|py| {
            let val = pyo3::types::PyFloat::new_bound(py, 2.5);
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert!(matches!(kv, KvValue::Float64(v) if (v - 2.5).abs() < 1e-10));
        });
    }

    #[test]
    fn py_to_kv_value_str() {
        init_python();
        Python::with_gil(|py| {
            let val = pyo3::types::PyString::new_bound(py, "hello");
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert_eq!(kv, KvValue::String("hello".to_string()));
        });
    }

    #[test]
    fn py_to_kv_value_bytes() {
        init_python();
        Python::with_gil(|py| {
            let val = pyo3::types::PyBytes::new_bound(py, b"raw");
            let kv = py_to_kv_value(val.as_any()).unwrap();
            assert_eq!(kv, KvValue::Bytes(b"raw".to_vec()));
        });
    }

    #[test]
    fn py_to_kv_value_list_of_ints() {
        init_python();
        Python::with_gil(|py| {
            let list = PyList::new_bound(py, [1i64, 2, 3]);
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
        Python::with_gil(|py| {
            let list: &[i64] = &[];
            let list = PyList::new_bound(py, list);
            let result = py_to_kv_value(list.as_any());
            assert!(result.is_err());
        });
    }

    #[test]
    fn file_and_stream_error_are_os_errors() {
        init_python();
        Python::with_gil(|py| {
            assert!(FileError::new_err("x").is_instance_of::<pyo3::exceptions::PyOSError>(py));
            assert!(crate::errors::StreamError::new_err("x")
                .is_instance_of::<pyo3::exceptions::PyOSError>(py));
        });
    }
}
