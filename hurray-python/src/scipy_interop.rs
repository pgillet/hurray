//! SciPy sparse matrix interop: `hurray.from_scipy`.
//!
//! SciPy is an optional dependency — importing `hurray` must not require it.
//! Functions in this module call `import scipy.sparse` lazily at call time.
//!
//! ## Supported conversions
//!
//! | Direction | CSR | CSC | COO |
//! |-----------|-----|-----|-----|
//! | `from_scipy` | yes | yes | no (D19) |
//! | `to_scipy` | yes | yes | no |
//!
//! COO is rejected by `from_scipy` because SciPy stores COO row/col as two
//! separate arrays while Hurray requires a single packed `[nnz, rank]` uint64
//! index buffer (D19).

use pyo3::prelude::*;

use hurray_core::{ElementType, Shape};

use crate::buffer::BufferStore;
use crate::errors::{InvalidDescriptorError, UnsupportedError};
use crate::sparse::{build_csc_descriptor, build_csr_descriptor, make_sparse_tensor};
use crate::tensor::Tensor;

// ── hurray.from_scipy ─────────────────────────────────────────────────────────

/// Wrap a `scipy.sparse` matrix as a `hurray.Tensor` without copying.
///
/// ## Supported formats
///
/// | SciPy type | Hurray format |
/// |------------|---------------|
/// | `csr_matrix` / `csr_array` | CSR |
/// | `csc_matrix` / `csc_array` | CSC |
///
/// COO format is not supported via zero-copy `from_scipy` because SciPy stores
/// COO row/col as two separate arrays while Hurray requires a single packed
/// `[nnz, rank]` `uint64` index buffer (D19). Repack first:
///
/// ```python
/// indices = np.stack([m.row, m.col], axis=1).astype(np.uint64)
/// ```
///
/// Then construct the `hurray.Tensor` directly from the component buffers.
///
/// ## Index dtype requirement (D16)
///
/// Hurray's CSR/CSC spec requires `uint64` index arrays. SciPy typically uses
/// `int32` or `int64`. If the index arrays are not already `uint64`, this
/// function raises `hurray.UnsupportedError` with a cast instruction.
///
/// ## Errors
///
/// - `ImportError` — SciPy is not installed.
/// - `hurray.UnsupportedError` — unsupported matrix format (BSR, DIA, LIL, DOK).
/// - `hurray.UnsupportedError` — COO format (see above, D19).
/// - `hurray.UnsupportedError` — index arrays are not `uint64` (D16).
/// - `hurray.UnsupportedError` — values dtype has no Hurray equivalent.
///
/// ## Examples
///
/// ```python
/// import numpy as np, scipy.sparse, hurray
///
/// m = scipy.sparse.csr_matrix(
///     np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
/// )
/// # Cast index arrays to uint64 first (D16).
/// m.indices = m.indices.astype(np.uint64)
/// m.indptr  = m.indptr.astype(np.uint64)
///
/// sparse = hurray.from_scipy(m)
/// assert sparse.layout == "csr"
/// assert sparse.nnz == 2
/// ```
#[pyfunction]
pub fn from_scipy(py: Python<'_>, matrix: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    // D7-pattern: import scipy lazily — hurray must not require scipy at module load.
    py.import("scipy.sparse").map_err(|_| {
        pyo3::exceptions::PyImportError::new_err(
            "scipy is not installed; install with: pip install scipy",
        )
    })?;

    let fmt: String = matrix.getattr("format")?.extract()?;
    match fmt.as_str() {
        // D19: COO rejected here — SciPy stores row/col separately while Hurray needs a
        // packed [nnz, rank] buffer, so it cannot be zero-copy from a coo_matrix. Repack and
        // use hurray.sparse_coo instead (which shares the packed arrays zero-copy).
        "coo" => {
            return Err(UnsupportedError::new_err(
                "from_scipy does not support COO (SciPy stores row/col as two separate arrays; \
                 Hurray needs a single packed [nnz, rank] uint64 buffer). Repack and use \
                 hurray.sparse_coo: \
                 indices = np.stack([m.row, m.col], axis=1).astype(np.uint64); \
                 hurray.sparse_coo(m.data, indices, m.shape)",
            ));
        }
        "csr" | "csc" => {}
        other => {
            return Err(UnsupportedError::new_err(format!(
                "from_scipy does not support '{}' format; only 'csr' and 'csc' are supported",
                other
            )));
        }
    }

    // Extract shape (must be rank-2 for CSR/CSC).
    let shape_obj = matrix.getattr("shape")?;
    let shape_vec: Vec<i64> = shape_obj.extract()?;
    if shape_vec.len() != 2 {
        return Err(InvalidDescriptorError::new_err(
            "scipy sparse matrix must be rank-2",
        ));
    }
    if shape_vec[0] < 0 || shape_vec[1] < 0 {
        return Err(InvalidDescriptorError::new_err(
            "scipy matrix shape must have non-negative dimensions",
        ));
    }
    let nrows = shape_vec[0] as u64;
    let ncols = shape_vec[1] as u64;
    let nnz: u64 = matrix.getattr("nnz")?.extract::<u64>()?;

    // Extract .data (values array).
    let data_arr = matrix.getattr("data")?;
    let (values_ptr, values_len, element_type) = extract_numpy_buffer(py, &data_arr, None)?;

    // Extract index arrays (.indices and .indptr for both CSR and CSC).
    let indices_arr = matrix.getattr("indices")?;
    let indptr_arr = matrix.getattr("indptr")?;

    let (aux0_ptr, aux0_len, aux0_et) = extract_numpy_buffer(py, &indices_arr, None)?;
    let (aux1_ptr, aux1_len, aux1_et) = extract_numpy_buffer(py, &indptr_arr, None)?;

    // D16: both index arrays must be uint64 — reject otherwise with cast instructions.
    if aux0_et != ElementType::Uint64 {
        return Err(UnsupportedError::new_err(format!(
            "from_scipy requires uint64 index arrays; got '{:?}' for 'indices'. \
             Cast first: matrix.indices = matrix.indices.astype(np.uint64)",
            aux0_et
        )));
    }
    if aux1_et != ElementType::Uint64 {
        return Err(UnsupportedError::new_err(format!(
            "from_scipy requires uint64 index arrays; got '{:?}' for 'indptr'. \
             Cast first: matrix.indptr = matrix.indptr.astype(np.uint64)",
            aux1_et
        )));
    }

    let hurray_shape = Shape::new(vec![nrows, ncols])
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid shape: {e}")))?;

    let descriptor = match fmt.as_str() {
        "csr" => build_csr_descriptor(
            element_type,
            hurray_shape,
            nnz,
            values_len as u64,
            aux0_len as u64,
            aux1_len as u64,
        )?,
        "csc" => build_csc_descriptor(
            element_type,
            hurray_shape,
            nnz,
            values_len as u64,
            aux0_len as u64,
            aux1_len as u64,
        )?,
        _ => unreachable!(),
    };

    // D14: hold a strong reference to the SciPy matrix to keep all three component
    // arrays alive. SciPy matrices own their .data/.indices/.indptr arrays; as long
    // as the matrix object lives, the arrays (and their buffers) remain valid.
    let base: Py<PyAny> = matrix.clone().unbind();

    // SAFETY: ptr values point into the SciPy matrix's NumPy sub-arrays.
    // `base` holds the matrix alive. This relies on SciPy (≥1.7) storing `.data`,
    // `.indices`, and `.indptr` as direct instance attributes, not computed properties
    // that return new arrays on each access. If a future SciPy version makes these
    // properties that allocate on demand, holding the matrix alone would not be
    // sufficient — the individual arrays would need to be stored as bases instead.
    let values_buf = unsafe { BufferStore::borrowed(values_ptr, values_len, base.clone_ref(py)) };
    let aux_buf_0 = unsafe { BufferStore::borrowed(aux0_ptr, aux0_len, base.clone_ref(py)) };
    let aux_buf_1 = unsafe { BufferStore::borrowed(aux1_ptr, aux1_len, base) };

    make_sparse_tensor(py, descriptor, values_buf, aux_buf_0, Some(aux_buf_1))
}

// ── Registration ──────────────────────────────────────────────────────────────

/// Register `from_scipy` on the `hurray` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_scipy, m)?)?;
    Ok(())
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Extract a raw pointer, byte length, and `ElementType` from a NumPy array.
///
/// Uses `__array_interface__` which is available on all NumPy arrays.
/// If `expected` is `Some(et)`, verifies the array's dtype matches.
pub(crate) fn extract_numpy_buffer(
    _py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    expected: Option<ElementType>,
) -> PyResult<(*mut u8, usize, ElementType)> {
    let iface = arr.getattr("__array_interface__")?;
    let typestr: String = iface.get_item("typestr")?.extract()?;
    let element_type =
        crate::interop::numpy_typestr_to_element_type(&typestr).ok_or_else(|| {
            UnsupportedError::new_err(format!(
                "NumPy dtype '{}' has no Hurray equivalent",
                typestr
            ))
        })?;
    if let Some(exp) = expected {
        if element_type != exp {
            return Err(UnsupportedError::new_err(format!(
                "expected dtype {:?}, got '{}'",
                exp, typestr
            )));
        }
    }
    let data_tuple = iface.get_item("data")?;
    let ptr_int: usize = data_tuple.get_item(0)?.extract()?;
    let nbytes: usize = arr.getattr("nbytes")?.extract()?;
    Ok((ptr_int as *mut u8, nbytes, element_type))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn init() {
        pyo3::Python::initialize();
    }

    /// Check that `from_scipy` raises `UnsupportedError` for COO matrices.
    /// This test requires scipy to be importable; it is skipped gracefully if not.
    #[test]
    fn from_scipy_rejects_coo() {
        init();
        Python::attach(|py| {
            // If scipy is not available, skip.
            if py.import("scipy.sparse").is_err() {
                return;
            }
            // Build a tiny COO matrix via Python eval.
            let code = "import scipy.sparse as sp, numpy as np; sp.coo_matrix(np.eye(3))";
            let m = py
                .eval(&std::ffi::CString::new(code).unwrap(), None, None)
                .unwrap();
            let result = from_scipy(py, &m);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<UnsupportedError>(py),
                "COO should raise UnsupportedError"
            );
        });
    }

    #[test]
    fn from_scipy_rejects_lil() {
        init();
        Python::attach(|py| {
            if py.import("scipy.sparse").is_err() {
                return;
            }
            let code = "import scipy.sparse as sp, numpy as np; sp.lil_matrix(np.eye(2))";
            let m = py
                .eval(&std::ffi::CString::new(code).unwrap(), None, None)
                .unwrap();
            let result = from_scipy(py, &m);
            assert!(result.is_err());
            assert!(result.unwrap_err().is_instance_of::<UnsupportedError>(py));
        });
    }

    #[test]
    fn from_scipy_rejects_non_uint64_indices() {
        init();
        Python::attach(|py| {
            if py.import("scipy.sparse").is_err() {
                return;
            }
            // Build a CSR matrix — SciPy defaults to int32 indices.
            let code = r#"
import scipy.sparse as sp, numpy as np
sp.csr_matrix(np.array([[1.0, 0.0],[0.0, 2.0]], dtype=np.float32))
"#;
            let m = py
                .eval(&std::ffi::CString::new(code).unwrap(), None, None)
                .unwrap();
            let result = from_scipy(py, &m);
            assert!(result.is_err());
            assert!(
                result.unwrap_err().is_instance_of::<UnsupportedError>(py),
                "non-uint64 indices should raise UnsupportedError"
            );
        });
    }

    #[test]
    fn from_scipy_csr_succeeds_with_uint64_indices() {
        init();
        Python::attach(|py| {
            if py.import("scipy.sparse").is_err() {
                return;
            }
            let code = r#"
import scipy.sparse as sp, numpy as np
m = sp.csr_matrix(np.array([[1.0, 0.0],[0.0, 2.0]], dtype=np.float32))
m.indices = m.indices.astype(np.uint64)
m.indptr  = m.indptr.astype(np.uint64)
m
"#;
            let m = py
                .eval(&std::ffi::CString::new(code).unwrap(), None, None)
                .unwrap();
            let sparse =
                from_scipy(py, &m).expect("from_scipy should succeed for CSR with uint64 indices");
            assert_eq!(crate::layout::layout_name(&sparse.descriptor.layout), "csr");
            assert_eq!(sparse.nnz().unwrap(), 2);
        });
    }
}
