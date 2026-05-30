//! Python bindings for the Hurray element type system.
//!
//! Exposes [`Dtype`] as a Python class (`hurray.Dtype`) and populates a
//! `hurray.dtype` submodule with one constant per element type.
//!
//! ## Type organisation
//!
//! - **Tier 1** types (Array API-compatible) are accessible both at the
//!   top-level (`hurray.float32`) and on the submodule (`hurray.dtype.float32`).
//!   Both names bind the **same Python object** (D3).
//! - **Tier 2** types (extended / sub-byte) are accessible only on the
//!   submodule (`hurray.dtype.int4`); no top-level alias is created.
//!
//! The `hurray.dtype` submodule is registered in `sys.modules` so
//! `from hurray.dtype import int4` works correctly (D4).

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr for functions
// returning PyResult<()> — suppress the false positive across this module.
#![allow(clippy::useless_conversion)]

use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use hurray_core::ElementType;

use crate::errors::InvalidDescriptorError;

/// A Hurray element type descriptor.
///
/// `Dtype` objects are singletons: `hurray.float32 is hurray.dtype.float32`.
/// They are immutable (`frozen`) and hashable — safe to use as dict keys.
///
/// ## Tier classification
///
/// | Tier | Meaning | Accessible at |
/// |------|---------|---------------|
/// | 1 | Array API-compatible | `hurray.<name>` and `hurray.dtype.<name>` |
/// | 2 | Extended / sub-byte | `hurray.dtype.<name>` only |
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// assert hurray.float32.name == "float32"
/// assert hurray.float32.is_array_api
/// assert hurray.float32 is hurray.dtype.float32   # singleton identity
///
/// assert hurray.dtype.int4.bit_width == 4
/// assert not hurray.dtype.int4.is_array_api
/// assert not hasattr(hurray, "int4")              # Tier 2: no top-level alias
///
/// # Usable as a dict key.
/// dtypes = {hurray.float32: "fp32", hurray.dtype.int4: "i4"}
/// assert dtypes[hurray.float32] == "fp32"
/// ```
#[pyclass(name = "Dtype", frozen)]
pub struct Dtype {
    /// The underlying hurray-core element type.
    pub inner: ElementType,
}

#[pymethods]
impl Dtype {
    // ── Getters ──────────────────────────────────────────────────────────────

    /// The canonical lowercase spec name for this type (e.g. `"float32"`, `"int4"`).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.float32.name == "float32"
    /// assert hurray.dtype.int4.name == "int4"
    /// ```
    #[getter]
    pub fn name(&self) -> &'static str {
        element_type_name(self.inner)
    }

    /// Bit width of one scalar element (e.g. 32 for `float32`, 4 for `int4`).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.float32.bit_width == 32
    /// assert hurray.dtype.int4.bit_width == 4
    /// ```
    #[getter]
    pub fn bit_width(&self) -> u32 {
        self.inner.bit_width()
    }

    /// `True` if this is a signed or unsigned integer type (not float, not bool).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.int32.is_integer
    /// assert not hurray.float32.is_integer
    /// ```
    #[getter]
    pub fn is_integer(&self) -> bool {
        self.inner.is_integer()
    }

    /// `True` if this is a floating-point or complex type.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.float32.is_float
    /// assert not hurray.int32.is_float
    /// ```
    #[getter]
    pub fn is_float(&self) -> bool {
        self.inner.is_float()
    }

    /// `True` if this type is signed (has a sign bit or two's-complement sign).
    ///
    /// Unsigned integers, `bool`, and `float8_e8m0` return `False`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.int32.is_signed
    /// assert not hurray.uint32.is_signed
    /// ```
    #[getter]
    pub fn is_signed(&self) -> bool {
        self.inner.is_signed()
    }

    /// `True` if elements occupy fewer than 8 bits (e.g. `int4`, `uint4`, `bool`).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.dtype.int4.is_sub_byte
    /// assert not hurray.float32.is_sub_byte
    /// ```
    #[getter]
    pub fn is_sub_byte(&self) -> bool {
        self.inner.is_sub_byte()
    }

    /// The tier of this type: `1` for Array API-compatible types, `2` for extended types.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.float32.tier == 1
    /// assert hurray.dtype.int4.tier == 2
    /// ```
    #[getter]
    pub fn tier(&self) -> u8 {
        self.inner.tier()
    }

    /// `True` iff this is a Tier 1 (Array API-compatible) type.
    ///
    /// Equivalent to `self.tier == 1`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.float32.is_array_api
    /// assert not hurray.dtype.int4.is_array_api
    /// ```
    #[getter]
    pub fn is_array_api(&self) -> bool {
        self.inner.tier() == 1
    }

    // ── Dunders ──────────────────────────────────────────────────────────────

    fn __repr__(&self) -> String {
        format!("hurray.Dtype('{}')", element_type_name(self.inner))
    }

    fn __str__(&self) -> &'static str {
        element_type_name(self.inner)
    }

    fn __richcmp__(&self, other: &Dtype, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            // Ordering is not meaningful for element types.
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => Err(
                pyo3::exceptions::PyNotImplementedError::new_err("Dtype does not support ordering"),
            ),
        }
    }

    fn __hash__(&self) -> u64 {
        // Use the wire tag as hash; unique per variant, small, and stable.
        self.inner.tag() as u64
    }

    // ── Class methods ─────────────────────────────────────────────────────────

    /// Parse a `Dtype` from its canonical name string.
    ///
    /// Raises `hurray.InvalidDescriptorError` for unknown names.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.Dtype.from_name("float32") == hurray.float32
    /// assert hurray.Dtype.from_name("int4") == hurray.dtype.int4
    ///
    /// try:
    ///     hurray.Dtype.from_name("not_a_type")
    /// except hurray.InvalidDescriptorError:
    ///     pass
    /// ```
    #[classmethod]
    pub fn from_name(_cls: &Bound<'_, pyo3::types::PyType>, name: &str) -> PyResult<Dtype> {
        let inner = element_type_from_name(name).ok_or_else(|| {
            InvalidDescriptorError::new_err(format!("unknown element type name: '{name}'"))
        })?;
        Ok(Dtype { inner })
    }
}

// ── Name ↔ ElementType helpers ────────────────────────────────────────────────

/// Returns the canonical spec name for an `ElementType`.
///
/// Uses `ElementType`'s `Display` impl which already produces the correct
/// lowercase underscore-separated names from `element-types.md`.
pub(crate) fn element_type_name(ty: ElementType) -> &'static str {
    // Static table; faster than heap-allocating a String on every getter call.
    match ty {
        ElementType::Float16 => "float16",
        ElementType::BFloat16 => "bfloat16",
        ElementType::Float32 => "float32",
        ElementType::Float64 => "float64",
        ElementType::Int8 => "int8",
        ElementType::Uint8 => "uint8",
        ElementType::Int16 => "int16",
        ElementType::Uint16 => "uint16",
        ElementType::Int32 => "int32",
        ElementType::Uint32 => "uint32",
        ElementType::Int64 => "int64",
        ElementType::Uint64 => "uint64",
        ElementType::Bool => "bool",
        ElementType::Float8E4M3 => "float8_e4m3",
        ElementType::Float8E5M2 => "float8_e5m2",
        ElementType::Float8E8M0 => "float8_e8m0",
        ElementType::Float4E2M1 => "float4_e2m1",
        ElementType::Float6E2M3 => "float6_e2m3",
        ElementType::Float6E3M2 => "float6_e3m2",
        ElementType::Float128 => "float128",
        ElementType::Int4 => "int4",
        ElementType::Uint4 => "uint4",
        ElementType::Int2 => "int2",
        ElementType::Uint2 => "uint2",
        ElementType::Complex64 => "complex64",
        ElementType::Complex128 => "complex128",
        // Extension types carry semantics out-of-band; no fixed string name.
        ElementType::Extension(_) => "extension",
    }
}

/// Reverse-maps a canonical name string to an `ElementType`.
///
/// Returns `None` for unknown names.
pub(crate) fn element_type_from_name(name: &str) -> Option<ElementType> {
    match name {
        "float16" => Some(ElementType::Float16),
        "bfloat16" => Some(ElementType::BFloat16),
        "float32" => Some(ElementType::Float32),
        "float64" => Some(ElementType::Float64),
        "int8" => Some(ElementType::Int8),
        "uint8" => Some(ElementType::Uint8),
        "int16" => Some(ElementType::Int16),
        "uint16" => Some(ElementType::Uint16),
        "int32" => Some(ElementType::Int32),
        "uint32" => Some(ElementType::Uint32),
        "int64" => Some(ElementType::Int64),
        "uint64" => Some(ElementType::Uint64),
        "bool" => Some(ElementType::Bool),
        "float8_e4m3" => Some(ElementType::Float8E4M3),
        "float8_e5m2" => Some(ElementType::Float8E5M2),
        "float8_e8m0" => Some(ElementType::Float8E8M0),
        "float4_e2m1" => Some(ElementType::Float4E2M1),
        "float6_e2m3" => Some(ElementType::Float6E2M3),
        "float6_e3m2" => Some(ElementType::Float6E3M2),
        "float128" => Some(ElementType::Float128),
        "int4" => Some(ElementType::Int4),
        "uint4" => Some(ElementType::Uint4),
        "int2" => Some(ElementType::Int2),
        "uint2" => Some(ElementType::Uint2),
        "complex64" => Some(ElementType::Complex64),
        "complex128" => Some(ElementType::Complex128),
        _ => None,
    }
}

/// All `ElementType` variants in declaration order, used to populate the submodule.
const ALL_ELEMENT_TYPES: &[ElementType] = &[
    // Tier 1 — float
    ElementType::Float16,
    ElementType::BFloat16,
    ElementType::Float32,
    ElementType::Float64,
    // Tier 1 — integer
    ElementType::Int8,
    ElementType::Uint8,
    ElementType::Int16,
    ElementType::Uint16,
    ElementType::Int32,
    ElementType::Uint32,
    ElementType::Int64,
    ElementType::Uint64,
    // Tier 1 — bool
    ElementType::Bool,
    // Tier 2 — float8
    ElementType::Float8E4M3,
    ElementType::Float8E5M2,
    ElementType::Float8E8M0,
    // Tier 2 — sub-byte float
    ElementType::Float4E2M1,
    ElementType::Float6E2M3,
    ElementType::Float6E3M2,
    // Tier 2 — extended float
    ElementType::Float128,
    // Tier 2 — sub-byte integer
    ElementType::Int4,
    ElementType::Uint4,
    ElementType::Int2,
    ElementType::Uint2,
    // Tier 2 — complex
    ElementType::Complex64,
    ElementType::Complex128,
];

/// Register `Dtype` and the `hurray.dtype` submodule on the parent module.
///
/// For Tier 1 types, the same `Py<Dtype>` object is bound on both the parent
/// module and the submodule, ensuring `hurray.float32 is hurray.dtype.float32`
/// (D3 singleton identity).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add_class::<Dtype>()?;

    // Create the `hurray.dtype` submodule.
    let dtype_mod = PyModule::new_bound(py, "dtype")?;

    for &ty in ALL_ELEMENT_TYPES {
        let name = element_type_name(ty);
        let obj = Py::new(py, Dtype { inner: ty })?;

        // Always add to `hurray.dtype.*`.
        dtype_mod.add(name, obj.clone_ref(py))?;

        // Tier 1 types also get a top-level alias — same Python object.
        if ty.tier() == 1 {
            m.add(name, obj)?;
        }
    }

    // Register `hurray.dtype` in `sys.modules` so `from hurray.dtype import int4` works.
    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("hurray.dtype", &dtype_mod)?;

    m.add_submodule(&dtype_mod)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn init() {
        // Required before any Python::with_gil call in test binaries.
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn name_round_trip() {
        // Test the underlying pure-Rust helper directly — no GIL needed.
        let tier1 = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Bool,
        ];
        for ty in tier1 {
            let name = element_type_name(ty);
            let roundtripped = element_type_from_name(name)
                .unwrap_or_else(|| panic!("from_name should succeed for '{name}'"));
            assert_eq!(
                element_type_name(roundtripped),
                name,
                "round-trip failed for {name}"
            );
        }
    }

    #[test]
    fn tier1_is_array_api() {
        let tier1 = [
            ElementType::Float16,
            ElementType::BFloat16,
            ElementType::Float32,
            ElementType::Float64,
            ElementType::Int8,
            ElementType::Uint8,
            ElementType::Int16,
            ElementType::Uint16,
            ElementType::Int32,
            ElementType::Uint32,
            ElementType::Int64,
            ElementType::Uint64,
            ElementType::Bool,
        ];
        for ty in tier1 {
            let d = Dtype { inner: ty };
            assert!(d.is_array_api(), "{} should be is_array_api", d.name());
        }
    }

    #[test]
    fn tier2_not_array_api() {
        let tier2 = [
            ElementType::Int4,
            ElementType::Uint4,
            ElementType::Float8E4M3,
            ElementType::Float8E5M2,
            ElementType::Float8E8M0,
            ElementType::Float4E2M1,
            ElementType::Float6E2M3,
            ElementType::Float6E3M2,
            ElementType::Float128,
            ElementType::Int2,
            ElementType::Uint2,
            ElementType::Complex64,
            ElementType::Complex128,
        ];
        for ty in tier2 {
            let d = Dtype { inner: ty };
            assert!(!d.is_array_api(), "{} should NOT be is_array_api", d.name());
        }
    }

    #[test]
    fn bit_width_float32() {
        let d = Dtype {
            inner: ElementType::Float32,
        };
        assert_eq!(d.bit_width(), 32);
    }

    #[test]
    fn bit_width_int4() {
        let d = Dtype {
            inner: ElementType::Int4,
        };
        assert_eq!(d.bit_width(), 4);
    }

    #[test]
    fn eq_same_type() {
        init();
        Python::with_gil(|_py| {
            let a = Dtype {
                inner: ElementType::Float32,
            };
            let b = Dtype {
                inner: ElementType::Float32,
            };
            assert_eq!(a.inner, b.inner);
        });
    }

    #[test]
    fn ne_different_types() {
        let a = Dtype {
            inner: ElementType::Float32,
        };
        let b = Dtype {
            inner: ElementType::Int32,
        };
        assert_ne!(a.inner, b.inner);
    }

    #[test]
    fn hash_consistent_with_eq() {
        let a = Dtype {
            inner: ElementType::Float32,
        };
        let b = Dtype {
            inner: ElementType::Float32,
        };
        assert_eq!(a.__hash__(), b.__hash__());
    }

    #[test]
    fn from_name_unknown() {
        // Test the pure Rust helper — no GIL needed.
        assert!(
            element_type_from_name("garbage").is_none(),
            "element_type_from_name('garbage') should return None"
        );
    }

    #[test]
    fn repr_format() {
        let d = Dtype {
            inner: ElementType::Float32,
        };
        let r = d.__repr__();
        assert!(
            r.contains("float32"),
            "repr should contain canonical name: got '{r}'"
        );
    }
}
