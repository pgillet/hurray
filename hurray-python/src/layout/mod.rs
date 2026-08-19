//! Python bindings for the layout descriptor hierarchy (ADR-032).
//!
//! `hurray.Tensor.layout` returns an object, not a string. The wire format models
//! layout as a tag plus per-layout parameters — `nnz`, `strides`, `page_size` — and
//! a string carries none of them.
//!
//! ## A data-carrying base
//!
//! [`Layout`] holds the core [`LayoutDescriptor`] and implements `tag`, `name`,
//! `buffer_count`, `is_dense`, `is_virtual`, equality, hashing and `repr` **once**.
//! Every subclass is a typed façade that reads its own fields back off that
//! descriptor; none of them store anything.
//!
//! The base is also the legal fallback object. `LayoutDescriptor` is
//! `#[non_exhaustive]`: when core gains a variant this build has not bound,
//! [`layout_to_py`] returns a bare `Layout` carrying `tag` and `name` rather than an
//! `UnknownLayout`, which would claim the tag is unrecognised when it is merely
//! unbound — the exact signal a permissive reader depends on.
//!
//! ## A descriptor, not a container
//!
//! A layout object holds no reference to its tensor and owns no buffers. The buffer
//! table is a *sibling* section of the layout section, not a child — which is why
//! quantization descriptors reference buffers by index into that shared table. The
//! component views (`values`, `indices`, `row_ptr`, …) and the generic
//! `buffer(index)` accessor stay on `hurray.Tensor`.
//!
//! ## Value semantics
//!
//! Layout objects are immutable, compare by value, and hash. `t.layout` builds a
//! fresh object per access, so `t.layout is t.layout` is `False` while
//! `t.layout == t.layout` is `True`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::PyClass;

use hurray_core::LayoutDescriptor;

use crate::errors::InvalidDescriptorError;

mod dense;
mod indirect;
mod sparse;
mod validate;

pub(crate) use validate::{validate_layout, validate_quantization_indices};

pub use dense::{
    ColMajorLayout, HilbertLayout, MortonLayout, RowMajorLayout, StridedLayout, TiledLayout,
};
pub use indirect::{BlockPagedLayout, CompositeLayout, PrivateExtensionLayout, UnknownLayout};
pub use sparse::{CooLayout, CscLayout, CsfLayout, CsrLayout};

// ── Base class ────────────────────────────────────────────────────────────────

/// The memory layout of a tensor: a tag plus that layout's parameters.
///
/// `Layout` is the base of every layout class and is **not constructible from
/// Python** — there is no layout that is only "a layout". It is returned directly
/// only as the fallback for a layout tag this build of `hurray` does not yet bind,
/// in which case `tag` and `name` are still available.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// t = hurray.Tensor(bytes(16), hurray.float32, [4])
/// assert isinstance(t.layout, hurray.Layout)
/// assert isinstance(t.layout, hurray.RowMajorLayout)
/// assert t.layout.name == "row_major"
/// assert t.layout.tag == 0x01
/// ```
#[pyclass(name = "Layout", subclass, frozen)]
#[derive(Debug)]
pub struct Layout {
    pub(crate) inner: LayoutDescriptor,
}

#[pymethods]
impl Layout {
    /// The layout tag byte, as defined by `docs/spec/memory-layout.md`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsrLayout(nnz=4).tag == 0x07
    /// ```
    #[getter]
    pub fn tag(&self) -> u8 {
        self.inner.tag()
    }

    /// The layout's name: `"row_major"`, `"csr"`, `"block_paged"`, and so on.
    ///
    /// Private and unrecognised tags both report `"extension"`; use `isinstance`
    /// to tell `PrivateExtensionLayout` from `UnknownLayout`, which is the
    /// distinction the wire format actually makes.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CooLayout(nnz=2).name == "coo"
    /// ```
    #[getter]
    pub fn name(&self) -> &'static str {
        layout_name(&self.inner)
    }

    /// The number of buffers this layout requires, or `None` when that is not
    /// statically knowable.
    ///
    /// `0` for a composite head, which owns no buffers; `None` for a private or
    /// unknown layout, whose buffer requirements only its definer knows.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.RowMajorLayout().buffer_count == 1
    /// assert hurray.CsrLayout(nnz=4).buffer_count == 3
    /// assert hurray.UnknownLayout(0x0C, b"").buffer_count is None
    /// ```
    #[getter]
    pub fn buffer_count(&self) -> Option<u8> {
        // Core reports None both for a composite head (a *known* zero that NonZeroU8
        // cannot represent) and for genuinely unknown counts; Python can say 0, so
        // the two cases must not collapse into one None here.
        if self.inner.is_virtual() {
            return Some(0);
        }
        self.inner.buffer_count().map(|n| n.get())
    }

    /// `True` if this layout stores elements in a directly addressable order — the
    /// layouts DLPack, NumPy and PyTorch can consume.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.RowMajorLayout().is_dense
    /// assert not hurray.CsrLayout(nnz=4).is_dense
    /// ```
    #[getter]
    pub fn is_dense(&self) -> bool {
        is_dense(&self.inner)
    }

    /// `True` if this layout owns no data buffer of its own — only a composite head.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert not hurray.RowMajorLayout().is_virtual
    /// ```
    #[getter]
    pub fn is_virtual(&self) -> bool {
        self.inner.is_virtual()
    }

    /// Value equality: two layouts are equal when their descriptors are.
    ///
    /// Comparing a layout to a string is always `False`. `t.layout == "csr"` was the
    /// lossy comparison this hierarchy replaces; keeping it alive as a special case
    /// would break the hash/equality contract.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CsrLayout(nnz=4) == hurray.CsrLayout(nnz=4)
    /// assert hurray.CsrLayout(nnz=4) != hurray.CsrLayout(nnz=5)
    /// assert hurray.CsrLayout(nnz=4) != "csr"
    /// ```
    pub fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        // Extracting the base matches every subclass, so equality is decided by the
        // descriptor alone — a subclass is a façade over a variant, never extra state.
        match other.extract::<PyRef<'_, Layout>>() {
            Ok(o) => self.inner == o.inner,
            Err(_) => false,
        }
    }

    /// Hash of the underlying descriptor, consistent with `__eq__`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert len({hurray.CsrLayout(nnz=4), hurray.CsrLayout(nnz=4)}) == 1
    /// ```
    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    /// The fallback representation, carrying only what an unbound tag can offer.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// print(repr(hurray.Tensor(bytes(16), hurray.float32, [4]).layout))
    /// # RowMajorLayout()
    /// ```
    pub fn __repr__(&self) -> String {
        format!("Layout(tag=0x{:02X}, name='{}')", self.tag(), self.name())
    }
}

impl Layout {
    /// Wrap a core descriptor for handing to a subclass constructor.
    pub(crate) fn of(inner: LayoutDescriptor) -> Self {
        Self { inner }
    }

    /// The descriptor this layout carries.
    pub(crate) fn descriptor(&self) -> &LayoutDescriptor {
        &self.inner
    }

    /// Build the initializer that places the façade `sub` on top of this base.
    ///
    /// Every subclass constructor goes through here, so the base is populated in
    /// exactly one place and a façade can never be built over the wrong variant by
    /// a caller assembling the pair itself.
    pub(crate) fn init<S>(self, sub: S) -> PyClassInitializer<S>
    where
        S: PyClass<BaseType = Layout>,
    {
        PyClassInitializer::from(self).add_subclass(sub)
    }
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Wrap a core [`LayoutDescriptor`] in its Python class.
///
/// A variant core defines but this build does not bind falls back to a bare
/// [`Layout`] — never to `UnknownLayout`, which means "the tag was unrecognised",
/// a different and load-bearing fact.
pub(crate) fn layout_to_py(py: Python<'_>, layout: &LayoutDescriptor) -> PyResult<Py<PyAny>> {
    let base = Layout::of(layout.clone());
    Ok(match layout {
        LayoutDescriptor::RowMajor => Py::new(py, base.init(RowMajorLayout))?.into_any(),
        LayoutDescriptor::ColMajor => Py::new(py, base.init(ColMajorLayout))?.into_any(),
        LayoutDescriptor::Strided(_) => Py::new(py, base.init(StridedLayout))?.into_any(),
        LayoutDescriptor::Tiled(_) => Py::new(py, base.init(TiledLayout))?.into_any(),
        LayoutDescriptor::Morton(_) => Py::new(py, base.init(MortonLayout))?.into_any(),
        LayoutDescriptor::Hilbert(_) => Py::new(py, base.init(HilbertLayout))?.into_any(),
        LayoutDescriptor::Coo(_) => Py::new(py, base.init(CooLayout))?.into_any(),
        LayoutDescriptor::Csr(_) => Py::new(py, base.init(CsrLayout))?.into_any(),
        LayoutDescriptor::Csc(_) => Py::new(py, base.init(CscLayout))?.into_any(),
        LayoutDescriptor::Csf(_) => Py::new(py, base.init(CsfLayout))?.into_any(),
        LayoutDescriptor::BlockPaged(_) => Py::new(py, base.init(BlockPagedLayout))?.into_any(),
        LayoutDescriptor::Composite(_) => Py::new(py, base.init(CompositeLayout))?.into_any(),
        LayoutDescriptor::PrivateExtension(_) => {
            Py::new(py, base.init(PrivateExtensionLayout))?.into_any()
        }
        LayoutDescriptor::Unknown(_) => Py::new(py, base.init(UnknownLayout))?.into_any(),
        // LayoutDescriptor is #[non_exhaustive]: a variant added to core but not yet
        // bound here reads as a bare Layout with a live tag and name.
        _ => Py::new(py, base)?.into_any(),
    })
}

/// Extract the core descriptor from any `hurray.Layout` instance.
///
/// One type check covers the whole hierarchy — the reason the classes share a base.
pub(crate) fn extract_layout(obj: &Bound<'_, PyAny>) -> PyResult<LayoutDescriptor> {
    match obj.extract::<PyRef<'_, Layout>>() {
        Ok(l) => Ok(l.inner.clone()),
        // A layout string cannot carry nnz or strides, so `layout="csr"` is a request
        // that cannot be honoured; accepting it would reopen the lossy path this
        // hierarchy exists to close.
        Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "layout must be a hurray.Layout instance (e.g. hurray.CsrLayout(nnz=4)), \
             got {}",
            obj.get_type().name()?
        ))),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// The Python-visible name of a layout.
///
/// Matches on the variant, not on `tag()`: an `UnknownLayout` may carry any tag
/// byte, and must never borrow a named layout's name on the strength of it.
///
/// Private and unrecognised tags collapse to `"extension"`: their tag byte is
/// meaningful only to whoever defined it, so naming them individually would imply
/// a support they do not have.
pub(crate) fn layout_name(layout: &LayoutDescriptor) -> &'static str {
    match layout {
        LayoutDescriptor::RowMajor => "row_major",
        LayoutDescriptor::ColMajor => "col_major",
        LayoutDescriptor::Strided(_) => "strided",
        LayoutDescriptor::Tiled(_) => "tiled",
        LayoutDescriptor::Morton(_) => "morton",
        LayoutDescriptor::Hilbert(_) => "hilbert",
        LayoutDescriptor::Coo(_) => "coo",
        LayoutDescriptor::Csr(_) => "csr",
        LayoutDescriptor::Csc(_) => "csc",
        LayoutDescriptor::Csf(_) => "csf",
        LayoutDescriptor::BlockPaged(_) => "block_paged",
        LayoutDescriptor::Composite(_) => "composite",
        // LayoutDescriptor is #[non_exhaustive]; a layout added to core but not yet
        // named here reads as an extension rather than failing to compile downstream.
        _ => "extension",
    }
}

/// The name of a layout named by **tag byte** rather than by descriptor.
///
/// Only for the tiled layout's `outer_layout` and `inner_layout` fields, which are
/// bare tag bytes on the wire with no descriptor of their own.
pub(crate) fn tag_name(tag: u8) -> &'static str {
    match tag {
        hurray_core::layout::TAG_ROW_MAJOR => "row_major",
        hurray_core::layout::TAG_COL_MAJOR => "col_major",
        hurray_core::layout::TAG_STRIDED => "strided",
        hurray_core::layout::TAG_TILED => "tiled",
        hurray_core::layout::TAG_MORTON => "morton",
        hurray_core::layout::TAG_HILBERT => "hilbert",
        _ => "extension",
    }
}

/// `true` for layouts whose single buffer holds elements in a directly addressable
/// order — the ones DLPack, NumPy and PyTorch can consume.
pub(crate) fn is_dense(layout: &LayoutDescriptor) -> bool {
    matches!(
        layout,
        LayoutDescriptor::RowMajor
            | LayoutDescriptor::ColMajor
            | LayoutDescriptor::Strided(_)
            | LayoutDescriptor::Tiled(_)
            | LayoutDescriptor::Morton(_)
            | LayoutDescriptor::Hilbert(_)
    )
}

/// The error for a façade whose base carries a different variant.
///
/// Unreachable in practice: `#[new]` is the only way to build each subclass and it
/// always stores the matching variant. Reported rather than panicked because a
/// panic would cross the FFI boundary.
pub(crate) fn variant_mismatch(class: &str, layout: &LayoutDescriptor) -> PyErr {
    InvalidDescriptorError::new_err(format!(
        "{class} wraps a {} descriptor",
        layout_name(layout)
    ))
}

/// Map a core layout error to the Python exception type.
pub(crate) fn layout_err(e: hurray_core::Error) -> PyErr {
    InvalidDescriptorError::new_err(e.to_string())
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Layout>()?;
    m.add_class::<RowMajorLayout>()?;
    m.add_class::<ColMajorLayout>()?;
    m.add_class::<StridedLayout>()?;
    m.add_class::<TiledLayout>()?;
    m.add_class::<MortonLayout>()?;
    m.add_class::<HilbertLayout>()?;
    m.add_class::<CooLayout>()?;
    m.add_class::<CsrLayout>()?;
    m.add_class::<CscLayout>()?;
    m.add_class::<CsfLayout>()?;
    m.add_class::<BlockPagedLayout>()?;
    m.add_class::<CompositeLayout>()?;
    m.add_class::<PrivateExtensionLayout>()?;
    m.add_class::<UnknownLayout>()?;
    Ok(())
}
