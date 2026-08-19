//! Indirect, virtual, and extension layout classes.
//!
//! `BlockPagedLayout` addresses its elements through a page table; `CompositeLayout`
//! is a virtual head that owns no buffers at all; `PrivateExtensionLayout` and
//! `UnknownLayout` carry tags this implementation does not define.
//!
//! ## Private and unknown are separate classes
//!
//! "A private layout I can identify by its extension id" and "a tag from a newer
//! spec version I could not parse" are different facts. Merging them would erase
//! the signal a permissive relay needs in order to pass a descriptor through
//! unchanged, so they stay separate classes — even though both report `"extension"`
//! as their `name`.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use hurray_core::layout::{
    PrivateExtensionLayout as CorePrivate, UnknownLayout as CoreUnknown, TAG_BLOCK_PAGED,
    TAG_COL_MAJOR, TAG_COMPOSITE, TAG_COO, TAG_CSC, TAG_CSF, TAG_CSR, TAG_HILBERT, TAG_MORTON,
    TAG_ROW_MAJOR, TAG_STRIDED, TAG_TILED,
};
use hurray_core::{
    BlockPagedLayout as CoreBlockPaged, BlockTableIndexType, CombineOp,
    CompositeLayout as CoreComposite, CompositionRule, KvRole, LayoutDescriptor,
};

use super::{layout_err, variant_mismatch, Layout};

// ── BlockPagedLayout ──────────────────────────────────────────────────────────

/// Block-paged indirect layout for a PagedAttention KV cache. Tag `0x0A`.
///
/// Three buffers: the page pool, the block table, and the sequence lengths. They
/// have no named accessors on `hurray.Tensor`; reach them with `t.buffer(index)`.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.BlockPagedLayout(page_size=16, num_pages=64, paged_axis=0, num_seqs=2)
/// assert l.page_size == 16
/// assert l.kv_role == "key"
/// assert l.buffer_count == 3
/// ```
#[pyclass(name = "BlockPagedLayout", extends = Layout, frozen)]
pub struct BlockPagedLayout;

#[pymethods]
impl BlockPagedLayout {
    /// Construct a block-paged layout.
    ///
    /// `kv_role` is `"key"`, `"value"`, or `"fused"`; `block_table_index_type` is
    /// `"uint32"` or `"uint64"`.
    ///
    /// ## Errors
    ///
    /// - `ValueError` — an unrecognised `kv_role` or `block_table_index_type`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// l = hurray.BlockPagedLayout(
    ///     page_size=16, num_pages=64, paged_axis=0, num_seqs=2,
    ///     kv_role="fused", layer_index=3, block_table_index_type="uint64",
    /// )
    /// assert l.layer_index == 3
    /// assert l.block_table_index_type == "uint64"
    /// ```
    #[new]
    #[pyo3(signature = (
        page_size,
        num_pages,
        paged_axis,
        num_seqs,
        kv_role = "key",
        layer_index = None,
        block_table_index_type = "uint32",
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_size: u32,
        num_pages: u64,
        paged_axis: u32,
        num_seqs: u32,
        kv_role: &str,
        layer_index: Option<u32>,
        block_table_index_type: &str,
    ) -> PyResult<PyClassInitializer<Self>> {
        let core = CoreBlockPaged::new(
            page_size,
            num_pages,
            paged_axis,
            num_seqs,
            kv_role_from_name(kv_role)?,
            layer_index,
            index_type_from_name(block_table_index_type)?,
        );
        Ok(Layout::of(LayoutDescriptor::BlockPaged(core)).init(Self))
    }

    /// Number of tokens per page.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2).page_size == 16
    /// ```
    #[getter]
    pub fn page_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(paged_of(&slf)?.page_size)
    }

    /// Number of pages in the shared pool.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2).num_pages == 64
    /// ```
    #[getter]
    pub fn num_pages(slf: PyRef<'_, Self>) -> PyResult<u64> {
        Ok(paged_of(&slf)?.num_pages)
    }

    /// The axis divided into pages.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2).paged_axis == 0
    /// ```
    #[getter]
    pub fn paged_axis(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(paged_of(&slf)?.paged_axis)
    }

    /// Number of sequences sharing the page pool.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2).num_seqs == 2
    /// ```
    #[getter]
    pub fn num_seqs(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(paged_of(&slf)?.num_seqs)
    }

    /// Which half of the KV cache this tensor holds: `"key"`, `"value"`, or `"fused"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2, kv_role="value").kv_role == "value"
    /// ```
    #[getter]
    pub fn kv_role(slf: PyRef<'_, Self>) -> PyResult<&'static str> {
        Ok(kv_role_name(paged_of(&slf)?.kv_role))
    }

    /// The transformer layer this cache belongs to, or `None` if unstated.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2).layer_index is None
    /// ```
    #[getter]
    pub fn layer_index(slf: PyRef<'_, Self>) -> PyResult<Option<u32>> {
        Ok(paged_of(&slf)?.layer_index)
    }

    /// The element type of the block table: `"uint32"` or `"uint64"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.BlockPagedLayout(16, 64, 0, 2).block_table_index_type == "uint32"
    /// ```
    #[getter]
    pub fn block_table_index_type(slf: PyRef<'_, Self>) -> PyResult<&'static str> {
        Ok(index_type_name(paged_of(&slf)?.block_table_index_type))
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.BlockPagedLayout(16, 64, 0, 2)).startswith("BlockPagedLayout(page_size=16")
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let l = paged_of(&slf)?;
        let layer = match l.layer_index {
            Some(i) => i.to_string(),
            None => "None".to_string(),
        };
        Ok(format!(
            "BlockPagedLayout(page_size={}, num_pages={}, paged_axis={}, num_seqs={}, \
             kv_role='{}', layer_index={layer}, block_table_index_type='{}')",
            l.page_size,
            l.num_pages,
            l.paged_axis,
            l.num_seqs,
            kv_role_name(l.kv_role),
            index_type_name(l.block_table_index_type),
        ))
    }
}

fn paged_of<'a>(slf: &'a PyRef<'_, BlockPagedLayout>) -> PyResult<&'a CoreBlockPaged> {
    match slf.as_super().descriptor() {
        LayoutDescriptor::BlockPaged(l) => Ok(l),
        other => Err(variant_mismatch("BlockPagedLayout", other)),
    }
}

fn kv_role_name(role: KvRole) -> &'static str {
    match role {
        KvRole::Key => "key",
        KvRole::Value => "value",
        KvRole::Fused => "fused",
        // KvRole is #[non_exhaustive]; an unbound role reads as unknown rather than
        // borrowing the name of one this build does understand.
        _ => "unknown",
    }
}

fn kv_role_from_name(name: &str) -> PyResult<KvRole> {
    match name {
        "key" => Ok(KvRole::Key),
        "value" => Ok(KvRole::Value),
        "fused" => Ok(KvRole::Fused),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid kv_role {other:?}: expected 'key', 'value', or 'fused'"
        ))),
    }
}

fn index_type_name(t: BlockTableIndexType) -> &'static str {
    match t {
        BlockTableIndexType::U32 => "uint32",
        BlockTableIndexType::U64 => "uint64",
        _ => "unknown",
    }
}

fn index_type_from_name(name: &str) -> PyResult<BlockTableIndexType> {
    match name {
        "uint32" => Ok(BlockTableIndexType::U32),
        "uint64" => Ok(BlockTableIndexType::U64),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid block_table_index_type {other:?}: expected 'uint32' or 'uint64'"
        ))),
    }
}

// ── CompositeLayout ───────────────────────────────────────────────────────────

/// Composite (virtual) head. Tag `0x0B`. Owns no buffers.
///
/// A composite head presents one logical view over an ordered set of member tensors
/// bound by stream adjacency. It is **readable** here so that a head decoded from a
/// stream reports its own layout truthfully; building a `hurray.Tensor` with it
/// raises `hurray.UnsupportedError`, because the Python `Tensor` cannot yet
/// represent a tensor that owns no buffers.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.CompositeLayout("overlay", member_count=3, combine_op="add")
/// assert l.composition_rule == "overlay"
/// assert l.combine_op == "add"
/// assert l.buffer_count == 0
/// assert l.is_virtual
/// ```
#[pyclass(name = "CompositeLayout", extends = Layout, frozen)]
pub struct CompositeLayout;

#[pymethods]
impl CompositeLayout {
    /// Construct a composite head descriptor.
    ///
    /// `combine_op` is required when `composition_rule` is `"overlay"` and MUST be
    /// omitted otherwise: for a partition or a group there is no combine operation
    /// to name, and inventing one would put a meaningless byte on the wire.
    ///
    /// The two are named after the spec's own fields, so the binding does not add a
    /// third vocabulary alongside the specification's and `hurray-inspect`'s.
    ///
    /// ## Errors
    ///
    /// - `ValueError` — an unrecognised rule or combine operation, a `combine_op` on
    ///   a non-overlay rule, or an overlay with no `combine_op`.
    /// - `hurray.InvalidDescriptorError` — a member count core rejects.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CompositeLayout("group", member_count=2).combine_op is None
    /// ```
    #[new]
    #[pyo3(signature = (composition_rule, member_count, combine_op = None))]
    pub fn new(
        composition_rule: &str,
        member_count: u32,
        combine_op: Option<&str>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let rule = parse_composition_rule(composition_rule, combine_op)?;
        let core = CoreComposite::new(rule, member_count).map_err(layout_err)?;
        Ok(Layout::of(LayoutDescriptor::Composite(core)).init(Self))
    }

    /// The composition rule: `"partition"`, `"overlay"`, or `"group"`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CompositeLayout("partition", 2).composition_rule == "partition"
    /// ```
    #[getter]
    pub fn composition_rule(slf: PyRef<'_, Self>) -> PyResult<&'static str> {
        Ok(rule_name(&composite_of(&slf)?.rule))
    }

    /// The combine operation for an overlay, or `None` for any other rule.
    ///
    /// Kept separate from `composition_rule` rather than flattened into one string:
    /// for a
    /// partition or a group the operation is not merely unset, it does not apply.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CompositeLayout("overlay", 2, combine_op="replace").combine_op == "replace"
    /// assert hurray.CompositeLayout("partition", 2).combine_op is None
    /// ```
    #[getter]
    pub fn combine_op(slf: PyRef<'_, Self>) -> PyResult<Option<&'static str>> {
        Ok(match &composite_of(&slf)?.rule {
            CompositionRule::Overlay(op) => Some(combine_name(*op)),
            _ => None,
        })
    }

    /// The number of member tensors this head composes.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.CompositeLayout("group", 4).member_count == 4
    /// ```
    #[getter]
    pub fn member_count(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(composite_of(&slf)?.member_count)
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.CompositeLayout("group", 2)) == \
    ///     "CompositeLayout(composition_rule='group', member_count=2)"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        let core = composite_of(&slf)?;
        let combine = match &core.rule {
            CompositionRule::Overlay(op) => format!(", combine_op='{}'", combine_name(*op)),
            _ => String::new(),
        };
        Ok(format!(
            "CompositeLayout(composition_rule='{}', member_count={}{combine})",
            rule_name(&core.rule),
            core.member_count,
        ))
    }
}

fn composite_of<'a>(slf: &'a PyRef<'_, CompositeLayout>) -> PyResult<&'a CoreComposite> {
    match slf.as_super().descriptor() {
        LayoutDescriptor::Composite(l) => Ok(l),
        other => Err(variant_mismatch("CompositeLayout", other)),
    }
}

fn rule_name(rule: &CompositionRule) -> &'static str {
    match rule {
        CompositionRule::Partition => "partition",
        CompositionRule::Overlay(_) => "overlay",
        CompositionRule::Group => "group",
        _ => "unknown",
    }
}

fn combine_name(op: CombineOp) -> &'static str {
    match op {
        CombineOp::Replace => "replace",
        CombineOp::Add => "add",
        _ => "unknown",
    }
}

fn parse_composition_rule(rule: &str, combine: Option<&str>) -> PyResult<CompositionRule> {
    let overlay_op = |op: Option<&str>| -> PyResult<CombineOp> {
        match op {
            Some("replace") => Ok(CombineOp::Replace),
            Some("add") => Ok(CombineOp::Add),
            Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid combine_op {other:?}: expected 'replace' or 'add'"
            ))),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                "an overlay composition requires combine_op='replace' or combine_op='add'",
            )),
        }
    };
    match (rule, combine) {
        ("overlay", op) => Ok(CompositionRule::Overlay(overlay_op(op)?)),
        ("partition", None) => Ok(CompositionRule::Partition),
        ("group", None) => Ok(CompositionRule::Group),
        ("partition" | "group", Some(_)) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "combine_op does not apply to a {rule:?} composition; it is only meaningful \
             for 'overlay'"
        ))),
        (other, _) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid composition rule {other:?}: expected 'partition', 'overlay', or 'group'"
        ))),
    }
}

// ── PrivateExtensionLayout ────────────────────────────────────────────────────

/// An implementation-private layout. Tags `0xF0`–`0xFE`.
///
/// The buffer count is unknown, so a tensor built with this layout **skips the
/// buffer-size check** every other layout gets: nothing in the descriptor says how
/// large its buffers should be. That hole is intrinsic to a private tag.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.PrivateExtensionLayout(0xF0, extension_layout_id=7, extension_data=b"\x01")
/// assert l.tag == 0xF0
/// assert l.extension_layout_id == 7
/// assert l.buffer_count is None
/// ```
#[pyclass(name = "PrivateExtensionLayout", extends = Layout, frozen)]
pub struct PrivateExtensionLayout;

#[pymethods]
impl PrivateExtensionLayout {
    /// Construct a private extension layout.
    ///
    /// ## Errors
    ///
    /// - `hurray.InvalidDescriptorError` — a tag outside the private range
    ///   `0xF0`–`0xFE`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.PrivateExtensionLayout(0xF1, 0, b"").name == "extension"
    /// ```
    #[new]
    #[pyo3(signature = (tag, extension_layout_id, extension_data = None))]
    pub fn new(
        tag: u8,
        extension_layout_id: u64,
        extension_data: Option<Vec<u8>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let core = CorePrivate::new(tag, extension_layout_id, extension_data.unwrap_or_default())
            .map_err(layout_err)?;
        Ok(Layout::of(LayoutDescriptor::PrivateExtension(core)).init(Self))
    }

    /// The private extension identifier.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.PrivateExtensionLayout(0xF0, 7, b"").extension_layout_id == 7
    /// ```
    #[getter]
    pub fn extension_layout_id(slf: PyRef<'_, Self>) -> PyResult<u64> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::PrivateExtension(l) => Ok(l.extension_layout_id),
            other => Err(variant_mismatch("PrivateExtensionLayout", other)),
        }
    }

    /// The opaque payload, as `bytes`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.PrivateExtensionLayout(0xF0, 7, b"\x01\x02").extension_data == b"\x01\x02"
    /// ```
    #[getter]
    pub fn extension_data(slf: PyRef<'_, Self>) -> PyResult<Py<PyBytes>> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::PrivateExtension(l) => {
                Ok(PyBytes::new(slf.py(), &l.extension_data).unbind())
            }
            other => Err(variant_mismatch("PrivateExtensionLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.PrivateExtensionLayout(0xF0, 7, b"")).startswith(
    ///     "PrivateExtensionLayout(tag=0xF0"
    /// )
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::PrivateExtension(l) => Ok(format!(
                "PrivateExtensionLayout(tag=0x{:02X}, extension_layout_id={}, \
                 extension_data={} bytes)",
                l.tag,
                l.extension_layout_id,
                l.extension_data.len()
            )),
            other => Err(variant_mismatch("PrivateExtensionLayout", other)),
        }
    }
}

// ── UnknownLayout ─────────────────────────────────────────────────────────────

/// A layout tag this implementation does not recognise, accepted in permissive mode.
///
/// Constructible so that a permissive relay can rebuild a descriptor it decoded and
/// write it back out unchanged. Like a private layout, its buffer count is unknown,
/// so a tensor built with it skips the buffer-size check.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// l = hurray.UnknownLayout(0x0C, b"\x00\x01")
/// assert l.tag == 0x0C
/// assert l.raw_bytes == b"\x00\x01"
/// assert l.name == "extension"
/// ```
#[pyclass(name = "UnknownLayout", extends = Layout, frozen)]
pub struct UnknownLayout;

#[pymethods]
impl UnknownLayout {
    /// Construct an unknown-layout passthrough.
    ///
    /// ## Errors
    ///
    /// - `ValueError` — a tag this build *does* recognise. Calling a known tag
    ///   "unknown" would smuggle a descriptor past every rank and buffer check that
    ///   the named class would have applied.
    /// - `hurray.InvalidDescriptorError` — the permanently-invalid sentinels `0x00`
    ///   and `0xFF`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.UnknownLayout(0x0C).raw_bytes == b""
    /// ```
    #[new]
    #[pyo3(signature = (tag, raw_bytes = None))]
    pub fn new(tag: u8, raw_bytes: Option<Vec<u8>>) -> PyResult<PyClassInitializer<Self>> {
        if let Some(known) = known_tag_name(tag) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tag 0x{tag:02X} is the {known} layout, not an unknown one; use \
                 hurray.{class} instead",
                class = known_tag_class(tag),
            )));
        }
        let core = CoreUnknown::new(tag, raw_bytes.unwrap_or_default()).map_err(layout_err)?;
        Ok(Layout::of(LayoutDescriptor::Unknown(core)).init(Self))
    }

    /// The unrecognised tag byte.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert hurray.UnknownLayout(0x0C).tag == 0x0C
    /// ```
    #[getter]
    pub fn raw_bytes(slf: PyRef<'_, Self>) -> PyResult<Py<PyBytes>> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Unknown(l) => Ok(PyBytes::new(slf.py(), &l.raw_bytes).unbind()),
            other => Err(variant_mismatch("UnknownLayout", other)),
        }
    }

    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// assert repr(hurray.UnknownLayout(0x0C)) == "UnknownLayout(tag=0x0C, raw_bytes=0 bytes)"
    /// ```
    pub fn __repr__(slf: PyRef<'_, Self>) -> PyResult<String> {
        match slf.as_super().descriptor() {
            LayoutDescriptor::Unknown(l) => Ok(format!(
                "UnknownLayout(tag=0x{:02X}, raw_bytes={} bytes)",
                l.tag,
                l.raw_bytes.len()
            )),
            other => Err(variant_mismatch("UnknownLayout", other)),
        }
    }
}

/// The name of the layout a tag denotes, or `None` if this build does not know it.
///
/// Private tags count as known: they belong to `PrivateExtensionLayout`, which can
/// carry an extension id, where `UnknownLayout` would throw that structure away.
fn known_tag_name(tag: u8) -> Option<&'static str> {
    match tag {
        TAG_ROW_MAJOR => Some("row_major"),
        TAG_COL_MAJOR => Some("col_major"),
        TAG_STRIDED => Some("strided"),
        TAG_TILED => Some("tiled"),
        TAG_MORTON => Some("morton"),
        TAG_COO => Some("coo"),
        TAG_CSR => Some("csr"),
        TAG_CSC => Some("csc"),
        TAG_CSF => Some("csf"),
        TAG_BLOCK_PAGED => Some("block_paged"),
        TAG_COMPOSITE => Some("composite"),
        TAG_HILBERT => Some("hilbert"),
        t if hurray_core::layout::is_private_tag(t) => Some("private extension"),
        _ => None,
    }
}

/// The Python class a known tag should have been built with.
fn known_tag_class(tag: u8) -> &'static str {
    match tag {
        TAG_ROW_MAJOR => "RowMajorLayout",
        TAG_COL_MAJOR => "ColMajorLayout",
        TAG_STRIDED => "StridedLayout",
        TAG_TILED => "TiledLayout",
        TAG_MORTON => "MortonLayout",
        TAG_COO => "CooLayout",
        TAG_CSR => "CsrLayout",
        TAG_CSC => "CscLayout",
        TAG_CSF => "CsfLayout",
        TAG_BLOCK_PAGED => "BlockPagedLayout",
        TAG_COMPOSITE => "CompositeLayout",
        TAG_HILBERT => "HilbertLayout",
        _ => "PrivateExtensionLayout",
    }
}
