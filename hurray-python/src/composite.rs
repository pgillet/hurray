//! Composite tensors from Python (ADR-036).
//!
//! ```python
//! composite = hurray.Composite(
//!     "partition", shape=[8, 8], dtype=hurray.float32, members=[tile0, tile1]
//! )
//! ```
//!
//! ## A container, not a tensor
//!
//! A composite *contains* tensors, where a sparse tensor *has* data. Its head owns
//! zero buffers — the format calls that addressing category **virtual** — so there is
//! no `.values`, no `__dlpack__`, and nothing to hand a consumer directly. The data
//! belongs to the members, each of which is an ordinary `hurray.Tensor`.
//!
//! That is why this is its own class rather than a `hurray.Tensor` with a composite
//! layout: `len(members)` means nothing on a tensor, and putting a second family of
//! inapplicable accessors on `Tensor` would break the `hasattr` discipline ADR-031 § 2
//! chose deliberately.
//!
//! ## Stated, not derived
//!
//! The head's `shape`, `dtype`, rule, and combine operation are required arguments.
//! A partition's shape could be computed from its members' shards; deriving it would
//! quietly reshape the head to match a caller's miscomputed offset, and hand the next
//! consumer something self-consistent and wrong. Validation is core's
//! `CompositeValidator` — the binding keeps no second copy of the coverage rules.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use hurray_core::{
    composite::CompositeTensor, layout::CompositeLayout, LayoutDescriptor, TensorDescriptor,
};

use crate::dtype::Dtype;
use crate::errors::InvalidDescriptorError;
use crate::tensor::Tensor;

use hurray_io::file::FileCompositeNode as FileNode;
use hurray_io::stream::CompositeNode as StreamNode;

/// One member of a composite: an ordinary tensor, or a nested composite.
///
/// Mirrors the wire, where a nested composite's *head* is the member its parent sees
/// (ADR-027 § Binding).
pub(crate) enum Member {
    Tensor(Py<Tensor>),
    Composite(Py<Composite>),
}

impl Member {
    /// The descriptor this member presents to its parent's validator.
    pub(crate) fn descriptor(&self, py: Python<'_>) -> TensorDescriptor {
        match self {
            Member::Tensor(t) => t.borrow(py).descriptor.clone(),
            Member::Composite(c) => c.borrow(py).head.clone(),
        }
    }

    /// The member as a Python object, for `members`.
    fn to_object(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Member::Tensor(t) => t.clone_ref(py).into_any(),
            Member::Composite(c) => c.clone_ref(py).into_any(),
        }
    }
}

/// A composite tensor: a head presenting one logical view over ordered members.
///
/// ## Examples (Python)
///
/// ```python
/// import hurray, struct
///
/// tile = hurray.Tensor(bytes(128), hurray.float32, [8, 4], shard=hurray.Shard([8, 8], [0, 0]))
/// other = hurray.Tensor(bytes(128), hurray.float32, [8, 4], shard=hurray.Shard([8, 8], [0, 4]))
///
/// composite = hurray.Composite(
///     "partition", shape=[8, 8], dtype=hurray.float32, members=[tile, other]
/// )
/// assert composite.member_count == 2
/// assert composite.layout.composition_rule == "partition"
/// ```
#[pyclass(name = "Composite")]
pub struct Composite {
    /// The composite head descriptor: `layout_tag = 0x0B`, no buffers.
    pub(crate) head: TensorDescriptor,
    /// Members in wire order.
    pub(crate) members: Vec<Member>,
    /// Python-side dtype handle, so `composite.dtype is hurray.float32` holds.
    dtype_py: Py<Dtype>,
}

#[pymethods]
impl Composite {
    /// Build a composite from its rule, its logical view, and its members.
    ///
    /// `combine_op` is required for `"overlay"` and MUST be omitted otherwise — the
    /// same rule `hurray.CompositeLayout` applies, because it is the same field.
    ///
    /// ## Errors
    ///
    /// - `TypeError` — a member that is neither a `hurray.Tensor` nor a
    ///   `hurray.Composite`.
    /// - `ValueError` — an unrecognised rule or combine operation.
    /// - `hurray.InvalidDescriptorError` — anything core's validator rejects: a
    ///   partition that does not cover its head's index space, an overlay out of
    ///   order, a member whose box falls outside the head, a dynamic dimension in a
    ///   partition or overlay head.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// group = hurray.Composite(
    ///     "group",
    ///     shape=[4],
    ///     dtype=hurray.float32,
    ///     members=[hurray.Tensor(bytes(16), hurray.float32, [4])],
    /// )
    /// assert group.member_count == 1
    /// ```
    #[new]
    #[pyo3(signature = (composition_rule, *, shape, dtype, members, combine_op = None))]
    pub fn new(
        py: Python<'_>,
        composition_rule: &str,
        shape: Vec<Option<i64>>,
        dtype: &Bound<'_, Dtype>,
        members: Vec<Py<PyAny>>,
        combine_op: Option<&str>,
    ) -> PyResult<Self> {
        let rule = crate::layout::parse_composition_rule(composition_rule, combine_op)?;

        // Static only: a head is checked against what its members cover, and an unknown
        // extent cannot be covered by anything.
        let head_shape = crate::creation::parse_static_shape(shape, "hurray.Composite()")?;

        let members: Vec<Member> = members
            .iter()
            .enumerate()
            .map(|(index, obj)| extract_member(py, obj, index))
            .collect::<PyResult<_>>()?;

        // member_count is the one head field taken from the members rather than stated:
        // it counts what was passed, so it cannot disagree with it.
        let member_count = u32::try_from(members.len()).map_err(|_| {
            InvalidDescriptorError::new_err("a composite cannot have more than 2^32-1 members")
        })?;

        let layout = CompositeLayout::new(rule, member_count)
            .map_err(|e| InvalidDescriptorError::new_err(e.to_string()))?;

        let head = TensorDescriptor::new(
            hurray_core::DESCRIPTOR_VERSION_MAJOR,
            hurray_core::DESCRIPTOR_VERSION_MINOR,
            dtype.get().inner,
            head_shape,
            0, // a head owns no data, so it has no byte offset
            LayoutDescriptor::Composite(layout),
            vec![], // and no buffers
            None,
            None,
            None,
            None,
        )
        .map_err(|e| InvalidDescriptorError::new_err(format!("invalid composite head: {e}")))?;

        // Every rule core enforces — per-member boxes, partition coverage, overlay
        // ordering, member count — runs here. The binding keeps no second copy.
        let member_descriptors: Vec<TensorDescriptor> =
            members.iter().map(|m| m.descriptor(py)).collect();
        CompositeTensor::new(head.clone(), member_descriptors)
            .map_err(|e| InvalidDescriptorError::new_err(e.to_string()))?;

        Ok(Self {
            head,
            members,
            dtype_py: dtype.clone().unbind(),
        })
    }

    /// The members, in wire order.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert len(c.members) == 1
    /// ```
    #[getter]
    pub fn members(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let items: Vec<Py<PyAny>> = self.members.iter().map(|m| m.to_object(py)).collect();
        Ok(PyTuple::new(py, items)?.unbind())
    }

    /// How many members this head declares.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert c.member_count == 1
    /// ```
    #[getter]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// The head's layout, as a `hurray.CompositeLayout`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert c.layout.composition_rule == "group"
    /// assert c.layout.combine_op is None
    /// ```
    #[getter]
    pub fn layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::layout::layout_to_py(py, &self.head.layout)
    }

    /// The logical shape the head presents.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert c.shape == (4,)
    /// ```
    #[getter]
    pub fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let dims: Vec<u64> = self.head.shape.dims().to_vec();
        Ok(PyTuple::new(py, dims)?.unbind())
    }

    /// The element type the head presents.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert c.dtype == hurray.float32
    /// ```
    #[getter]
    pub fn dtype(&self, py: Python<'_>) -> Py<Dtype> {
        self.dtype_py.clone_ref(py)
    }

    /// Number of dimensions of the head's logical view.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert c.ndim == 1
    /// ```
    #[getter]
    pub fn ndim(&self) -> usize {
        self.head.shape.rank()
    }

    /// Two composites are equal when their heads and members are.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// def group():
    ///     t = hurray.Tensor(bytes(16), hurray.float32, [4])
    ///     return hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    ///
    /// assert group() == group()
    /// ```
    pub fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.extract::<PyRef<'_, Composite>>() else {
            return false;
        };
        if self.head != other.head || self.members.len() != other.members.len() {
            return false;
        }
        self.members
            .iter()
            .zip(other.members.iter())
            .all(|(a, b)| match (a, b) {
                (Member::Tensor(x), Member::Tensor(y)) => {
                    x.borrow(py).descriptor == y.borrow(py).descriptor
                }
                (Member::Composite(x), Member::Composite(y)) => {
                    x.borrow(py).__eq__(py, y.bind(py).as_any())
                }
                _ => false,
            })
    }

    /// A depth-aware representation, since composites nest.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// t = hurray.Tensor(bytes(16), hurray.float32, [4])
    /// c = hurray.Composite("group", shape=[4], dtype=hurray.float32, members=[t])
    /// assert repr(c).startswith("hurray.Composite(rule='group'")
    /// ```
    pub fn __repr__(&self, py: Python<'_>) -> String {
        self.repr_at(py, 0)
    }
}

impl Composite {
    /// `repr` with a depth budget, so a deep tree prints rather than runs away.
    fn repr_at(&self, py: Python<'_>, depth: usize) -> String {
        const MAX_REPR_DEPTH: usize = 3;
        let rule = crate::layout::composition_rule_name(&self.head.layout);
        let shape: Vec<String> = self
            .head
            .shape
            .dims()
            .iter()
            .map(|d| d.to_string())
            .collect();
        let members = if depth + 1 >= MAX_REPR_DEPTH {
            "…".to_string()
        } else {
            self.members
                .iter()
                .map(|m| match m {
                    Member::Tensor(_) => "Tensor".to_string(),
                    Member::Composite(c) => c.borrow(py).repr_at(py, depth + 1),
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        // Python spelling: single-quoted rule, and a trailing comma on a 1-tuple.
        let shape = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            format!("({})", shape.join(", "))
        };
        format!("hurray.Composite(rule='{rule}', shape={shape}, members=[{members}])")
    }
}

/// Extracts one member, rejecting anything that is neither a tensor nor a composite.
fn extract_member(py: Python<'_>, obj: &Py<PyAny>, index: usize) -> PyResult<Member> {
    let bound = obj.bind(py);
    if let Ok(t) = bound.extract::<Py<Tensor>>() {
        return Ok(Member::Tensor(t));
    }
    if let Ok(c) = bound.extract::<Py<Composite>>() {
        return Ok(Member::Composite(c));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "members[{index}] must be a hurray.Tensor or a hurray.Composite, got {}",
        bound.get_type().name()?
    )))
}

// ── Registration ──────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Composite>()?;
    Ok(())
}

// ── Writing a composite ───────────────────────────────────────────────────────

/// A buffer's raw parts, carried across a GIL release.
///
/// Same contract as `stream::SendSlice`: the bytes belong to a tensor the caller's
/// composite holds, so they outlive the call, and only the detached thread reads them.
pub(crate) struct SendSlice(pub(crate) *const u8, pub(crate) usize);

// SAFETY: see the type's documentation.
unsafe impl Send for SendSlice {}

/// A composite tree pulled out of Python: descriptors by value, buffers by raw parts.
pub(crate) enum OwnedNode {
    Tensor {
        descriptor: TensorDescriptor,
        buffers: Vec<SendSlice>,
    },
    Composite {
        head: TensorDescriptor,
        members: Vec<OwnedNode>,
    },
}

/// A composite's head plus its owned member tree.
pub(crate) struct OwnedTree {
    pub(crate) head: TensorDescriptor,
    pub(crate) members: Vec<OwnedNode>,
}

/// Pulls a composite out of Python while the GIL is held.
pub(crate) fn owned_tree(py: Python<'_>, composite: &Composite) -> OwnedTree {
    OwnedTree {
        head: composite.head.clone(),
        members: composite
            .members
            .iter()
            .map(|m| owned_node(py, m))
            .collect(),
    }
}

fn owned_node(py: Python<'_>, member: &Member) -> OwnedNode {
    match member {
        Member::Tensor(t) => {
            let tensor = t.borrow(py);
            OwnedNode::Tensor {
                descriptor: tensor.descriptor.clone(),
                buffers: tensor
                    .buffers()
                    .map(|store| {
                        // SAFETY: the GIL is held, so the store and its base are alive.
                        let slice = unsafe { store.as_slice() };
                        SendSlice(slice.as_ptr(), slice.len())
                    })
                    .collect(),
            }
        }
        Member::Composite(c) => {
            let nested = c.borrow(py);
            OwnedNode::Composite {
                head: nested.head.clone(),
                members: nested.members.iter().map(|m| owned_node(py, m)).collect(),
            }
        }
    }
}

/// Copies a node. The variants hold only shared references, so this is a shallow move.
///
/// Written by hand because `CompositeNode` derives neither `Clone` nor `Copy`.
fn dup<'a>(node: &StreamNode<'a>) -> StreamNode<'a> {
    match *node {
        StreamNode::Tensor {
            descriptor,
            buffers,
        } => StreamNode::Tensor {
            descriptor,
            buffers,
        },
        StreamNode::Composite { head, members } => StreamNode::Composite { head, members },
    }
}

/// Builds the borrowed node tree `write_composite` wants, and calls `f` with it.
///
/// Continuation-passing, because the borrowed tree cannot outlive the frames that own
/// its slice vectors: each node's `&[&[u8]]` lives in the stack frame that built it, so
/// every node has to be constructed *around* the call that uses it rather than returned
/// from one. `prefix` accumulates the siblings already built.
pub(crate) fn with_stream_nodes<R>(
    owned: &[OwnedNode],
    prefix: &[StreamNode<'_>],
    f: &mut dyn FnMut(&[StreamNode<'_>]) -> R,
) -> R {
    let Some((first, rest)) = owned.split_first() else {
        return f(prefix);
    };
    with_stream_node(first, &mut |node| {
        let mut extended: Vec<StreamNode<'_>> = prefix.iter().map(dup).collect();
        extended.push(dup(&node));
        with_stream_nodes(rest, &extended, f)
    })
}

fn with_stream_node<R>(owned: &OwnedNode, f: &mut dyn FnMut(StreamNode<'_>) -> R) -> R {
    match owned {
        OwnedNode::Tensor {
            descriptor,
            buffers,
        } => {
            // SAFETY: each SendSlice points into a live tensor buffer — see its docs.
            let slices: Vec<&[u8]> = buffers
                .iter()
                .map(|s| unsafe { std::slice::from_raw_parts(s.0, s.1) })
                .collect();
            f(StreamNode::Tensor {
                descriptor,
                buffers: &slices,
            })
        }
        OwnedNode::Composite { head, members } => {
            with_stream_nodes(members, &[], &mut |children| {
                f(StreamNode::Composite {
                    head,
                    members: children,
                })
            })
        }
    }
}

// ── Reading a composite ───────────────────────────────────────────────────────

/// Builds a `hurray.Composite` from a head and already-built member objects.
///
/// Used by the readers, whose trees came through core's own validator on decode —
/// revalidating here would re-run the identical checks on data the format already
/// accepted.
pub(crate) fn composite_from_parts(
    py: Python<'_>,
    head: TensorDescriptor,
    members: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let members: Vec<Member> = members
        .iter()
        .enumerate()
        .map(|(index, obj)| extract_member(py, obj, index))
        .collect::<PyResult<_>>()?;
    let dtype_py = Py::new(
        py,
        Dtype {
            inner: head.element_type,
        },
    )?;
    Ok(Py::new(
        py,
        Composite {
            head,
            members,
            dtype_py,
        },
    )?
    .into_any())
}

// ── Writing a composite to a file ─────────────────────────────────────────────

/// A composite tree with a name per node, which the file container requires: every
/// tensor, head and member alike, gets its own footer-index entry (ADR-027 § Binding).
///
/// Member names are derived as `"{parent}.{index}"`. They are an artifact of the file
/// container rather than of the composite — nothing on the wire or in the descriptor
/// carries them — so they are generated rather than asked for.
pub(crate) enum NamedNode {
    Tensor {
        name: String,
        descriptor: TensorDescriptor,
        buffers: Vec<Vec<u8>>,
    },
    Composite {
        name: String,
        head: TensorDescriptor,
        members: Vec<NamedNode>,
    },
}

/// Pulls a composite out of Python with names attached and bytes copied.
///
/// Copied rather than borrowed because `save` releases the GIL for the whole write and
/// the file writer holds its slices across every await; the existing `save` path copies
/// for the same reason.
pub(crate) fn named_tree(
    py: Python<'_>,
    composite: &Composite,
    head_name: &str,
) -> (TensorDescriptor, Vec<NamedNode>) {
    let members = composite
        .members
        .iter()
        .enumerate()
        .map(|(index, member)| named_node(py, member, &format!("{head_name}.{index}")))
        .collect();
    (composite.head.clone(), members)
}

fn named_node(py: Python<'_>, member: &Member, name: &str) -> NamedNode {
    match member {
        Member::Tensor(t) => {
            let tensor = t.borrow(py);
            NamedNode::Tensor {
                name: name.to_string(),
                descriptor: tensor.descriptor.clone(),
                // SAFETY: the GIL is held, so every store and its base are alive.
                buffers: tensor
                    .buffers()
                    .map(|store| unsafe { store.as_slice() }.to_vec())
                    .collect(),
            }
        }
        Member::Composite(c) => {
            let nested = c.borrow(py);
            NamedNode::Composite {
                name: name.to_string(),
                head: nested.head.clone(),
                members: nested
                    .members
                    .iter()
                    .enumerate()
                    .map(|(index, m)| named_node(py, m, &format!("{name}.{index}")))
                    .collect(),
            }
        }
    }
}

/// Copies a file node. Shallow, for the same reason as [`dup`].
fn dup_file<'a>(node: &FileNode<'a>) -> FileNode<'a> {
    match *node {
        FileNode::Tensor {
            name,
            descriptor,
            buffers,
        } => FileNode::Tensor {
            name,
            descriptor,
            buffers,
        },
        FileNode::Composite {
            name,
            head,
            members,
        } => FileNode::Composite {
            name,
            head,
            members,
        },
    }
}

/// Builds the borrowed node tree `FileWriter::write_composite` wants.
///
/// Continuation-passing for the same reason as [`with_stream_nodes`]: each node's
/// slice vector lives in the frame that built it.
pub(crate) fn with_file_nodes<R>(
    owned: &[NamedNode],
    prefix: &[FileNode<'_>],
    f: &mut dyn FnMut(&[FileNode<'_>]) -> R,
) -> R {
    let Some((first, rest)) = owned.split_first() else {
        return f(prefix);
    };
    with_file_node(first, &mut |node| {
        let mut extended: Vec<FileNode<'_>> = prefix.iter().map(dup_file).collect();
        extended.push(dup_file(&node));
        with_file_nodes(rest, &extended, f)
    })
}

fn with_file_node<R>(owned: &NamedNode, f: &mut dyn FnMut(FileNode<'_>) -> R) -> R {
    match owned {
        NamedNode::Tensor {
            name,
            descriptor,
            buffers,
        } => {
            let slices: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
            f(FileNode::Tensor {
                name,
                descriptor,
                buffers: &slices,
            })
        }
        NamedNode::Composite {
            name,
            head,
            members,
        } => with_file_nodes(members, &[], &mut |children| {
            f(FileNode::Composite {
                name,
                head,
                members: children,
            })
        }),
    }
}
