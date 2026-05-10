//! Buffer protocol types for the Hurray tensor format.
//!
//! A **buffer handle** is the unit by which a tensor descriptor references a
//! contiguous region of memory. This module defines the in-memory representation
//! of buffer handles, the device tag that identifies where a buffer resides, and
//! a colocation validator that enforces the spec's rule that all buffers within
//! a single tensor descriptor must reside on the same device.
//!
//! The binary encoding of a buffer handle in the descriptor wire format is
//! defined in `docs/spec/metadata.md § Buffer Table`. The normative rules
//! governing alignment, device tags, ownership, and zero-copy invariants are
//! in `docs/spec/buffer-protocol.md`.
//!
//! ## Quick reference
//!
//! | Item | Description |
//! |------|-------------|
//! | [`DeviceTag`] | Identifies the memory space a buffer resides in |
//! | [`BufferHandle`] | Declares a buffer's size, alignment, and device |
//! | [`validate_colocation`] | Checks that a set of handles all share the same device |
//! | [`MIN_BUFFER_ALIGNMENT`] | 64-byte SIMD minimum for non-empty buffers |
//! | [`PAGE_ALIGNMENT`] | 4096-byte recommendation for GPU / IPC buffers |

use std::fmt;

use crate::Error;

// ── PrivateTag ────────────────────────────────────────────────────────────────

/// An implementation-private device tag byte, guaranteed to be in `0xF0`–`0xFE`.
///
/// Values of this type are constructible only via [`DeviceTag::from_byte`];
/// direct construction is not possible from outside the crate, preventing
/// callers from forging an out-of-range private tag.
///
/// # Examples
///
/// ```
/// use hurray_core::{DeviceTag, PrivateTag};
///
/// let tag = DeviceTag::from_byte(0xF2).unwrap();
/// if let DeviceTag::Private(pt) = tag {
///     assert_eq!(pt.byte(), 0xF2);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrivateTag(u8);

impl PrivateTag {
    /// Returns the raw wire byte for this private device tag (`0xF0`–`0xFE`).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// let tag = DeviceTag::from_byte(0xF5).unwrap();
    /// if let DeviceTag::Private(pt) = tag {
    ///     assert_eq!(pt.byte(), 0xF5);
    /// }
    /// ```
    #[inline]
    pub fn byte(self) -> u8 {
        self.0
    }
}

// ── Alignment constants ───────────────────────────────────────────────────────

/// Minimum buffer alignment for SIMD compatibility (64 bytes).
///
/// The base address of every **non-empty** buffer MUST be aligned to at least
/// this many bytes. This ensures compatibility with all current SIMD instruction
/// sets (AVX-512, NEON, SVE) without per-operation alignment negotiation.
///
/// See `docs/spec/buffer-protocol.md § Minimum Alignment`.
///
/// # Examples
///
/// ```
/// use hurray_core::MIN_BUFFER_ALIGNMENT;
///
/// assert_eq!(MIN_BUFFER_ALIGNMENT, 64);
/// ```
pub const MIN_BUFFER_ALIGNMENT: u32 = 64;

/// Recommended alignment for GPU, IPC, and RDMA buffers (one host page = 4096 bytes).
///
/// Buffers shared across process boundaries (IPC) or placed in device memory
/// (GPU) SHOULD be aligned to at least this value. Writers targeting RDMA MUST
/// set `alignment` to at least `4096`.
///
/// See `docs/spec/buffer-protocol.md § Page Alignment for GPU and IPC`.
///
/// # Examples
///
/// ```
/// use hurray_core::PAGE_ALIGNMENT;
///
/// assert_eq!(PAGE_ALIGNMENT, 4096);
/// ```
pub const PAGE_ALIGNMENT: u32 = 4096;

// ── DeviceTag ─────────────────────────────────────────────────────────────────

/// Identifies the memory space in which a buffer resides.
///
/// The `device_tag` field in the binary buffer handle is a single `uint8`.
/// This enum is the typed representation of that byte; use [`DeviceTag::from_byte`]
/// to parse and [`DeviceTag::to_byte`] to serialize.
///
/// | Wire value | Variant |
/// |------------|---------|
/// | `0x00` | [`Cpu`][DeviceTag::Cpu] |
/// | `0x01` | [`Cuda`][DeviceTag::Cuda] |
/// | `0x02` | [`Rocm`][DeviceTag::Rocm] |
/// | `0x03` | [`Metal`][DeviceTag::Metal] |
/// | `0x04` | [`Vulkan`][DeviceTag::Vulkan] |
/// | `0x05` | [`WebGpu`][DeviceTag::WebGpu] |
/// | `0x06` | [`Hexagon`][DeviceTag::Hexagon] |
/// | `0x07` | [`LevelZero`][DeviceTag::LevelZero] |
/// | `0x08` | [`OpenCl`][DeviceTag::OpenCl] |
/// | `0x09`–`0xEF` | reserved — yields [`Error::ReservedDeviceTag`] |
/// | `0xF0`–`0xFE` | [`Private(b)`][DeviceTag::Private] |
/// | `0xFF` | permanently invalid — yields [`Error::InvalidDeviceTag`] |
///
/// See `docs/spec/buffer-protocol.md § Device Tags` for the normative table.
///
/// # Examples
///
/// ```
/// use hurray_core::DeviceTag;
///
/// let tag = DeviceTag::from_byte(0x00).unwrap();
/// assert_eq!(tag, DeviceTag::Cpu);
/// assert_eq!(tag.to_byte(), 0x00);
/// assert_eq!(tag.to_string(), "cpu");
///
/// let private = DeviceTag::from_byte(0xF2).unwrap();
/// assert!(private.is_private());
/// assert_eq!(private.to_byte(), 0xF2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceTag {
    /// CPU host memory. Wire byte `0x00`.
    Cpu,
    /// CUDA device memory. Wire byte `0x01`.
    Cuda,
    /// ROCm device memory. Wire byte `0x02`.
    Rocm,
    /// Metal device memory (Apple Silicon unified memory). Wire byte `0x03`.
    Metal,
    /// Vulkan device memory (cross-vendor GPU API). Wire byte `0x04`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// let tag = DeviceTag::from_byte(0x04).unwrap();
    /// assert_eq!(tag, DeviceTag::Vulkan);
    /// assert_eq!(tag.to_byte(), 0x04);
    /// assert_eq!(tag.to_string(), "vulkan");
    /// ```
    Vulkan,
    /// WebGPU device memory (browser and native WebGPU API). Wire byte `0x05`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// let tag = DeviceTag::from_byte(0x05).unwrap();
    /// assert_eq!(tag, DeviceTag::WebGpu);
    /// assert_eq!(tag.to_byte(), 0x05);
    /// assert_eq!(tag.to_string(), "webgpu");
    /// ```
    WebGpu,
    /// Qualcomm Hexagon DSP memory. Wire byte `0x06`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// let tag = DeviceTag::from_byte(0x06).unwrap();
    /// assert_eq!(tag, DeviceTag::Hexagon);
    /// assert_eq!(tag.to_byte(), 0x06);
    /// assert_eq!(tag.to_string(), "hexagon");
    /// ```
    Hexagon,
    /// Intel oneAPI Level Zero device memory. Wire byte `0x07`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// let tag = DeviceTag::from_byte(0x07).unwrap();
    /// assert_eq!(tag, DeviceTag::LevelZero);
    /// assert_eq!(tag.to_byte(), 0x07);
    /// assert_eq!(tag.to_string(), "level_zero");
    /// ```
    LevelZero,
    /// OpenCL device memory (cross-vendor compute API). Wire byte `0x08`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// let tag = DeviceTag::from_byte(0x08).unwrap();
    /// assert_eq!(tag, DeviceTag::OpenCl);
    /// assert_eq!(tag.to_byte(), 0x08);
    /// assert_eq!(tag.to_string(), "opencl");
    /// ```
    OpenCl,
    /// Implementation-private device type. Wire byte in `0xF0`–`0xFE`.
    ///
    /// Descriptors carrying private device tags MUST NOT be exchanged between
    /// independent implementations unless both parties have agreed on the
    /// semantics out of band.
    ///
    /// Use [`DeviceTag::from_byte`] to construct; direct construction of the
    /// inner [`PrivateTag`] from outside the crate is not possible.
    Private(PrivateTag),
}

impl DeviceTag {
    /// Parses a [`DeviceTag`] from its one-byte wire representation.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidDeviceTag`] — byte is `0xFF` (permanently reserved).
    /// - [`Error::ReservedDeviceTag`] — byte is in `0x09`–`0xEF` (reserved for
    ///   future specification versions).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{DeviceTag, Error};
    ///
    /// assert_eq!(DeviceTag::from_byte(0x00).unwrap(), DeviceTag::Cpu);
    /// assert_eq!(DeviceTag::from_byte(0x01).unwrap(), DeviceTag::Cuda);
    /// assert_eq!(DeviceTag::from_byte(0x02).unwrap(), DeviceTag::Rocm);
    /// assert_eq!(DeviceTag::from_byte(0x03).unwrap(), DeviceTag::Metal);
    /// assert_eq!(DeviceTag::from_byte(0x04).unwrap(), DeviceTag::Vulkan);
    /// assert_eq!(DeviceTag::from_byte(0x05).unwrap(), DeviceTag::WebGpu);
    /// assert_eq!(DeviceTag::from_byte(0x06).unwrap(), DeviceTag::Hexagon);
    /// assert_eq!(DeviceTag::from_byte(0x07).unwrap(), DeviceTag::LevelZero);
    /// assert_eq!(DeviceTag::from_byte(0x08).unwrap(), DeviceTag::OpenCl);
    /// assert!(DeviceTag::from_byte(0xF0).unwrap().is_private());
    /// assert_eq!(DeviceTag::from_byte(0xF0).unwrap().to_byte(), 0xF0);
    /// assert_eq!(DeviceTag::from_byte(0xFE).unwrap().to_byte(), 0xFE);
    /// assert!(matches!(DeviceTag::from_byte(0x09), Err(Error::ReservedDeviceTag(0x09))));
    /// assert!(matches!(DeviceTag::from_byte(0xEF), Err(Error::ReservedDeviceTag(0xEF))));
    /// assert!(matches!(DeviceTag::from_byte(0xFF), Err(Error::InvalidDeviceTag(0xFF))));
    /// ```
    pub fn from_byte(b: u8) -> crate::Result<Self> {
        match b {
            0x00 => Ok(Self::Cpu),
            0x01 => Ok(Self::Cuda),
            0x02 => Ok(Self::Rocm),
            0x03 => Ok(Self::Metal),
            0x04 => Ok(Self::Vulkan),
            0x05 => Ok(Self::WebGpu),
            0x06 => Ok(Self::Hexagon),
            0x07 => Ok(Self::LevelZero),
            0x08 => Ok(Self::OpenCl),
            0x09..=0xEF => Err(Error::ReservedDeviceTag(b)),
            0xF0..=0xFE => Ok(Self::Private(PrivateTag(b))),
            0xFF => Err(Error::InvalidDeviceTag(b)),
        }
    }

    /// Returns the one-byte wire representation of this device tag.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// assert_eq!(DeviceTag::Cpu.to_byte(), 0x00);
    /// assert_eq!(DeviceTag::Cuda.to_byte(), 0x01);
    /// assert_eq!(DeviceTag::Rocm.to_byte(), 0x02);
    /// assert_eq!(DeviceTag::Metal.to_byte(), 0x03);
    /// assert_eq!(DeviceTag::Vulkan.to_byte(), 0x04);
    /// assert_eq!(DeviceTag::WebGpu.to_byte(), 0x05);
    /// assert_eq!(DeviceTag::Hexagon.to_byte(), 0x06);
    /// assert_eq!(DeviceTag::LevelZero.to_byte(), 0x07);
    /// assert_eq!(DeviceTag::OpenCl.to_byte(), 0x08);
    /// assert_eq!(DeviceTag::from_byte(0xF5).unwrap().to_byte(), 0xF5);
    /// ```
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Cpu => 0x00,
            Self::Cuda => 0x01,
            Self::Rocm => 0x02,
            Self::Metal => 0x03,
            Self::Vulkan => 0x04,
            Self::WebGpu => 0x05,
            Self::Hexagon => 0x06,
            Self::LevelZero => 0x07,
            Self::OpenCl => 0x08,
            Self::Private(t) => t.0,
        }
    }

    /// Returns `true` if this tag is a private/experimental device type
    /// (`0xF0`–`0xFE`).
    ///
    /// Private tags MAY be used by implementations for experimental device
    /// types but MUST NOT be exchanged between independent implementations
    /// without an out-of-band agreement.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// assert!(DeviceTag::from_byte(0xF0).unwrap().is_private());
    /// assert!(!DeviceTag::Cpu.is_private());
    /// assert!(!DeviceTag::Cuda.is_private());
    /// ```
    pub fn is_private(self) -> bool {
        matches!(self, Self::Private(_))
    }
}

impl fmt::Display for DeviceTag {
    /// Formats the device tag as a human-readable lowercase string.
    ///
    /// Private tags are formatted as `private(0xNN)` where `NN` is the
    /// hex wire byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::DeviceTag;
    ///
    /// assert_eq!(DeviceTag::Cpu.to_string(), "cpu");
    /// assert_eq!(DeviceTag::Cuda.to_string(), "cuda");
    /// assert_eq!(DeviceTag::Rocm.to_string(), "rocm");
    /// assert_eq!(DeviceTag::Metal.to_string(), "metal");
    /// assert_eq!(DeviceTag::Vulkan.to_string(), "vulkan");
    /// assert_eq!(DeviceTag::WebGpu.to_string(), "webgpu");
    /// assert_eq!(DeviceTag::Hexagon.to_string(), "hexagon");
    /// assert_eq!(DeviceTag::LevelZero.to_string(), "level_zero");
    /// assert_eq!(DeviceTag::OpenCl.to_string(), "opencl");
    /// assert_eq!(DeviceTag::from_byte(0xF3).unwrap().to_string(), "private(0xF3)");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda => f.write_str("cuda"),
            Self::Rocm => f.write_str("rocm"),
            Self::Metal => f.write_str("metal"),
            Self::Vulkan => f.write_str("vulkan"),
            Self::WebGpu => f.write_str("webgpu"),
            Self::Hexagon => f.write_str("hexagon"),
            Self::LevelZero => f.write_str("level_zero"),
            Self::OpenCl => f.write_str("opencl"),
            Self::Private(t) => write!(f, "private(0x{:02X})", t.0),
        }
    }
}

// ── BufferHandle ──────────────────────────────────────────────────────────────

/// A declaration of a buffer's size, alignment, and device location.
///
/// A `BufferHandle` is the in-memory representation of the 16-byte buffer
/// handle entry in the tensor descriptor's buffer table (see
/// `docs/spec/metadata.md § Buffer Table`). It carries the metadata needed to
/// locate and access a buffer but does not itself hold a pointer — the actual
/// memory address is communicated out-of-band via the interchange protocol or
/// the C ABI (see `docs/impl/c-ffi.md`).
///
/// # Alignment rules
///
/// - `alignment` MUST be a power of two.
/// - For **non-empty** buffers (`byte_size > 0`): `alignment` MUST be at least
///   [`MIN_BUFFER_ALIGNMENT`] (64 bytes).
/// - For **empty** buffers (`byte_size == 0`): any power-of-two alignment is
///   valid, including `1`. A reader MUST NOT dereference the pointer of an empty
///   buffer.
///
/// See `docs/spec/buffer-protocol.md § Alignment` for the normative rules.
///
/// # Examples
///
/// ```
/// use hurray_core::{BufferHandle, DeviceTag, MIN_BUFFER_ALIGNMENT};
///
/// // Non-empty CPU buffer, minimum SIMD alignment.
/// let handle = BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu).unwrap();
/// assert_eq!(handle.byte_size(), 1024);
/// assert_eq!(handle.alignment(), 64);
/// assert_eq!(handle.device_tag(), DeviceTag::Cpu);
/// assert!(!handle.is_empty());
///
/// // Empty buffer — alignment 1 is valid.
/// let empty = BufferHandle::empty(DeviceTag::Cuda);
/// assert!(empty.is_empty());
/// assert_eq!(empty.alignment(), 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferHandle {
    byte_size: u64,
    alignment: u32,
    device_tag: DeviceTag,
}

impl BufferHandle {
    /// Creates a new [`BufferHandle`] with the given size, alignment, and device.
    ///
    /// # Errors
    ///
    /// - [`Error::AlignmentNotPowerOfTwo`] — `alignment` is not a power of two.
    /// - [`Error::AlignmentBelowMinimum`] — `byte_size > 0` and `alignment` is
    ///   less than [`MIN_BUFFER_ALIGNMENT`] (64).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, Error, MIN_BUFFER_ALIGNMENT};
    ///
    /// // Valid: non-empty buffer with minimum SIMD alignment.
    /// assert!(BufferHandle::new(512, 64, DeviceTag::Cpu).is_ok());
    ///
    /// // Valid: empty buffer with alignment 1.
    /// assert!(BufferHandle::new(0, 1, DeviceTag::Cpu).is_ok());
    ///
    /// // Error: alignment is not a power of two.
    /// assert!(matches!(
    ///     BufferHandle::new(512, 3, DeviceTag::Cpu),
    ///     Err(Error::AlignmentNotPowerOfTwo { alignment: 3 })
    /// ));
    ///
    /// // Error: non-empty buffer with alignment below the 64-byte minimum.
    /// assert!(matches!(
    ///     BufferHandle::new(512, 32, DeviceTag::Cpu),
    ///     Err(Error::AlignmentBelowMinimum { alignment: 32, minimum: 64 })
    /// ));
    /// ```
    pub fn new(byte_size: u64, alignment: u32, device_tag: DeviceTag) -> crate::Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(Error::AlignmentNotPowerOfTwo { alignment });
        }
        if byte_size > 0 && alignment < MIN_BUFFER_ALIGNMENT {
            return Err(Error::AlignmentBelowMinimum {
                alignment,
                minimum: MIN_BUFFER_ALIGNMENT,
            });
        }
        Ok(Self {
            byte_size,
            alignment,
            device_tag,
        })
    }

    /// Creates an **empty** [`BufferHandle`] (zero bytes) on the given device.
    ///
    /// The alignment is set to `1` — the minimum valid power-of-two for an
    /// empty buffer. This constructor is infallible.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag};
    ///
    /// let handle = BufferHandle::empty(DeviceTag::Cpu);
    /// assert!(handle.is_empty());
    /// assert_eq!(handle.byte_size(), 0);
    /// assert_eq!(handle.alignment(), 1);
    /// assert_eq!(handle.device_tag(), DeviceTag::Cpu);
    /// ```
    pub fn empty(device_tag: DeviceTag) -> Self {
        Self {
            byte_size: 0,
            alignment: 1,
            device_tag,
        }
    }

    /// Returns the size of the buffer in bytes.
    ///
    /// A value of `0` denotes an empty buffer whose backing pointer MUST NOT
    /// be dereferenced.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag};
    ///
    /// let handle = BufferHandle::new(4096, 4096, DeviceTag::Cuda).unwrap();
    /// assert_eq!(handle.byte_size(), 4096);
    /// ```
    pub fn byte_size(self) -> u64 {
        self.byte_size
    }

    /// Returns the minimum alignment of the buffer's base address in bytes.
    ///
    /// Always a power of two. For non-empty buffers, always at least
    /// [`MIN_BUFFER_ALIGNMENT`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, PAGE_ALIGNMENT};
    ///
    /// let handle = BufferHandle::new(8192, PAGE_ALIGNMENT, DeviceTag::Cuda).unwrap();
    /// assert_eq!(handle.alignment(), 4096);
    /// ```
    pub fn alignment(self) -> u32 {
        self.alignment
    }

    /// Returns the [`DeviceTag`] identifying the memory space this buffer resides in.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag};
    ///
    /// let handle = BufferHandle::new(256, 64, DeviceTag::Metal).unwrap();
    /// assert_eq!(handle.device_tag(), DeviceTag::Metal);
    /// ```
    pub fn device_tag(self) -> DeviceTag {
        self.device_tag
    }

    /// Returns `true` if this buffer has zero bytes (`byte_size == 0`).
    ///
    /// Readers MUST NOT dereference the pointer of an empty buffer. In C ABI
    /// contexts an empty buffer MAY be represented by a null pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag};
    ///
    /// assert!(BufferHandle::empty(DeviceTag::Cpu).is_empty());
    /// assert!(!BufferHandle::new(1, 64, DeviceTag::Cpu).unwrap().is_empty());
    /// ```
    pub fn is_empty(self) -> bool {
        self.byte_size == 0
    }
}

// ── validate_colocation ───────────────────────────────────────────────────────

/// Checks that all buffer handles in `handles` share the same [`DeviceTag`].
///
/// All buffers referenced by a single tensor descriptor — the data buffer plus
/// all quantization-parameter buffers — MUST share the same device tag. A
/// reader MUST reject a descriptor whose buffers carry different `device_tag`
/// values (see `docs/spec/buffer-protocol.md § Device Colocation`).
///
/// Returns the common [`DeviceTag`] on success.
///
/// # Errors
///
/// - [`Error::EmptyBufferList`] — `handles` is empty (at least one handle is
///   required to determine a common device).
/// - [`Error::DeviceTagMismatch`] — two or more handles carry different device
///   tags; the error captures the first tag seen (`expected`) and the first
///   mismatching tag (`found`), both as raw wire bytes.
///
/// # Examples
///
/// ```
/// use hurray_core::{BufferHandle, DeviceTag, Error, validate_colocation};
///
/// // All handles on CPU — succeeds.
/// let handles = [
///     BufferHandle::new(1024, 64, DeviceTag::Cpu).unwrap(),
///     BufferHandle::new(256, 64, DeviceTag::Cpu).unwrap(),
/// ];
/// assert_eq!(validate_colocation(&handles).unwrap(), DeviceTag::Cpu);
///
/// // Empty slice — error.
/// assert!(matches!(validate_colocation(&[]), Err(Error::EmptyBufferList)));
///
/// // Mixed devices — error.
/// let mixed = [
///     BufferHandle::new(1024, 64, DeviceTag::Cpu).unwrap(),
///     BufferHandle::new(256, 64, DeviceTag::Cuda).unwrap(),
/// ];
/// assert!(matches!(
///     validate_colocation(&mixed),
///     Err(Error::DeviceTagMismatch { expected: 0x00, found: 0x01 })
/// ));
/// ```
pub fn validate_colocation(handles: &[BufferHandle]) -> crate::Result<DeviceTag> {
    let first = handles.first().ok_or(Error::EmptyBufferList)?;
    let expected_tag = first.device_tag;
    let expected_byte = expected_tag.to_byte();

    for handle in handles.iter().skip(1) {
        if handle.device_tag != expected_tag {
            return Err(Error::DeviceTagMismatch {
                expected: expected_byte,
                found: handle.device_tag.to_byte(),
            });
        }
    }

    Ok(expected_tag)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeviceTag::from_byte — named variants ────────────────────────────────

    /// Spec § buffer-protocol.md Device Tags: 0x00 decodes to Cpu.
    #[test]
    fn from_byte_cpu() {
        assert_eq!(DeviceTag::from_byte(0x00).unwrap(), DeviceTag::Cpu);
    }

    /// Spec § buffer-protocol.md Device Tags: 0x01 decodes to Cuda.
    #[test]
    fn from_byte_cuda() {
        assert_eq!(DeviceTag::from_byte(0x01).unwrap(), DeviceTag::Cuda);
    }

    /// Spec § buffer-protocol.md Device Tags: 0x02 decodes to Rocm.
    #[test]
    fn from_byte_rocm() {
        assert_eq!(DeviceTag::from_byte(0x02).unwrap(), DeviceTag::Rocm);
    }

    /// Spec § buffer-protocol.md Device Tags: 0x03 decodes to Metal.
    #[test]
    fn from_byte_metal() {
        assert_eq!(DeviceTag::from_byte(0x03).unwrap(), DeviceTag::Metal);
    }

    // ── DeviceTag::from_byte — reserved range 0x09–0xEF ─────────────────────

    /// Spec § buffer-protocol.md Device Tags: 0x09 (lower bound of reserved range
    /// after ADR-016 assigned 0x04–0x08) must return ReservedDeviceTag.
    #[test]
    fn from_byte_reserved_lower_bound() {
        assert!(matches!(
            DeviceTag::from_byte(0x09),
            Err(Error::ReservedDeviceTag(0x09))
        ));
    }

    /// Spec § buffer-protocol.md Device Tags: 0xEF (upper bound of reserved range)
    /// must return ReservedDeviceTag.
    #[test]
    fn from_byte_reserved_upper_bound() {
        assert!(matches!(
            DeviceTag::from_byte(0xEF),
            Err(Error::ReservedDeviceTag(0xEF))
        ));
    }

    /// Spot-check mid-reserved byte 0x80 returns ReservedDeviceTag.
    #[test]
    fn from_byte_reserved_mid_range() {
        assert!(matches!(
            DeviceTag::from_byte(0x80),
            Err(Error::ReservedDeviceTag(0x80))
        ));
    }

    // ── DeviceTag::from_byte — private range 0xF0–0xFE ──────────────────────

    /// Spec § buffer-protocol.md Device Tags: 0xF0 (lower bound of private range)
    /// must decode to Private(0xF0).
    #[test]
    fn from_byte_private_lower_bound() {
        let tag = DeviceTag::from_byte(0xF0).unwrap();
        assert!(tag.is_private());
        assert_eq!(tag.to_byte(), 0xF0);
    }

    /// Spec § buffer-protocol.md Device Tags: 0xFE (upper bound of private range)
    /// must decode to Private(0xFE).
    #[test]
    fn from_byte_private_upper_bound() {
        let tag = DeviceTag::from_byte(0xFE).unwrap();
        assert!(tag.is_private());
        assert_eq!(tag.to_byte(), 0xFE);
    }

    // ── DeviceTag::from_byte — permanently invalid sentinel 0xFF ────────────

    /// Spec § buffer-protocol.md Device Tags: 0xFF is permanently reserved and
    /// MUST be rejected with InvalidDeviceTag.
    #[test]
    fn from_byte_invalid_sentinel() {
        assert!(matches!(
            DeviceTag::from_byte(0xFF),
            Err(Error::InvalidDeviceTag(0xFF))
        ));
    }

    // ── DeviceTag::to_byte — round-trip for named variants ──────────────────

    /// Each named variant must serialize back to its documented wire byte.
    #[test]
    fn to_byte_cpu() {
        assert_eq!(DeviceTag::Cpu.to_byte(), 0x00);
    }

    #[test]
    fn to_byte_cuda() {
        assert_eq!(DeviceTag::Cuda.to_byte(), 0x01);
    }

    #[test]
    fn to_byte_rocm() {
        assert_eq!(DeviceTag::Rocm.to_byte(), 0x02);
    }

    #[test]
    fn to_byte_metal() {
        assert_eq!(DeviceTag::Metal.to_byte(), 0x03);
    }

    /// Private(b) must serialize to exactly b.
    #[test]
    fn to_byte_private() {
        assert_eq!(DeviceTag::from_byte(0xF2).unwrap().to_byte(), 0xF2);
    }

    // ── DeviceTag round-trip: from_byte → to_byte ────────────────────────────

    /// For every valid byte (named variants + private range), from_byte then
    /// to_byte must be the identity.
    #[test]
    fn round_trip_named_variants() {
        for b in [0x00u8, 0x01, 0x02, 0x03] {
            let tag = DeviceTag::from_byte(b).unwrap();
            assert_eq!(tag.to_byte(), b, "round-trip failed for byte 0x{b:02X}");
        }
    }

    #[test]
    fn round_trip_private_range() {
        for b in 0xF0u8..=0xFE {
            let tag = DeviceTag::from_byte(b).unwrap();
            assert_eq!(tag.to_byte(), b, "round-trip failed for byte 0x{b:02X}");
        }
    }

    // ── DeviceTag::is_private ────────────────────────────────────────────────

    #[test]
    fn is_private_true_for_private_variant() {
        assert!(DeviceTag::from_byte(0xF0).unwrap().is_private());
    }

    #[test]
    fn is_private_false_for_cpu() {
        assert!(!DeviceTag::Cpu.is_private());
    }

    #[test]
    fn is_private_false_for_cuda() {
        assert!(!DeviceTag::Cuda.is_private());
    }

    #[test]
    fn is_private_false_for_rocm() {
        assert!(!DeviceTag::Rocm.is_private());
    }

    #[test]
    fn is_private_false_for_metal() {
        assert!(!DeviceTag::Metal.is_private());
    }

    // ── DeviceTag Display ────────────────────────────────────────────────────

    /// Spec § buffer-protocol.md Display: named variants use lowercase ASCII.
    #[test]
    fn display_cpu() {
        assert_eq!(DeviceTag::Cpu.to_string(), "cpu");
    }

    #[test]
    fn display_cuda() {
        assert_eq!(DeviceTag::Cuda.to_string(), "cuda");
    }

    #[test]
    fn display_rocm() {
        assert_eq!(DeviceTag::Rocm.to_string(), "rocm");
    }

    #[test]
    fn display_metal() {
        assert_eq!(DeviceTag::Metal.to_string(), "metal");
    }

    /// Private tags display as "private(0xNN)" with uppercase hex digits.
    #[test]
    fn display_private() {
        assert_eq!(
            DeviceTag::from_byte(0xF1).unwrap().to_string(),
            "private(0xF1)"
        );
    }

    #[test]
    fn display_private_lower_bound() {
        assert_eq!(
            DeviceTag::from_byte(0xF0).unwrap().to_string(),
            "private(0xF0)"
        );
    }

    #[test]
    fn display_private_upper_bound() {
        assert_eq!(
            DeviceTag::from_byte(0xFE).unwrap().to_string(),
            "private(0xFE)"
        );
    }

    // ── MIN_BUFFER_ALIGNMENT and PAGE_ALIGNMENT constants ───────────────────

    /// Spec § buffer-protocol.md Minimum Alignment: SIMD minimum is 64 bytes.
    #[test]
    fn min_buffer_alignment_is_64() {
        assert_eq!(MIN_BUFFER_ALIGNMENT, 64);
    }

    /// Spec § buffer-protocol.md Page Alignment: page-aligned value is 4096 bytes.
    #[test]
    fn page_alignment_is_4096() {
        assert_eq!(PAGE_ALIGNMENT, 4096);
    }

    /// Both constants must be powers of two (required by the alignment contract).
    #[test]
    fn min_buffer_alignment_is_power_of_two() {
        assert!(MIN_BUFFER_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn page_alignment_is_power_of_two() {
        assert!(PAGE_ALIGNMENT.is_power_of_two());
    }

    // ── BufferHandle::new — success cases ────────────────────────────────────

    /// Spec § buffer-protocol.md Alignment: non-empty buffer with minimum SIMD
    /// alignment (64) on CPU is valid.
    #[test]
    fn new_nonempty_cpu_min_alignment() {
        let result = BufferHandle::new(1024, 64, DeviceTag::Cpu);
        assert!(result.is_ok());
    }

    /// A non-empty CUDA buffer at page alignment is valid.
    #[test]
    fn new_nonempty_cuda_page_alignment() {
        let result = BufferHandle::new(4096, 4096, DeviceTag::Cuda);
        assert!(result.is_ok());
    }

    /// Spec § buffer-protocol.md Alignment: an empty buffer (byte_size == 0)
    /// may have any power-of-two alignment, including 1.
    #[test]
    fn new_empty_alignment_1_is_valid() {
        let result = BufferHandle::new(0, 1, DeviceTag::Cpu);
        assert!(result.is_ok());
    }

    /// An empty buffer with the SIMD minimum alignment is also valid.
    #[test]
    fn new_empty_alignment_64_is_valid() {
        let result = BufferHandle::new(0, 64, DeviceTag::Cpu);
        assert!(result.is_ok());
    }

    // ── BufferHandle::new — failure cases ────────────────────────────────────

    /// alignment=0 is not a power of two; must return AlignmentNotPowerOfTwo.
    #[test]
    fn new_alignment_zero_not_power_of_two() {
        assert!(matches!(
            BufferHandle::new(1024, 0, DeviceTag::Cpu),
            Err(Error::AlignmentNotPowerOfTwo { alignment: 0 })
        ));
    }

    /// alignment=63 is not a power of two; must return AlignmentNotPowerOfTwo.
    #[test]
    fn new_alignment_63_not_power_of_two() {
        assert!(matches!(
            BufferHandle::new(1024, 63, DeviceTag::Cpu),
            Err(Error::AlignmentNotPowerOfTwo { alignment: 63 })
        ));
    }

    /// alignment=100 is not a power of two; must return AlignmentNotPowerOfTwo.
    #[test]
    fn new_alignment_100_not_power_of_two() {
        assert!(matches!(
            BufferHandle::new(1024, 100, DeviceTag::Cpu),
            Err(Error::AlignmentNotPowerOfTwo { alignment: 100 })
        ));
    }

    /// Non-empty buffer with alignment=32 (power of two but below MIN_BUFFER_ALIGNMENT)
    /// must return AlignmentBelowMinimum.
    #[test]
    fn new_nonempty_alignment_below_minimum_32() {
        assert!(matches!(
            BufferHandle::new(1, 32, DeviceTag::Cpu),
            Err(Error::AlignmentBelowMinimum {
                alignment: 32,
                minimum: 64
            })
        ));
    }

    /// Non-empty buffer with alignment=1 (power of two but well below MIN_BUFFER_ALIGNMENT)
    /// must return AlignmentBelowMinimum.
    #[test]
    fn new_nonempty_alignment_below_minimum_1() {
        assert!(matches!(
            BufferHandle::new(1, 1, DeviceTag::Cpu),
            Err(Error::AlignmentBelowMinimum {
                alignment: 1,
                minimum: 64
            })
        ));
    }

    // ── BufferHandle::empty ──────────────────────────────────────────────────

    /// BufferHandle::empty must produce a zero-sized handle with alignment 1.
    #[test]
    fn empty_byte_size_is_zero() {
        assert_eq!(BufferHandle::empty(DeviceTag::Cpu).byte_size(), 0);
    }

    #[test]
    fn empty_alignment_is_one() {
        assert_eq!(BufferHandle::empty(DeviceTag::Cpu).alignment(), 1);
    }

    #[test]
    fn empty_is_empty_returns_true() {
        assert!(BufferHandle::empty(DeviceTag::Cpu).is_empty());
    }

    /// The device tag passed to empty() must be preserved.
    #[test]
    fn empty_preserves_device_tag_cpu() {
        assert_eq!(
            BufferHandle::empty(DeviceTag::Cpu).device_tag(),
            DeviceTag::Cpu
        );
    }

    #[test]
    fn empty_preserves_device_tag_cuda() {
        assert_eq!(
            BufferHandle::empty(DeviceTag::Cuda).device_tag(),
            DeviceTag::Cuda
        );
    }

    #[test]
    fn empty_preserves_device_tag_private() {
        let private = DeviceTag::from_byte(0xF5).unwrap();
        assert_eq!(BufferHandle::empty(private).device_tag(), private);
    }

    // ── BufferHandle accessors ───────────────────────────────────────────────

    /// byte_size(), alignment(), device_tag(), and is_empty() must return the
    /// values that were passed to new().
    #[test]
    fn accessors_byte_size() {
        let handle = BufferHandle::new(8192, 4096, DeviceTag::Cuda).unwrap();
        assert_eq!(handle.byte_size(), 8192);
    }

    #[test]
    fn accessors_alignment() {
        let handle = BufferHandle::new(256, 256, DeviceTag::Metal).unwrap();
        assert_eq!(handle.alignment(), 256);
    }

    #[test]
    fn accessors_device_tag() {
        let handle = BufferHandle::new(512, 64, DeviceTag::Rocm).unwrap();
        assert_eq!(handle.device_tag(), DeviceTag::Rocm);
    }

    #[test]
    fn accessors_is_empty_false_for_nonempty() {
        let handle = BufferHandle::new(1, 64, DeviceTag::Cpu).unwrap();
        assert!(!handle.is_empty());
    }

    #[test]
    fn accessors_is_empty_true_for_zero_size() {
        let handle = BufferHandle::new(0, 1, DeviceTag::Cpu).unwrap();
        assert!(handle.is_empty());
    }

    // ── validate_colocation ──────────────────────────────────────────────────

    /// Empty slice must return EmptyBufferList.
    #[test]
    fn colocation_empty_slice_returns_error() {
        assert!(matches!(
            validate_colocation(&[]),
            Err(Error::EmptyBufferList)
        ));
    }

    /// Single-element slice must succeed and return the handle's device tag.
    #[test]
    fn colocation_single_handle_returns_its_tag() {
        let handle = BufferHandle::new(1024, 64, DeviceTag::Cpu).unwrap();
        let result = validate_colocation(&[handle]);
        assert_eq!(result.unwrap(), DeviceTag::Cpu);
    }

    /// All-CPU slice of three handles must succeed and return Cpu.
    #[test]
    fn colocation_all_cpu_three_handles() {
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cpu).unwrap(),
            BufferHandle::new(512, 64, DeviceTag::Cpu).unwrap(),
            BufferHandle::new(256, 64, DeviceTag::Cpu).unwrap(),
        ];
        assert_eq!(validate_colocation(&handles).unwrap(), DeviceTag::Cpu);
    }

    /// Mixed Cpu + Cuda must return DeviceTagMismatch with correct wire bytes.
    #[test]
    fn colocation_mixed_cpu_cuda_returns_mismatch() {
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cpu).unwrap(),
            BufferHandle::new(256, 64, DeviceTag::Cuda).unwrap(),
        ];
        assert!(matches!(
            validate_colocation(&handles),
            Err(Error::DeviceTagMismatch {
                expected: 0x00,
                found: 0x01
            })
        ));
    }

    /// All-Private(0xF0) slice must succeed and return Private(0xF0).
    #[test]
    fn colocation_all_private_same_tag() {
        let private = DeviceTag::from_byte(0xF0).unwrap();
        let handles = [
            BufferHandle::new(1024, 64, private).unwrap(),
            BufferHandle::new(512, 64, private).unwrap(),
        ];
        assert_eq!(validate_colocation(&handles).unwrap(), private);
    }

    /// Mixed named + private must return DeviceTagMismatch.
    #[test]
    fn colocation_named_and_private_returns_mismatch() {
        let private = DeviceTag::from_byte(0xF0).unwrap();
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cpu).unwrap(),
            BufferHandle::new(512, 64, private).unwrap(),
        ];
        assert!(matches!(
            validate_colocation(&handles),
            Err(Error::DeviceTagMismatch {
                expected: 0x00,
                found: 0xF0
            })
        ));
    }

    /// colocation validates the first mismatch position: 2nd handle agrees,
    /// 3rd handle disagrees. Error captures the first vs. third wire bytes.
    #[test]
    fn colocation_mismatch_at_third_element() {
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cuda).unwrap(),
            BufferHandle::new(512, 64, DeviceTag::Cuda).unwrap(),
            BufferHandle::new(256, 64, DeviceTag::Metal).unwrap(),
        ];
        assert!(matches!(
            validate_colocation(&handles),
            Err(Error::DeviceTagMismatch {
                expected: 0x01,
                found: 0x03
            })
        ));
    }
}
