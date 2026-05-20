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
//! | [`SyncMode`] | Describes how the producer–consumer memory ordering is established |
//! | [`BufferHandle`] | Declares a buffer's size, alignment, device, and sync mode |
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

// ── SyncMode ──────────────────────────────────────────────────────────────────

/// Describes how the producer–consumer memory ordering guarantee is established
/// for a buffer.
///
/// The `sync_mode` field in the binary buffer handle is a single `uint8` at
/// wire offset 13. This enum is the typed representation of that byte; use
/// [`SyncMode::from_byte`] to parse and [`SyncMode::to_byte`] to serialize.
///
/// CPU buffers (`device_tag == 0x00`) MUST use [`SyncMode::ProducerSynced`];
/// [`BufferHandle::new`] enforces this and returns [`Error::InvalidSyncMode`]
/// if any other mode is combined with [`DeviceTag::Cpu`].
///
/// | Wire value | Variant |
/// |------------|---------|
/// | `0x00` | [`ProducerSynced`][SyncMode::ProducerSynced] |
/// | `0x01` | [`Event`][SyncMode::Event] |
/// | `0x02` | [`ConsumerStream`][SyncMode::ConsumerStream] |
/// | `0x03`–`0xFF` | reserved / permanently invalid → [`Error::InvalidSyncMode`] |
///
/// See `docs/spec/buffer-protocol.md § Synchronization Mode` and ADR-018.
///
/// # Examples
///
/// ```
/// use hurray_core::{SyncMode, Error};
///
/// let mode = SyncMode::from_byte(0x00).unwrap();
/// assert_eq!(mode, SyncMode::ProducerSynced);
/// assert_eq!(mode.to_byte(), 0x00);
/// assert_eq!(mode.to_string(), "producer_synced");
///
/// let event = SyncMode::from_byte(0x01).unwrap();
/// assert_eq!(event, SyncMode::Event);
///
/// assert!(matches!(SyncMode::from_byte(0x03), Err(Error::InvalidSyncMode(0x03))));
/// assert!(matches!(SyncMode::from_byte(0xFF), Err(Error::InvalidSyncMode(0xFF))));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SyncMode {
    /// Producer has issued a host-side wait; consumer may access on any stream.
    ///
    /// This is the only valid mode for CPU buffers (`device_tag == 0x00`).
    /// Wire byte `0x00`.
    ProducerSynced,
    /// Producer recorded a device event; consumer must wait on it via the C ABI.
    ///
    /// Wire byte `0x01`.
    Event,
    /// Consumer declared a target stream; producer ordered it device-side.
    ///
    /// Wire byte `0x02`.
    ConsumerStream,
}

impl SyncMode {
    /// Parses a [`SyncMode`] from its one-byte wire representation.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSyncMode`] — byte is `0x03`–`0xFF` (reserved or
    ///   permanently invalid).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{SyncMode, Error};
    ///
    /// assert_eq!(SyncMode::from_byte(0x00).unwrap(), SyncMode::ProducerSynced);
    /// assert_eq!(SyncMode::from_byte(0x01).unwrap(), SyncMode::Event);
    /// assert_eq!(SyncMode::from_byte(0x02).unwrap(), SyncMode::ConsumerStream);
    /// assert!(matches!(SyncMode::from_byte(0x03), Err(Error::InvalidSyncMode(0x03))));
    /// assert!(matches!(SyncMode::from_byte(0xFE), Err(Error::InvalidSyncMode(0xFE))));
    /// assert!(matches!(SyncMode::from_byte(0xFF), Err(Error::InvalidSyncMode(0xFF))));
    /// ```
    pub fn from_byte(b: u8) -> crate::Result<Self> {
        match b {
            0x00 => Ok(Self::ProducerSynced),
            0x01 => Ok(Self::Event),
            0x02 => Ok(Self::ConsumerStream),
            // 0x03–0xFF: all reserved or permanently invalid; reject unconditionally.
            _ => Err(Error::InvalidSyncMode(b)),
        }
    }

    /// Returns the one-byte wire representation of this sync mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::SyncMode;
    ///
    /// assert_eq!(SyncMode::ProducerSynced.to_byte(), 0x00);
    /// assert_eq!(SyncMode::Event.to_byte(), 0x01);
    /// assert_eq!(SyncMode::ConsumerStream.to_byte(), 0x02);
    /// ```
    pub fn to_byte(self) -> u8 {
        match self {
            Self::ProducerSynced => 0x00,
            Self::Event => 0x01,
            Self::ConsumerStream => 0x02,
        }
    }
}

impl fmt::Display for SyncMode {
    /// Formats the sync mode as a human-readable lowercase string.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::SyncMode;
    ///
    /// assert_eq!(SyncMode::ProducerSynced.to_string(), "producer_synced");
    /// assert_eq!(SyncMode::Event.to_string(), "event");
    /// assert_eq!(SyncMode::ConsumerStream.to_string(), "consumer_stream");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProducerSynced => f.write_str("producer_synced"),
            Self::Event => f.write_str("event"),
            Self::ConsumerStream => f.write_str("consumer_stream"),
        }
    }
}

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

// ── MemoryClass ───────────────────────────────────────────────────────────────

/// An implementation-private memory class byte, guaranteed to be in `0xF0`–`0xFE`.
///
/// Constructible only via [`MemoryClass::from_byte`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrivateMemoryClass(u8);

impl PrivateMemoryClass {
    /// Returns the raw wire byte for this private memory class.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::MemoryClass;
    ///
    /// let cls = MemoryClass::from_byte(0xF1).unwrap();
    /// if let MemoryClass::Private(p) = cls {
    ///     assert_eq!(p.to_byte(), 0xF1);
    /// }
    /// ```
    pub fn to_byte(self) -> u8 {
        self.0
    }
}

/// The memory access class of a buffer handle.
///
/// Identifies *how* a buffer is accessible — whether it is device-exclusive or
/// can be accessed by the CPU and/or peer accelerators without copying.
///
/// The `memory_class` field in the binary buffer handle is a single `uint8` at
/// offset 14. Use [`MemoryClass::from_byte`] to parse and [`MemoryClass::to_byte`]
/// to serialize. See `docs/spec/buffer-protocol.md § Memory Class` for the
/// normative definition.
///
/// | Value | Variant |
/// |-------|---------|
/// | `0x00` | [`Standard`][MemoryClass::Standard] |
/// | `0x01` | [`HostPinned`][MemoryClass::HostPinned] |
/// | `0x02` | [`Unified`][MemoryClass::Unified] |
/// | `0x03` | [`Peer`][MemoryClass::Peer] |
/// | `0x04`–`0xEF` | reserved → [`Error::ReservedMemoryClass`] |
/// | `0xF0`–`0xFE` | [`Private(b)`][MemoryClass::Private] |
/// | `0xFF` | permanently invalid → [`Error::InvalidMemoryClass`] |
///
/// # Examples
///
/// ```
/// use hurray_core::{MemoryClass, Error};
///
/// assert_eq!(MemoryClass::from_byte(0x00).unwrap(), MemoryClass::Standard);
/// assert_eq!(MemoryClass::from_byte(0x02).unwrap(), MemoryClass::Unified);
/// assert!(matches!(MemoryClass::from_byte(0x04), Err(Error::ReservedMemoryClass(0x04))));
/// assert!(matches!(MemoryClass::from_byte(0xFF), Err(Error::InvalidMemoryClass(0xFF))));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryClass {
    /// Device-exclusive memory. Only the primary compute unit of the tagged device
    /// can access this buffer without a copy. Default for all device types; backward-
    /// compatible with pre-ADR-020 descriptors whose `_reserved[0]` byte was `0x00`.
    Standard,
    /// CPU-accessible, device-mapped. No hardware-managed coherency. Examples:
    /// `cudaMallocHost`, `hipHostMalloc`, `CL_MEM_ALLOC_HOST_PTR`.
    HostPinned,
    /// Hardware-managed unified or coherent memory. CPU and device can access this
    /// buffer simultaneously; the hardware ensures coherency. Examples:
    /// `cudaMallocManaged`, ROCm HMM, Metal `MTLStorageModeShared`.
    Unified,
    /// Peer-to-peer device memory. Accessible by a set of peer accelerators agreed
    /// out of band (NVLink, xGMI, PCIe BAR). Not CPU-accessible without a copy.
    Peer,
    /// Implementation-private memory class (`0xF0`–`0xFE`). Semantics agreed out of band.
    Private(PrivateMemoryClass),
}

impl MemoryClass {
    /// Parses a [`MemoryClass`] from its one-byte wire representation.
    ///
    /// # Errors
    ///
    /// - [`Error::ReservedMemoryClass`] — byte is `0x04`–`0xEF`.
    /// - [`Error::InvalidMemoryClass`] — byte is `0xFF`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{MemoryClass, Error};
    ///
    /// assert_eq!(MemoryClass::from_byte(0x00).unwrap(), MemoryClass::Standard);
    /// assert_eq!(MemoryClass::from_byte(0x01).unwrap(), MemoryClass::HostPinned);
    /// assert_eq!(MemoryClass::from_byte(0x02).unwrap(), MemoryClass::Unified);
    /// assert_eq!(MemoryClass::from_byte(0x03).unwrap(), MemoryClass::Peer);
    /// assert!(matches!(MemoryClass::from_byte(0x04), Err(Error::ReservedMemoryClass(0x04))));
    /// assert!(matches!(MemoryClass::from_byte(0xEF), Err(Error::ReservedMemoryClass(0xEF))));
    /// assert!(matches!(MemoryClass::from_byte(0xFF), Err(Error::InvalidMemoryClass(0xFF))));
    /// ```
    pub fn from_byte(b: u8) -> crate::Result<Self> {
        match b {
            0x00 => Ok(Self::Standard),
            0x01 => Ok(Self::HostPinned),
            0x02 => Ok(Self::Unified),
            0x03 => Ok(Self::Peer),
            0x04..=0xEF => Err(Error::ReservedMemoryClass(b)),
            0xF0..=0xFE => Ok(Self::Private(PrivateMemoryClass(b))),
            0xFF => Err(Error::InvalidMemoryClass(b)),
        }
    }

    /// Returns the one-byte wire representation of this memory class.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::MemoryClass;
    ///
    /// assert_eq!(MemoryClass::Standard.to_byte(), 0x00);
    /// assert_eq!(MemoryClass::HostPinned.to_byte(), 0x01);
    /// assert_eq!(MemoryClass::Unified.to_byte(), 0x02);
    /// assert_eq!(MemoryClass::Peer.to_byte(), 0x03);
    /// ```
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Standard => 0x00,
            Self::HostPinned => 0x01,
            Self::Unified => 0x02,
            Self::Peer => 0x03,
            Self::Private(p) => p.0,
        }
    }

    /// Returns `true` if this is an implementation-private memory class (`0xF0`–`0xFE`).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::MemoryClass;
    ///
    /// assert!(!MemoryClass::Standard.is_private());
    /// assert!(MemoryClass::from_byte(0xF0).unwrap().is_private());
    /// ```
    pub fn is_private(self) -> bool {
        matches!(self, Self::Private(_))
    }
}

impl fmt::Display for MemoryClass {
    /// # Examples
    ///
    /// ```
    /// use hurray_core::MemoryClass;
    ///
    /// assert_eq!(MemoryClass::Standard.to_string(), "standard");
    /// assert_eq!(MemoryClass::Unified.to_string(), "unified");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => f.write_str("standard"),
            Self::HostPinned => f.write_str("host_pinned"),
            Self::Unified => f.write_str("unified"),
            Self::Peer => f.write_str("peer"),
            Self::Private(p) => write!(f, "private(0x{:02X})", p.0),
        }
    }
}

// ── BufferHandle ──────────────────────────────────────────────────────────────

/// A declaration of a buffer's size, alignment, device location, and sync mode.
///
/// A `BufferHandle` is the in-memory representation of the 16-byte buffer
/// handle entry in the tensor descriptor's buffer table (see
/// `docs/spec/metadata.md § Buffer Table`). It carries the metadata needed to
/// locate and access a buffer but does not itself hold a pointer — the actual
/// memory address is communicated out-of-band via the interchange protocol or
/// the C ABI (see `docs/impl/c-ffi.md`).
///
/// ## Wire layout (ADR-018 § 3, ADR-020)
///
/// | Offset | Field | Type | Size |
/// |--------|-------|------|------|
/// | 0 | `byte_size` | uint64 LE | 8 |
/// | 8 | `alignment` | uint32 LE | 4 |
/// | 12 | `device_tag` | uint8 | 1 |
/// | 13 | `sync_mode` | uint8 | 1 |
/// | 14 | `memory_class` | uint8 | 1 |
/// | 15 | `_reserved` | uint8 | 1 |
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
/// use hurray_core::{BufferHandle, DeviceTag, MemoryClass, SyncMode, MIN_BUFFER_ALIGNMENT};
///
/// // Non-empty CPU buffer, minimum SIMD alignment, default Standard class.
/// let handle = BufferHandle::new(1024, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
/// assert_eq!(handle.byte_size(), 1024);
/// assert_eq!(handle.alignment(), 64);
/// assert_eq!(handle.device_tag(), DeviceTag::Cpu);
/// assert_eq!(handle.sync_mode(), SyncMode::ProducerSynced);
/// assert_eq!(handle.memory_class(), MemoryClass::Standard);
/// assert!(!handle.is_empty());
///
/// // CUDA buffer with Unified memory class.
/// let unified = BufferHandle::with_memory_class(
///     4096, 4096, DeviceTag::Cuda, SyncMode::ProducerSynced, MemoryClass::Unified,
/// ).unwrap();
/// assert_eq!(unified.memory_class(), MemoryClass::Unified);
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
    sync_mode: SyncMode,
    memory_class: MemoryClass,
}

impl BufferHandle {
    /// Creates a new [`BufferHandle`] with the given size, alignment, device, and sync mode.
    ///
    /// # Errors
    ///
    /// - [`Error::AlignmentNotPowerOfTwo`] — `alignment` is not a power of two.
    /// - [`Error::AlignmentBelowMinimum`] — `byte_size > 0` and `alignment` is
    ///   less than [`MIN_BUFFER_ALIGNMENT`] (64).
    /// - [`Error::InvalidSyncMode`] — `device_tag` is [`DeviceTag::Cpu`] and
    ///   `sync_mode` is not [`SyncMode::ProducerSynced`] (CPU buffers MUST use
    ///   `SYNC_PRODUCER_SYNCED` per the spec).
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, Error, SyncMode, MIN_BUFFER_ALIGNMENT};
    ///
    /// // Valid: non-empty CPU buffer with minimum SIMD alignment.
    /// assert!(BufferHandle::new(512, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).is_ok());
    ///
    /// // Valid: CUDA buffer with Event sync.
    /// assert!(BufferHandle::new(512, 64, DeviceTag::Cuda, SyncMode::Event).is_ok());
    ///
    /// // Valid: empty buffer with alignment 1.
    /// assert!(BufferHandle::new(0, 1, DeviceTag::Cpu, SyncMode::ProducerSynced).is_ok());
    ///
    /// // Error: alignment is not a power of two.
    /// assert!(matches!(
    ///     BufferHandle::new(512, 3, DeviceTag::Cpu, SyncMode::ProducerSynced),
    ///     Err(Error::AlignmentNotPowerOfTwo { alignment: 3 })
    /// ));
    ///
    /// // Error: non-empty buffer with alignment below the 64-byte minimum.
    /// assert!(matches!(
    ///     BufferHandle::new(512, 32, DeviceTag::Cpu, SyncMode::ProducerSynced),
    ///     Err(Error::AlignmentBelowMinimum { alignment: 32, minimum: 64 })
    /// ));
    ///
    /// // Error: CPU buffer with non-ProducerSynced mode.
    /// assert!(matches!(
    ///     BufferHandle::new(512, 64, DeviceTag::Cpu, SyncMode::Event),
    ///     Err(Error::InvalidSyncMode(0x01))
    /// ));
    /// ```
    pub fn new(
        byte_size: u64,
        alignment: u32,
        device_tag: DeviceTag,
        sync_mode: SyncMode,
    ) -> crate::Result<Self> {
        Self::with_memory_class(
            byte_size,
            alignment,
            device_tag,
            sync_mode,
            MemoryClass::Standard,
        )
    }

    /// Creates a new [`BufferHandle`] with an explicit memory class.
    ///
    /// Identical to [`BufferHandle::new`] but accepts a [`MemoryClass`] value.
    /// Use this constructor when the buffer's memory class is not [`MemoryClass::Standard`]
    /// (e.g., CUDA managed memory, Metal shared storage, or peer-to-peer memory).
    ///
    /// # Errors
    ///
    /// - [`Error::AlignmentNotPowerOfTwo`] — `alignment` is not a power of two.
    /// - [`Error::AlignmentBelowMinimum`] — `byte_size > 0` and `alignment < 64`.
    /// - [`Error::InvalidSyncMode`] — `device_tag` is [`DeviceTag::Cpu`] and
    ///   `sync_mode` is not [`SyncMode::ProducerSynced`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, MemoryClass, SyncMode};
    ///
    /// // CUDA unified (managed) memory buffer.
    /// let handle = BufferHandle::with_memory_class(
    ///     4096, 4096, DeviceTag::Cuda, SyncMode::ProducerSynced, MemoryClass::Unified,
    /// ).unwrap();
    /// assert_eq!(handle.memory_class(), MemoryClass::Unified);
    ///
    /// // CPU host-pinned buffer.
    /// let pinned = BufferHandle::with_memory_class(
    ///     512, 64, DeviceTag::Cpu, SyncMode::ProducerSynced, MemoryClass::HostPinned,
    /// ).unwrap();
    /// assert_eq!(pinned.memory_class(), MemoryClass::HostPinned);
    /// ```
    pub fn with_memory_class(
        byte_size: u64,
        alignment: u32,
        device_tag: DeviceTag,
        sync_mode: SyncMode,
        memory_class: MemoryClass,
    ) -> crate::Result<Self> {
        if !alignment.is_power_of_two() {
            return Err(Error::AlignmentNotPowerOfTwo { alignment });
        }
        if byte_size > 0 && alignment < MIN_BUFFER_ALIGNMENT {
            return Err(Error::AlignmentBelowMinimum {
                alignment,
                minimum: MIN_BUFFER_ALIGNMENT,
            });
        }
        // CPU buffers must use ProducerSynced: no device-side sync primitives exist.
        if device_tag == DeviceTag::Cpu && sync_mode != SyncMode::ProducerSynced {
            return Err(Error::InvalidSyncMode(sync_mode.to_byte()));
        }
        Ok(Self {
            byte_size,
            alignment,
            device_tag,
            sync_mode,
            memory_class,
        })
    }

    /// Creates an **empty** [`BufferHandle`] (zero bytes) on the given device.
    ///
    /// The alignment is set to `1` — the minimum valid power-of-two for an
    /// empty buffer — and `sync_mode` is always [`SyncMode::ProducerSynced`].
    /// This constructor is infallible.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, SyncMode};
    ///
    /// let handle = BufferHandle::empty(DeviceTag::Cpu);
    /// assert!(handle.is_empty());
    /// assert_eq!(handle.byte_size(), 0);
    /// assert_eq!(handle.alignment(), 1);
    /// assert_eq!(handle.device_tag(), DeviceTag::Cpu);
    /// assert_eq!(handle.sync_mode(), SyncMode::ProducerSynced);
    /// ```
    pub fn empty(device_tag: DeviceTag) -> Self {
        // ProducerSynced is the safest universal default: empty buffers carry no
        // data and no device-side synchronisation is required.
        Self {
            byte_size: 0,
            alignment: 1,
            device_tag,
            sync_mode: SyncMode::ProducerSynced,
            memory_class: MemoryClass::Standard,
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
    /// use hurray_core::{BufferHandle, DeviceTag, SyncMode};
    ///
    /// let handle = BufferHandle::new(4096, 4096, DeviceTag::Cuda, SyncMode::Event).unwrap();
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
    /// use hurray_core::{BufferHandle, DeviceTag, SyncMode, PAGE_ALIGNMENT};
    ///
    /// let handle = BufferHandle::new(8192, PAGE_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event).unwrap();
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
    /// use hurray_core::{BufferHandle, DeviceTag, SyncMode};
    ///
    /// let handle = BufferHandle::new(256, 64, DeviceTag::Metal, SyncMode::Event).unwrap();
    /// assert_eq!(handle.device_tag(), DeviceTag::Metal);
    /// ```
    pub fn device_tag(self) -> DeviceTag {
        self.device_tag
    }

    /// Returns the [`SyncMode`] describing the producer–consumer ordering guarantee
    /// for this buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, SyncMode};
    ///
    /// let handle = BufferHandle::new(1024, 64, DeviceTag::Cuda, SyncMode::ConsumerStream).unwrap();
    /// assert_eq!(handle.sync_mode(), SyncMode::ConsumerStream);
    ///
    /// // CPU buffers are always ProducerSynced.
    /// let cpu = BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
    /// assert_eq!(cpu.sync_mode(), SyncMode::ProducerSynced);
    /// ```
    pub fn sync_mode(self) -> SyncMode {
        self.sync_mode
    }

    /// Returns the [`MemoryClass`] describing how this buffer is accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, MemoryClass, SyncMode};
    ///
    /// // new() defaults to Standard.
    /// let handle = BufferHandle::new(1024, 64, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap();
    /// assert_eq!(handle.memory_class(), MemoryClass::Standard);
    ///
    /// // with_memory_class() sets an explicit class.
    /// let unified = BufferHandle::with_memory_class(
    ///     1024, 64, DeviceTag::Cuda, SyncMode::ProducerSynced, MemoryClass::Unified,
    /// ).unwrap();
    /// assert_eq!(unified.memory_class(), MemoryClass::Unified);
    /// ```
    pub fn memory_class(self) -> MemoryClass {
        self.memory_class
    }

    /// Returns `true` if this buffer has zero bytes (`byte_size == 0`).
    ///
    /// Readers MUST NOT dereference the pointer of an empty buffer. In C ABI
    /// contexts an empty buffer MAY be represented by a null pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::{BufferHandle, DeviceTag, SyncMode};
    ///
    /// assert!(BufferHandle::empty(DeviceTag::Cpu).is_empty());
    /// assert!(!BufferHandle::new(1, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap().is_empty());
    /// ```
    pub fn is_empty(self) -> bool {
        self.byte_size == 0
    }
}

// ── validate_colocation ───────────────────────────────────────────────────────

/// Checks that all buffer handles in `handles` share the same [`DeviceTag`] and [`MemoryClass`].
///
/// All buffers referenced by a single tensor descriptor — the data buffer plus
/// all quantization-parameter buffers — MUST share the same `device_tag` AND the
/// same `memory_class` (see `docs/spec/buffer-protocol.md § Device Colocation`).
///
/// Returns the common [`DeviceTag`] on success.
///
/// # Errors
///
/// - [`Error::EmptyBufferList`] — `handles` is empty.
/// - [`Error::DeviceTagMismatch`] — two or more handles carry different device tags.
/// - [`Error::MemoryClassMismatch`] — two or more handles carry different memory classes.
///
/// # Examples
///
/// ```
/// use hurray_core::{BufferHandle, DeviceTag, Error, MemoryClass, SyncMode, validate_colocation};
///
/// // All handles on CPU, all Standard — succeeds.
/// let handles = [
///     BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
///     BufferHandle::new(256, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
/// ];
/// assert_eq!(validate_colocation(&handles).unwrap(), DeviceTag::Cpu);
///
/// // Empty slice — error.
/// assert!(matches!(validate_colocation(&[]), Err(Error::EmptyBufferList)));
///
/// // Mixed devices — error.
/// let mixed_device = [
///     BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
///     BufferHandle::new(256, 64, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap(),
/// ];
/// assert!(matches!(
///     validate_colocation(&mixed_device),
///     Err(Error::DeviceTagMismatch { expected: 0x00, found: 0x01 })
/// ));
///
/// // Mixed memory classes — error.
/// let mixed_class = [
///     BufferHandle::new(1024, 64, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap(),
///     BufferHandle::with_memory_class(256, 64, DeviceTag::Cuda, SyncMode::ProducerSynced, MemoryClass::Unified).unwrap(),
/// ];
/// assert!(matches!(
///     validate_colocation(&mixed_class),
///     Err(Error::MemoryClassMismatch { expected: 0x00, found: 0x02 })
/// ));
/// ```
pub fn validate_colocation(handles: &[BufferHandle]) -> crate::Result<DeviceTag> {
    let first = handles.first().ok_or(Error::EmptyBufferList)?;
    let expected_tag = first.device_tag;
    let expected_tag_byte = expected_tag.to_byte();
    let expected_class = first.memory_class;
    let expected_class_byte = expected_class.to_byte();

    for handle in handles.iter().skip(1) {
        if handle.device_tag != expected_tag {
            return Err(Error::DeviceTagMismatch {
                expected: expected_tag_byte,
                found: handle.device_tag.to_byte(),
            });
        }
        if handle.memory_class != expected_class {
            return Err(Error::MemoryClassMismatch {
                expected: expected_class_byte,
                found: handle.memory_class.to_byte(),
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
        let result = BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced);
        assert!(result.is_ok());
    }

    /// A non-empty CUDA buffer at page alignment is valid.
    #[test]
    fn new_nonempty_cuda_page_alignment() {
        let result = BufferHandle::new(4096, 4096, DeviceTag::Cuda, SyncMode::Event);
        assert!(result.is_ok());
    }

    /// Spec § buffer-protocol.md Alignment: an empty buffer (byte_size == 0)
    /// may have any power-of-two alignment, including 1.
    #[test]
    fn new_empty_alignment_1_is_valid() {
        let result = BufferHandle::new(0, 1, DeviceTag::Cpu, SyncMode::ProducerSynced);
        assert!(result.is_ok());
    }

    /// An empty buffer with the SIMD minimum alignment is also valid.
    #[test]
    fn new_empty_alignment_64_is_valid() {
        let result = BufferHandle::new(0, 64, DeviceTag::Cpu, SyncMode::ProducerSynced);
        assert!(result.is_ok());
    }

    // ── BufferHandle::new — failure cases ────────────────────────────────────

    /// alignment=0 is not a power of two; must return AlignmentNotPowerOfTwo.
    #[test]
    fn new_alignment_zero_not_power_of_two() {
        assert!(matches!(
            BufferHandle::new(1024, 0, DeviceTag::Cpu, SyncMode::ProducerSynced),
            Err(Error::AlignmentNotPowerOfTwo { alignment: 0 })
        ));
    }

    /// alignment=63 is not a power of two; must return AlignmentNotPowerOfTwo.
    #[test]
    fn new_alignment_63_not_power_of_two() {
        assert!(matches!(
            BufferHandle::new(1024, 63, DeviceTag::Cpu, SyncMode::ProducerSynced),
            Err(Error::AlignmentNotPowerOfTwo { alignment: 63 })
        ));
    }

    /// alignment=100 is not a power of two; must return AlignmentNotPowerOfTwo.
    #[test]
    fn new_alignment_100_not_power_of_two() {
        assert!(matches!(
            BufferHandle::new(1024, 100, DeviceTag::Cpu, SyncMode::ProducerSynced),
            Err(Error::AlignmentNotPowerOfTwo { alignment: 100 })
        ));
    }

    /// Non-empty buffer with alignment=32 (power of two but below MIN_BUFFER_ALIGNMENT)
    /// must return AlignmentBelowMinimum.
    #[test]
    fn new_nonempty_alignment_below_minimum_32() {
        assert!(matches!(
            BufferHandle::new(1, 32, DeviceTag::Cpu, SyncMode::ProducerSynced),
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
            BufferHandle::new(1, 1, DeviceTag::Cpu, SyncMode::ProducerSynced),
            Err(Error::AlignmentBelowMinimum {
                alignment: 1,
                minimum: 64
            })
        ));
    }

    /// ADR-018: CPU buffer with non-ProducerSynced mode must return InvalidSyncMode.
    #[test]
    fn new_cpu_with_event_sync_rejected() {
        assert!(matches!(
            BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::Event),
            Err(Error::InvalidSyncMode(0x01))
        ));
    }

    /// ADR-018: CPU buffer with ConsumerStream must return InvalidSyncMode.
    #[test]
    fn new_cpu_with_consumer_stream_rejected() {
        assert!(matches!(
            BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ConsumerStream),
            Err(Error::InvalidSyncMode(0x02))
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

    /// byte_size(), alignment(), device_tag(), sync_mode(), and is_empty() must
    /// return the values that were passed to new().
    #[test]
    fn accessors_byte_size() {
        let handle = BufferHandle::new(8192, 4096, DeviceTag::Cuda, SyncMode::Event).unwrap();
        assert_eq!(handle.byte_size(), 8192);
    }

    #[test]
    fn accessors_alignment() {
        let handle = BufferHandle::new(256, 256, DeviceTag::Metal, SyncMode::Event).unwrap();
        assert_eq!(handle.alignment(), 256);
    }

    #[test]
    fn accessors_device_tag() {
        let handle = BufferHandle::new(512, 64, DeviceTag::Rocm, SyncMode::ConsumerStream).unwrap();
        assert_eq!(handle.device_tag(), DeviceTag::Rocm);
    }

    #[test]
    fn accessors_sync_mode_producer_synced() {
        let handle = BufferHandle::new(512, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
        assert_eq!(handle.sync_mode(), SyncMode::ProducerSynced);
    }

    #[test]
    fn accessors_sync_mode_event() {
        let handle = BufferHandle::new(512, 64, DeviceTag::Cuda, SyncMode::Event).unwrap();
        assert_eq!(handle.sync_mode(), SyncMode::Event);
    }

    #[test]
    fn accessors_sync_mode_consumer_stream() {
        let handle = BufferHandle::new(512, 64, DeviceTag::Cuda, SyncMode::ConsumerStream).unwrap();
        assert_eq!(handle.sync_mode(), SyncMode::ConsumerStream);
    }

    #[test]
    fn accessors_is_empty_false_for_nonempty() {
        let handle = BufferHandle::new(1, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
        assert!(!handle.is_empty());
    }

    #[test]
    fn accessors_is_empty_true_for_zero_size() {
        let handle = BufferHandle::new(0, 1, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
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
        let handle = BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap();
        let result = validate_colocation(&[handle]);
        assert_eq!(result.unwrap(), DeviceTag::Cpu);
    }

    /// All-CPU slice of three handles must succeed and return Cpu.
    #[test]
    fn colocation_all_cpu_three_handles() {
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
            BufferHandle::new(512, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
            BufferHandle::new(256, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
        ];
        assert_eq!(validate_colocation(&handles).unwrap(), DeviceTag::Cpu);
    }

    /// Mixed Cpu + Cuda must return DeviceTagMismatch with correct wire bytes.
    #[test]
    fn colocation_mixed_cpu_cuda_returns_mismatch() {
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
            BufferHandle::new(256, 64, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap(),
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
            BufferHandle::new(1024, 64, private, SyncMode::ProducerSynced).unwrap(),
            BufferHandle::new(512, 64, private, SyncMode::ProducerSynced).unwrap(),
        ];
        assert_eq!(validate_colocation(&handles).unwrap(), private);
    }

    /// Mixed named + private must return DeviceTagMismatch.
    #[test]
    fn colocation_named_and_private_returns_mismatch() {
        let private = DeviceTag::from_byte(0xF0).unwrap();
        let handles = [
            BufferHandle::new(1024, 64, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap(),
            BufferHandle::new(512, 64, private, SyncMode::ProducerSynced).unwrap(),
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
            BufferHandle::new(1024, 64, DeviceTag::Cuda, SyncMode::Event).unwrap(),
            BufferHandle::new(512, 64, DeviceTag::Cuda, SyncMode::Event).unwrap(),
            BufferHandle::new(256, 64, DeviceTag::Metal, SyncMode::Event).unwrap(),
        ];
        assert!(matches!(
            validate_colocation(&handles),
            Err(Error::DeviceTagMismatch {
                expected: 0x01,
                found: 0x03
            })
        ));
    }

    // ── SyncMode ──────────────────────────────────────────────────────────────

    mod sync_mode {
        use super::*;
        use std::collections::HashSet;

        // ── from_byte — named values ─────────────────────────────────────────

        /// Spec § buffer-protocol.md Sync Mode: 0x00 decodes to ProducerSynced.
        #[test]
        fn from_byte_producer_synced() {
            assert_eq!(SyncMode::from_byte(0x00).unwrap(), SyncMode::ProducerSynced);
        }

        /// Spec § buffer-protocol.md Sync Mode: 0x01 decodes to Event.
        #[test]
        fn from_byte_event() {
            assert_eq!(SyncMode::from_byte(0x01).unwrap(), SyncMode::Event);
        }

        /// Spec § buffer-protocol.md Sync Mode: 0x02 decodes to ConsumerStream.
        #[test]
        fn from_byte_consumer_stream() {
            assert_eq!(SyncMode::from_byte(0x02).unwrap(), SyncMode::ConsumerStream);
        }

        // ── from_byte — reserved / invalid bytes ─────────────────────────────

        /// Spec § buffer-protocol.md Sync Mode: 0x03 (first reserved) must
        /// return InvalidSyncMode(0x03).
        #[test]
        fn from_byte_0x03_is_invalid() {
            assert!(matches!(
                SyncMode::from_byte(0x03),
                Err(Error::InvalidSyncMode(0x03))
            ));
        }

        /// Spec § buffer-protocol.md Sync Mode: 0xFE (reserved) must return
        /// InvalidSyncMode(0xFE).
        #[test]
        fn from_byte_0xfe_is_invalid() {
            assert!(matches!(
                SyncMode::from_byte(0xFE),
                Err(Error::InvalidSyncMode(0xFE))
            ));
        }

        /// Spec § buffer-protocol.md Sync Mode: 0xFF (reserved) must return
        /// InvalidSyncMode(0xFF).
        #[test]
        fn from_byte_0xff_is_invalid() {
            assert!(matches!(
                SyncMode::from_byte(0xFF),
                Err(Error::InvalidSyncMode(0xFF))
            ));
        }

        // ── to_byte — wire byte values ────────────────────────────────────────

        /// ProducerSynced serializes to wire byte 0x00.
        #[test]
        fn to_byte_producer_synced() {
            assert_eq!(SyncMode::ProducerSynced.to_byte(), 0x00);
        }

        /// Event serializes to wire byte 0x01.
        #[test]
        fn to_byte_event() {
            assert_eq!(SyncMode::Event.to_byte(), 0x01);
        }

        /// ConsumerStream serializes to wire byte 0x02.
        #[test]
        fn to_byte_consumer_stream() {
            assert_eq!(SyncMode::ConsumerStream.to_byte(), 0x02);
        }

        // ── from_byte → to_byte identity ─────────────────────────────────────

        /// For each named value, from_byte(m.to_byte()) == m.
        #[test]
        fn from_byte_to_byte_identity_producer_synced() {
            let m = SyncMode::ProducerSynced;
            assert_eq!(SyncMode::from_byte(m.to_byte()).unwrap(), m);
        }

        #[test]
        fn from_byte_to_byte_identity_event() {
            let m = SyncMode::Event;
            assert_eq!(SyncMode::from_byte(m.to_byte()).unwrap(), m);
        }

        #[test]
        fn from_byte_to_byte_identity_consumer_stream() {
            let m = SyncMode::ConsumerStream;
            assert_eq!(SyncMode::from_byte(m.to_byte()).unwrap(), m);
        }

        // ── Display ───────────────────────────────────────────────────────────

        /// Spec § buffer-protocol.md Display: ProducerSynced displays as
        /// "producer_synced".
        #[test]
        fn display_producer_synced() {
            assert_eq!(SyncMode::ProducerSynced.to_string(), "producer_synced");
        }

        /// Event displays as "event".
        #[test]
        fn display_event() {
            assert_eq!(SyncMode::Event.to_string(), "event");
        }

        /// ConsumerStream displays as "consumer_stream".
        #[test]
        fn display_consumer_stream() {
            assert_eq!(SyncMode::ConsumerStream.to_string(), "consumer_stream");
        }

        // ── Clone, Copy, PartialEq, Eq, Hash smoke tests ─────────────────────

        /// Clone produces an equal value for each variant.
        #[test]
        fn clone_equals_original() {
            for m in [
                SyncMode::ProducerSynced,
                SyncMode::Event,
                SyncMode::ConsumerStream,
            ] {
                assert_eq!(m, m.clone());
            }
        }

        /// Copy: assigning to a new binding leaves the original usable (Copy
        /// semantics verified by using both after the assignment).
        #[test]
        fn copy_semantics() {
            let original = SyncMode::Event;
            let copied = original;
            assert_eq!(original, copied);
        }

        /// PartialEq: distinct variants are not equal.
        #[test]
        fn partial_eq_distinct_variants() {
            assert_ne!(SyncMode::ProducerSynced, SyncMode::Event);
            assert_ne!(SyncMode::ProducerSynced, SyncMode::ConsumerStream);
            assert_ne!(SyncMode::Event, SyncMode::ConsumerStream);
        }

        /// Hash: all three variants produce distinct hashes (smoke test — hash
        /// collisions are possible in theory but the stdlib hasher avoids them
        /// for small integers).
        #[test]
        fn hash_all_variants_distinct() {
            let set: HashSet<SyncMode> = [
                SyncMode::ProducerSynced,
                SyncMode::Event,
                SyncMode::ConsumerStream,
            ]
            .into_iter()
            .collect();
            assert_eq!(set.len(), 3);
        }
    }

    // ── DeviceTag new variants (ADR-016) ──────────────────────────────────────

    mod device_tag_new_variants {
        use super::*;

        // ── Vulkan (0x04) ────────────────────────────────────────────────────

        /// ADR-016: 0x04 decodes to Vulkan.
        #[test]
        fn vulkan_from_byte() {
            assert_eq!(DeviceTag::from_byte(0x04).unwrap(), DeviceTag::Vulkan);
        }

        /// Vulkan serializes to wire byte 0x04.
        #[test]
        fn vulkan_to_byte() {
            assert_eq!(DeviceTag::Vulkan.to_byte(), 0x04);
        }

        /// from_byte(0x04) → to_byte() == 0x04.
        #[test]
        fn vulkan_round_trip() {
            let tag = DeviceTag::from_byte(0x04).unwrap();
            assert_eq!(tag.to_byte(), 0x04);
        }

        /// Vulkan displays as "vulkan".
        #[test]
        fn vulkan_display() {
            assert_eq!(DeviceTag::Vulkan.to_string(), "vulkan");
        }

        /// Vulkan is not a private tag.
        #[test]
        fn vulkan_is_not_private() {
            assert!(!DeviceTag::Vulkan.is_private());
        }

        // ── WebGpu (0x05) ────────────────────────────────────────────────────

        /// ADR-016: 0x05 decodes to WebGpu.
        #[test]
        fn webgpu_from_byte() {
            assert_eq!(DeviceTag::from_byte(0x05).unwrap(), DeviceTag::WebGpu);
        }

        /// WebGpu serializes to wire byte 0x05.
        #[test]
        fn webgpu_to_byte() {
            assert_eq!(DeviceTag::WebGpu.to_byte(), 0x05);
        }

        /// from_byte(0x05) → to_byte() == 0x05.
        #[test]
        fn webgpu_round_trip() {
            let tag = DeviceTag::from_byte(0x05).unwrap();
            assert_eq!(tag.to_byte(), 0x05);
        }

        /// WebGpu displays as "webgpu".
        #[test]
        fn webgpu_display() {
            assert_eq!(DeviceTag::WebGpu.to_string(), "webgpu");
        }

        /// WebGpu is not a private tag.
        #[test]
        fn webgpu_is_not_private() {
            assert!(!DeviceTag::WebGpu.is_private());
        }

        // ── Hexagon (0x06) ───────────────────────────────────────────────────

        /// ADR-016: 0x06 decodes to Hexagon.
        #[test]
        fn hexagon_from_byte() {
            assert_eq!(DeviceTag::from_byte(0x06).unwrap(), DeviceTag::Hexagon);
        }

        /// Hexagon serializes to wire byte 0x06.
        #[test]
        fn hexagon_to_byte() {
            assert_eq!(DeviceTag::Hexagon.to_byte(), 0x06);
        }

        /// from_byte(0x06) → to_byte() == 0x06.
        #[test]
        fn hexagon_round_trip() {
            let tag = DeviceTag::from_byte(0x06).unwrap();
            assert_eq!(tag.to_byte(), 0x06);
        }

        /// Hexagon displays as "hexagon".
        #[test]
        fn hexagon_display() {
            assert_eq!(DeviceTag::Hexagon.to_string(), "hexagon");
        }

        /// Hexagon is not a private tag.
        #[test]
        fn hexagon_is_not_private() {
            assert!(!DeviceTag::Hexagon.is_private());
        }

        // ── LevelZero (0x07) ─────────────────────────────────────────────────

        /// ADR-016: 0x07 decodes to LevelZero.
        #[test]
        fn level_zero_from_byte() {
            assert_eq!(DeviceTag::from_byte(0x07).unwrap(), DeviceTag::LevelZero);
        }

        /// LevelZero serializes to wire byte 0x07.
        #[test]
        fn level_zero_to_byte() {
            assert_eq!(DeviceTag::LevelZero.to_byte(), 0x07);
        }

        /// from_byte(0x07) → to_byte() == 0x07.
        #[test]
        fn level_zero_round_trip() {
            let tag = DeviceTag::from_byte(0x07).unwrap();
            assert_eq!(tag.to_byte(), 0x07);
        }

        /// LevelZero displays as "level_zero".
        #[test]
        fn level_zero_display() {
            assert_eq!(DeviceTag::LevelZero.to_string(), "level_zero");
        }

        /// LevelZero is not a private tag.
        #[test]
        fn level_zero_is_not_private() {
            assert!(!DeviceTag::LevelZero.is_private());
        }

        // ── OpenCl (0x08) ────────────────────────────────────────────────────

        /// ADR-016: 0x08 decodes to OpenCl.
        #[test]
        fn opencl_from_byte() {
            assert_eq!(DeviceTag::from_byte(0x08).unwrap(), DeviceTag::OpenCl);
        }

        /// OpenCl serializes to wire byte 0x08.
        #[test]
        fn opencl_to_byte() {
            assert_eq!(DeviceTag::OpenCl.to_byte(), 0x08);
        }

        /// from_byte(0x08) → to_byte() == 0x08.
        #[test]
        fn opencl_round_trip() {
            let tag = DeviceTag::from_byte(0x08).unwrap();
            assert_eq!(tag.to_byte(), 0x08);
        }

        /// OpenCl displays as "opencl".
        #[test]
        fn opencl_display() {
            assert_eq!(DeviceTag::OpenCl.to_string(), "opencl");
        }

        /// OpenCl is not a private tag.
        #[test]
        fn opencl_is_not_private() {
            assert!(!DeviceTag::OpenCl.is_private());
        }

        // ── Edge cases ───────────────────────────────────────────────────────

        /// 0x08 is the last named tag; it must succeed.
        #[test]
        fn from_byte_0x08_is_last_named_tag() {
            assert!(DeviceTag::from_byte(0x08).is_ok());
        }

        /// 0x09 is the first reserved byte after the new range; must return
        /// ReservedDeviceTag(0x09).
        #[test]
        fn from_byte_0x09_is_first_reserved_after_new_range() {
            assert!(matches!(
                DeviceTag::from_byte(0x09),
                Err(Error::ReservedDeviceTag(0x09))
            ));
        }

        /// The full new range 0x04–0x08 all round-trip cleanly.
        #[test]
        fn round_trip_new_range_all_bytes() {
            for b in 0x04u8..=0x08 {
                let tag = DeviceTag::from_byte(b)
                    .unwrap_or_else(|e| panic!("from_byte(0x{b:02X}) failed: {e}"));
                assert_eq!(tag.to_byte(), b, "round-trip failed for byte 0x{b:02X}");
            }
        }
    }

    // ── MemoryClass ───────────────────────────────────────────────────────────

    mod memory_class {
        use super::*;

        /// Spec § buffer-protocol.md Memory Class: 0x00 decodes to Standard.
        #[test]
        fn from_byte_standard() {
            assert_eq!(MemoryClass::from_byte(0x00).unwrap(), MemoryClass::Standard);
        }

        /// Spec § buffer-protocol.md Memory Class: 0x01 decodes to HostPinned.
        #[test]
        fn from_byte_host_pinned() {
            assert_eq!(
                MemoryClass::from_byte(0x01).unwrap(),
                MemoryClass::HostPinned
            );
        }

        /// Spec § buffer-protocol.md Memory Class: 0x02 decodes to Unified.
        #[test]
        fn from_byte_unified() {
            assert_eq!(MemoryClass::from_byte(0x02).unwrap(), MemoryClass::Unified);
        }

        /// Spec § buffer-protocol.md Memory Class: 0x03 decodes to Peer.
        #[test]
        fn from_byte_peer() {
            assert_eq!(MemoryClass::from_byte(0x03).unwrap(), MemoryClass::Peer);
        }

        /// 0x04 is the first reserved byte; must return ReservedMemoryClass.
        #[test]
        fn from_byte_reserved_lower_bound() {
            assert!(matches!(
                MemoryClass::from_byte(0x04),
                Err(Error::ReservedMemoryClass(0x04))
            ));
        }

        /// 0xEF is the upper bound of the reserved range.
        #[test]
        fn from_byte_reserved_upper_bound() {
            assert!(matches!(
                MemoryClass::from_byte(0xEF),
                Err(Error::ReservedMemoryClass(0xEF))
            ));
        }

        /// 0xF0 (lower bound of private range) decodes to Private(0xF0).
        #[test]
        fn from_byte_private_lower_bound() {
            let cls = MemoryClass::from_byte(0xF0).unwrap();
            assert!(cls.is_private());
            assert_eq!(cls.to_byte(), 0xF0);
        }

        /// 0xFE (upper bound of private range) decodes to Private(0xFE).
        #[test]
        fn from_byte_private_upper_bound() {
            let cls = MemoryClass::from_byte(0xFE).unwrap();
            assert!(cls.is_private());
            assert_eq!(cls.to_byte(), 0xFE);
        }

        /// 0xFF is permanently reserved; must return InvalidMemoryClass.
        #[test]
        fn from_byte_invalid_sentinel() {
            assert!(matches!(
                MemoryClass::from_byte(0xFF),
                Err(Error::InvalidMemoryClass(0xFF))
            ));
        }

        /// All four named variants round-trip through to_byte → from_byte.
        #[test]
        fn named_variants_round_trip() {
            for cls in [
                MemoryClass::Standard,
                MemoryClass::HostPinned,
                MemoryClass::Unified,
                MemoryClass::Peer,
            ] {
                assert_eq!(MemoryClass::from_byte(cls.to_byte()).unwrap(), cls);
            }
        }

        /// BufferHandle::new() defaults to MemoryClass::Standard.
        #[test]
        fn buffer_handle_new_defaults_to_standard() {
            let h = BufferHandle::new(64, 64, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap();
            assert_eq!(h.memory_class(), MemoryClass::Standard);
        }

        /// BufferHandle::with_memory_class() stores the given class.
        #[test]
        fn buffer_handle_with_memory_class_unified() {
            let h = BufferHandle::with_memory_class(
                64,
                64,
                DeviceTag::Cuda,
                SyncMode::ProducerSynced,
                MemoryClass::Unified,
            )
            .unwrap();
            assert_eq!(h.memory_class(), MemoryClass::Unified);
        }

        /// BufferHandle::empty() always uses Standard.
        #[test]
        fn empty_handle_is_standard() {
            assert_eq!(
                BufferHandle::empty(DeviceTag::Cpu).memory_class(),
                MemoryClass::Standard
            );
        }

        /// validate_colocation rejects mixed memory classes.
        #[test]
        fn validate_colocation_rejects_mixed_memory_class() {
            let standard =
                BufferHandle::new(64, 64, DeviceTag::Cuda, SyncMode::ProducerSynced).unwrap();
            let unified = BufferHandle::with_memory_class(
                64,
                64,
                DeviceTag::Cuda,
                SyncMode::ProducerSynced,
                MemoryClass::Unified,
            )
            .unwrap();
            assert!(matches!(
                validate_colocation(&[standard, unified]),
                Err(Error::MemoryClassMismatch {
                    expected: 0x00,
                    found: 0x02
                })
            ));
        }

        /// validate_colocation accepts uniform memory class.
        #[test]
        fn validate_colocation_accepts_uniform_memory_class() {
            let a = BufferHandle::with_memory_class(
                64,
                64,
                DeviceTag::Cuda,
                SyncMode::ProducerSynced,
                MemoryClass::Unified,
            )
            .unwrap();
            let b = BufferHandle::with_memory_class(
                128,
                64,
                DeviceTag::Cuda,
                SyncMode::ProducerSynced,
                MemoryClass::Unified,
            )
            .unwrap();
            assert_eq!(validate_colocation(&[a, b]).unwrap(), DeviceTag::Cuda);
        }
    }

    // ── BufferHandle + SyncMode integration ───────────────────────────────────

    mod buffer_handle_sync_mode {
        use super::*;
        use crate::descriptor::TensorDescriptor;
        use crate::layout::LayoutDescriptor;
        use crate::{ElementType, Shape, MIN_BUFFER_ALIGNMENT};

        // ── Non-CPU buffers accept Event and ConsumerStream ───────────────────

        /// A non-CPU buffer with SyncMode::Event must be accepted and the
        /// sync_mode() accessor must return Event.
        #[test]
        fn new_with_event_sync_non_cpu_buffer() {
            let handle =
                BufferHandle::new(512, MIN_BUFFER_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event)
                    .unwrap();
            assert_eq!(handle.sync_mode(), SyncMode::Event);
            assert_eq!(handle.device_tag(), DeviceTag::Cuda);
        }

        /// A non-CPU buffer with SyncMode::ConsumerStream must be accepted and
        /// the sync_mode() accessor must return ConsumerStream.
        #[test]
        fn new_with_consumer_stream_sync_non_cpu_buffer() {
            let handle = BufferHandle::new(
                512,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Rocm,
                SyncMode::ConsumerStream,
            )
            .unwrap();
            assert_eq!(handle.sync_mode(), SyncMode::ConsumerStream);
            assert_eq!(handle.device_tag(), DeviceTag::Rocm);
        }

        // ── CPU buffers must reject non-ProducerSynced modes ─────────────────

        /// ADR-018: CPU buffer with SyncMode::Event must be rejected with
        /// InvalidSyncMode(0x01).
        #[test]
        fn new_cpu_rejects_event() {
            let result =
                BufferHandle::new(512, MIN_BUFFER_ALIGNMENT, DeviceTag::Cpu, SyncMode::Event);
            assert!(matches!(result, Err(Error::InvalidSyncMode(0x01))));
        }

        /// ADR-018: CPU buffer with SyncMode::ConsumerStream must be rejected
        /// with InvalidSyncMode(0x02).
        #[test]
        fn new_cpu_rejects_consumer_stream() {
            let result = BufferHandle::new(
                512,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ConsumerStream,
            );
            assert!(matches!(result, Err(Error::InvalidSyncMode(0x02))));
        }

        /// ADR-018: CPU buffer with SyncMode::ProducerSynced must succeed.
        #[test]
        fn new_cpu_allows_producer_synced() {
            let result = BufferHandle::new(
                512,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Cpu,
                SyncMode::ProducerSynced,
            );
            assert!(result.is_ok());
        }

        // ── BufferHandle::empty always uses ProducerSynced ────────────────────

        /// BufferHandle::empty always sets sync_mode to ProducerSynced,
        /// even for non-CPU devices.
        #[test]
        fn empty_has_producer_synced() {
            assert_eq!(
                BufferHandle::empty(DeviceTag::Cuda).sync_mode(),
                SyncMode::ProducerSynced
            );
        }

        // ── encode/decode round-trip preserves sync_mode ─────────────────────

        /// A TensorDescriptor containing a buffer with SyncMode::Event must
        /// survive encode → decode with sync_mode unchanged.
        #[test]
        fn encode_decode_preserves_sync_mode_event() {
            let shape = Shape::new(vec![4u64]).unwrap();
            let buffer =
                BufferHandle::new(64, MIN_BUFFER_ALIGNMENT, DeviceTag::Cuda, SyncMode::Event)
                    .unwrap();
            let desc = TensorDescriptor::new(
                1,
                0,
                ElementType::Float32,
                shape,
                0,
                LayoutDescriptor::RowMajor,
                vec![buffer],
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let encoded = desc.encode().unwrap();
            let decoded = TensorDescriptor::decode(&encoded).unwrap();

            assert_eq!(decoded.buffers[0].sync_mode(), SyncMode::Event);
            assert_eq!(decoded.buffers[0].device_tag(), DeviceTag::Cuda);
            assert_eq!(decoded, desc);
        }

        /// A TensorDescriptor containing a buffer with SyncMode::ConsumerStream
        /// must survive encode → decode with sync_mode unchanged.
        #[test]
        fn encode_decode_preserves_sync_mode_consumer_stream() {
            let shape = Shape::new(vec![8u64]).unwrap();
            let buffer = BufferHandle::new(
                128,
                MIN_BUFFER_ALIGNMENT,
                DeviceTag::Vulkan,
                SyncMode::ConsumerStream,
            )
            .unwrap();
            let desc = TensorDescriptor::new(
                1,
                0,
                ElementType::Float32,
                shape,
                0,
                LayoutDescriptor::RowMajor,
                vec![buffer],
                None,
                None,
                None,
                None,
            )
            .unwrap();

            let encoded = desc.encode().unwrap();
            let decoded = TensorDescriptor::decode(&encoded).unwrap();

            assert_eq!(decoded.buffers[0].sync_mode(), SyncMode::ConsumerStream);
            assert_eq!(decoded.buffers[0].device_tag(), DeviceTag::Vulkan);
            assert_eq!(decoded, desc);
        }
    }
}
