//! Python bindings for the Hurray device and memory class system.
//!
//! Exposes [`Device`] as a Python class (`hurray.Device`) and populates a
//! `hurray.device` submodule with well-known device constants (`cpu`, `cuda`, …).
//!
//! The `hurray.device` submodule is registered in `sys.modules` so
//! `from hurray.device import cpu` works correctly (D4).
//!
//! See `docs/spec/buffer-protocol.md § Device Tags` for the normative definitions
//! of the device tag and memory class enumerations.

// PyO3 0.22 macro expansion emits a redundant .into() on PyErr for functions
// returning PyResult<()> — suppress the false positive across this module.
#![allow(clippy::useless_conversion)]

use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use hurray_core::{DeviceTag, MemoryClass};

use crate::errors::InvalidDescriptorError;

/// A Hurray device descriptor: a (kind, device_id, memory_class) triple.
///
/// `Device` objects are immutable (`frozen`) and hashable — safe to use as dict keys.
///
/// ## Constructor
///
/// ```python
/// hurray.Device(kind, device_id=0, memory_class="standard")
/// ```
///
/// | Parameter | Type | Default | Accepted values |
/// |-----------|------|---------|-----------------|
/// | `kind` | `str` | — | `"cpu"`, `"cuda"`, `"rocm"`, `"metal"`, `"vulkan"`, `"webgpu"`, `"hexagon"`, `"level_zero"`, `"opencl"` |
/// | `device_id` | `int` | `0` | Non-negative integer |
/// | `memory_class` | `str` | `"standard"` | `"standard"`, `"host_pinned"`, `"unified"`, `"peer"` |
///
/// ## Examples (Python)
///
/// ```python
/// import hurray
///
/// cpu = hurray.Device("cpu")
/// assert cpu.kind == "cpu"
/// assert cpu.device_id == 0
/// assert cpu.memory_class == "standard"
///
/// gpu = hurray.Device("cuda", 1)
/// assert gpu.kind == "cuda"
/// assert gpu.device_id == 1
///
/// # Well-known constants
/// assert hurray.Device("cpu") == hurray.device.cpu
///
/// # Hashable
/// d = {hurray.device.cpu: "cpu0"}
/// assert d[hurray.Device("cpu")] == "cpu0"
/// ```
#[pyclass(name = "Device", frozen)]
#[derive(Debug)]
pub struct Device {
    /// The hurray-core device tag.
    pub tag: DeviceTag,
    /// The memory access class of this buffer.
    pub memory_class: MemoryClass,
    /// Device ordinal (0 = first device of this kind).
    pub device_id: i32,
}

#[pymethods]
impl Device {
    // ── Constructor ───────────────────────────────────────────────────────────

    /// Create a new `Device`.
    ///
    /// ## Examples
    ///
    /// ```python
    /// import hurray
    ///
    /// cpu = hurray.Device("cpu")
    /// gpu = hurray.Device("cuda", 1)
    /// unified = hurray.Device("cuda", 0, "unified")
    /// ```
    #[new]
    #[pyo3(signature = (kind, device_id = None, memory_class = None))]
    pub fn new(kind: &str, device_id: Option<i32>, memory_class: Option<&str>) -> PyResult<Self> {
        let tag = device_tag_from_str(kind).ok_or_else(|| {
            InvalidDescriptorError::new_err(format!(
                "unknown device kind '{kind}'; accepted: cpu, cuda, rocm, metal, \
                 vulkan, webgpu, hexagon, level_zero, opencl"
            ))
        })?;

        let mc = memory_class_from_str(memory_class.unwrap_or("standard")).ok_or_else(|| {
            InvalidDescriptorError::new_err(format!(
                "unknown memory_class '{}'; accepted: standard, host_pinned, unified, peer",
                memory_class.unwrap_or("standard")
            ))
        })?;

        let id = device_id.unwrap_or(0);
        if id < 0 {
            return Err(InvalidDescriptorError::new_err(format!(
                "device_id must be non-negative, got {id}"
            )));
        }

        Ok(Self {
            tag,
            memory_class: mc,
            device_id: id,
        })
    }

    // ── Getters ───────────────────────────────────────────────────────────────

    /// The device kind string (e.g. `"cpu"`, `"cuda"`).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Device("cuda", 1).kind == "cuda"
    /// ```
    #[getter]
    pub fn kind(&self) -> &'static str {
        device_tag_to_str(self.tag)
    }

    /// The device ordinal (0-based index within devices of the same kind).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Device("cuda", 2).device_id == 2
    /// ```
    #[getter]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// The memory access class string (e.g. `"standard"`, `"unified"`).
    ///
    /// ## Examples
    ///
    /// ```python
    /// assert hurray.Device("cpu").memory_class == "standard"
    /// ```
    #[getter]
    pub fn memory_class(&self) -> &'static str {
        memory_class_to_str(self.memory_class)
    }

    // ── Dunders ───────────────────────────────────────────────────────────────

    fn __repr__(&self) -> String {
        format!(
            "hurray.Device(kind='{}', device_id={}, memory_class='{}')",
            device_tag_to_str(self.tag),
            self.device_id,
            memory_class_to_str(self.memory_class),
        )
    }

    fn __richcmp__(&self, other: &Device, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.tag == other.tag
                && self.memory_class == other.memory_class
                && self.device_id == other.device_id),
            CompareOp::Ne => Ok(self.tag != other.tag
                || self.memory_class != other.memory_class
                || self.device_id != other.device_id),
            // Ordering is not meaningful for device descriptors.
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {
                Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Device does not support ordering",
                ))
            }
        }
    }

    fn __hash__(&self) -> u64 {
        // Non-zero seed ensures the most common device (cpu/standard/0) doesn't hash to 0,
        // which would degrade dict/set performance for cpu-heavy workloads.
        const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
        SEED ^ (self.tag.to_byte() as u64).wrapping_mul(2_654_435_761)
            ^ ((self.memory_class.to_byte() as u64).wrapping_mul(2_246_822_519))
            ^ (self.device_id as u64)
    }
}

// ── String ↔ DeviceTag / MemoryClass helpers ──────────────────────────────────

fn device_tag_from_str(s: &str) -> Option<DeviceTag> {
    match s {
        "cpu" => Some(DeviceTag::Cpu),
        "cuda" => Some(DeviceTag::Cuda),
        "rocm" => Some(DeviceTag::Rocm),
        "metal" => Some(DeviceTag::Metal),
        "vulkan" => Some(DeviceTag::Vulkan),
        "webgpu" => Some(DeviceTag::WebGpu),
        "hexagon" => Some(DeviceTag::Hexagon),
        "level_zero" => Some(DeviceTag::LevelZero),
        "opencl" => Some(DeviceTag::OpenCl),
        _ => None,
    }
}

pub(crate) fn device_tag_to_str(tag: DeviceTag) -> &'static str {
    match tag {
        DeviceTag::Cpu => "cpu",
        DeviceTag::Cuda => "cuda",
        DeviceTag::Rocm => "rocm",
        DeviceTag::Metal => "metal",
        DeviceTag::Vulkan => "vulkan",
        DeviceTag::WebGpu => "webgpu",
        DeviceTag::Hexagon => "hexagon",
        DeviceTag::LevelZero => "level_zero",
        DeviceTag::OpenCl => "opencl",
        // Private tags have no fixed string name in the public API.
        DeviceTag::Private(_) => "private",
    }
}

fn memory_class_from_str(s: &str) -> Option<MemoryClass> {
    match s {
        "standard" => Some(MemoryClass::Standard),
        "host_pinned" => Some(MemoryClass::HostPinned),
        "unified" => Some(MemoryClass::Unified),
        "peer" => Some(MemoryClass::Peer),
        _ => None,
    }
}

pub(crate) fn memory_class_to_str(mc: MemoryClass) -> &'static str {
    match mc {
        MemoryClass::Standard => "standard",
        MemoryClass::HostPinned => "host_pinned",
        MemoryClass::Unified => "unified",
        MemoryClass::Peer => "peer",
        MemoryClass::Private(_) => "private",
    }
}

/// Well-known device constants registered on `hurray.device`.
///
/// Each entry is `(python_name, kind_str, device_id, memory_class_str)`.
const DEVICE_CONSTANTS: &[(&str, &str)] = &[
    ("cpu", "cpu"),
    ("cuda", "cuda"),
    ("rocm", "rocm"),
    ("metal", "metal"),
    ("vulkan", "vulkan"),
    ("webgpu", "webgpu"),
    ("hexagon", "hexagon"),
    ("level_zero", "level_zero"),
    ("opencl", "opencl"),
];

/// Return the `repr()` string for a `Device`, usable from sibling modules.
pub(crate) fn device_repr(dev: &Device) -> String {
    format!(
        "hurray.Device(kind='{}', device_id={}, memory_class='{}')",
        device_tag_to_str(dev.tag),
        dev.device_id,
        memory_class_to_str(dev.memory_class),
    )
}

/// Register `Device` and the `hurray.device` submodule on the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add_class::<Device>()?;

    // Create the `hurray.device` submodule.
    let device_mod = PyModule::new_bound(py, "device")?;

    for &(name, kind) in DEVICE_CONSTANTS {
        let tag = device_tag_from_str(kind).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "DEVICE_CONSTANTS contains invalid kind '{kind}'; this is a bug"
            ))
        })?;
        let obj = Py::new(
            py,
            Device {
                tag,
                memory_class: MemoryClass::Standard,
                device_id: 0,
            },
        )?;
        device_mod.add(name, obj)?;
    }

    // Register `hurray.device` in `sys.modules` so `from hurray.device import cpu` works.
    let sys = py.import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("hurray.device", &device_mod)?;

    m.add_submodule(&device_mod)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    fn init() {
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn constructor_succeeds_for_all_known_kinds() {
        let kinds = [
            "cpu",
            "cuda",
            "rocm",
            "metal",
            "vulkan",
            "webgpu",
            "hexagon",
            "level_zero",
            "opencl",
        ];
        for kind in kinds {
            Device::new(kind, None, None)
                .unwrap_or_else(|_| panic!("Device::new should succeed for kind='{kind}'"));
        }
    }

    #[test]
    fn constructor_fails_unknown_kind() {
        init();
        Python::with_gil(|py| {
            let result = Device::new("tpu", None, None);
            assert!(result.is_err(), "unknown kind should return Err");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<InvalidDescriptorError>(py),
                "error should be InvalidDescriptorError"
            );
        });
    }

    #[test]
    fn constructor_fails_unknown_memory_class() {
        init();
        Python::with_gil(|py| {
            let result = Device::new("cpu", None, Some("device_local"));
            assert!(result.is_err(), "unknown memory_class should return Err");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<InvalidDescriptorError>(py),
                "error should be InvalidDescriptorError"
            );
        });
    }

    #[test]
    fn constructor_fails_negative_device_id() {
        init();
        Python::with_gil(|py| {
            let result = Device::new("cuda", Some(-1), None);
            assert!(result.is_err(), "negative device_id should return Err");
            let err = result.unwrap_err();
            assert!(
                err.is_instance_of::<InvalidDescriptorError>(py),
                "error should be InvalidDescriptorError"
            );
        });
    }

    #[test]
    fn properties_round_trip() {
        let d = Device::new("cuda", Some(2), Some("unified")).unwrap();
        assert_eq!(d.kind(), "cuda");
        assert_eq!(d.device_id(), 2);
        assert_eq!(d.memory_class(), "unified");
    }

    #[test]
    fn eq_covers_all_three_fields() {
        let a = Device::new("cuda", Some(0), Some("standard")).unwrap();
        let b = Device::new("cuda", Some(0), Some("standard")).unwrap();
        let c = Device::new("cuda", Some(1), Some("standard")).unwrap();
        let d_dev = Device::new("rocm", Some(0), Some("standard")).unwrap();
        let e = Device::new("cuda", Some(0), Some("unified")).unwrap();

        // Same triple — equal.
        assert!(
            a.__richcmp__(&b, CompareOp::Eq).unwrap(),
            "same triple should be equal"
        );
        // Different device_id.
        assert!(
            !a.__richcmp__(&c, CompareOp::Eq).unwrap(),
            "different device_id should be unequal"
        );
        // Different kind.
        assert!(
            !a.__richcmp__(&d_dev, CompareOp::Eq).unwrap(),
            "different kind should be unequal"
        );
        // Different memory_class.
        assert!(
            !a.__richcmp__(&e, CompareOp::Eq).unwrap(),
            "different memory_class should be unequal"
        );
    }

    #[test]
    fn hash_consistent_with_eq() {
        let a = Device::new("cpu", Some(0), Some("standard")).unwrap();
        let b = Device::new("cpu", Some(0), Some("standard")).unwrap();
        assert_eq!(
            a.__hash__(),
            b.__hash__(),
            "equal devices must have equal hashes"
        );
    }

    #[test]
    fn repr_format() {
        let d = Device::new("cuda", Some(1), Some("unified")).unwrap();
        let r = d.__repr__();
        assert!(r.contains("cuda"), "repr should contain kind: got '{r}'");
        assert!(r.contains('1'), "repr should contain device_id: got '{r}'");
        assert!(
            r.contains("unified"),
            "repr should contain memory_class: got '{r}'"
        );
    }
}
