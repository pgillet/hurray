# API Reference

Generated reference documentation for this version of Hurray. Both references are built
from this version's own source tree, so they describe exactly the code the rest of this
book describes.

> **Note (non-normative):** The links below are relative to the documentation site and
> resolve only there — they are generated output, not files in the repository. Reading this
> page on GitHub, follow it on the [documentation site](https://pgillet.github.io/hurray/)
> instead.

## Rust

[`cargo doc`](api/) output for every crate in the workspace.

| Crate | What it covers |
|-------|----------------|
| [`hurray_core`](api/hurray_core/index.html) | Element types, shape, buffer handles, quantization and layout descriptors, `TensorDescriptor` encoding. No I/O. |
| [`hurray_io`](api/hurray_io/index.html) | Async streaming interchange and the `HRRYFILE` container. |
| [`hurray_ffi`](api/hurray_ffi/index.html) | The C ABI: opaque handles, function table, release callbacks. |
| [`hurray`](api/hurray/index.html) | The Rust side of the Python bindings — the PyO3 classes behind the `hurray` module. |
| [`hurray_inspect`](api/hurray_inspect/index.html) | The descriptor hex-viewer CLI. |

## Python

[The `hurray` module](python-api/) — every class, function, and constant the bindings
expose, with signatures and examples.

| Page | What it covers |
|------|----------------|
| [`hurray`](python-api/hurray.html) | `Tensor`, `Composite`, `Descriptor`, layout and quantization classes, creation and interop functions, file and stream I/O, exceptions. |
| [`hurray.dtype`](python-api/hurray/dtype.html) | The element-type constants and `Dtype`. |
| [`hurray.device`](python-api/hurray/device.html) | The device constants and `Device`. |

For task-shaped introductions rather than an exhaustive surface, start from the cookbook's
Python entries — [Dtype, Device, and Tensor](cookbook/hurray-python-tensor-basics.md) is
the first one.
