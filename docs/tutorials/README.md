# Tutorials

Guided, end-to-end walkthroughs of Hurray. Where the [Cookbook](../cookbook/layer-0-element-types-and-shape.md)
gives focused recipes for one feature at a time, tutorials string those recipes together
into a start-to-finish path.

## Guided path: build up a Hurray implementation, one layer at a time

The layered cookbook is written to be read in order. Follow it top to bottom for a complete
tour of the format, from element types to the C FFI:

1. [Layer 0 — Element Types and Shape](../cookbook/layer-0-element-types-and-shape.md)
2. [Layer 1 — Buffer Protocol](../cookbook/layer-1-buffer-protocol.md)
3. [Layer 2 — Quantization Descriptors](../cookbook/layer-2-quantization-descriptors.md)
4. [Layer 3 — Layout Descriptors](../cookbook/layer-3-layout-descriptors.md)
5. [Layer 4 — Tensor Descriptor Encoding](../cookbook/layer-4-tensor-descriptor-encoding.md)
6. [Layer 5 — Streaming Interchange](../cookbook/layer-5-streaming-interchange.md)
7. [Layer 6 — File Format](../cookbook/layer-6-file-format.md)
8. [Layer 7 — C FFI](../cookbook/layer-7-c-ffi.md)

## Task-focused tutorials

- [Integrating a Python Library with Hurray](python-interop-paths.md) — the four
  interop paths available to a library with its own array type, and how to choose
  between them.

More tutorials will be added here over time. To propose or contribute one, open an issue or
pull request on [GitHub](https://github.com/pgillet/hurray).
