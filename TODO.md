# TODO

- Data-driven layout: the reference implementation may repack/convert a tensor to a more optimized layout by introspecting the tensor data itself or the attached tensor statistics
- Redis module: use Hurray as the native tensor type for a Redis module (storage + serving, format-aware, no inference execution — narrower and more useful than the abandoned RedisAI approach)
- Cookbook: a collection of recipes demonstrating common Hurray usage patterns in Rust and Python (e.g., creating and reading tensors, quantized inference, zero-copy interop with NumPy/PyTorch, IPC exchange)
- Tensor view / reshape annotation (v2 candidate): a mechanism for a descriptor to reference an existing buffer with a different shape, strides, or rank (zero-copy reshape, slice, transpose, broadcast). Key design constraint: conflicts with the "no back-references" streaming rule — likely restricted to file format and in-process buffer protocol, not the streaming wire format. Would subsume compound element types as a special case (rank r-1 view with named components). Deferred from v1; revisit when streaming model constraints are better understood.
