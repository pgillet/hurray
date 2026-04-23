# TODO

- Data-driven layout: the reference implementation may repack/convert a tensor to a more optimized layout by introspecting the tensor data itself or the attached tensor statistics
- Redis module: use Hurray as the native tensor type for a Redis module (storage + serving, format-aware, no inference execution — narrower and more useful than the abandoned RedisAI approach)
- In-process SQL/MDA engine: an embeddable SQL query engine (ISO 9075-15 SQL/MDA) backed by Hurray as its native memory model — MDA columns backed by Hurray buffers, zero-copy handoff from query result to ML inference pipeline; open questions: quantized types as first-class SQL types vs opaque storage, sparse layout semantics, arithmetic on quantized arrays
- hurray-archive: a named/indexed multi-tensor collection format (analogous to SafeTensors/GGUF) as a future sibling specification — deferred from v1 per ADR-010; design should be informed by production experience with the runtime format
- Cookbook: a collection of recipes demonstrating common Hurray usage patterns in Rust and Python (e.g., creating and reading tensors, quantized inference, zero-copy interop with NumPy/PyTorch, IPC exchange)
