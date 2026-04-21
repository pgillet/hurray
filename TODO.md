# TODO

- Data-driven layout: the reference implementation may repack/convert a tensor to a more optimized layout by introspecting the tensor data itself or the attached tensor statistics
- Redis module: use Hurray as the native tensor type for a Redis module (storage + serving, format-aware, no inference execution — narrower and more useful than the abandoned RedisAI approach)
- In-process SQL/MDA engine: an embeddable SQL query engine (ISO 9075-15 SQL/MDA) backed by Hurray as its native memory model — MDA columns backed by Hurray buffers, zero-copy handoff from query result to ML inference pipeline; open questions: quantized types as first-class SQL types vs opaque storage, sparse layout semantics, arithmetic on quantized arrays
