# Project Structure

```
hurray/
├── README.md                   # Project overview and format contract
├── PROJECT_STRUCTURE.md        # This file
├── CLAUDE.md                   # AI agent configuration and conventions
├── Cargo.toml                  # Workspace root
├── docs/
│   ├── prior-art.md            # Research snapshot: formats, protocols, libraries
│   ├── spec/                   # Format specification (source of truth)
│   │   ├── README.md           # Scope, goals, RFC 2119 notice, versioning
│   │   ├── data-model.md       # Shape/dimension model
│   │   ├── element-types.md    # Element type system (int, float, quantized, custom)
│   │   ├── quantization.md     # Quantization schemes: per-tensor, per-channel, per-block, NF4, MXFP
│   │   ├── memory-layout.md    # Layout index and overview
│   │   ├── layouts/            # Per-layout spec files
│   │   │   ├── row-major.md
│   │   │   ├── column-major.md
│   │   │   ├── strided.md
│   │   │   ├── tiled.md
│   │   │   ├── morton.md
│   │   │   ├── hilbert.md
│   │   │   ├── subpaving.md
│   │   │   ├── coo.md          # Sparse: Coordinate list
│   │   │   ├── csr.md          # Sparse: Compressed Sparse Row
│   │   │   ├── csc.md          # Sparse: Compressed Sparse Column
│   │   │   ├── csf.md          # Sparse: Compressed Sparse Fiber (rank-N)
│   │   │   └── block-paged.md  # Indirect: PagedAttention KV cache
│   │   ├── buffer-protocol.md  # Zero-copy semantics, alignment, device memory
│   │   ├── metadata.md         # Tensor descriptor binary encoding
│   │   ├── interchange.md      # Runtime interchange: in-process, IPC, cross-machine
│   │   ├── versioning.md       # Format version field, compatibility policy
│   │   └── references.md       # Normative references
│   ├── impl/                   # Implementation requirements
│   │   ├── README.md           # Overview of implementation requirement docs
│   │   ├── compliance.md       # Compliance checklist for implementors
│   │   ├── rust-reference.md   # Rust reference implementation guide
│   │   ├── c-ffi.md            # C FFI implementation guide
│   │   └── python-bindings.md  # Python bindings guide
│   └── adr/                    # Architecture Decision Records
│       └── ADR-NNN-*.md
├── hurray-core/                # Core types, no I/O, no async
├── hurray-io/                  # Async I/O: streaming + file format (tokio)
├── hurray-ffi/                 # C ABI layer for language bindings
├── hurray-python/              # Python bindings (PyO3)
└── hurray-inspect/             # CLI hex viewer for Hurray descriptor files
```
