+++
title = "Introducing Hurray"
date = 2026-07-26
description = "A zero-copy, streamable tensor interchange format for AI/ML inference — and the documentation site that goes with it."
+++

Modern inference pipelines move tensors constantly: from disk into host memory, across
process boundaries, between machines, and onto accelerators. The formats we reach for were
each designed for one leg of that journey. **DLPack** nails in-process hand-off but has no
notion of quantization or on-disk storage. **SafeTensors** and **GGUF** store weights well
but say nothing about runtime interchange. **Apache Arrow** is superb for columnar
analytics but isn't tensor-shaped.

**Hurray** is an attempt to cover the whole path with a single tensor descriptor.

## Two formats, one descriptor

Hurray defines two binary formats that share the same descriptor encoding:

- a **streaming format** for runtime interchange — self-delimiting, no seek required, safe
  to start processing before the payload finishes arriving; and
- a **file format** for on-disk model storage — named tensors, a footer index for random
  access, and 4 KiB-aligned buffers for mmap-to-GPU zero-copy loading.

Write a descriptor parser once and it works for both.

## Built for inference

Hurray treats layout diversity, quantization, and device memory as first-class concerns:

- **Twelve memory layouts**, including tiled/blocked for GEMM, Morton and Hilbert curves
  for spatial locality, sparse COO/CSR/CSC/CSF, block-paged for PagedAttention KV caches,
  and composite tensors for partitioning and overlays.
- **Five quantization schemes** — per-tensor, per-channel, and per-block affine, NF4
  (QLoRA), and MXFP (OCP Microscaling / Blackwell) — with normative dequantization formulas.
- **Zero-copy** buffer sharing through a stable, language-agnostic C ABI.

## Where we are

The specification is at `0.1.0-draft` and the Rust reference implementation is underway.
The format is designed to be **evolvable**: backward-compatible and forward-additive across
the whole `1.x` line, with public tags never rebound once allocated.

This site is where the specification, implementation requirements, cookbook, tutorials, and
design decisions (ADRs) live — versioned alongside every release. Have a look at the
[docs](/docs/stable/), and come say hello on
[GitHub](https://github.com/pgillet/hurray).
