//! The binding's blocking drivers over `hurray-io`'s sans-io cores (ADR-041 § 1, § 7).
//!
//! The synchronous Python surface drives the cores itself over `std::io`, rather than
//! driving `hurray-io`'s tokio drivers through a `block_on`. The cores hold the wire
//! format and the read or write position; these types only move bytes and report how
//! many moved.
//!
//! ## Why the drivers live here
//!
//! They are *this binding's* drivers, exactly as the blocking drivers in `hurray-ffi` are
//! the C binding's. Hoisting them into `hurray-io` would publish a driver family no Rust
//! consumer has asked for, and ADR-041 § 1 puts drivers at the edge.
//!
//! ## Why `read`/`write` and never `read_exact`/`write_all`
//!
//! A bulk helper cannot report how many bytes it moved when it fails, and that count *is*
//! the position (ADR-041 § 6). Per call the count is always known, because `std`
//! guarantees a failed read or write moved nothing — so an I/O error leaves the core's
//! position intact and `Unusable::LostReadCount` and `LostWriteCount` are unreachable
//! from here, the conclusion step 2 reached for the Tokio drivers and step 3 for the C
//! transport. `IncompleteOutput` stays reachable: a write that fails *after* its first
//! byte has torn the item whether or not its count is known. The price is absorbing
//! `ErrorKind::Interrupted` by hand, which the bulk helpers did internally.
//!
//! ## Why the transfer and its report are never separated
//!
//! Each `fill` reads and reports in one unsuspendable step, which is the shape ADR-041
//! § 6a requires of every driver. Nothing can interleave here, where the thread blocks;
//! the rule exists for the event-loop driver that reuses these cores.

use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};

use hurray_core::TensorDescriptor;
use hurray_io::file::{
    flatten_composite, FileComposite, FileCompositeNode, FileReaderCore, FileStep, FileTensor,
    FileWriteOp, FileWriterCore, KvValue,
};
use hurray_io::stream::{
    CompositeAssembler, CompositeNode, ReadStep, StreamItem, StreamReaderCore, StreamReaderOptions,
    StreamWriterCore, Unusable, WriteOp, WriteStep,
};
use hurray_io::{Error, Result};

// ── Stream ────────────────────────────────────────────────────────────────────

/// Reads items from a blocking source in the Hurray streaming wire format.
pub(crate) struct StreamReaderDriver<R> {
    inner: R,
    core: StreamReaderCore,
    /// Held across calls, not created per call: between successful `next_item` calls it
    /// is provably empty, since a call returns only once a top-level item closes.
    assembler: CompositeAssembler,
}

impl<R: Read> StreamReaderDriver<R> {
    /// Creates a reader with the limits `options` carries.
    pub(crate) fn with_options(inner: R, options: StreamReaderOptions) -> Self {
        Self {
            inner,
            // The core ignores `max_composite_depth`; it bounds the assembler (ADR-041
            // § 10, step 2).
            assembler: CompositeAssembler::new(options.max_composite_depth),
            core: StreamReaderCore::with_options(options),
        }
    }

    /// Reads the next item, assembling a composite head and its members into one.
    ///
    /// Returns `Ok(None)` at a clean end of stream.
    pub(crate) fn next_item(&mut self) -> Result<Option<StreamItem>> {
        loop {
            match self.core.step()? {
                ReadStep::Input { .. } => self.fill()?,
                ReadStep::Tensor => {
                    // `take_tensor` yields `Some` exactly when `step` reported `Tensor`.
                    let tensor = self.core.take_tensor().ok_or(Error::CoreFailed)?;
                    if let Some(item) = self.assembler.push(tensor)? {
                        return Ok(Some(item));
                    }
                }
                ReadStep::Finished => {
                    self.assembler.finish()?;
                    return Ok(None);
                }
            }
        }
    }

    /// Answers one [`ReadStep::Input`]: read into the core's destination, report the count.
    fn fill(&mut self) -> Result<()> {
        let n = loop {
            // Disjoint field borrows: the destination comes from `core`, the bytes from
            // `inner`.
            match self.inner.read(self.core.input_dst()) {
                Ok(n) => break n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(Error::Io(e)),
            }
        };
        // `n == 0` is end of input: the core decides whether that is clean or torn.
        self.core.report_filled(n)
    }
}

/// Writes items to a blocking sink in the Hurray streaming wire format.
pub(crate) struct StreamWriterDriver<W> {
    inner: W,
    core: StreamWriterCore,
}

impl<W: Write> StreamWriterDriver<W> {
    /// Creates a writer that accepts any valid sync mode.
    pub(crate) fn new(inner: W) -> Self {
        Self {
            inner,
            core: StreamWriterCore::new(),
        }
    }

    /// Creates a writer that rejects buffers which are not `ProducerSynced`.
    pub(crate) fn cross_machine(inner: W) -> Self {
        Self {
            inner,
            core: StreamWriterCore::cross_machine(),
        }
    }

    /// Writes one tensor: its descriptor, then one payload slice per buffer.
    pub(crate) fn write_tensor(
        &mut self,
        descriptor: &TensorDescriptor,
        buffers: &[&[u8]],
    ) -> Result<()> {
        let mut op = self.core.tensor(descriptor, buffers)?;
        drain_stream(&mut op, &mut self.inner)
    }

    /// Writes a composite as head-then-members, validating the tree before any byte moves.
    pub(crate) fn write_composite(
        &mut self,
        head: &TensorDescriptor,
        members: &[CompositeNode<'_>],
    ) -> Result<()> {
        let mut op = self.core.composite(head, members)?;
        drain_stream(&mut op, &mut self.inner)
    }

    /// Flushes the sink and returns it.
    pub(crate) fn finish(mut self) -> Result<W> {
        self.inner.flush().map_err(Error::Io)?;
        Ok(self.inner)
    }
}

/// Writes every byte of one submitted stream item to `sink`.
///
/// A free function rather than a method: the operation holds a mutable borrow of the
/// core for as long as it lives, so only disjoint *field* borrows let the sink be used at
/// the same time. Deliberately not shared with [`drain_file`] — the two op types have no
/// common trait upstream, and a local one would be more surface than the lines it saved
/// (the conclusion ADR-041 step 4b reached for the same pair).
fn drain_stream<W: Write>(op: &mut WriteOp<'_>, sink: &mut W) -> Result<()> {
    while let WriteStep::Output { .. } = op.step()? {
        let n = {
            let src = op.output_src();
            loop {
                match sink.write(src) {
                    Ok(n) => break n,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    // This call put no bytes on the wire, so its count is known to be
                    // zero. That is not the same as the item being intact: if an earlier
                    // call in this item moved bytes, the core is already torn and the next
                    // call on this writer reports `CoreUnusable` (ADR-041 § 6b).
                    Err(e) => return Err(Error::Io(e)),
                }
            }
        };
        if n == 0 {
            // § 4: a zero-length write is never progress. A closed sink is its own signal,
            // not an I/O error to retry.
            return Err(op.report_sink_closed());
        }
        op.report_written(n)?;
    }
    Ok(())
}

// ── File ──────────────────────────────────────────────────────────────────────

/// Reads tensors from a seekable Hurray file by name.
pub(crate) struct FileReaderDriver<R> {
    inner: R,
    core: FileReaderCore,
}

impl<R: Read + Seek> FileReaderDriver<R> {
    /// Opens a file: header, trailer, index, and KV section.
    pub(crate) fn open(mut inner: R) -> Result<Self> {
        // Size is a constructor argument, not a step: every driver knows it, and here
        // seek-to-end is the answer (ADR-041 step 4a).
        let size = inner.seek(SeekFrom::End(0))?;
        let mut core = FileReaderCore::new(size);

        loop {
            match core.step()? {
                FileStep::Range { offset, len } => fill_range(&mut inner, &mut core, offset, len)?,
                FileStep::Ready => break,
                // Until the file is open the core answers nothing else.
                _ => return Err(Error::CoreFailed),
            }
        }
        // No `index`/`kv` copies here: `hurray_io::file::FileReader` keeps them only
        // because 0.1.0 published them as public fields. Nothing published this driver.
        Ok(Self { inner, core })
    }

    /// The names of every tensor in the file, in index order.
    pub(crate) fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.core.index().iter().map(|e| e.name.as_str())
    }

    /// The file-level KV metadata, empty when the file carries no KV section.
    pub(crate) fn kv(&self) -> &[(String, KvValue)] {
        self.core.kv()
    }

    /// Decodes one descriptor without reading any buffer data.
    pub(crate) fn read_descriptor(&mut self, name: &str) -> Result<TensorDescriptor> {
        self.core.begin_descriptor(name)?;
        match self.drive()? {
            FileStep::Descriptor => self.core.take_descriptor().ok_or(Error::CoreFailed),
            _ => Err(Error::CoreFailed),
        }
    }

    /// Reads one complete tensor: descriptor and every buffer.
    pub(crate) fn read_tensor(&mut self, name: &str) -> Result<FileTensor> {
        self.core.begin_tensor(name)?;
        match self.drive()? {
            FileStep::Tensor => self.core.take_tensor().ok_or(Error::CoreFailed),
            _ => Err(Error::CoreFailed),
        }
    }

    /// Reads a composite by its head name, reassembling it from file-offset adjacency.
    pub(crate) fn read_composite(&mut self, head_name: &str) -> Result<FileComposite> {
        self.core.begin_composite(head_name)?;
        match self.drive()? {
            FileStep::Composite => self.core.take_composite().ok_or(Error::CoreFailed),
            _ => Err(Error::CoreFailed),
        }
    }

    /// Answers every range the core asks for, and returns the step that ends the job.
    fn drive(&mut self) -> Result<FileStep> {
        loop {
            match self.core.step()? {
                FileStep::Range { offset, len } => {
                    fill_range(&mut self.inner, &mut self.core, offset, len)?
                }
                other => return Ok(other),
            }
        }
    }
}

/// Answers one [`FileStep::Range`]: seek, read into the core's destination, report the
/// count.
///
/// A free function rather than a method so the seek and the destination can come from
/// disjoint field borrows, which is also what `open` needs before the driver exists.
fn fill_range<R: Read + Seek>(
    inner: &mut R,
    core: &mut FileReaderCore,
    offset: u64,
    len: usize,
) -> Result<()> {
    inner.seek(SeekFrom::Start(offset))?;
    let n = loop {
        match inner.read(&mut core.input_dst()[..len]) {
            Ok(n) => break n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e)),
        }
    };
    // `n == 0` means the file is shorter than its own index claims; the core calls it.
    core.report_filled(n)
}

/// Writes a Hurray file in a single forward pass, without seeking back.
pub(crate) struct FileWriterDriver<W> {
    inner: W,
    core: FileWriterCore,
}

impl<W: Write> FileWriterDriver<W> {
    /// Creates a writer and emits the file header.
    pub(crate) fn new(mut inner: W) -> Result<Self> {
        let mut core = FileWriterCore::new();
        drain_file(&mut core.header()?, &mut inner)?;
        Ok(Self { inner, core })
    }

    /// Encodes and writes one named tensor.
    pub(crate) fn write_tensor(
        &mut self,
        name: &str,
        descriptor: &TensorDescriptor,
        buffers: &[&[u8]],
    ) -> Result<()> {
        let mut op = self.core.tensor(name, descriptor, buffers)?;
        drain_file(&mut op, &mut self.inner)
    }

    /// Writes a composite: its head, then every member, contiguously and in order.
    pub(crate) fn write_composite(
        &mut self,
        head_name: &str,
        head: &TensorDescriptor,
        members: &[FileCompositeNode<'_>],
    ) -> Result<()> {
        // Validated and flattened into write order first, so a torn or invalid composite
        // never reaches the sink.
        for (index, node) in flatten_composite(head_name, head, members)?
            .into_iter()
            .enumerate()
        {
            // The op is bound to a local and dropped before the poison below: it borrows
            // the core mutably for its whole life, so `self.core` is unreachable until it
            // is gone. That is also why `mark_unusable` on an op cannot serve here — a
            // *submission* failure never produces one.
            let result = match self.core.tensor(node.name, node.descriptor, node.buffers) {
                Ok(mut op) => drain_file(&mut op, &mut self.inner),
                Err(e) => Err(e),
            };
            if let Err(e) = result {
                // A group that began and did not finish leaves a head with fewer than
                // `member_count` members after it, which `file-format.md § Composite
                // Tensors` calls invalid. No `finish` can rescue that, so the writer must
                // not go on to write a clean index over it.
                // `index > 0` is exactly "an earlier node completed": the loop returns on
                // the first failure, so every node before this one is on the sink.
                if index > 0 {
                    self.core.poison(Unusable::IncompleteOutput);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Writes the KV section, footer index, and trailer, then flushes.
    pub(crate) fn finish(mut self, kv: Vec<(String, KvValue)>) -> Result<()> {
        let mut op = self.core.finish(kv)?;
        drain_file(&mut op, &mut self.inner)?;
        self.inner.flush().map_err(Error::Io)
    }
}

/// Writes every byte of one submitted file item to `sink`. See [`drain_stream`].
fn drain_file<W: Write>(op: &mut FileWriteOp<'_>, sink: &mut W) -> Result<()> {
    while let WriteStep::Output { .. } = op.step()? {
        let n = {
            let src = op.output_src();
            loop {
                match sink.write(src) {
                    Ok(n) => break n,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::Io(e)),
                }
            }
        };
        if n == 0 {
            return Err(op.report_sink_closed());
        }
        op.report_written(n)?;
    }
    Ok(())
}
