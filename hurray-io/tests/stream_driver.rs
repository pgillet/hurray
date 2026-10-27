#![cfg(feature = "tokio")]

//! What the tokio drivers must do with an awkward transport.
//!
//! The drivers answer the core one transport call at a time — `read`, not `read_exact`;
//! `write`, not `write_all` — because a single call reports its own byte count and a
//! bulk helper does not (ADR-041 § 4). That choice is only correct if the drivers handle
//! everything the bulk helpers used to absorb: short transfers, `Interrupted`, a closed
//! sink, and a mid-item failure.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode, TensorDescriptor,
    MIN_BUFFER_ALIGNMENT,
};
use hurray_io::stream::{StreamReader, StreamWriter, Unusable};
use hurray_io::Error;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

// ── transports ────────────────────────────────────────────────────────────────

/// Hands out at most `max` bytes per read, however much was asked for.
struct Trickle<'a> {
    data: &'a [u8],
    max: usize,
}

impl AsyncRead for Trickle<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        let data = me.data;
        let n = me.max.min(buf.remaining()).min(data.len());
        buf.put_slice(&data[..n]);
        me.data = &data[n..];
        Poll::Ready(Ok(()))
    }
}

/// Fails with `Interrupted` the first `interrupts` times, then reads normally.
struct Interrupting<'a> {
    data: &'a [u8],
    interrupts: usize,
}

impl AsyncRead for Interrupting<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        if me.interrupts > 0 {
            me.interrupts -= 1;
            return Poll::Ready(Err(io::Error::new(io::ErrorKind::Interrupted, "simulated")));
        }
        let data = me.data;
        let n = buf.remaining().min(data.len());
        buf.put_slice(&data[..n]);
        me.data = &data[n..];
        Poll::Ready(Ok(()))
    }
}

/// Accepts at most `max` bytes per write.
#[derive(Default)]
struct ShortSink {
    written: Vec<u8>,
    max: usize,
}

impl AsyncWrite for ShortSink {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        let n = me.max.min(buf.len());
        me.written.extend_from_slice(&buf[..n]);
        Poll::Ready(Ok(n))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Accepts nothing, without raising an error: a sink that has gone away.
struct ClosedSink;

impl AsyncWrite for ClosedSink {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(Ok(0))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Fails with `Interrupted` the first `interrupts` times, then accepts everything.
struct InterruptingSink {
    written: Vec<u8>,
    interrupts: usize,
}

impl AsyncWrite for InterruptingSink {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        if me.interrupts > 0 {
            me.interrupts -= 1;
            return Poll::Ready(Err(io::Error::new(io::ErrorKind::Interrupted, "simulated")));
        }
        me.written.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Fails the first write with an error, then accepts everything. Unlike
/// `InterruptingSink`, the error is not `Interrupted`, so the driver surfaces it instead
/// of retrying — which is what makes this the "failed before the first byte" case.
struct FailOnce {
    written: Vec<u8>,
    failed: bool,
}

impl AsyncWrite for FailOnce {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        if !me.failed {
            me.failed = true;
            return Poll::Ready(Err(io::Error::other("simulated")));
        }
        me.written.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Takes `limit` bytes, then fails every write.
struct FailAfter {
    accepted: usize,
    limit: usize,
}

impl AsyncWrite for FailAfter {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        if me.accepted >= me.limit {
            return Poll::Ready(Err(io::Error::other("simulated transport failure")));
        }
        let n = buf.len().min(me.limit - me.accepted);
        me.accepted += n;
        Poll::Ready(Ok(n))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

// ── fixtures ──────────────────────────────────────────────────────────────────

fn desc() -> TensorDescriptor {
    let handle = BufferHandle::new(
        256,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap();
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![handle],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

async fn one_tensor_wire(payload: &[u8]) -> Vec<u8> {
    let d = desc();
    let buffers: [&[u8]; 1] = [payload];
    let mut wire = Vec::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&d, &buffers).await.unwrap();
    writer.finish().await.unwrap();
    wire
}

// ── read side ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_trickling_transport_still_yields_the_whole_tensor() {
    let payload: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let wire = one_tensor_wire(&payload).await;

    for max in [1usize, 3, 7, 64] {
        let mut reader = StreamReader::new(Trickle { data: &wire, max });
        let tensor = reader.next_tensor().await.unwrap().expect("one tensor");
        assert_eq!(tensor.buffers[0].as_ref(), &payload[..], "max {max}");
        assert!(
            reader.next_tensor().await.unwrap().is_none(),
            "clean EOF at max {max}"
        );
    }
}

/// `read_exact` absorbed `Interrupted` internally; a per-call loop has to do it by hand,
/// or a transport that yields it would surface as a read failure.
#[tokio::test]
async fn an_interrupted_read_is_retried_not_surfaced() {
    let payload = vec![0xA5u8; 256];
    let wire = one_tensor_wire(&payload).await;

    let mut reader = StreamReader::new(Interrupting {
        data: &wire,
        interrupts: 3,
    });
    let tensor = reader.next_tensor().await.unwrap().expect("one tensor");
    assert_eq!(tensor.buffers[0].as_ref(), &payload[..]);
}

#[tokio::test]
async fn a_truncated_stream_is_an_unexpected_eof_not_a_clean_one() {
    let payload = vec![0x5Au8; 256];
    let wire = one_tensor_wire(&payload).await;

    // One byte short of the payload: end of input mid-item.
    let mut reader = StreamReader::new(Trickle {
        data: &wire[..wire.len() - 1],
        max: 8,
    });
    assert!(matches!(
        reader.next_tensor().await,
        Err(Error::UnexpectedEof)
    ));
}

// ── write side ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_short_sink_receives_every_byte_in_order() {
    let payload: Vec<u8> = (0..256).map(|i| (i * 7 % 256) as u8).collect();
    let expected = one_tensor_wire(&payload).await;

    for max in [1usize, 5, 100] {
        let d = desc();
        let buffers: [&[u8]; 1] = [&payload];
        let mut writer = StreamWriter::new(ShortSink {
            written: Vec::new(),
            max,
        });
        writer.write_tensor(&d, &buffers).await.unwrap();
        let sink = writer.finish().await.unwrap();
        assert_eq!(sink.written, expected, "max {max}");
    }
}

/// `write_all` absorbed `Interrupted` internally, same as `read_exact` did.
#[tokio::test]
async fn an_interrupted_write_is_retried_not_surfaced() {
    let payload: Vec<u8> = (0..256).map(|i| i as u8).collect();
    let expected = one_tensor_wire(&payload).await;

    let d = desc();
    let buffers: [&[u8]; 1] = [&payload];
    let mut writer = StreamWriter::new(InterruptingSink {
        written: Vec::new(),
        interrupts: 3,
    });
    writer.write_tensor(&d, &buffers).await.unwrap();
    let sink = writer.finish().await.unwrap();
    assert_eq!(sink.written, expected);
}

/// § 4: a zero-length write is never progress. A sink that accepts nothing is its own
/// signal, and since a partial item is already on the wire it poisons the writer — a
/// second attempt over the same connection is not a recovery (§ 6a).
#[tokio::test]
async fn a_closed_sink_is_reported_and_poisons_the_writer() {
    let d = desc();
    let payload = vec![0u8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    let mut writer = StreamWriter::new(ClosedSink);
    assert!(matches!(
        writer.write_tensor(&d, &buffers).await,
        Err(Error::SinkClosed)
    ));
    assert!(matches!(
        writer.write_tensor(&d, &buffers).await,
        Err(Error::CoreUnusable(_))
    ));
}

/// A write that fails *after* its first byte leaves a truncated item on the wire, and a
/// stream has no resynchronisation point, so the writer is poisoned (§ 6b). The failing
/// call still reports the I/O error that caused it — the poison surfaces on the next call.
#[tokio::test]
async fn a_write_that_fails_after_its_first_byte_poisons_the_writer() {
    let d = desc();
    let payload = vec![0u8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    // Enough for the descriptor, not enough for the payload.
    let limit = d.encode().unwrap().len() + 64;
    let mut writer = StreamWriter::new(FailAfter { accepted: 0, limit });
    assert!(matches!(
        writer.write_tensor(&d, &buffers).await,
        Err(Error::Io(_))
    ));
    assert!(matches!(
        writer.write_tensor(&d, &buffers).await,
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
}

/// The other half of § 6b: a write that fails *before* its first byte emitted nothing, so
/// there is nothing to un-send. The writer stays usable, the same tensor may be written
/// again, and the wire is byte-identical to one written with no failure at all.
#[tokio::test]
async fn a_write_that_fails_before_its_first_byte_is_retryable() {
    let d = desc();
    let payload = vec![0u8; 256];
    let buffers: [&[u8]; 1] = [&payload];

    let mut writer = StreamWriter::new(FailOnce {
        written: Vec::new(),
        failed: false,
    });
    // The very first write of the item fails, so nothing reached the sink.
    assert!(matches!(
        writer.write_tensor(&d, &buffers).await,
        Err(Error::Io(_))
    ));
    writer.write_tensor(&d, &buffers).await.unwrap();
    let retried = writer.finish().await.unwrap().written;

    let mut clean = StreamWriter::new(Vec::new());
    clean.write_tensor(&d, &buffers).await.unwrap();
    assert_eq!(retried, clean.finish().await.unwrap());
}
