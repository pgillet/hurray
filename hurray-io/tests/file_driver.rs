//! The tokio `FileWriter` against sinks that fail, pinning ADR-041 § 6b at the driver
//! level.
//!
//! The sans-io tests in `sans_io_file_core.rs` cover the rule on the core. These cover the
//! driver, because the driver is where the rule was previously *lost*: a mid-item I/O error
//! returned without poisoning the core, so a later `finish` could write a clean footer
//! index over a torn tensor region (issue #248).

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use hurray_core::{
    buffer_size_bytes, BufferHandle, CompositeLayout, CompositionRule, DeviceTag, ElementType,
    LayoutDescriptor, Shape, ShardDescriptor, SyncMode, TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileCompositeNode, FileWriter};
use hurray_io::stream::Unusable;
use hurray_io::Error;
use tokio::io::AsyncWrite;

fn buf(byte_size: u64, alignment: u32) -> BufferHandle {
    BufferHandle::new(byte_size, alignment, DeviceTag::Cpu, SyncMode::ProducerSynced).unwrap()
}

fn row_major() -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(256, MIN_BUFFER_ALIGNMENT)],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn composite_head(member_count: u32) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 8]).unwrap(),
        0,
        LayoutDescriptor::Composite(
            CompositeLayout::new(CompositionRule::Partition, member_count).unwrap(),
        ),
        vec![],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

fn partition_member(offset: u64) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![8u64, 4]).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![buf(
            buffer_size_bytes(ElementType::Float32, 32),
            MIN_BUFFER_ALIGNMENT,
        )],
        None,
        Some(ShardDescriptor::new(vec![8, 8], vec![0, offset]).unwrap()),
        None,
        None,
    )
    .unwrap()
}

/// Accepts `accept` bytes, then fails every write until `rearm` is called. The error is
/// not `Interrupted`, so the driver surfaces it rather than retrying.
struct FailAfter {
    written: Vec<u8>,
    accepted: usize,
    limit: usize,
}

impl FailAfter {
    fn new(limit: usize) -> Self {
        Self {
            written: Vec::new(),
            accepted: 0,
            limit,
        }
    }
}

impl AsyncWrite for FailAfter {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        if me.accepted >= me.limit {
            // Unlimited from here: the test only needs the first failure.
            me.limit = usize::MAX;
            return Poll::Ready(Err(io::Error::other("simulated")));
        }
        let take = buf.len().min(me.limit - me.accepted);
        me.written.extend_from_slice(&buf[..take]);
        me.accepted += take;
        Poll::Ready(Ok(take))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

/// Issue #248 through the driver: a failure before the tensor's first byte leaves the file
/// completable under the same name, and the result is byte-identical to a clean write.
#[tokio::test]
async fn a_write_that_fails_before_its_first_byte_is_retryable() {
    let d = row_major();
    let data = vec![0x11u8; 256];
    let buffers: [&[u8]; 1] = [&data];

    // 64 bytes is exactly the header, so the tensor's own first write is the one that
    // fails.
    let mut writer = FileWriter::new(FailAfter::new(64)).await.unwrap();
    assert!(matches!(
        writer.write_tensor("weight", &d, &buffers).await,
        Err(Error::Io(_))
    ));
    writer.write_tensor("weight", &d, &buffers).await.unwrap();
    let retried = writer.finish(vec![]).await.unwrap().written;

    let mut clean = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    clean.write_tensor("weight", &d, &buffers).await.unwrap();
    assert_eq!(retried, clean.finish(vec![]).await.unwrap());
}

/// The bug the issue did not name: a mid-item failure used to leave the core usable, so
/// `finish` wrote a clean footer index over a torn tensor region — a self-consistent index
/// on a non-conforming file, with no error raised anywhere.
#[tokio::test]
async fn a_write_that_fails_mid_item_poisons_the_writer_and_refuses_the_footer() {
    let d = row_major();
    let data = vec![0x11u8; 256];
    let buffers: [&[u8]; 1] = [&data];

    // Past the header and into the tensor, so bytes of the item are already out.
    let mut writer = FileWriter::new(FailAfter::new(96)).await.unwrap();
    assert!(matches!(
        writer.write_tensor("weight", &d, &buffers).await,
        Err(Error::Io(_))
    ));
    // The failing call reports the I/O error; the poison surfaces on the next call.
    assert!(matches!(
        writer.write_tensor("other", &d, &buffers).await,
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
    assert!(matches!(
        writer.finish(vec![]).await,
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
}

/// A composite is a group, not an item: a head on the sink with fewer than `member_count`
/// members after it is an invalid file (`file-format.md § Composite Tensors`), and no
/// `finish` can rescue it. The member here fails *before its first byte*, so nothing but
/// the group rule makes this terminal.
#[tokio::test]
async fn a_composite_whose_member_fails_after_the_head_poisons_the_writer() {
    let head = composite_head(2);
    let (m0, m1) = (partition_member(0), partition_member(4));
    let d0 = vec![0xAAu8; 128];
    let d1 = vec![0xBBu8; 128];
    let b0: [&[u8]; 1] = [&d0];
    let b1: [&[u8]; 1] = [&d1];
    let members = vec![
        FileCompositeNode::Tensor {
            name: "w.tile0",
            descriptor: &m0,
            buffers: &b0,
        },
        FileCompositeNode::Tensor {
            name: "w.tile1",
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    // Enough for the header and the whole head descriptor, nothing more: the head commits
    // and the first member fails before any of its bytes go out.
    let head_len = head.encode().unwrap().len();
    let mut writer = FileWriter::new(FailAfter::new(64 + head_len)).await.unwrap();
    assert!(writer.write_composite("w", &head, &members).await.is_err());

    assert!(matches!(
        writer.finish(vec![]).await,
        Err(Error::CoreUnusable(Unusable::IncompleteOutput))
    ));
}

/// The other side of the group rule: if the *head* fails before its first byte, nothing of
/// the group reached the sink, so the whole composite may be written again.
#[tokio::test]
async fn a_composite_whose_head_fails_before_its_first_byte_is_retryable() {
    let head = composite_head(2);
    let (m0, m1) = (partition_member(0), partition_member(4));
    let d0 = vec![0xAAu8; 128];
    let d1 = vec![0xBBu8; 128];
    let b0: [&[u8]; 1] = [&d0];
    let b1: [&[u8]; 1] = [&d1];
    let members = vec![
        FileCompositeNode::Tensor {
            name: "w.tile0",
            descriptor: &m0,
            buffers: &b0,
        },
        FileCompositeNode::Tensor {
            name: "w.tile1",
            descriptor: &m1,
            buffers: &b1,
        },
    ];

    let mut writer = FileWriter::new(FailAfter::new(64)).await.unwrap();
    assert!(writer.write_composite("w", &head, &members).await.is_err());

    // Every member name is free too, because no member was ever submitted.
    writer.write_composite("w", &head, &members).await.unwrap();
    let retried = writer.finish(vec![]).await.unwrap().written;

    let mut clean = FileWriter::new(Vec::<u8>::new()).await.unwrap();
    clean.write_composite("w", &head, &members).await.unwrap();
    assert_eq!(retried, clean.finish(vec![]).await.unwrap());
}
