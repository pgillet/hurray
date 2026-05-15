use bytes::BytesMut;
use tokio::io::{AsyncRead, AsyncReadExt};

use hurray_core::TensorDescriptor;

use crate::{Error, Result};

// magic(4) + version(1) + flags(1) + descriptor_length(4)
const PREFIX_LEN: usize = 10;

/// Reads a single [`TensorDescriptor`] from `reader`.
///
/// Returns `Ok(None)` on a clean EOF (zero bytes available before any descriptor byte).
/// Returns `Err(Error::UnexpectedEof)` if the stream ends mid-descriptor.
pub(crate) async fn read_descriptor<R>(
    reader: &mut R,
    max_descriptor_bytes: u64,
) -> Result<Option<TensorDescriptor>>
where
    R: AsyncRead + Unpin,
{
    // A 0-byte read before any descriptor byte is a clean EOF.
    let mut first = [0u8; 1];
    if reader.read(&mut first).await? == 0 {
        return Ok(None);
    }

    let mut prefix = [0u8; PREFIX_LEN];
    prefix[0] = first[0];
    reader
        .read_exact(&mut prefix[1..])
        .await
        .map_err(map_unexpected_eof)?;

    let descriptor_length =
        u32::from_le_bytes(prefix[6..10].try_into().expect("slice is exactly 4 bytes")) as u64;

    if descriptor_length < PREFIX_LEN as u64 {
        return Err(Error::InvalidHeader(format!(
            "descriptor_length {descriptor_length} is below minimum {PREFIX_LEN}"
        )));
    }

    if descriptor_length > max_descriptor_bytes {
        return Err(Error::FrameTooLarge {
            kind: "descriptor",
            value: descriptor_length,
            limit: max_descriptor_bytes,
        });
    }

    let total = descriptor_length as usize;
    let mut buf = BytesMut::with_capacity(total);
    buf.resize(total, 0);
    buf[..PREFIX_LEN].copy_from_slice(&prefix);
    reader
        .read_exact(&mut buf[PREFIX_LEN..])
        .await
        .map_err(map_unexpected_eof)?;

    let desc = TensorDescriptor::decode(&buf)?;
    Ok(Some(desc))
}

pub(crate) fn map_unexpected_eof(e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        Error::UnexpectedEof
    } else {
        Error::Io(e)
    }
}
