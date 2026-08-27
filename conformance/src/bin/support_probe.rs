//! Reports, as JSON, what the Rust reference implementation actually supports.
//!
//! Run from anywhere in the workspace:
//! ```text
//! cargo run -q -p hurray-conformance --bin support-probe
//! ```
//!
//! Consumed by `website/check-coverage-matrix.py`, which turns this report — plus the
//! Python module's surface and the C header's exported symbols — into the Implementation
//! Status page. Nothing here is a hand-kept list: tag coverage is read back out of the
//! `ElementType` / layout-tag / quantization-scheme decoders one byte at a time, and every
//! capability is proven by an actual encode → decode round-trip rather than by asserting
//! that a type exists. A claim on the published page can therefore only be wrong if the
//! round-trip that backs it is wrong too.
//!
//! Adding a capability here without adding its row to `website/coverage-matrix.toml` fails
//! the checker: an orphan probe result is drift in the same way a stale cell is.

use hurray_core::{
    buffer_size_bytes,
    descriptor::{
        CompositeMemberDescriptor, ExtensionTypeDescriptor, MemberRole, Statistics, StatisticsMask,
    },
    layout::{self, CompositeLayout, CompositionRule},
    BufferHandle, DeviceTag, ElementType, LayoutDescriptor, PerTensorAffine,
    QuantizationDescriptor, QuantizationSchemeTag, Shape, ShardDescriptor, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::{
    file::{FileCompositeNode, FileReader, FileWriter, KvValue},
    stream::{CompositeNode, StreamItem, StreamReader, StreamWriter},
};
use serde_json::{json, Map, Value};

type Failure = Box<dyn std::error::Error>;

fn main() -> Result<(), Failure> {
    let report = json!({
        "element_types": element_types(),
        "layout_tags": layout_tags(),
        "quantization_scheme_tags": quantization_scheme_tags(),
        "capabilities": capabilities()?,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

// ── Tag-space probes ─────────────────────────────────────────────────────────

/// Every element-type tag the decoder accepts, mapped to its canonical spec name.
///
/// The extension range `0xF0`–`0xFE` decodes to `ElementType::Extension`, which names no
/// single type — it is reported as the `extension_types` capability instead, so the map
/// holds only concrete types.
fn element_types() -> Map<String, Value> {
    let mut out = Map::new();
    for tag in 0x00..=0xFFu8 {
        if let Ok(ty) = ElementType::from_tag(tag) {
            if !matches!(ty, ElementType::Extension(_)) {
                out.insert(format!("0x{tag:02X}"), Value::from(ty.to_string()));
            }
        }
    }
    out
}

/// Every named layout tag the implementation knows — `is_named_tag` is the one list the
/// strict validator and the layout decoder both dispatch on.
fn layout_tags() -> Vec<String> {
    (0x00..=0xFFu8)
        .filter(|&tag| layout::is_named_tag(tag))
        .map(|tag| format!("0x{tag:02X}"))
        .collect()
}

/// Every quantization scheme tag `QuantizationSchemeTag::from_byte` resolves to a scheme.
fn quantization_scheme_tags() -> Vec<String> {
    (0x00..=0xFFu8)
        .filter(|&tag| QuantizationSchemeTag::from_byte(tag).is_ok())
        .map(|tag| format!("0x{tag:02X}"))
        .collect()
}

// ── Capability probes ────────────────────────────────────────────────────────

/// Runs every capability round-trip and records whether it succeeded.
///
/// A probe reports `false` on any failure rather than aborting: a report that stops at the
/// first gap cannot describe the gap, and describing it is the whole point of the page.
fn capabilities() -> Result<Map<String, Value>, Failure> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let mut out = Map::new();
    let mut record = |name: &str, outcome: Result<(), Failure>| {
        out.insert(name.to_string(), Value::from(outcome.is_ok()));
    };

    record("descriptor_encode", descriptor_encode());
    record("descriptor_decode", descriptor_decode());
    record("section_quantization", section_quantization());
    record("section_shard", section_shard());
    record("section_statistics", section_statistics());
    record("section_extension_type", section_extension_type());
    record("section_composite_member", section_composite_member());
    record("stream_write", runtime.block_on(stream_write()));
    record("stream_read", runtime.block_on(stream_read()));
    record("stream_composite", runtime.block_on(stream_composite()));
    record("file_write", runtime.block_on(file_write()));
    record("file_read", runtime.block_on(file_read()));
    record("file_kv", runtime.block_on(file_kv()));
    record("file_composite", runtime.block_on(file_composite()));
    Ok(out)
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

fn buffer(byte_size: u64) -> Result<BufferHandle, Failure> {
    Ok(BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )?)
}

/// A `[2, 3]` row-major tensor of `element_type`, with a correctly sized buffer handle.
fn dense(element_type: ElementType) -> Result<TensorDescriptor, Failure> {
    Ok(TensorDescriptor::new(
        1,
        0,
        element_type,
        Shape::new(vec![2u64, 3])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer(buffer_size_bytes(element_type, 6))?],
        None,
        None,
        None,
        None,
    )?)
}

/// Encodes `descriptor`, decodes the bytes back, and requires the result to be identical.
fn round_trip(descriptor: &TensorDescriptor) -> Result<TensorDescriptor, Failure> {
    let decoded = TensorDescriptor::decode(&descriptor.encode()?)?;
    if &decoded != descriptor {
        return Err("descriptor changed across an encode/decode round-trip".into());
    }
    Ok(decoded)
}

// ── Descriptor and its optional sections ─────────────────────────────────────

fn descriptor_encode() -> Result<(), Failure> {
    let bytes = dense(ElementType::Float32)?.encode()?;
    if bytes.is_empty() {
        return Err("encode produced no bytes".into());
    }
    Ok(())
}

fn descriptor_decode() -> Result<(), Failure> {
    round_trip(&dense(ElementType::Float32)?).map(|_| ())
}

fn section_quantization() -> Result<(), Failure> {
    // int8 storage: the affine schemes reject a float storage type, so f32 would prove
    // nothing about the section and everything about the validator.
    let mut descriptor = dense(ElementType::Int8)?;
    let scheme = QuantizationDescriptor::PerTensorAffine(PerTensorAffine::new(0.125, -7)?);
    descriptor.quantization = Some(scheme.encode_to_vec());
    let decoded = round_trip(&descriptor)?;
    let payload = decoded.quantization.ok_or("quantization section lost")?;
    let (recovered, _) = QuantizationDescriptor::decode(&payload)?;
    if recovered != scheme {
        return Err("quantization descriptor changed across a round-trip".into());
    }
    Ok(())
}

fn section_shard() -> Result<(), Failure> {
    let mut descriptor = dense(ElementType::Float32)?;
    descriptor.shard = Some(ShardDescriptor::new(vec![4, 3], vec![2, 0])?);
    round_trip(&descriptor).map(|_| ())
}

fn section_statistics() -> Result<(), Failure> {
    let mut descriptor = dense(ElementType::Float32)?;
    descriptor.statistics = Some(Statistics {
        computed_mask: StatisticsMask(StatisticsMask::NNZ_VALID),
        nnz: 5,
        sparsity_ratio: 0.0,
        value_min: 0.0,
        value_max: 0.0,
        value_abs_max: 0.0,
        value_mean: 0.0,
        value_stddev: 0.0,
        nm_n: 0,
        nm_m: 0,
        has_nan: false,
        has_inf: false,
    });
    round_trip(&descriptor).map(|_| ())
}

/// An extension element type and its describing section. One probe covers both: the two
/// are inseparable on the wire, where `HAS_EXTENSION_TYPE` is set iff the type tag falls
/// in `0xF0`–`0xFE`.
fn section_extension_type() -> Result<(), Failure> {
    let element_type = ElementType::from_tag(0xF0)?;
    let descriptor = TensorDescriptor::new(
        1,
        0,
        element_type,
        Shape::new(vec![2u64, 3])?,
        0,
        LayoutDescriptor::RowMajor,
        vec![buffer(6)?],
        None,
        None,
        None,
        Some(ExtensionTypeDescriptor::new(
            8, 1, false, true, 0, 0, 0, 0, false, false,
        )?),
    )?;
    round_trip(&descriptor).map(|_| ())
}

fn section_composite_member() -> Result<(), Failure> {
    let descriptor = dense(ElementType::Float32)?
        .with_composite_member(CompositeMemberDescriptor::new(MemberRole::Correction));
    round_trip(&descriptor).map(|_| ())
}

// ── Streaming interchange ────────────────────────────────────────────────────

/// Writes one tensor to an in-memory wire and returns the bytes.
async fn stream_bytes() -> Result<(Vec<u8>, TensorDescriptor, Vec<u8>), Failure> {
    let descriptor = dense(ElementType::Float32)?;
    let data = vec![0xA5u8; 24];
    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_tensor(&descriptor, &[&data]).await?;
    writer.finish().await?;
    Ok((wire, descriptor, data))
}

async fn stream_write() -> Result<(), Failure> {
    let (wire, _, _) = stream_bytes().await?;
    if wire.is_empty() {
        return Err("stream writer emitted no bytes".into());
    }
    Ok(())
}

async fn stream_read() -> Result<(), Failure> {
    let (wire, descriptor, data) = stream_bytes().await?;
    let mut reader = StreamReader::new(wire.as_slice());
    let tensor = reader
        .next_tensor()
        .await?
        .ok_or("stream reader returned no tensor")?;
    if tensor.descriptor != descriptor {
        return Err("descriptor changed across the stream".into());
    }
    if tensor.buffers.first().map(|b| b.as_ref()) != Some(data.as_slice()) {
        return Err("buffer bytes changed across the stream".into());
    }
    Ok(())
}

/// A partition composite: a `[4, 4]` head over two `[4, 2]` members that exactly cover it.
fn composite_parts() -> Result<(TensorDescriptor, [TensorDescriptor; 2], Vec<u8>), Failure> {
    let head = TensorDescriptor::new(
        1,
        0,
        ElementType::Float32,
        Shape::new(vec![4u64, 4])?,
        0,
        LayoutDescriptor::Composite(CompositeLayout::new(CompositionRule::Partition, 2)?),
        vec![],
        None,
        None,
        None,
        None,
    )?;
    let tile = |column: u64| -> Result<TensorDescriptor, Failure> {
        Ok(TensorDescriptor::new(
            1,
            0,
            ElementType::Float32,
            Shape::new(vec![4u64, 2])?,
            0,
            LayoutDescriptor::RowMajor,
            vec![buffer(buffer_size_bytes(ElementType::Float32, 8))?],
            None,
            Some(ShardDescriptor::new(vec![4, 4], vec![0, column])?),
            None,
            None,
        )?)
    };
    Ok((head, [tile(0)?, tile(2)?], vec![0x5Au8; 32]))
}

async fn stream_composite() -> Result<(), Failure> {
    let (head, tiles, data) = composite_parts()?;
    let buffers: [&[u8]; 1] = [data.as_slice()];
    let members = vec![
        CompositeNode::Tensor {
            descriptor: &tiles[0],
            buffers: &buffers,
        },
        CompositeNode::Tensor {
            descriptor: &tiles[1],
            buffers: &buffers,
        },
    ];

    let mut wire = Vec::<u8>::new();
    let mut writer = StreamWriter::new(&mut wire);
    writer.write_composite(&head, &members).await?;
    writer.finish().await?;

    let mut reader = StreamReader::new(wire.as_slice());
    match reader.next_item().await? {
        Some(StreamItem::Composite(group)) => {
            if group.members.len() != 2 {
                return Err("composite lost a member across the stream".into());
            }
            Ok(())
        }
        _ => Err("stream reader did not reassemble the composite".into()),
    }
}

// ── File container ───────────────────────────────────────────────────────────

/// Writes one named tensor plus `kv` into an in-memory HRRYFILE.
async fn file_bytes(kv: Vec<(String, KvValue)>) -> Result<(Vec<u8>, TensorDescriptor), Failure> {
    let descriptor = dense(ElementType::Float32)?;
    let data = vec![0x3Cu8; 24];
    let mut writer = FileWriter::new(Vec::<u8>::new()).await?;
    writer
        .write_tensor("weights", &descriptor, &[&data])
        .await?;
    Ok((writer.finish(kv).await?, descriptor))
}

async fn file_write() -> Result<(), Failure> {
    let (bytes, _) = file_bytes(vec![]).await?;
    if bytes.is_empty() {
        return Err("file writer emitted no bytes".into());
    }
    Ok(())
}

async fn file_read() -> Result<(), Failure> {
    let (bytes, descriptor) = file_bytes(vec![]).await?;
    let mut reader = FileReader::open(std::io::Cursor::new(bytes)).await?;
    let tensor = reader.read_tensor("weights").await?;
    if tensor.descriptor != descriptor {
        return Err("descriptor changed across the file".into());
    }
    Ok(())
}

async fn file_kv() -> Result<(), Failure> {
    let kv = vec![("model".to_string(), KvValue::String("probe".to_string()))];
    let (bytes, _) = file_bytes(kv.clone()).await?;
    let reader = FileReader::open(std::io::Cursor::new(bytes)).await?;
    if reader.kv() != kv.as_slice() {
        return Err("KV metadata changed across the file".into());
    }
    Ok(())
}

async fn file_composite() -> Result<(), Failure> {
    let (head, tiles, data) = composite_parts()?;
    let buffers: [&[u8]; 1] = [data.as_slice()];
    let members = vec![
        FileCompositeNode::Tensor {
            name: "tile.0",
            descriptor: &tiles[0],
            buffers: &buffers,
        },
        FileCompositeNode::Tensor {
            name: "tile.1",
            descriptor: &tiles[1],
            buffers: &buffers,
        },
    ];

    let mut writer = FileWriter::new(Vec::<u8>::new()).await?;
    writer.write_composite("tiles", &head, &members).await?;
    let bytes = writer.finish(vec![]).await?;

    let mut reader = FileReader::open(std::io::Cursor::new(bytes)).await?;
    let group = reader.read_composite("tiles").await?;
    if group.members.len() != 2 {
        return Err("composite lost a member across the file".into());
    }
    Ok(())
}
