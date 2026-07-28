//! Generates the committed golden-vector corpus under `conformance/vectors/`.
//!
//! Run from anywhere in the workspace:
//! ```text
//! cargo run -p hurray-conformance --bin generate-vectors
//! ```
//! The output is deterministic; committing its result is the source of truth that the
//! Rust (`tests/verify.rs`) and Python conformance suites check against.

use std::path::PathBuf;

use hurray_conformance::{
    derive_expect, descriptor_vectors, element_type_name, DescriptorVector, FileTensorExpect,
    FileVector, KvExpect, Manifest, FORMAT_VERSION,
};
use hurray_core::{
    buffer_size_bytes, BufferHandle, DeviceTag, ElementType, LayoutDescriptor, Shape, SyncMode,
    TensorDescriptor, MIN_BUFFER_ALIGNMENT,
};
use hurray_io::file::{FileWriter, KvValue};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vectors")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = vectors_dir();
    std::fs::create_dir_all(root.join("descriptors"))?;
    std::fs::create_dir_all(root.join("files"))?;

    // ── Descriptor vectors ──────────────────────────────────────────────────
    let mut descriptors = Vec::new();
    for nv in descriptor_vectors() {
        let bytes = nv.descriptor.encode()?;
        let rel = format!("descriptors/{}.bin", nv.stem);
        std::fs::write(root.join(&rel), &bytes)?;
        descriptors.push(DescriptorVector {
            file: rel,
            description: nv.description,
            expect: derive_expect(&nv.descriptor),
        });
    }

    // ── File vector: two dense tensors + KV metadata ────────────────────────
    let file_vector = build_basic_file(&root).await?;

    let manifest = Manifest {
        format_version: FORMAT_VERSION.to_string(),
        descriptors,
        files: vec![file_vector],
    };

    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(root.join("manifest.json"), json + "\n")?;

    println!(
        "Wrote {} descriptor vector(s) + {} file vector(s) to {}",
        manifest.descriptors.len(),
        manifest.files.len(),
        root.display()
    );
    Ok(())
}

fn cpu_buf(byte_size: u64) -> BufferHandle {
    BufferHandle::new(
        byte_size,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .unwrap()
}

fn dense_desc(et: ElementType, dims: Vec<u64>, num_elements: u64) -> TensorDescriptor {
    TensorDescriptor::new(
        1,
        0,
        et,
        Shape::new(dims).unwrap(),
        0,
        LayoutDescriptor::RowMajor,
        vec![cpu_buf(buffer_size_bytes(et, num_elements))],
        None,
        None,
        None,
        None,
    )
    .unwrap()
}

async fn build_basic_file(
    root: &std::path::Path,
) -> Result<FileVector, Box<dyn std::error::Error>> {
    let weights = dense_desc(ElementType::Float32, vec![2, 2], 4);
    let bias = dense_desc(ElementType::Int8, vec![3], 3);
    let w_data = vec![0u8; 16]; // 4 × float32
    let b_data = vec![0u8; 3]; // 3 × int8

    let out = Vec::<u8>::new();
    let mut writer = FileWriter::new(out).await?;
    writer.write_tensor("weights", &weights, &[&w_data]).await?;
    writer.write_tensor("bias", &bias, &[&b_data]).await?;
    let bytes = writer
        .finish(vec![
            (
                "producer".to_string(),
                KvValue::String("hurray-conformance".to_string()),
            ),
            ("tensor_count".to_string(), KvValue::Int64(2)),
        ])
        .await?;

    std::fs::write(root.join("files/basic.hrry"), &bytes)?;

    Ok(FileVector {
        file: "files/basic.hrry".to_string(),
        description: "Two dense tensors (float32 [2,2], int8 [3]) with string + int64 KV"
            .to_string(),
        tensors: vec![
            FileTensorExpect {
                name: "weights".to_string(),
                element_type: element_type_name(ElementType::Float32),
                element_type_tag: ElementType::Float32.tag(),
                shape: vec![2, 2],
                data_bytes: 16,
            },
            FileTensorExpect {
                name: "bias".to_string(),
                element_type: element_type_name(ElementType::Int8),
                element_type_tag: ElementType::Int8.tag(),
                shape: vec![3],
                data_bytes: 3,
            },
        ],
        kv: vec![
            KvExpect {
                key: "producer".to_string(),
                kind: "string".to_string(),
                value_string: Some("hurray-conformance".to_string()),
                value_int: None,
            },
            KvExpect {
                key: "tensor_count".to_string(),
                kind: "int64".to_string(),
                value_string: None,
                value_int: Some(2),
            },
        ],
    })
}
