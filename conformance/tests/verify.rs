//! Conformance check: decode every committed golden vector and assert it matches the
//! committed `manifest.json`. This validates the reference decoder against fixed bytes —
//! the same corpus the Python binding's test suite consumes.
//!
//! Reads only the committed artifacts (no generator code), so it catches any drift between
//! the bytes and the manifest.

use std::io::Cursor;
use std::path::PathBuf;

use hurray_conformance::{derive_expect, element_type_name, FileTensorExpect, KvExpect, Manifest};
use hurray_core::TensorDescriptor;
use hurray_io::file::{FileReader, KvValue};

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vectors")
}

fn load_manifest() -> Manifest {
    let path = vectors_dir().join("manifest.json");
    let json =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&json).expect("manifest.json is valid")
}

#[test]
fn manifest_targets_expected_format_version() {
    assert_eq!(load_manifest().format_version, "1.0");
}

#[test]
fn descriptor_vectors_decode_to_manifest() {
    let manifest = load_manifest();
    let root = vectors_dir();
    assert!(
        !manifest.descriptors.is_empty(),
        "corpus has no descriptor vectors"
    );

    for v in &manifest.descriptors {
        let bytes =
            std::fs::read(root.join(&v.file)).unwrap_or_else(|e| panic!("read {}: {e}", v.file));
        let desc = TensorDescriptor::decode(&bytes)
            .unwrap_or_else(|e| panic!("decode {} ({}): {e:?}", v.file, v.description));
        let observed = derive_expect(&desc);
        assert_eq!(
            observed, v.expect,
            "vector {} ({}) decoded to unexpected properties",
            v.file, v.description
        );
    }
}

#[tokio::test]
async fn file_vectors_match_manifest() {
    let manifest = load_manifest();
    let root = vectors_dir();
    assert!(!manifest.files.is_empty(), "corpus has no file vectors");

    for fv in &manifest.files {
        let bytes =
            std::fs::read(root.join(&fv.file)).unwrap_or_else(|e| panic!("read {}: {e}", fv.file));
        let mut reader = FileReader::open(Cursor::new(bytes))
            .await
            .unwrap_or_else(|e| panic!("open {} ({}): {e:?}", fv.file, fv.description));

        // Tensors, in index order.
        let observed_names: Vec<String> = reader.tensor_names().map(|s| s.to_string()).collect();
        let expected_names: Vec<String> = fv.tensors.iter().map(|t| t.name.clone()).collect();
        assert_eq!(observed_names, expected_names, "{}: tensor names", fv.file);

        for expected in &fv.tensors {
            let t = reader
                .read_tensor(&expected.name)
                .await
                .unwrap_or_else(|e| panic!("read_tensor {}: {e:?}", expected.name));
            let data_bytes: u64 = t.buffers.iter().map(|b| b.len() as u64).sum();
            let observed = FileTensorExpect {
                name: t.name.clone(),
                element_type: element_type_name(t.descriptor.element_type),
                element_type_tag: t.descriptor.element_type.tag(),
                shape: t.descriptor.shape.dims().to_vec(),
                data_bytes,
            };
            assert_eq!(
                observed, *expected,
                "{}: tensor {} properties",
                fv.file, expected.name
            );
        }

        // Key-value metadata, in order.
        let observed_kv: Vec<KvExpect> = reader
            .kv()
            .iter()
            .map(|(key, value)| match value {
                KvValue::String(s) => KvExpect {
                    key: key.clone(),
                    kind: "string".to_string(),
                    value_string: Some(s.clone()),
                    value_int: None,
                },
                KvValue::Int64(i) => KvExpect {
                    key: key.clone(),
                    kind: "int64".to_string(),
                    value_string: None,
                    value_int: Some(*i),
                },
                other => panic!("{}: unexpected KV value kind {other:?}", fv.file),
            })
            .collect();
        assert_eq!(observed_kv, fv.kv, "{}: KV metadata", fv.file);
    }
}
