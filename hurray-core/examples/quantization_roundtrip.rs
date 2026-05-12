//! Quantization descriptor roundtrip example.
//!
//! Demonstrates encoding and decoding all five quantization schemes: per-tensor
//! affine, per-channel affine (symmetric and asymmetric), per-block affine, NF4,
//! and MXFP. Each scheme is constructed, encoded to bytes, decoded back, and
//! verified for equality.

use hurray_core::{
    BufferHandle, DeviceTag, ElementType, Mxfp, Nf4, PerBlockAffine, PerChannelAffine,
    PerTensorAffine, QuantizationDescriptor, SyncMode, MIN_BUFFER_ALIGNMENT, NF4_LUT,
};

fn main() {
    println!("Quantization Descriptor Roundtrip Examples\n");

    // ── Per-tensor affine (INT8) ────────────────────────────────────────────

    let pta = QuantizationDescriptor::PerTensorAffine(
        PerTensorAffine::new(0.015625, 128).expect("valid per-tensor affine"),
    );

    let pta_encoded = pta.encode_to_vec();
    let (pta_decoded, pta_consumed) =
        QuantizationDescriptor::decode(&pta_encoded).expect("valid per-tensor decode");
    assert_eq!(pta_decoded, pta, "per-tensor roundtrip mismatch");
    assert_eq!(
        pta_consumed,
        pta_encoded.len(),
        "per-tensor consumed mismatch"
    );

    println!("Per-Tensor Affine:");
    println!(
        "  Encoded: {} bytes, scale={:.6}, zero_point=128",
        pta_encoded.len(),
        0.015625f32
    );
    println!("  Roundtrip: ✓\n");

    // ── Per-channel affine (symmetric) ──────────────────────────────────────

    let pca_sym = QuantizationDescriptor::PerChannelAffine(
        PerChannelAffine::new_symmetric(0, 1).expect("valid per-channel symmetric"),
    );

    let pca_sym_encoded = pca_sym.encode_to_vec();
    let (pca_sym_decoded, pca_sym_consumed) = QuantizationDescriptor::decode(&pca_sym_encoded)
        .expect("valid per-channel symmetric decode");
    assert_eq!(
        pca_sym_decoded, pca_sym,
        "per-channel symmetric roundtrip mismatch"
    );
    assert_eq!(
        pca_sym_consumed,
        pca_sym_encoded.len(),
        "per-channel symmetric consumed mismatch"
    );

    println!("Per-Channel Affine (Symmetric):");
    println!(
        "  Encoded: {} bytes, axis=0, scale_buffer=1, zero_point=implicit",
        pca_sym_encoded.len()
    );
    println!("  Roundtrip: ✓\n");

    // ── Per-channel affine (asymmetric) ─────────────────────────────────────

    let pca_asym = QuantizationDescriptor::PerChannelAffine(
        PerChannelAffine::new_asymmetric(1, 2, 3).expect("valid per-channel asymmetric"),
    );

    let pca_asym_encoded = pca_asym.encode_to_vec();
    let (pca_asym_decoded, pca_asym_consumed) = QuantizationDescriptor::decode(&pca_asym_encoded)
        .expect("valid per-channel asymmetric decode");
    assert_eq!(
        pca_asym_decoded, pca_asym,
        "per-channel asymmetric roundtrip mismatch"
    );
    assert_eq!(
        pca_asym_consumed,
        pca_asym_encoded.len(),
        "per-channel asymmetric consumed mismatch"
    );

    println!("Per-Channel Affine (Asymmetric):");
    println!(
        "  Encoded: {} bytes, axis=1, scale_buffer=2, zero_point_buffer=3",
        pca_asym_encoded.len()
    );
    println!("  Roundtrip: ✓\n");

    // ── Per-block affine (QLoRA-style) ──────────────────────────────────────

    let pba = QuantizationDescriptor::PerBlockAffine(
        PerBlockAffine::new_symmetric(2, 64, 4, ElementType::Float32)
            .expect("valid per-block affine"),
    );

    let pba_encoded = pba.encode_to_vec();
    let (pba_decoded, pba_consumed) =
        QuantizationDescriptor::decode(&pba_encoded).expect("valid per-block decode");
    assert_eq!(pba_decoded, pba, "per-block roundtrip mismatch");
    assert_eq!(
        pba_consumed,
        pba_encoded.len(),
        "per-block consumed mismatch"
    );

    println!("Per-Block Affine (Symmetric):");
    println!(
        "  Encoded: {} bytes, axis=2, block_size=64, scale_buffer=4, scale_type=Float32",
        pba_encoded.len()
    );
    println!("  Roundtrip: ✓\n");

    // ── NF4 (NormalFloat4) block quantization ───────────────────────────────

    let nf4 = QuantizationDescriptor::Nf4(Nf4::new(0, 128, 5).expect("valid NF4"));

    let nf4_encoded = nf4.encode_to_vec();
    let (nf4_decoded, nf4_consumed) =
        QuantizationDescriptor::decode(&nf4_encoded).expect("valid NF4 decode");
    assert_eq!(nf4_decoded, nf4, "NF4 roundtrip mismatch");
    assert_eq!(nf4_consumed, nf4_encoded.len(), "NF4 consumed mismatch");

    // Demonstrate NF4_LUT: show a few quantization levels
    println!("NF4 (NormalFloat4) Block Quantization:");
    println!(
        "  Encoded: {} bytes, axis=0, block_size=128, scale_buffer=5",
        nf4_encoded.len()
    );
    println!("  NF4_LUT sample (16 levels indexed 0-15):");
    println!(
        "    code 0 (min): {:.4}, code 7 (zero): {:.4}, code 15 (max): {:.4}",
        NF4_LUT[0], NF4_LUT[7], NF4_LUT[15]
    );
    println!("  Roundtrip: ✓\n");

    // ── MXFP (OCP Microscaling) block quantization ──────────────────────────

    let mxfp = QuantizationDescriptor::Mxfp(Mxfp::new(1, 32, 6).expect("valid MXFP"));

    let mxfp_encoded = mxfp.encode_to_vec();
    let (mxfp_decoded, mxfp_consumed) =
        QuantizationDescriptor::decode(&mxfp_encoded).expect("valid MXFP decode");
    assert_eq!(mxfp_decoded, mxfp, "MXFP roundtrip mismatch");
    assert_eq!(mxfp_consumed, mxfp_encoded.len(), "MXFP consumed mismatch");

    println!("MXFP (OCP Microscaling) Block Quantization:");
    println!(
        "  Encoded: {} bytes, axis=1, block_size=32, scale_buffer=6",
        mxfp_encoded.len()
    );
    println!("  Roundtrip: ✓\n");

    // ── Validate buffer placement ───────────────────────────────────────────

    println!("Validating buffer placement constraints...");

    let cpu_buffer_data = BufferHandle::new(
        4096,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid buffer");
    let cpu_buffer_scale = BufferHandle::new(
        256,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid buffer");
    let cpu_buffer_zp = BufferHandle::new(
        256,
        MIN_BUFFER_ALIGNMENT,
        DeviceTag::Cpu,
        SyncMode::ProducerSynced,
    )
    .expect("valid buffer");

    let buffers = [cpu_buffer_data, cpu_buffer_scale, cpu_buffer_zp];

    // Per-block affine with scale at index 1 and zero_point at index 2
    let pba_with_zp = QuantizationDescriptor::PerBlockAffine(
        PerBlockAffine::new_asymmetric(0, 32, 1, 2, ElementType::Float16)
            .expect("valid per-block with ZP"),
    );
    hurray_core::validate_buffer_placement(&pba_with_zp, &buffers, 0)
        .expect("valid buffer placement");

    println!("  Per-block affine asymmetric: scale_buffer=1, zp_buffer=2 — ✓");

    // NF4 with scale at index 1 (data buffer is 0)
    let nf4_validation = QuantizationDescriptor::Nf4(Nf4::new(0, 64, 1).expect("valid NF4"));
    hurray_core::validate_buffer_placement(&nf4_validation, &buffers, 0)
        .expect("valid buffer placement");

    println!("  NF4: scale_buffer=1 (distinct from data=0) — ✓\n");

    // ── Encode into pre-allocated buffer (zero-alloc path) ────────────────────

    println!("Zero-alloc encode_into demonstration:");

    let desc = QuantizationDescriptor::PerTensorAffine(
        PerTensorAffine::new(0.5, 0).expect("valid per-tensor affine"),
    );

    let mut pre_allocated = vec![0u8; desc.encoded_len()];
    let written = desc
        .encode_into(&mut pre_allocated)
        .expect("valid encode_into");

    println!("  Pre-allocated buffer: {} bytes", pre_allocated.len());
    println!("  Bytes written: {}", written);
    println!("  encode_into: ✓\n");

    println!("All roundtrip tests passed!");
}
