//! Hilbert curve element offset computation.
//!
//! Implements the normative Skilling (2004) algorithm as specified in
//! `docs/spec/layouts/hilbert.md § Normative Index Mapping (CoordsToHilbert)`.

use crate::layout::HilbertLayout;
use crate::{Error, Result, Shape};

use super::{validate_index, ElementAddress};

impl ElementAddress for HilbertLayout {
    fn element_offset(&self, index: &[u64], shape: &Shape) -> Result<u64> {
        validate_index(index, shape)?;

        let r = self.hilbert_rank as usize;
        let p = self.hilbert_order as usize;

        // Hilbert index has r×p bits; guard against u64 overflow.
        if r.saturating_mul(p) > 64 {
            return Err(Error::IndexArithmeticOverflow);
        }

        // Working copy of coordinates; Skilling algorithm mutates them in place.
        let mut x: Vec<u64> = index.to_vec();

        let m = 1u64 << (p - 1); // 2^(p-1)

        // --- CoordsToHilbert (Skilling 2004) ---
        // needless_range_loop: index arithmetic over x[] is intentional per the
        // Skilling algorithm — elements at arbitrary positions (x[0], x[i], x[i-1])
        // are mixed in a way that does not map cleanly to an iterator pattern.
        #[allow(clippy::needless_range_loop)]
        let mut q = m;
        while q > 1 {
            let mask = q - 1;
            for i in 0..r {
                if x[i] & q != 0 {
                    x[0] ^= mask;
                } else {
                    let t = (x[0] ^ x[i]) & mask;
                    x[0] ^= t;
                    x[i] ^= t;
                }
            }
            q >>= 1;
        }

        for i in 1..r {
            x[i] ^= x[i - 1];
        }

        let mut t: u64 = 0;
        q = m;
        while q > 1 {
            if x[r - 1] & q != 0 {
                t ^= q - 1;
            }
            q >>= 1;
        }
        for xi in x.iter_mut().take(r) {
            *xi ^= t;
        }

        // Bit packing: bit (b*r + (r-1-d)) of h holds bit b of X[d].
        let mut h: u64 = 0;
        for b in 0..p {
            for (d, &xd) in x.iter().enumerate().take(r) {
                let bit = (xd >> b) & 1;
                let shift = b * r + (r - 1 - d);
                if shift >= 64 {
                    return Err(Error::IndexArithmeticOverflow);
                }
                h |= bit << shift;
            }
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Shape};

    // Spec docs/spec/layouts/hilbert.md § Normative Index Mapping (CoordsToHilbert):
    // Full 4×4 conformance table for hilbert_rank=2, hilbert_order=2.
    // The spec lists HilbertToCoords; we test the inverse CoordsToHilbert.
    fn layout_4x4() -> (HilbertLayout, Shape) {
        let layout = HilbertLayout::new(2, 2).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        (layout, shape)
    }

    #[test]
    fn hilbert_4x4_full_table() {
        // All 16 entries produced by the normative Skilling (2004) algorithm.
        // NOTE: The spec conformance table (docs/spec/layouts/hilbert.md) has
        // entries h=13 and h=15 swapped relative to the algorithm — see the
        // spec finding filed in hilbert.md. The algorithm is normative (MUST);
        // the table is SHOULD-level. These values are what the algorithm produces.
        let table: &[([u64; 2], u64)] = &[
            ([0, 0], 0),
            ([1, 0], 1),
            ([1, 1], 2),
            ([0, 1], 3),
            ([0, 2], 4),
            ([0, 3], 5),
            ([1, 3], 6),
            ([1, 2], 7),
            ([2, 2], 8),
            ([2, 3], 9),
            ([3, 3], 10),
            ([3, 2], 11),
            ([3, 1], 12),
            ([2, 1], 13), // algorithm: [2,1]→13 (table says [3,0]→13 — swapped)
            ([2, 0], 14),
            ([3, 0], 15), // algorithm: [3,0]→15 (table says [2,1]→15 — swapped)
        ];
        let (layout, shape) = layout_4x4();
        for &(index, expected) in table {
            let got = layout.element_offset(&index, &shape).unwrap();
            assert_eq!(
                got, expected,
                "hilbert[{},{}]: expected {expected}, got {got}",
                index[0], index[1]
            );
        }
    }

    // Hilbert locality invariant from spec:
    // Consecutive Hilbert indices differ by 1 in exactly one coordinate
    // (L∞ distance between the coordinates is exactly 1).
    #[test]
    fn hilbert_4x4_consecutive_indices_differ_by_one_coordinate() {
        let (layout, shape) = layout_4x4();

        // Build a map from hilbert index → coordinates by inverting the full table.
        let table: &[([u64; 2], u64)] = &[
            ([0, 0], 0),
            ([1, 0], 1),
            ([1, 1], 2),
            ([0, 1], 3),
            ([0, 2], 4),
            ([0, 3], 5),
            ([1, 3], 6),
            ([1, 2], 7),
            ([2, 2], 8),
            ([2, 3], 9),
            ([3, 3], 10),
            ([3, 2], 11),
            ([3, 1], 12),
            ([2, 1], 13), // algorithm output (spec table has h=13/15 swapped)
            ([2, 0], 14),
            ([3, 0], 15), // algorithm output
        ];

        // Verify the layout produces the table values first.
        for &(index, expected) in table {
            assert_eq!(layout.element_offset(&index, &shape).unwrap(), expected);
        }

        // Build h → coords map.
        let mut h_to_coords = [(0u64, 0u64); 16];
        for &(coords, h) in table {
            h_to_coords[h as usize] = (coords[0], coords[1]);
        }

        // Check each consecutive pair.
        for h in 0..15u64 {
            let (r0, c0) = h_to_coords[h as usize];
            let (r1, c1) = h_to_coords[(h + 1) as usize];
            let dr = r0.abs_diff(r1);
            let dc = c0.abs_diff(c1);
            // Exactly one coordinate must change, and by exactly 1.
            assert_eq!(
                dr + dc,
                1,
                "h={h} → h={}: coords ({r0},{c0}) and ({r1},{c1}) — L1 distance must be 1",
                h + 1
            );
        }
    }

    // Error: index rank mismatch.
    #[test]
    fn hilbert_rank_mismatch() {
        let (layout, shape) = layout_4x4();
        let err = layout.element_offset(&[0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexRankMismatch { .. }),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index out of range (index[0]=4 >= dim[0]=4).
    #[test]
    fn hilbert_index_out_of_range() {
        let (layout, shape) = layout_4x4();
        let err = layout.element_offset(&[4, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexOutOfRange { dim: 0, .. }),
            "expected IndexOutOfRange, got {err:?}"
        );
    }

    // Overflow guard: hilbert_rank × hilbert_order > 64 → IndexArithmeticOverflow.
    // hilbert_rank=8, hilbert_order=9 → 8×9=72 > 64.
    #[test]
    fn hilbert_overflow_rank_times_order_exceeds_64() {
        // rank=8, order=9 → 72 bits needed, overflows u64.
        let layout = HilbertLayout::new(9, 8).unwrap(); // order=9, rank=8
        let shape = Shape::new(vec![512u64; 8]).unwrap(); // 2^9 = 512 per dim
        let err = layout.element_offset(&[0u64; 8], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexArithmeticOverflow),
            "expected IndexArithmeticOverflow, got {err:?}"
        );
    }
}
