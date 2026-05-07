//! Morton (Z-order curve) element offset computation.
//!
//! Spec: docs/spec/layouts/morton.md § Morton Index Computation

use crate::layout::MortonLayout;
use crate::{Error, Result, Shape};

use super::{validate_index, ElementAddress};

impl ElementAddress for MortonLayout {
    fn element_offset(&self, index: &[u64], shape: &Shape) -> Result<u64> {
        validate_index(index, shape)?;
        if self.morton_bits.len() != shape.rank() {
            return Err(Error::IndexRankMismatch {
                index_rank: self.morton_bits.len(),
                shape_rank: shape.rank(),
            });
        }

        let rank = index.len();
        let total_bits: u32 = self.morton_bits.iter().sum();
        if total_bits > 64 {
            return Err(Error::IndexArithmeticOverflow);
        }

        let max_bits = self.morton_bits.iter().copied().max().unwrap_or(0);
        let mut morton_code: u64 = 0;

        // Bit interleaving: for each bit position b and dimension d, place bit b
        // of index[d] at output position b*rank+d. Round-robin LSB-first per spec.
        for b in 0..max_bits {
            for (d, (&idx_val, &bits_for_dim)) in
                index.iter().zip(self.morton_bits.iter()).enumerate()
            {
                if b < bits_for_dim {
                    let bit = (idx_val >> b) & 1;
                    let shift = b * rank as u32 + d as u32;
                    if shift >= 64 {
                        return Err(Error::IndexArithmeticOverflow);
                    }
                    morton_code |= bit << shift;
                }
            }
        }
        Ok(morton_code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Shape};

    // Spec docs/spec/layouts/morton.md § Morton Index Computation:
    // Bit interleaving: for index [i_0, i_1, …], place bit b of i_d at output
    // bit position b*rank + d (round-robin, LSB-first).
    //
    // Full 4×4 conformance table (morton_bits=[2,2], shape=[4,4]):
    //   [i_0, i_1] → morton code
    //   i_0 occupies bits 0,2; i_1 occupies bits 1,3.
    fn layout_2d_2bits() -> (MortonLayout, Shape) {
        let layout = MortonLayout::new(vec![2, 2]).unwrap();
        let shape = Shape::new(vec![4, 4]).unwrap();
        (layout, shape)
    }

    #[test]
    fn morton_4x4_full_table() {
        // Expected offsets from the spec conformance table.
        // Entry format: ([i_0, i_1], expected_offset)
        let table: &[([u64; 2], u64)] = &[
            ([0, 0], 0),
            ([1, 0], 1),
            ([0, 1], 2),
            ([1, 1], 3),
            ([2, 0], 4),
            ([3, 0], 5),
            ([2, 1], 6),
            ([3, 1], 7),
            ([0, 2], 8),
            ([1, 2], 9),
            ([0, 3], 10),
            ([1, 3], 11),
            ([2, 2], 12),
            ([3, 2], 13),
            ([2, 3], 14),
            ([3, 3], 15),
        ];
        let (layout, shape) = layout_2d_2bits();
        for &(index, expected) in table {
            let got = layout.element_offset(&index, &shape).unwrap();
            assert_eq!(
                got, expected,
                "morton[{},{}]: expected {expected}, got {got}",
                index[0], index[1]
            );
        }
    }

    // Spec example from task description:
    // shape [4,4], morton_bits [2,2], element [2,3]:
    // i_0=2=0b10, i_1=3=0b11
    // bit0(i_0)=0, bit0(i_1)=1, bit1(i_0)=1, bit1(i_1)=1 → 0b1110 = 14
    #[test]
    fn morton_4x4_spec_example_2_3() {
        let (layout, shape) = layout_2d_2bits();
        assert_eq!(layout.element_offset(&[2, 3], &shape).unwrap(), 14);
    }

    // Rank-3 Morton (morton_bits=[1,1,1], shape=[2,2,2]):
    // i_0 → bit 0,3; i_1 → bit 1,4; i_2 → bit 2,5
    // Each dimension contributes 1 bit.
    #[test]
    fn morton_2x2x2_full_table() {
        let layout = MortonLayout::new(vec![1, 1, 1]).unwrap();
        let shape = Shape::new(vec![2, 2, 2]).unwrap();
        let table: &[([u64; 3], u64)] = &[
            ([0, 0, 0], 0),
            ([1, 0, 0], 1),
            ([0, 1, 0], 2),
            ([1, 1, 0], 3),
            ([0, 0, 1], 4),
            ([1, 0, 1], 5),
            ([0, 1, 1], 6),
            ([1, 1, 1], 7),
        ];
        for &(index, expected) in table {
            let got = layout.element_offset(&index, &shape).unwrap();
            assert_eq!(
                got, expected,
                "morton[{},{},{}]: expected {expected}, got {got}",
                index[0], index[1], index[2]
            );
        }
    }

    // Overflow guard: total_bits > 64 must return IndexArithmeticOverflow.
    // morton_bits=[33, 33] → total = 66 > 64.
    #[test]
    fn morton_overflow_total_bits_exceeds_64() {
        let layout = MortonLayout::new(vec![33, 33]).unwrap();
        // Shape must accommodate 2^33 per dimension; use DYNAMIC to avoid
        // a shape-construction overflow.
        let shape = Shape::new(vec![crate::DYNAMIC, crate::DYNAMIC]).unwrap();
        // validate_index will reject DYNAMIC before the overflow check; use a
        // shape whose dims are within index range but trigger the overflow path.
        // Since validate_index fires first on DYNAMIC, construct with large static dims.
        // 2^33 = 8_589_934_592
        let shape2 = Shape::new(vec![8_589_934_592u64, 8_589_934_592u64]).unwrap();
        let err = layout.element_offset(&[0, 0], &shape2).unwrap_err();
        assert!(
            matches!(err, Error::IndexArithmeticOverflow),
            "expected IndexArithmeticOverflow, got {err:?}"
        );
    }

    // Error: index rank mismatch.
    #[test]
    fn morton_rank_mismatch() {
        let (layout, shape) = layout_2d_2bits();
        let err = layout.element_offset(&[1], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexRankMismatch { .. }),
            "expected IndexRankMismatch, got {err:?}"
        );
    }

    // Error: index out of range.
    #[test]
    fn morton_index_out_of_range() {
        let (layout, shape) = layout_2d_2bits();
        // Index [4,0] where shape[0]=4 → out of range.
        let err = layout.element_offset(&[4, 0], &shape).unwrap_err();
        assert!(
            matches!(err, Error::IndexOutOfRange { dim: 0, .. }),
            "expected IndexOutOfRange, got {err:?}"
        );
    }
}
