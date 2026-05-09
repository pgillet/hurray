//! Shard descriptor — binary encode/decode.
//!
//! A shard descriptor identifies this tensor as a rectangular sub-region of a
//! larger logical parent tensor. It is present in the wire format when the
//! `HAS_SHARD` flag is set.
//!
//! Wire layout (spec § Shard Section):
//! ```text
//! parent_shape  uint64[rank]   (rank × 8 bytes)
//! shard_offset  uint64[rank]   (rank × 8 bytes)
//! ```
//! Constraint: `shard_offset[k] + shape[k] <= parent_shape[k]` for all `k`.

use crate::descriptor::cursor::{ByteCursor, ByteWriter};
use crate::{Error, Result, Shape};

/// Declares that this tensor is a rectangular shard of a larger parent tensor.
///
/// The shard descriptor carries the parent tensor's shape and the starting
/// index of this shard within the parent along each dimension.
///
/// # Examples
///
/// ```
/// use hurray_core::descriptor::ShardDescriptor;
/// use hurray_core::Shape;
///
/// let parent = vec![10u64, 20];
/// let offset = vec![0u64, 5];
/// let shard  = ShardDescriptor::new(parent, offset).unwrap();
/// assert_eq!(shard.parent_shape, [10, 20]);
/// assert_eq!(shard.shard_offset, [0, 5]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardDescriptor {
    /// Shape of the logical parent tensor. Same length as the tensor rank.
    pub parent_shape: Vec<u64>,
    /// Starting index of this shard within the parent along each dimension.
    pub shard_offset: Vec<u64>,
}

impl ShardDescriptor {
    /// Creates a new [`ShardDescriptor`], validating that `parent_shape` and
    /// `shard_offset` have the same length.
    ///
    /// Shape-vs-offset bound checking (`offset[k] + shape[k] <= parent[k]`)
    /// requires the tensor shape and is performed by
    /// [`ShardDescriptor::validate_against_shape`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidShape`] if `parent_shape.len() != shard_offset.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::descriptor::ShardDescriptor;
    ///
    /// let s = ShardDescriptor::new(vec![10, 20], vec![0, 5]).unwrap();
    /// assert_eq!(s.parent_shape, [10, 20]);
    ///
    /// // Mismatched lengths are rejected.
    /// assert!(ShardDescriptor::new(vec![10], vec![0, 5]).is_err());
    /// ```
    pub fn new(parent_shape: Vec<u64>, shard_offset: Vec<u64>) -> Result<Self> {
        if parent_shape.len() != shard_offset.len() {
            return Err(Error::InvalidShape(format!(
                "shard: parent_shape.len() ({}) != shard_offset.len() ({})",
                parent_shape.len(),
                shard_offset.len()
            )));
        }
        Ok(Self {
            parent_shape,
            shard_offset,
        })
    }

    /// Checks that `shard_offset[k] + shape[k] <= parent_shape[k]` for every `k`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShardOutOfBounds`] for the first dimension `k` that
    /// violates the constraint.
    ///
    /// # Examples
    ///
    /// ```
    /// use hurray_core::descriptor::ShardDescriptor;
    /// use hurray_core::Shape;
    ///
    /// let shape  = Shape::new(vec![4, 4]).unwrap();
    /// let shard  = ShardDescriptor::new(vec![10, 10], vec![0, 6]).unwrap();
    ///
    /// // offset[1]=6 + shape[1]=4 = 10 == parent[1]=10 → valid
    /// assert!(shard.validate_against_shape(&shape).is_ok());
    ///
    /// let bad = ShardDescriptor::new(vec![10, 10], vec![0, 7]).unwrap();
    /// // offset[1]=7 + shape[1]=4 = 11 > parent[1]=10 → rejected
    /// assert!(bad.validate_against_shape(&shape).is_err());
    /// ```
    pub fn validate_against_shape(&self, shape: &Shape) -> Result<()> {
        for k in 0..self.parent_shape.len() {
            let dim = shape.dims()[k];
            let offset = self.shard_offset[k];
            let parent = self.parent_shape[k];
            // Use saturating_add to avoid u64 overflow on pathological inputs.
            if offset.saturating_add(dim) > parent {
                return Err(Error::ShardOutOfBounds {
                    dim: k,
                    offset,
                    size: dim,
                    parent,
                });
            }
        }
        Ok(())
    }

    /// Encodes the shard section into `w` as `uint64[rank]` × 2.
    pub(crate) fn encode_into(&self, w: &mut ByteWriter) {
        for &v in &self.parent_shape {
            w.write_u64_le(v);
        }
        for &v in &self.shard_offset {
            w.write_u64_le(v);
        }
    }

    /// Decodes a shard section from `cursor`, reading `2 × rank` u64 values.
    pub(crate) fn decode_from(cursor: &mut ByteCursor<'_>, rank: u32) -> Result<Self> {
        let rank = rank as usize;
        let mut parent_shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            parent_shape.push(cursor.read_u64_le()?);
        }
        let mut shard_offset = Vec::with_capacity(rank);
        for _ in 0..rank {
            shard_offset.push(cursor.read_u64_le()?);
        }
        Ok(Self {
            parent_shape,
            shard_offset,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::cursor::ByteWriter;
    use crate::Shape;

    fn round_trip(parent: Vec<u64>, offset: Vec<u64>) -> ShardDescriptor {
        let rank = parent.len() as u32;
        let shard = ShardDescriptor::new(parent, offset).unwrap();
        let mut w = ByteWriter::new();
        shard.encode_into(&mut w);
        let bytes = w.into_vec();
        let mut c = ByteCursor::new(&bytes, bytes.len());
        ShardDescriptor::decode_from(&mut c, rank).unwrap()
    }

    #[test]
    fn shard_round_trip() {
        let parent = vec![100u64, 200, 300];
        let offset = vec![10u64, 20, 30];
        let decoded = round_trip(parent.clone(), offset.clone());
        assert_eq!(decoded.parent_shape, parent);
        assert_eq!(decoded.shard_offset, offset);
    }

    #[test]
    fn shard_out_of_bounds_rejected() {
        let shape = Shape::new(vec![3u64, 4]).unwrap();
        // offset[1]=7 + shape[1]=4 = 11 > parent[1]=10
        let shard = ShardDescriptor::new(vec![10, 10], vec![0, 7]).unwrap();
        let err = shard.validate_against_shape(&shape).unwrap_err();
        assert!(matches!(err, Error::ShardOutOfBounds { dim: 1, .. }));
    }

    #[test]
    fn shard_length_mismatch_rejected() {
        let err = ShardDescriptor::new(vec![10], vec![0, 5]).unwrap_err();
        assert!(matches!(err, Error::InvalidShape(_)));
    }

    #[test]
    fn shard_exact_bound_is_valid() {
        // offset[k] + shape[k] == parent[k] is valid (not strictly less than).
        let shape = Shape::new(vec![5u64]).unwrap();
        let shard = ShardDescriptor::new(vec![10], vec![5]).unwrap();
        assert!(shard.validate_against_shape(&shape).is_ok());
    }
}
