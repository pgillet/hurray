//! CSC (Compressed Sparse Column) sparse layout element offset — stub.
//!
//! Full implementation is deferred pending API shape validation by the first
//! consumer (Layer 4). See OQ-014.1.

use crate::layout::CscLayout;
use crate::Result;

use super::{SparseBuffers, SparseElementAddress};

impl SparseElementAddress for CscLayout {
    fn sparse_element_offset(
        &self,
        _index: &[u64],
        _buffers: &SparseBuffers<'_>,
    ) -> Result<Option<u64>> {
        todo!("CSC sparse element addressing is deferred to Layer 4 (OQ-014.1)")
    }
}
