//! Row-major (C-order) layout descriptor.
//!
//! Tag `0x01`. No extra fields — strides are implicit.
//! See `docs/spec/layouts/row-major.md`.

#[cfg(test)]
mod tests {
    use crate::layout::LayoutDescriptor;

    #[test]
    fn row_major_tag_is_0x01() {
        assert_eq!(LayoutDescriptor::RowMajor.tag(), 0x01);
    }

    #[test]
    fn row_major_buffer_count_is_1() {
        use std::num::NonZeroU8;
        assert_eq!(
            LayoutDescriptor::RowMajor.buffer_count(),
            Some(NonZeroU8::new(1).unwrap())
        );
    }
}
