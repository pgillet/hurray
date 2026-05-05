//! Column-major (Fortran-order) layout descriptor.
//!
//! Tag `0x02`. No extra fields — strides are implicit.
//! See `docs/spec/layouts/column-major.md`.

#[cfg(test)]
mod tests {
    use crate::layout::LayoutDescriptor;

    #[test]
    fn col_major_tag_is_0x02() {
        assert_eq!(LayoutDescriptor::ColMajor.tag(), 0x02);
    }

    #[test]
    fn col_major_buffer_count_is_1() {
        use std::num::NonZeroU8;
        assert_eq!(
            LayoutDescriptor::ColMajor.buffer_count(),
            Some(NonZeroU8::new(1).unwrap())
        );
    }
}
