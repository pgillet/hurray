//! The auto traits the public reader and writer types carry.
//!
//! These are not decorative. A `Box<dyn Trait>` anywhere inside a type erases `Sync`,
//! `UnwindSafe` and `RefUnwindSafe` from it and from everything that contains it, silently
//! and at a distance — a field added three types down is enough. It compiles, every test
//! passes, and the loss only surfaces in a downstream crate or in `cargo semver-checks`,
//! which is where it surfaced the first time (ADR-041 step 2).
//!
//! So the guarantees are asserted here, where a regression is a local compile error.
//! `StreamReaderCore` is generic over its allocator precisely so these hold: with the
//! default allocator nothing is boxed, and a binding that supplies its own gets whichever
//! auto traits its own type has.

use std::panic::{RefUnwindSafe, UnwindSafe};

use hurray_io::stream::{StreamReaderCore, StreamWriterCore};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}
fn assert_unwind_safe<T: UnwindSafe>() {}
fn assert_ref_unwind_safe<T: RefUnwindSafe>() {}

/// All four, for a type that should be fully transferable.
fn assert_all<T: Send + Sync + UnwindSafe + RefUnwindSafe>() {
    assert_send::<T>();
    assert_sync::<T>();
    assert_unwind_safe::<T>();
    assert_ref_unwind_safe::<T>();
}

#[test]
fn the_sans_io_cores_carry_every_auto_trait() {
    assert_all::<StreamReaderCore>();
    assert_all::<StreamWriterCore>();
}

#[cfg(feature = "tokio")]
#[test]
fn the_stream_drivers_carry_every_auto_trait_their_transport_does() {
    use hurray_io::stream::{StreamReader, StreamWriter};

    // `&[u8]` and `Vec<u8>` have all four, so the drivers must not be the thing that
    // takes them away.
    assert_all::<StreamReader<&[u8]>>();
    assert_all::<StreamWriter<Vec<u8>>>();
}

#[cfg(feature = "tokio")]
#[test]
fn the_file_drivers_carry_every_auto_trait_their_transport_does() {
    use hurray_io::file::{FileReader, FileWriter};

    assert_all::<FileWriter<Vec<u8>>>();
    assert_all::<FileReader<std::io::Cursor<Vec<u8>>>>();
}
