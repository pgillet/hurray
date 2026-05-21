fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let config =
        cbindgen::Config::from_file(std::path::Path::new(&crate_dir).join("cbindgen.toml"))
            .expect("unable to read cbindgen.toml");
    cbindgen::generate_with_config(&crate_dir, config)
        .expect("unable to generate bindings")
        .write_to_file(std::path::Path::new(&crate_dir).join("include/hurray.h"));
}
