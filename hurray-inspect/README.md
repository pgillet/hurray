# hurray-inspect

A command-line tool to inspect **[Hurray](https://www.pascalgillet.net/hurray/)** binary
tensor descriptor files as a human-readable, annotated hex table.

Hurray is a zero-copy, streamable, language-agnostic tensor interchange format for AI/ML
inference.

## Install

```sh
cargo install hurray-inspect
```

## Usage

```sh
hurray-inspect <file>
```

It prints the parsed tensor descriptor — magic, version, element type, shape, layout,
buffer table, and optional sections — alongside the raw bytes, for debugging and format
exploration.

## Documentation

- Format specification, cookbook, and guides: <https://www.pascalgillet.net/hurray/>
- Source: <https://github.com/pgillet/hurray>

## License

Licensed under either of MIT or Apache-2.0 at your option.
