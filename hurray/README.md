# hurray

The umbrella name for **[Hurray](https://www.pascalgillet.net/hurray/)**, a zero-copy,
streamable, language-agnostic tensor interchange format for AI/ML inference pipelines and
scientific arrays.

**This crate contains no code.** It holds the name next to the crates that do the work.
Depend on the one you need:

| Crate | What it gives you |
|---|---|
| [`hurray-core`](https://crates.io/crates/hurray-core) | Format types: tensor descriptor, element types, layouts, quantization, buffer handles. No I/O. |
| [`hurray-io`](https://crates.io/crates/hurray-io) | Async streaming interchange and the `HRRYFILE` container. |
| [`hurray-ffi`](https://crates.io/crates/hurray-ffi) | The C ABI, for bindings in other languages. |
| [`hurray-inspect`](https://crates.io/crates/hurray-inspect) | A CLI that prints a descriptor as an annotated hex table. |

```sh
cargo add hurray-core
```

From Python, install [`pyhurray`](https://pypi.org/project/pyhurray/) — the distribution
name differs because `hurray` on PyPI is an unrelated package, but the module is imported
as `hurray`.

Specification, cookbook and API reference: <https://www.pascalgillet.net/hurray/>

## Licence

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE), at your
option.
