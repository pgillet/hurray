# hurray-python

Python bindings for the [Hurray](https://github.com/pgillet/hurray) tensor interchange format.

Built with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs).

## Quick start

```bash
pip install maturin
maturin develop
```

```python
import hurray

t = hurray.zeros((3, 4), dtype=hurray.float32)
print(t.shape, t.dtype)

hurray.save("model.hrry", {"weights": t})
loaded = hurray.load("model.hrry")
```

See the project [README](../README.md) for the full documentation.
