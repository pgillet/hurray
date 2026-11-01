"""Python bindings for the Hurray tensor interchange format.

The whole synchronous API lives in the compiled extension and is re-exported here.
`hurray.aio` holds the asyncio surface and is **not** imported with the package: it is
descriptor-driven and therefore Unix-only (ADR-042 § 2), and `import hurray` must work
everywhere. Import it when you want it::

    import hurray.aio
"""

from .hurray import *  # noqa: F403
from .hurray import __all__ as _extension_all
from .hurray import __doc__ as _extension_doc

# What `from hurray import *` gives, and what `help(hurray)` prints, are the extension's
# — this file adds no name of its own. Spelled out rather than left to maturin's generated
# `__init__.py`, which this replaces so that `aio.py` can ship beside it.
__doc__ = _extension_doc
__all__ = _extension_all
