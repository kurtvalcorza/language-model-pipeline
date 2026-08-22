"""Validation dataset suite: registry, deterministic conversion, packaging.

Validation data is treated like code and model provenance — versioned, deterministic,
licensed, auditable, reproducible. The point is not to have "some data that trains", but to
be able to say exactly which data validated which pipeline and to rebuild it later byte for
byte.

Everything here fails closed: a revision that moved, a licence that forbids redistribution,
or a digest that does not match stops the build rather than quietly producing different
data.
"""

from __future__ import annotations

from .convert import CONVERTERS, to_canonical
from .package import build_manifest, write_dimer_zip
from .registry import DatasetRegistry, DatasetSource, Profile
from .subset import select

__all__ = [
    "CONVERTERS",
    "DatasetRegistry",
    "DatasetSource",
    "Profile",
    "build_manifest",
    "select",
    "to_canonical",
    "write_dimer_zip",
]
