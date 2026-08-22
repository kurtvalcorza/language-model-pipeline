"""Load and apply the JSON Schemas that describe this pipeline's documents.

The schemas live INSIDE the package rather than in a top-level `schemas/` directory, which
is a deliberate departure from the blueprint's layout. The reason is the vendoring
arrangement: consumers carry a copy of this package, not a copy of this repository, so a
schema outside the package would not reach the containers that emit the documents it
describes — and a schema the emitter cannot see is a schema that silently stops matching.

`jsonschema` is a TEST dependency, not a runtime one. Nothing in the containers validates on
the hot path; the gate is CI, run against documents produced by real runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

JOB = "job"
VALIDATION_RESULT = "validation-result"
TRAINING_RESULT = "training-result"
MODEL_REGISTRY = "model-registry"

SCHEMA_NAMES = (JOB, VALIDATION_RESULT, TRAINING_RESULT, MODEL_REGISTRY)


class SchemaError(RuntimeError):
    """A document that does not match its schema."""


def schema_path(name: str) -> Path:
    path = SCHEMA_DIR / f"{name}.schema.json"
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in SCHEMA_DIR.glob("*.schema.json")))
        raise SchemaError(f"No schema {name!r}. Available: {available}")
    return path


def load_schema(name: str) -> dict[str, Any]:
    return json.loads(schema_path(name).read_text(encoding="utf-8"))


def _validator(name: str):
    """Build a validator that can resolve the shared `result-common` references.

    The result schemas `$ref` a common file for the DIMER outer shape, so the validator
    needs a registry that can find it. Resolved from the local directory rather than over
    the network: a schema check that reaches the internet is a schema check that fails in a
    disconnected build.
    """
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except ImportError as exc:  # pragma: no cover - test-only dependency
        raise SchemaError(
            "Schema validation needs `jsonschema`: pip install 'jsonschema>=4.20'"
        ) from exc

    registry = Registry()
    for path in SCHEMA_DIR.glob("*.schema.json"):
        contents = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(contents)
        # Registered under the bare filename as well as its $id, because the sibling refs
        # are written as relative filenames.
        registry = resource @ registry
        registry = registry.with_resource(uri=path.name, resource=resource)

    return Draft202012Validator(load_schema(name), registry=registry)


def validate_document(name: str, document: Any) -> None:
    """Raise SchemaError listing every problem, not just the first.

    Reporting one error at a time turns a shape change into a sequence of rebuild-and-retry
    cycles; the whole point of having the schema is to see the full divergence at once.
    """
    errors = sorted(_validator(name).iter_errors(document), key=lambda e: list(e.path))
    if not errors:
        return

    problems = []
    for error in errors:
        location = "/".join(str(part) for part in error.path) or "(root)"
        problems.append(f"{location}: {error.message}")
    raise SchemaError(
        f"Document does not match {name}.schema.json:\n  " + "\n  ".join(problems[:20])
    )


def is_valid(name: str, document: Any) -> bool:
    try:
        validate_document(name, document)
    except SchemaError:
        return False
    return True
