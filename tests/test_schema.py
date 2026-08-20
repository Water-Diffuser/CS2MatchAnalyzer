"""Guards the contract between the CV pipeline, the AI adapters, and storage.

Run: python tests/test_schema.py   (or under pytest)
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas" / "clip_analysis.v1.schema.json").read_text())


def test_schema_is_well_formed():
    Draft202012Validator.check_schema(SCHEMA)


def test_documented_example_validates():
    """The record in ARCHITECTURE.md must stay in sync with the schema."""
    md = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    blocks = re.findall(r"```json\n(.*?)```", md, re.S)
    example = json.loads(next(b for b in blocks if '"schema_version"' in b))
    errors = list(Draft202012Validator(SCHEMA).iter_errors(example))
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors]


def test_openai_strict_mode_conformance():
    """OpenAI strict mode is the tightest of the three dialects.

    It requires every property to appear in `required` and additionalProperties
    to be false on every object. Conforming here means Gemini's response_schema
    and Anthropic's tool input_schema accept the same document unchanged, which
    is the whole basis of the swappable-provider design.
    """
    problems: list[str] = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: additionalProperties must be false")
                missing = set(node["properties"]) - set(node.get("required", []))
                if missing:
                    problems.append(f"{path}: absent from `required`: {sorted(missing)}")
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(SCHEMA)
    assert not problems, problems


def test_model_never_receives_ai_status_as_an_output_field():
    """`ai_status` is app bookkeeping. If the model can write it, it will
    report "ok" on a response we are about to reject."""
    sys.path.insert(0, str(ROOT / "reference"))
    from ai_engine.providers import _assessed_schema

    schema = _assessed_schema()
    assert "ai_status" not in schema["properties"]
    assert "ai_status" not in schema["required"]


def test_measured_is_not_in_the_model_output_target():
    """The AI must never be able to overwrite instrumented CV numbers."""
    sys.path.insert(0, str(ROOT / "reference"))
    from ai_engine.providers import _assessed_schema

    forbidden = {"reaction_time_ms", "jitter_ratio", "smoothness_sparc", "overshoot_count"}
    assert not (forbidden & set(_assessed_schema()["properties"]))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{'all passed' if not failed else f'{failed} failed'}")
    raise SystemExit(1 if failed else 0)
