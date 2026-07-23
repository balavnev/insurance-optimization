"""Robust LLM-response parsing: JSON extraction and schema validation.

Pure string/dict functions, no `LLMClient` involved -- unit-testable in
isolation (tests/test_llm_parsing.py). `extract_json`'s algorithm (strip
<think> reasoning blocks, then markdown fences, then parse, falling back to
slicing the outermost {...}) is copied from vendor-examples/examples/
prompt_lab.py's `extract_json`, since it's directly why this project's Qwen
deployment needs it -- the model emits visible chain-of-thought that has to
be stripped before anything else. `validate_against_schema` mirrors that same
file's `validate_schema` shape (return error strings, don't raise), wired to
real JSON-Schema validation via `jsonschema` instead of ad hoc dict checks.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = _THINK_RE.sub("", cleaned).strip()
    cleaned = _FENCE_OPEN_RE.sub("", cleaned)
    cleaned = _FENCE_CLOSE_RE.sub("", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("root JSON value must be an object")
    return parsed


def validate_against_schema(obj: dict, schema: dict) -> list[str]:
    validator = jsonschema.Draft7Validator(schema)
    return [e.message for e in validator.iter_errors(obj)]
