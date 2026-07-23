"""One function per LLM step, each producing `(system, user, json_schema)`.
No call site builds a raw prompt string inline. Static instructions live in
versioned template files under `llm/prompts/` (house style observed in
vendor-examples/examples/prompt_lab.py: prompts as files, not inline
strings); each function loads its file for the system prompt and
interpolates only the dynamic, per-call parts into the user message.

Populated in Phase 4 with the constraint-classification step, Phase 6 adds
the verifier-codegen step's; Phase 7 adds the remaining steps' prompt
functions here too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from offer_opt.llm.schemas import CONSTRAINT_TYPE_SCHEMA, VERIFIER_CODEGEN_SCHEMA

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def constraint_type_prompt(key: str, populated_columns: list[str], dims: tuple[str, ...],
                            example: dict[str, Any]) -> tuple[str, str, dict]:
    system = _load("constraint_type_v1.txt")
    user = (
        f"Constraint type string: {key!r}\n"
        f"Known target dimension vocabulary for this dataset: {list(dims)}\n"
        f"Row's populated raw columns: {populated_columns}\n"
        f"Example row: {example}"
    )
    return system, user, CONSTRAINT_TYPE_SCHEMA


def verifier_codegen_prompt(function_name: str, description: str, scope: dict,
                             measure: str, min_: float | None, max_: float | None,
                             per_client: bool) -> tuple[str, str, dict]:
    # Plain substring replace, not str.format(): the template file's own
    # JSON-example text contains literal "{"..."}" braces that would
    # otherwise collide with format-string field syntax.
    system = _load("verifier_codegen_v1.txt").replace("{function_name}", function_name)
    user = (
        f"Function name: {function_name}\n"
        f"Constraint description: {description}\n"
        f"Scope: {scope}\n"
        f"Measure: {measure}\n"
        f"Min bound: {min_}\n"
        f"Max bound: {max_}\n"
        f"Per-client (bounds apply per individual client_id, not campaign-wide): {per_client}"
    )
    return system, user, VERIFIER_CODEGEN_SCHEMA
