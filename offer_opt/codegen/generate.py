"""Generate a verifier function per constraint -- template-fill for the
overwhelming majority (every constraint `constraints.py` produces today is
already fully structured), an LLM path for genuinely novel/unclear shapes.

`verify.py` stays the ground truth throughout: the final pass/fail decision
always comes from it, never from generated code -- `cross_check()` compares
a generated function's verdict against it on real data, and disagreement is
treated as a bug in the generated code, never as "the LLM found something
new" (see codegen/sandbox.py's docstring for the full trust pipeline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from offer_opt.codegen import sandbox
from offer_opt.codegen.templates import render
from offer_opt.schema import ConstraintSet, ConstraintSpec

_UNSAFE_CHARS_RE = re.compile(r"\W+")


class CodegenError(Exception):
    """Generated verifier source failed a safety check, a golden fixture, or
    otherwise couldn't be trusted -- never silently ignored."""


@dataclass
class GeneratedCheck:
    constraint: ConstraintSpec
    function_name: str
    source: str
    fn: object  # callable(option_table, selection) -> bool
    origin: str  # "template" | "llm"


def _safe_identifier(raw_id: str) -> str:
    cleaned = _UNSAFE_CHARS_RE.sub("_", raw_id).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return f"check_{cleaned}"


def _describe(constraint: ConstraintSpec) -> str:
    bound = []
    if constraint.min is not None:
        bound.append(f"min={constraint.min:g}")
    if constraint.max is not None:
        bound.append(f"max={constraint.max:g}")
    level = "per client" if constraint.per_client else "campaign-wide"
    return f"{constraint.raw_type}: {constraint.measure} {', '.join(bound)} over {constraint.scope} ({level})"


def generate_one(constraint: ConstraintSpec) -> GeneratedCheck:
    """Deterministic template-fill -- the path every constraint produced by
    `constraints.py` takes today, since all of them are already fully
    structured (scope/measure/min/max/per_client) by the time they reach
    here. No LLM call, no network, nothing that could stall."""
    function_name = _safe_identifier(constraint.id)
    source = render(function_name, _describe(constraint), constraint.scope,
                     constraint.measure, constraint.min, constraint.max, constraint.per_client)
    fn = sandbox.compile_check_function(source, function_name)
    sandbox.check_golden_fixtures(fn, constraint)
    return GeneratedCheck(constraint=constraint, function_name=function_name, source=source, fn=fn, origin="template")


def generate_via_llm(constraint: ConstraintSpec, llm_client, max_retries: int = 1) -> GeneratedCheck:
    """For a constraint flagged as structurally novel (not produced by
    today's pipeline, but the extension point for one that someday is): ask
    an LLM to write the function body directly under the same fixed
    signature, then run it through the exact same safety+golden-fixture
    gate as the template path -- being LLM-authored earns it no less
    scrutiny, if anything more."""
    from offer_opt.llm import prompts as _prompts
    from offer_opt.llm.client import LLMUnavailable
    from offer_opt.llm.parsing import validate_against_schema

    function_name = _safe_identifier(constraint.id)
    description = _describe(constraint)
    system, user, schema = _prompts.verifier_codegen_prompt(
        function_name, description, constraint.scope, constraint.measure,
        constraint.min, constraint.max, constraint.per_client,
    )

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = llm_client.complete_json(system=system, user=user, json_schema=schema)
        except LLMUnavailable as exc:
            raise CodegenError(f"{constraint.id}: no LLM client available to generate verifier code") from exc

        errors = validate_against_schema(response, schema)
        if errors:
            last_error = CodegenError(f"{constraint.id}: invalid codegen response: {errors}")
            user = user + "\n\nPrevious response was invalid: " + "; ".join(errors) + ". Try again."
            continue

        source = response["source"]
        try:
            fn = sandbox.compile_check_function(source, function_name)
            sandbox.check_golden_fixtures(fn, constraint)
            return GeneratedCheck(constraint=constraint, function_name=function_name, source=source,
                                   fn=fn, origin="llm")
        except Exception as exc:  # noqa: BLE001 -- any sandbox rejection is a retry signal, not a crash
            last_error = exc
            user = user + f"\n\nPrevious response failed: {exc}. Try again, following the rules exactly."

    raise CodegenError(f"{constraint.id}: LLM-generated verifier code never passed the sandbox") from last_error


def generate_all(constraint_set: ConstraintSet) -> dict[str, GeneratedCheck]:
    return {c.id: generate_one(c) for c in constraint_set.constraints}


def cross_check(generated: GeneratedCheck, offer_table: pd.DataFrame, selection: np.ndarray,
                 eps: float = 1e-6) -> bool:
    """Run a generated checker and compare its verdict against verify.py's
    ground-truth verdict for the SAME constraint in isolation, on the same
    real dataset+selection. True iff they agree."""
    from offer_opt.verify import verify

    generated_ok = bool(sandbox.run_with_timeout(generated.fn, (offer_table, selection)))
    isolated = ConstraintSet(constraints=[generated.constraint], parameters=[])
    report = verify(offer_table, isolated, selection, eps=eps)
    return generated_ok == report.ok
