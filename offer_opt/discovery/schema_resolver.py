"""Orchestrates column_roles' cheap heuristic tier with an LLM fallback for
columns the heuristics can't confidently place. Produces a `SchemaMapping`:
the canonical description of which raw column plays which role, so nothing
downstream ever needs to know a raw column's literal name again -- the
generic replacement for `features.py`'s hand-written `_load_low/_load_med/
_load_hard`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from offer_opt.discovery.column_roles import VALUE_ROLES, RoleGuess, classify_columns
from offer_opt.llm.client import LLMClient, LLMUnavailable, NullClient

CONFIDENCE_THRESHOLD = 0.6

_ROLE_SCHEMA = {
    "type": "object",
    "properties": {"role": {"type": "string", "enum": ["subject_id", "dimension", *VALUE_ROLES, "ignore"]}},
    "required": ["role"],
}


@dataclass
class SchemaMapping:
    subject_id_column: str
    dimension_columns: list[str] = field(default_factory=list)
    value_component_columns: dict[str, str] = field(default_factory=dict)   # role -> column
    unresolved_columns: list[str] = field(default_factory=list)
    precomputed_value_column: str | None = None
    objective_formula: str | None = None


def resolve_schema(df: pd.DataFrame, llm_client: LLMClient | None = None) -> SchemaMapping:
    llm_client = llm_client or NullClient()
    guesses = classify_columns(df)

    low_confidence = {col: g for col, g in guesses.items() if g.confidence < CONFIDENCE_THRESHOLD}
    if low_confidence:
        for col, role in _resolve_via_llm(df, low_confidence, llm_client).items():
            guesses[col] = RoleGuess(role, confidence=1.0)

    subject_id_column = next(col for col, g in guesses.items() if g.role == "subject_id")
    dimension_columns: list[str] = []
    value_component_columns: dict[str, str] = {}
    unresolved: list[str] = []

    for col, g in guesses.items():
        if g.role == "subject_id":
            continue
        elif g.role == "dimension":
            dimension_columns.append(col)
        elif g.role in VALUE_ROLES:
            value_component_columns[g.role] = col
        else:
            unresolved.append(col)  # "ignore", or anything the LLM fallback couldn't place

    return SchemaMapping(
        subject_id_column=subject_id_column,
        dimension_columns=dimension_columns,
        value_component_columns=value_component_columns,
        unresolved_columns=unresolved,
    )


def _resolve_via_llm(df: pd.DataFrame, low_confidence: dict[str, RoleGuess],
                      llm_client: LLMClient, max_retries: int = 2) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for col, guess in low_confidence.items():
        samples = df[col].dropna().unique()[:5].tolist()
        prompt = (f"Column {col!r}: dtype={df[col].dtype}, cardinality={df[col].nunique()}, "
                  f"sample values={samples}. Classify its role in a resource-allocation pipeline: "
                  f"subject_id, dimension, score, premium, margin, cost, or ignore.")
        role = None
        for _ in range(max_retries + 1):
            try:
                response = llm_client.complete_json(
                    system="Classify a data column's role for a generic resource-allocation pipeline.",
                    user=prompt, json_schema=_ROLE_SCHEMA,
                )
                candidate = response.get("role")
                if candidate in ("subject_id", "dimension", *VALUE_ROLES, "ignore"):
                    role = candidate
                    break
                prompt += f"\nInvalid role {candidate!r} -- must be one of subject_id/dimension/{'/'.join(VALUE_ROLES)}/ignore."
            except LLMUnavailable:
                break
        if role is None:
            # No usable LLM response (unavailable, or exhausted retries) --
            # degrade to the heuristic's best guess rather than blocking.
            role = guess.role if guess.role != "value:unknown" else "ignore"
        resolved[col] = role
    return resolved
