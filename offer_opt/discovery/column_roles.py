"""Cheap, no-LLM heuristic tier for classifying an already-tabular offer
table's columns into roles: subject id, dimension, or value component.

Value-component roles are named after schema.py's canonical
`OFFER_TABLE_COLUMNS` fields (score/premium/margin/cost) so discovery output
slots directly into the vocabulary the rest of the pipeline already uses --
no separate "probability" vs "score" naming to reconcile later. `score` is
deliberately the umbrella name for either a raw response probability or an
already-combined precomputed value (med's SCORE column is `premium*proba*
margin` combined) -- schema.py's own OFFER_TABLE_COLUMNS comment already
treats those as the same slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

VALUE_ROLES = ("score", "premium", "margin", "cost")

_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "score": ("score", "proba", "probability", "response"),
    "premium": ("premium", "avg_check", "avgcheck", "check"),
    "margin": ("margin",),
    "cost": ("cost",),
}


@dataclass
class RoleGuess:
    role: str          # "subject_id" | "dimension" | one of VALUE_ROLES | "value:unknown" | "ignore"
    confidence: float   # 0..1 -- schema_resolver falls back to the LLM below a threshold


def _name_hint_role(column: str) -> str | None:
    lowered = column.lower()
    for role, hints in _NAME_HINTS.items():
        if any(h in lowered for h in hints):
            return role
    return None


def _is_positional_index(series: pd.Series, n: int) -> bool:
    """A literal 0..n-1 (or 1..n) serial index -- a common CSV-export
    artifact (e.g. a pandas index written out as an unnamed first column),
    not a real value component. Recognized directly rather than left for
    the LLM fallback to puzzle over."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.to_numpy()
    if len(values) != n or np.isnan(values).any():
        return False
    return np.array_equal(np.sort(values), np.arange(n)) or np.array_equal(np.sort(values), np.arange(1, n + 1))


def _is_continuous_measurement(series: pd.Series) -> bool:
    """Numeric with genuine fractional values (a premium, a probability, a
    monetary amount) -- as opposed to an identifier that just happens to be
    stored as float64 (e.g. "98303053.0"). An id column's cardinality can
    legitimately exceed the true subject count's "repeat ratio" threshold
    purely from float rounding coincidences on a continuous column with
    limited decimal precision (observed on real data: a probability column
    with ~5 significant digits over 150k rows repeats at a ~1.7x ratio by
    chance alone) -- this is the signal that actually distinguishes them,
    not cardinality or repeat-ratio."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.to_numpy()
    valid = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
    if len(valid) == 0:
        return False
    return not np.all(valid == np.floor(valid))


def classify_columns(df: pd.DataFrame) -> dict[str, RoleGuess]:
    n = len(df)
    cardinalities = {col: df[col].nunique(dropna=False) for col in df.columns}

    # subject_id: the highest-cardinality column that still *repeats*
    # (1 < nunique < n) -- a real subject shows up in multiple option rows.
    # Two refinements over "just pick the highest cardinality repeater":
    #  1. Exclude continuous measurement columns first (see
    #     _is_continuous_measurement) -- a value component with high decimal
    #     precision over many rows can have MORE distinct values than there
    #     are actual subjects, and would otherwise win on raw cardinality.
    #  2. Within what's left, prefer a repeat ratio (n / cardinality)
    #     meaningfully above 1x, so a column that's "not literally every
    #     value unique" only by a coincidental tie or two doesn't outrank a
    #     real, structurally-repeating id.
    # If nothing clears these bars, relax tier by tier, and finally fall
    # back to a fully-unique column (one-row-per-subject data, e.g.
    # already-reshaped wide data) that isn't a positional-index artifact.
    MIN_REPEAT_RATIO = 1.5
    repeaters = {c: card for c, card in cardinalities.items() if 1 < card < n}
    id_like_repeaters = {c: card for c, card in repeaters.items() if not _is_continuous_measurement(df[c])}
    strong_id_like = {c: card for c, card in id_like_repeaters.items() if n / card >= MIN_REPEAT_RATIO}
    strong_any = {c: card for c, card in repeaters.items() if n / card >= MIN_REPEAT_RATIO}

    repeating_candidates = strong_id_like or id_like_repeaters or strong_any or repeaters
    if repeating_candidates:
        subject_id_col = max(repeating_candidates, key=repeating_candidates.get)
        subject_id_confidence = 1.0 if repeating_candidates is strong_id_like else 0.7
    else:
        unique_candidates = [c for c, card in cardinalities.items()
                              if card == n and not _is_positional_index(df[c], n)]
        subject_id_col = unique_candidates[0] if unique_candidates else max(cardinalities, key=cardinalities.get)
        subject_id_confidence = 0.6

    guesses: dict[str, RoleGuess] = {subject_id_col: RoleGuess("subject_id", subject_id_confidence)}

    sqrt_n = math.sqrt(n) if n else 0
    for col in df.columns:
        if col == subject_id_col:
            continue
        card = cardinalities[col]
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        hinted = _name_hint_role(col)

        if is_numeric and _is_positional_index(df[col], n):
            guesses[col] = RoleGuess("ignore", confidence=0.95)
        elif not is_numeric and 2 <= card <= max(sqrt_n, 2):
            guesses[col] = RoleGuess("dimension", confidence=0.95 if hinted is None else 0.9)
        elif is_numeric and hinted is not None:
            guesses[col] = RoleGuess(hinted, confidence=0.9)
        elif is_numeric:
            # Numeric, no name hint -- ambiguous: an unrecognized value
            # component, or bookkeeping. schema_resolver's LLM fallback (or,
            # absent one, a heuristic best-guess) decides.
            guesses[col] = RoleGuess("value:unknown", confidence=0.3)
        else:
            # Non-numeric but outside the "clean dimension" cardinality
            # band -- also ambiguous.
            guesses[col] = RoleGuess("dimension", confidence=0.4)

    return guesses
