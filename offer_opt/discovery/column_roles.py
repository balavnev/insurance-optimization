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

    # subject_id: ranked from a single unified candidate pool, not two
    # mutually-exclusive "repeats" vs "fully unique" branches -- an earlier
    # two-branch version could get pre-empted by a handful of continuous
    # metric columns that happen to repeat by sheer decimal-rounding
    # coincidence (nunique just below n), which took over the *entire*
    # "repeating" tier and never even considered the fully-unique tier where
    # the real (wide-shaped) subject id actually lived. Unified instead:
    #  1. Exclude continuous-measurement columns first (see
    #     _is_continuous_measurement) -- a value component with high decimal
    #     precision can have a cardinality suspiciously close to (or exactly)
    #     the real subject count, purely by chance, and would otherwise win.
    #  2. Exclude columns that look like a *dimension* -- non-numeric with
    #     cardinality in the same ~2..sqrt(n) band the second loop below
    #     uses to call something a dimension. Without this, a low-cardinality
    #     categorical column (e.g. a 5-value SEGMENT on a small long/tidy
    #     table) can beat a real subject id on raw cardinality alone once the
    #     id is *also* excluded by step 3 below for merely looking sequential
    #     -- a subject id and a dimension are mutually exclusive roles, so a
    #     column this obviously dimension-shaped should never even compete.
    #  3. Among what's left, prefer columns that aren't a positional-index
    #     artifact (med's unnamed pandas-index column) -- but only as a
    #     soft preference, never a hard exclusion: a real subject id that
    #     happens to be assigned sequentially (e.g. 1..n in a synthetic
    #     dataset) must not be thrown out just for "looking like" one when
    #     it's the only legitimate candidate available.
    #  4. Rank what's left by cardinality -- valid whether the column
    #     "repeats" (long/tidy: many option-rows per subject) or is fully
    #     unique (wide: one row per subject) meaning it's a legitimate
    #     subject-id shape, since ordinary dimension columns are bounded to
    #     a much smaller cardinality (~sqrt(n)) by construction.
    sqrt_n_probe = math.sqrt(n) if n else 0
    dimension_like = {c for c in cardinalities
                      if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= cardinalities[c] <= max(sqrt_n_probe, 2)}
    non_continuous = {c: card for c, card in cardinalities.items() if not _is_continuous_measurement(df[c])}
    id_candidates = {c: card for c, card in non_continuous.items() if card > 1 and c not in dimension_like} or \
        {c: card for c, card in cardinalities.items() if card > 1 and c not in dimension_like} or \
        {c: card for c, card in non_continuous.items() if card > 1}
    non_index = {c: card for c, card in id_candidates.items() if not _is_positional_index(df[c], n)}
    ranked_pool = non_index or id_candidates

    subject_id_col = max(ranked_pool, key=ranked_pool.get)
    subject_id_confidence = 1.0 if ranked_pool is non_index else 0.6

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
