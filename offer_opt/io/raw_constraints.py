"""Turn a constraint-table DataFrame (med/hard dialect) into RawConstraintRow
tuples. Column names are matched case-insensitively since med uses
`Constraints;Channel;Product;min;max` and hard uses
`CONSTRAINTS;CHANNEL;PRODUCT;MIN;MAX` -- the only per-case difference here is
casing, not semantics.
"""

from __future__ import annotations

import math

import pandas as pd

from offer_opt.schema import RawConstraintRow

_COLUMN_ALIASES = {
    "type": "Constraints",
    "constraints": "Constraints",
    "channel": "Channel",
    "product": "Product",
    "min": "min",
    "max": "max",
}


def _find_column(df: pd.DataFrame, wanted: str) -> str:
    wanted_lower = wanted.lower()
    for col in df.columns:
        if col.strip().lower() == wanted_lower:
            return col
    raise KeyError(f"column matching {wanted!r} not found in {list(df.columns)}")


def _clean_str(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().strip('"')
    return s if s else None


def _clean_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return float(v)


def from_table(df: pd.DataFrame) -> list[RawConstraintRow]:
    type_col = _find_column(df, "Constraints")
    channel_col = _find_column(df, "Channel")
    product_col = _find_column(df, "Product")
    min_col = _find_column(df, "min")
    max_col = _find_column(df, "max")

    rows = []
    for _, r in df.iterrows():
        rows.append(
            RawConstraintRow(
                raw_type=str(r[type_col]).strip().strip('"'),
                channel=_clean_str(r[channel_col]),
                product=_clean_str(r[product_col]),
                min=_clean_float(r[min_col]),
                max=_clean_float(r[max_col]),
            )
        )
    return rows
