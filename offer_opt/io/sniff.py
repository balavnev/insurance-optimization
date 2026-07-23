"""Generic dialect/shape auto-detection for arbitrary raw input files.

Where `io/dialects.py` is a hardcoded per-case lookup table (sep/decimal/
encoding keyed by "low"/"med"/"hard"), this module *discovers* the same
information from the file itself, so a brand-new dataset -- a different
insurance case, or a wholly different domain like the farm example in
system_design_overview.md -- doesn't need a hand-written dialect entry
before it can be read at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SEP_CANDIDATES = (",", ";", "\t")
_DECIMAL_CANDIDATES = (".", ",")
_ENCODING_CANDIDATES = ("utf-8-sig", "cp1251")


@dataclass(frozen=True)
class SniffedDialect:
    sep: str
    decimal: str
    encoding: str


def is_key_value_format(path: str | Path, sample_lines: int = 20) -> bool:
    """True if the file looks like `KEY=VALUE` lines (e.g. this repo's low
    case constraint file) rather than a delimited table -- checked before
    trying any delimiter, since KEY=VALUE lines have no column structure."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()][:sample_lines]
    if not lines:
        return False
    return all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line) for line in lines)


def sniff_dialect(path: str | Path, nrows: int = 200) -> SniffedDialect:
    """Try each (sep, decimal, encoding) combination and score by: parses
    without error, has more than one column (a real delimiter was found),
    maximizes the fraction of columns that parse as true numeric dtype
    (the tell for a correct decimal separator -- wrong-decimal numbers
    parse as strings, not NaN, so NaN-rate alone can't distinguish them),
    and minimizes NaN rate as the final tiebreak."""
    best: tuple[tuple[int, float, float], SniffedDialect] | None = None
    for encoding in _ENCODING_CANDIDATES:
        for sep in _SEP_CANDIDATES:
            for decimal in _DECIMAL_CANDIDATES:
                if sep == decimal:
                    continue
                try:
                    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding,
                                      engine="c", nrows=nrows)
                except Exception:
                    continue
                n_cols = df.shape[1]
                if n_cols <= 1:
                    continue  # no real delimiter found with this candidate
                numeric_frac = float(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)) / n_cols
                nan_frac = float(df.isna().mean().mean()) if n_cols else 1.0
                score = (n_cols, numeric_frac, -nan_frac)
                if best is None or score > best[0]:
                    best = (score, SniffedDialect(sep=sep, decimal=decimal, encoding=encoding))
    if best is None:
        # No multi-column delimiter worked -- likely a KEY=VALUE file or
        # single-column data, handled by the caller via is_key_value_format.
        return SniffedDialect(sep=",", decimal=".", encoding="utf-8-sig")
    return best[1]


def read_sniffed(path: str | Path, dialect: SniffedDialect | None = None, **kwargs) -> pd.DataFrame:
    dialect = dialect or sniff_dialect(path)
    df = pd.read_csv(path, sep=dialect.sep, decimal=dialect.decimal,
                      encoding=dialect.encoding, engine="c", **kwargs)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def detect_shape(df: pd.DataFrame, subject_id_column: str) -> str:
    """"wide" (one row per subject, dimensions embedded in column names --
    this repo's low case) vs "long" (multiple option-rows per subject,
    dimensions are their own columns -- med/hard). A subject that never
    repeats and a table where every row is a distinct subject is the
    structural signature of "wide"."""
    n = len(df)
    n_unique = df[subject_id_column].nunique(dropna=False)
    return "wide" if n_unique == n else "long"


_METRIC_CLUSTER_RE = re.compile(r"^([A-Za-z]+)_(.+)$")


def discover_metric_clusters(columns: list[str]) -> dict[str, dict[str, str]]:
    """Generic version of reshape_low.py's hardcoded `_METRIC_RE` alternation
    (`^(PREMIUM|PROBA|MARGIN|COST)_(.+)$`): split each column name at its
    first underscore into (metric, suffix), group by metric. A metric with
    >=2 members is a real value-component cluster -- discovers
    {"PREMIUM": {...}, "PROBA": {...}, "MARGIN": {...}, "COST": {...}}
    without ever naming those words in code, so a differently-worded wide
    dataset clusters the same way."""
    clusters: dict[str, dict[str, str]] = {}
    for col in columns:
        m = _METRIC_CLUSTER_RE.match(col)
        if not m:
            continue
        metric, suffix = m.group(1), m.group(2)
        clusters.setdefault(metric, {})[suffix] = col
    return {metric: suffixes for metric, suffixes in clusters.items() if len(suffixes) >= 2}


def _split_suffix(suffix: str, anchor_values: set[str]) -> tuple[str | None, str]:
    if suffix in anchor_values:
        return None, suffix
    for v in anchor_values:
        if suffix.endswith("_" + v):
            return suffix[: -(len(v) + 1)], v
    raise ValueError(f"could not resolve column suffix {suffix!r} against anchor values {anchor_values}")


def reshape_wide_to_long(wide: pd.DataFrame, subject_id_column: str) -> pd.DataFrame:
    """Generic version of reshape_low.py's `load_low_offers`: cluster
    columns into value-component metrics by shared prefix
    (`discover_metric_clusters`), pick the cluster with the fewest distinct
    suffixes as the "anchor" dimension (generalizes reshape_low.py's
    hardcoded insight that "COST_* columns are channel-only" -- the anchor
    is whichever metric doesn't vary by the other, larger dimension), and
    split every other metric's suffix into (residual_dim, anchor_dim) using
    the anchor's own vocabulary. Produces a long/tidy frame with columns
    `subject_id`, `dim_anchor`, `dim_residual` (only if more than one
    dimension was actually found), plus one column per discovered metric."""
    clusters = discover_metric_clusters(list(wide.columns))
    if not clusters:
        raise ValueError("no wide-pivoted metric clusters discovered in columns")

    anchor_metric = min(clusters, key=lambda m: len(clusters[m]))
    anchor_values = set(clusters[anchor_metric].keys())
    non_anchor_metrics = [m for m in clusters if m != anchor_metric]

    # (residual, anchor) -> source column, one map per non-anchor metric.
    parsed: dict[str, dict[tuple[str | None, str], str]] = {}
    for metric in non_anchor_metrics:
        parsed[metric] = {}
        for suffix, col in clusters[metric].items():
            key = _split_suffix(suffix, anchor_values)
            parsed[metric][key] = col

    # Combos are keys present in *every* non-anchor metric (mirrors
    # reshape_low.py's PREMIUM/PROBA/MARGIN intersection) -- the anchor
    # metric's value is looked up per combo separately below, since it's
    # expected to broadcast across the residual dimension rather than
    # define its own combos.
    combos: set[tuple[str | None, str]] | None = None
    for mapping in parsed.values():
        keys = set(mapping.keys())
        combos = keys if combos is None else combos & keys
    combos = combos or set()

    has_residual = any(residual is not None for residual, _ in combos)
    frames = []
    for offer_idx, (residual, anchor) in enumerate(
        sorted(combos, key=lambda k: (str(k[0]), k[1])), start=1
    ):
        data: dict[str, object] = {"subject_id": wide[subject_id_column], "dim_anchor": anchor}
        if has_residual:
            data["dim_residual"] = residual
        for metric in non_anchor_metrics:
            data[metric] = wide[parsed[metric][(residual, anchor)]]
        anchor_col = clusters[anchor_metric].get(anchor) or clusters[anchor_metric].get(f"{residual}_{anchor}")
        if anchor_col is not None:
            data[anchor_metric] = wide[anchor_col]
        data["_offer_slot"] = offer_idx
        frames.append(pd.DataFrame(data))

    long_df = pd.concat(frames, ignore_index=True)
    return long_df.sort_values(["subject_id", "_offer_slot"]).reset_index(drop=True)
