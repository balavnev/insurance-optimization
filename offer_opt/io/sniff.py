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
    dimensions are their own columns -- med/hard).

    "subject_id never repeats" alone is NOT sufficient to call a table
    wide/pivoted: a long/tidy table where every subject simply happens to
    have exactly one candidate option (a valid, if degenerate, business
    shape) looks identical under that test alone, but has ordinary
    dimension columns of its own and no PREMIUM_X_Y-style column names to
    reshape. The real signature of "wide" is metric-name column clusters
    actually being discoverable (`discover_metric_clusters`) -- require
    both."""
    n = len(df)
    n_unique = df[subject_id_column].nunique(dropna=False)
    if n_unique != n:
        return "long"
    return "wide" if discover_metric_clusters(list(df.columns)) else "long"


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

    # Preserve *source column order*, not an alphabetical sort: a vendor's
    # own reference/output format can encode a selection positionally (this
    # repo's low case: sample_out_low.csv's Offer column is 1..4 by which
    # source column a combo came from, not by product/channel name) -- the
    # first non-anchor metric's dict already preserves that order, since
    # discover_metric_clusters built it by iterating wide.columns in order.
    ordered_keys = list(parsed[non_anchor_metrics[0]].keys())
    combos_ordered = [k for k in ordered_keys if k in combos]

    has_residual = any(residual is not None for residual, _ in combos)
    frames = []
    for offer_idx, (residual, anchor) in enumerate(combos_ordered, start=1):
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


def parse_key_value_constraints(path: str | Path, anchor_values: set[str], residual_values: set[str],
                                  raw_type: str = "offers_per_product"):
    """Generic version of reshape_low.py's KEY=VALUE constraint parsing:
    `<PREFIX>_<residual>_<anchor>=<max>` lines (e.g. "CNT_OSG_SMS=10000"),
    reusing the anchor/residual vocabularies already discovered while
    reshaping the wide offers table (`reshape_wide_to_long`) rather than
    rediscovering them from the constraint file itself.

    The file's own type prefix (e.g. "CNT") is discarded rather than
    per-line classified: a KEY=VALUE constraints file is, by construction,
    always a per-(residual, anchor) count cap -- that's a property of the
    file *shape* (the same spirit as this module's docstring: the pattern is
    generalized, not the literal words), not a per-row classification
    problem, so it's translated directly to `raw_type` and flows through
    `constraints.resolve_one` exactly like any other "offers_per_product"
    row -- no new code needed there."""
    from offer_opt.schema import RawConstraintRow

    text = Path(path).read_text(encoding="utf-8-sig")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        residual_part, anchor = _split_suffix(key, anchor_values)
        product = None
        if residual_part is not None and residual_values:
            _, product = _split_suffix(residual_part, residual_values)
        rows.append(RawConstraintRow(raw_type=raw_type, channel=anchor, product=product,
                                       min=None, max=float(value)))
    return rows
