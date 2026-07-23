"""Build the canonical OfferTable (+ its ConstraintSet) for any of the 3
cases. Each case's EV formula genuinely differs (this is a fact of the
source data, stated in the readme, not something to paper over) -- that
case-specific arithmetic lives here, once, so that everything downstream
(solver, verifier) only ever sees the resulting `base_ev`/`cost` columns and
never needs to know which formula produced them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from offer_opt.constraints import resolve_all
from offer_opt.io import raw_constraints as rc
from offer_opt.io import reshape_low
from offer_opt.io.dialects import read_offers
from offer_opt.io.dialects import read_constraints_table
from offer_opt.schema import ConstraintSet, ConstraintSpec


def _join_parameter(offer_table: pd.DataFrame, constraint_set: ConstraintSet, kind: str, dim: str) -> pd.Series:
    lookup = {p.scope[dim]: p.value for p in constraint_set.parameters if p.kind == kind}
    missing = set(offer_table[dim].unique()) - set(lookup)
    if missing:
        raise KeyError(f"no {kind!r} parameter found for {dim}={missing}")
    return offer_table[dim].map(lookup).astype("float64")


def _load_low() -> tuple[pd.DataFrame, ConstraintSet]:
    offer_table, _combos = reshape_low.load_low_offers()
    constraint_set = resolve_all(reshape_low.load_low_constraints())
    # low's own reference solution is a single categorical Offer column --
    # its source table is a genuinely pivoted single-decision table, and
    # that structural shape (not a general business-rule default) is what
    # implies a client picks at most one of its up-to-4 candidate offers.
    # This is synthesized only here, never for med/hard.
    constraint_set.constraints.append(
        ConstraintSpec(id="client_total_cap{}", raw_type="_synthetic_low_pivot_cap",
                        scope={}, measure="count", min=None, max=1.0, per_client=True)
    )
    # EV = Margin * Premium * Response(proba) - Cost (readme: "формула ЕВ=премиум*проба*маржин -кост")
    offer_table["base_ev"] = offer_table["margin"] * offer_table["premium"] * offer_table["score"] - offer_table["cost"]
    return offer_table, constraint_set


def _load_med() -> tuple[pd.DataFrame, ConstraintSet]:
    raw = read_offers("med")
    constraint_set = resolve_all(rc.from_table(read_constraints_table("med")))

    offer_table = pd.DataFrame(
        {
            "client_id": raw["SUBJISN"].astype("int64"),
            "product": raw["PRODUCT"].astype(str),
            "channel": raw["CHANNEL"].astype(str),
            "segment": raw["SEGMENT"].astype(str),
            "score": raw["SCORE"].astype("float64"),
            "premium": np.nan,
            "margin": np.nan,
        }
    )
    offer_table["cost"] = _join_parameter(offer_table, constraint_set, "cost", "channel")
    # SCORE already = premium*proba*margin combined (readme: "скор= он же премиум*проба*маржин").
    # EV = SCORE - Cost (readme: "формула ЕВ=скор -кост").
    offer_table["base_ev"] = offer_table["score"] - offer_table["cost"]
    offer_table["offer_uid"] = np.arange(len(offer_table), dtype="int64")
    return offer_table, constraint_set


def _load_hard() -> tuple[pd.DataFrame, ConstraintSet]:
    raw = read_offers("hard")
    constraint_set = resolve_all(rc.from_table(read_constraints_table("hard")))

    offer_table = pd.DataFrame(
        {
            "client_id": raw["SUBJISN"].astype("int64"),
            "product": raw["PRODUCT"].astype(str),
            "channel": raw["CHANNEL"].astype(str),
            "segment": raw["SEGMENT"].astype(str),
            "score": raw["SCORE"].astype("float64"),
            "premium": raw["AVG_CHECK"].astype("float64"),
        }
    )
    offer_table["cost"] = _join_parameter(offer_table, constraint_set, "cost", "channel")
    offer_table["margin"] = _join_parameter(offer_table, constraint_set, "margin", "product")
    # EV = Score * AVG_CHECK * Margin - Cost (readme: "формула ЕВ=скор*авг_чек*маржин -кост")
    offer_table["base_ev"] = offer_table["score"] * offer_table["premium"] * offer_table["margin"] - offer_table["cost"]
    offer_table["offer_uid"] = np.arange(len(offer_table), dtype="int64")
    return offer_table, constraint_set


_LOADERS = {"low": _load_low, "med": _load_med, "hard": _load_hard}


def load_case(case: str) -> tuple[pd.DataFrame, ConstraintSet]:
    if case not in _LOADERS:
        raise ValueError(f"unknown case {case!r}, expected one of {list(_LOADERS)}")
    return _LOADERS[case]()


def _build_offer_table_from_mapping(df: pd.DataFrame, mapping) -> tuple[pd.DataFrame, list[str]]:
    """Canonical offer_table columns from a `SchemaMapping` instead of
    hand-written per-case column names. Every discovered dimension column is
    carried through under its own (lowercased) name -- not just a hardcoded
    product/channel/segment -- so a 4th (or 5th, ...) dimension a future
    dataset introduces needs no code change here. `dim_anchor`/`dim_residual`
    are `io/sniff.py::reshape_wide_to_long`'s own naming for a reshaped wide
    table's two dimensions -- known here specifically because this module
    controls that reshaping (channel is always the "anchor" broadcast
    dimension in the wide cases seen so far) -- remapped to channel/product
    as the one deliberate special case; segment is synthesized the same way
    `reshape_low.py` already does when no explicit segment column exists,
    since a wide/pivoted source never carries one.

    Returns the built frame plus the ordered list of canonical dimension
    names actually present, for callers (`pipeline.py::run_dataset()`) that
    need to know what to build a `DimensionTree`/scope over without
    re-deriving it from scratch."""
    out = pd.DataFrame({"client_id": df[mapping.subject_id_column].astype("int64")})

    dim_names: list[str] = []
    for col in mapping.dimension_columns:
        if col == "dim_anchor":
            canonical = "channel"
        elif col == "dim_residual":
            canonical = "product"
        else:
            canonical = col.lower()
        out[canonical] = df[col].astype(str)
        dim_names.append(canonical)

    if "segment" not in out.columns and "product" in out.columns and "channel" in out.columns:
        out["segment"] = out["product"] + "_" + out["channel"]
        dim_names.append("segment")

    for role, col in mapping.value_component_columns.items():
        out[role] = df[col].astype("float64")

    return out, dim_names


def _resolve_value(role: str, join_dim: str, offer_table: pd.DataFrame, constraint_set: ConstraintSet) -> pd.Series:
    """A value component either came straight off the source table (already
    present in `offer_table`), or has to be joined from a `ParameterSpec`
    (an EV-formula input disguised as a constraint row -- `cost_of_
    communication`/`margin_per_product`), or is genuinely absent, in which
    case it defaults to the formula's identity element (1.0 for a
    multiplicative factor, 0.0 for the subtracted cost) so the single
    generic formula below produces the same result as each case's own
    hand-written formula regardless of which components it actually uses."""
    if role in offer_table.columns:
        return offer_table[role]
    try:
        return _join_parameter(offer_table, constraint_set, role, join_dim)
    except KeyError:
        return pd.Series(1.0 if role in ("premium", "margin") else 0.0, index=offer_table.index)


def load_from_paths(offers_path, constraints_path, llm_client=None) -> tuple[pd.DataFrame, ConstraintSet, list[str]]:
    """The arbitrary-new-dataset entrypoint: takes raw file *paths* directly
    (not a known `case` name), sniffs their dialect/shape, runs them through
    discovery (schema resolution, generic reshaping) and constraint
    resolution, and returns the same canonical `(offer_table, ConstraintSet)`
    shape every other loader in this module produces -- plus the ordered
    list of dimension names actually discovered, for `pipeline.py::
    run_dataset()` to build a `DimensionTree` over without re-deriving it.

    This is `load_generic()`'s core, extracted so it no longer needs a
    `case` name / `CASE_FILES` lookup at all -- `load_generic(case)` is now
    a thin wrapper calling this with `CASE_FILES[case]`'s paths."""
    from offer_opt.discovery.schema_resolver import resolve_schema
    from offer_opt.io import sniff

    dialect = sniff.sniff_dialect(offers_path)
    raw_offers = sniff.read_sniffed(offers_path, dialect)

    probe_mapping = resolve_schema(raw_offers, llm_client=llm_client)
    shape = sniff.detect_shape(raw_offers, probe_mapping.subject_id_column)

    if shape == "wide":
        working = sniff.reshape_wide_to_long(raw_offers, subject_id_column=probe_mapping.subject_id_column)
        mapping = resolve_schema(working, llm_client=llm_client)
    else:
        working = raw_offers
        mapping = probe_mapping

    offer_table, dim_names = _build_offer_table_from_mapping(working, mapping)
    if "_offer_slot" in working.columns:
        # Not part of the canonical schema -- only consumed by
        # metrics.py::reference_selection_low, which needs to know each
        # row's source-column position to decode the vendor's p1..p4/Offer
        # reference format. Harmless passenger column for every other case.
        offer_table["_offer_slot"] = working["_offer_slot"].to_numpy()

    if sniff.is_key_value_format(constraints_path):
        anchor_values = set(working["dim_anchor"].unique())
        residual_values = set(working["dim_residual"].unique()) if "dim_residual" in working.columns else set()
        raw_rows = sniff.parse_key_value_constraints(constraints_path, anchor_values, residual_values)
    else:
        cdialect = sniff.sniff_dialect(constraints_path)
        raw_constraints_df = sniff.read_sniffed(constraints_path, cdialect)
        raw_rows = rc.from_table(raw_constraints_df)

    constraint_set = resolve_all(raw_rows, dims=tuple(dim_names), llm_client=llm_client)

    if shape == "wide":
        # Mirrors _load_low's synthetic per-client cap: a wide/pivoted
        # source table (one row per client, one column per option) implies
        # a client picks at most one of its candidate offers -- a structural
        # fact about this file shape, not a general business-rule default.
        constraint_set.constraints.append(
            ConstraintSpec(id="client_total_cap{}", raw_type="_synthetic_wide_pivot_cap",
                            scope={}, measure="count", min=None, max=1.0, per_client=True)
        )

    offer_table["premium"] = _resolve_value("premium", "product", offer_table, constraint_set)
    offer_table["margin"] = _resolve_value("margin", "product", offer_table, constraint_set)
    offer_table["cost"] = _resolve_value("cost", "channel", offer_table, constraint_set)
    # Same generic formula for every case: EV = Margin * Premium * Score - Cost.
    # A component genuinely absent from the source (Section "_resolve_value")
    # defaults to its identity element, so this reduces to exactly each
    # case's own hand-written formula -- e.g. med's premium/margin default to
    # 1.0, giving base_ev = score - cost, matching _load_med precisely.
    offer_table["base_ev"] = offer_table["margin"] * offer_table["premium"] * offer_table["score"] - offer_table["cost"]
    offer_table["offer_uid"] = np.arange(len(offer_table), dtype="int64")

    return offer_table, constraint_set, dim_names


def load_generic(case: str, llm_client=None) -> tuple[pd.DataFrame, ConstraintSet]:
    """Generic entrypoint for the 3 known cases specifically: the same
    `(offer_table, ConstraintSet)` shape as `load_case()`, built via
    `load_from_paths()` from `CASE_FILES[case]`'s paths -- a generalization
    *proof* that the discovery/constraints pieces built in Phases 2-4
    compose correctly on data whose columns are read generically rather
    than assumed. For an arbitrary new dataset, use `load_from_paths()`
    directly (or `pipeline.py::run_dataset()`, which wraps it further)."""
    from offer_opt.io.dialects import CASE_FILES

    offer_table, constraint_set, _dim_names = load_from_paths(
        CASE_FILES[case]["offers"], CASE_FILES[case]["constraints"], llm_client=llm_client)
    return offer_table, constraint_set


def encode_dims(offer_table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add a dense 0..C-1 `client_idx` column (what the solver groups on) and
    return the number of distinct clients."""
    codes, _uniques = pd.factorize(offer_table["client_id"], sort=False)
    offer_table = offer_table.copy()
    offer_table["client_idx"] = codes.astype("int64")
    num_clients = int(codes.max()) + 1 if len(codes) else 0
    return offer_table, num_clients
