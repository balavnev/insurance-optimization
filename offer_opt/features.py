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


def encode_dims(offer_table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add a dense 0..C-1 `client_idx` column (what the solver groups on) and
    return the number of distinct clients."""
    codes, _uniques = pd.factorize(offer_table["client_id"], sort=False)
    offer_table = offer_table.copy()
    offer_table["client_idx"] = codes.astype("int64")
    num_clients = int(codes.max()) + 1 if len(codes) else 0
    return offer_table, num_clients
