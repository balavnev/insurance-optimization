"""Generic constraint verifier: given ANY offer table + parsed ConstraintSet
+ candidate 0/1 selection, check every constraint uniformly. Dispatches
purely on (scope, measure, min, max, per_client) -- never on constraint
type name or which case produced the data -- so it works unmodified on our
own solver's output, on a reconstructed vendor reference solution, or on a
constraint type that didn't exist when this module was written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from offer_opt.schema import ConstraintSet, DimensionTree
from offer_opt.scope import SCOPE_DIMS, ScopeIndex

EPS = 1e-6


@dataclass(frozen=True)
class Violation:
    constraint_id: str
    scope: dict
    measure: str
    bound: str  # "min" | "max"
    limit: float
    observed: float
    n_offending_clients: int | None = None

    def __str__(self) -> str:
        who = f" ({self.n_offending_clients} clients)" if self.n_offending_clients is not None else ""
        return (f"[{self.constraint_id}] {self.bound}={self.limit:g} violated: "
                f"observed={self.observed:g}{who}")


@dataclass
class VerificationReport:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    total_ev: float = 0.0
    n_selected: int = 0

    def __str__(self) -> str:
        lines = [f"{'PASS' if self.ok else 'FAIL'} -- total_ev={self.total_ev:,.2f}, n_selected={self.n_selected}"]
        for v in self.violations:
            lines.append(f"  {v}")
        return "\n".join(lines)


def _usage(offer_table: pd.DataFrame, mask: np.ndarray, measure: str) -> np.ndarray:
    if measure == "count":
        return mask.astype("float64")
    if measure == "cost":
        return np.where(mask, offer_table["cost"].to_numpy(), 0.0)
    raise ValueError(f"unknown measure {measure!r}")


def verify(offer_table: pd.DataFrame, constraint_set: ConstraintSet, selection: np.ndarray,
           eps: float = EPS, trees: dict[str, DimensionTree] | None = None,
           dims: tuple[str, ...] | None = None) -> VerificationReport:
    """`trees`/`dims` are optional and default to `None`/`SCOPE_DIMS` exactly
    as before -- every existing call site (none of which pass them) keeps
    verifying against flat/trivial-tree scopes unchanged. A caller that
    discovered a real dimension hierarchy (`pipeline.py::run_dataset()`)
    passes both through so scope matching is ancestor-aware here too."""
    sel = np.asarray(selection, dtype="float64")
    if sel.shape[0] != len(offer_table):
        raise ValueError(f"selection length {sel.shape[0]} != offer_table length {len(offer_table)}")

    violations: list[Violation] = []
    client_ids = offer_table["client_id"].to_numpy()
    scope_index = ScopeIndex(offer_table, trees=trees, dims=dims or SCOPE_DIMS)

    for c in constraint_set.constraints:
        mask = scope_index.mask(c.scope)
        usage = _usage(offer_table, mask, c.measure) * sel

        if c.per_client:
            agg = pd.Series(usage).groupby(client_ids, sort=False).sum()
            if c.max is not None:
                bad = agg[agg > c.max + eps]
                if len(bad):
                    violations.append(Violation(c.id, c.scope, c.measure, "max", c.max,
                                                 float(bad.max()), int(len(bad))))
            if c.min is not None:
                bad = agg[agg < c.min - eps]
                if len(bad):
                    violations.append(Violation(c.id, c.scope, c.measure, "min", c.min,
                                                 float(bad.min()), int(len(bad))))
        else:
            total = float(usage.sum())
            if c.max is not None and total > c.max + eps:
                violations.append(Violation(c.id, c.scope, c.measure, "max", c.max, total))
            if c.min is not None and total < c.min - eps:
                violations.append(Violation(c.id, c.scope, c.measure, "min", c.min, total))

    total_ev = float((offer_table["base_ev"].to_numpy() * sel).sum())
    return VerificationReport(ok=not violations, violations=violations,
                               total_ev=total_ev, n_selected=int(sel.sum()))
