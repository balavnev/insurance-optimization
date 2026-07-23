"""Deterministic post-convergence feasibility repair.

Local (per_client) constraints are always exactly satisfied by construction
(local_select enforces them every outer iteration) -- this module only has
to fix GLOBAL constraints, which can still be off after the subgradient loop
stops (dual iterates oscillate around feasibility, they don't land on it
exactly).

Max violations: drop the least value-efficient selected offers in scope
until satisfied -- monotone-safe, this only ever reduces usage everywhere,
so it can't create a new violation.

Min violations: greedily add the highest-value eligible offers in scope,
but only if doing so doesn't push any OTHER global constraint over its max,
and doesn't push the offer's client over any LOCAL per-client cap it
participates in. If a bound truly can't be met without breaking something
else, it's reported rather than silently violated or forced.

Every check here is O(1) against a running `totals`/`local_usage` dict,
never a fresh O(M) array sum -- recomputing a full sum inside the per-
candidate loop is what made this unusable on the 5M-row hard case (a single
repair() call would run for minutes instead of ~1s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from offer_opt.schema import ConstraintSet
from offer_opt.scope import ScopeIndex

EPS = 1e-6


def _usage_vec(mask: np.ndarray, measure: str, cost: np.ndarray) -> np.ndarray:
    return mask.astype("float64") if measure == "count" else np.where(mask, cost, 0.0)


def repair(offer_table: pd.DataFrame, constraint_set: ConstraintSet, selection: np.ndarray,
           scope_index: ScopeIndex | None = None, eps: float = EPS) -> tuple[np.ndarray, list[str]]:
    sel = np.asarray(selection, dtype=bool).copy()
    base_ev = offer_table["base_ev"].to_numpy()
    cost = offer_table["cost"].to_numpy()
    client_ids = offer_table["client_id"].to_numpy()
    if scope_index is None:
        scope_index = ScopeIndex(offer_table)

    log: list[str] = []
    global_constraints = [c for c in constraint_set.constraints if not c.per_client]
    local_constraints = [c for c in constraint_set.constraints if c.per_client and c.max is not None]

    g_masks = {id(c): scope_index.mask(c.scope) for c in global_constraints}
    l_masks = {id(c): scope_index.mask(c.scope) for c in local_constraints}
    usage_vecs = {id(c): _usage_vec(g_masks[id(c)], c.measure, cost) for c in global_constraints}
    totals = {id(c): float((usage_vecs[id(c)] * sel).sum()) for c in global_constraints}

    def _touching(j: int) -> list:
        # Computed lazily, only for offers actually added/removed below --
        # bounded by the number of repair actions, never all M offers.
        return [c for c in global_constraints if g_masks[id(c)][j]]

    def _apply(j: int, add: bool) -> None:
        sel[j] = add
        sign = 1.0 if add else -1.0
        for c2 in _touching(j):
            uv2 = usage_vecs[id(c2)][j]
            if uv2:
                totals[id(c2)] += sign * uv2

    # --- 1) fix max violations (monotone: only ever removes offers) ---
    for c in global_constraints:
        if c.max is None or totals[id(c)] <= c.max + eps:
            continue
        uv = usage_vecs[id(c)]
        idx = np.nonzero(g_masks[id(c)] & sel)[0]
        efficiency = base_ev[idx] / np.maximum(cost[idx], 1e-9) if c.measure == "cost" else base_ev[idx]
        order = idx[np.argsort(efficiency)]  # ascending -- worst first
        for j in order:
            if totals[id(c)] <= c.max + eps:
                break
            _apply(int(j), add=False)
        if totals[id(c)] > c.max + eps:
            log.append(f"UNRESOLVED max violation: {c.id} total={totals[id(c)]:.2f} > {c.max:.2f}")

    # --- 2) fix min violations (greedy add, O(1)-checked against every other global max + local caps) ---
    needs_min_fix = [c for c in global_constraints if c.min is not None]
    if needs_min_fix:
        local_usage = {
            id(c): pd.Series((l_masks[id(c)] & sel).astype("float64")).groupby(client_ids).sum()
            for c in local_constraints
        }
        for c in needs_min_fix:
            if totals[id(c)] >= c.min - eps:
                continue
            uv = usage_vecs[id(c)]
            candidates = np.nonzero(g_masks[id(c)] & ~sel)[0]
            order = candidates[np.argsort(-base_ev[candidates])]  # descending value
            for j in order:
                if totals[id(c)] >= c.min - eps:
                    break
                j = int(j)
                cj = client_ids[j]
                ok = all(
                    c2 is c or c2.max is None or totals[id(c2)] + usage_vecs[id(c2)][j] <= c2.max + eps
                    for c2 in _touching(j)
                )
                if ok:
                    for c2 in local_constraints:
                        if l_masks[id(c2)][j] and local_usage[id(c2)].get(cj, 0.0) + 1.0 > c2.max + eps:
                            ok = False
                            break
                if not ok:
                    continue
                _apply(j, add=True)
                for c2 in local_constraints:
                    if l_masks[id(c2)][j]:
                        local_usage[id(c2)][cj] = local_usage[id(c2)].get(cj, 0.0) + 1.0
            if totals[id(c)] < c.min - eps:
                log.append(f"UNRESOLVED min violation: {c.id} total={totals[id(c)]:.2f} < {c.min:.2f}")

    return sel, log
