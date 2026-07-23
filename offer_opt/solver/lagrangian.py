"""Orchestration: dualize global constraints, alternate the vectorized local
selection step with a subgradient multiplier update, track the best
near-feasible incumbent, and repair it once at the end into a strictly
feasible final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from offer_opt.device import dtype_for
from offer_opt.schema import ConstraintSet
from offer_opt.scope import ScopeIndex
from offer_opt.solver.dual import build_global_states, compute_penalty, update_multipliers
from offer_opt.solver.local import local_select
from offer_opt.solver.repair import repair


@dataclass
class SolveResult:
    selection: np.ndarray  # bool[M], final feasible (or best-effort) answer
    total_ev: float
    iterations: int
    converged: bool
    repair_log: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def solve(offer_table: pd.DataFrame, constraint_set: ConstraintSet, device: torch.device,
          max_iters: int = 1200, step0: float = 3.0, tol: float = 1e-3,
          stable_patience: int = 5, plateau_patience: int = 40, repair_every: int = 10) -> SolveResult:
    n = len(offer_table)
    if "client_idx" not in offer_table.columns:
        raise ValueError("offer_table must be pre-encoded via features.encode_dims()")

    dtype = dtype_for(device)
    client_idx = torch.from_numpy(offer_table["client_idx"].to_numpy().copy()).to(device=device, dtype=torch.int64)
    num_clients = int(offer_table["client_idx"].max()) + 1 if n else 0
    base_ev_np = offer_table["base_ev"].to_numpy()
    base_ev = torch.from_numpy(base_ev_np.copy()).to(device=device, dtype=dtype)
    cost = torch.from_numpy(offer_table["cost"].to_numpy().copy()).to(device=device, dtype=dtype)

    scope_index = ScopeIndex(offer_table)
    local_constraints = constraint_set.local()
    global_constraints = constraint_set.global_()
    global_states = build_global_states(global_constraints, cost, scope_index, device)

    # A raw iterate is almost never *exactly* feasible while the dual is
    # still oscillating (diminishing-step subgradient methods don't land on
    # feasibility, they converge around it) -- gating the incumbent on
    # max_rel_violation < tol means it fires on rare, low-EV, coincidentally
    # "clean" snapshots instead of the much richer near-converged ones.
    # Repairing periodically and comparing the REPAIRED objective is what
    # actually tracks solution quality.
    best_ev = -float("inf")
    best_selection_np: np.ndarray | None = None
    best_repair_log: list[str] = []
    history: list[dict] = []
    stable_count = 0
    plateau_count = 0
    t = -1

    def _checkpoint(selected: torch.Tensor) -> None:
        nonlocal best_ev, best_selection_np, best_repair_log
        sel_np = selected.cpu().numpy().astype(bool)
        repaired_np, log = repair(offer_table, constraint_set, sel_np, scope_index)
        ev = float((base_ev_np * repaired_np).sum())
        if ev > best_ev + 1e-6:
            best_ev = ev
            best_selection_np = repaired_np
            best_repair_log = log

    for t in range(max_iters):
        penalty = compute_penalty(global_states, n, device, base_ev.dtype)
        value_t = base_ev + penalty

        selected = local_select(value_t, client_idx, num_clients, local_constraints, scope_index, device)

        step_t = step0 / ((t + 1) ** 0.5)
        max_rel_violation = update_multipliers(global_states, selected, step_t)
        raw_ev = float(torch.dot(base_ev, selected.to(base_ev.dtype)).item())
        history.append(dict(iter=t, max_rel_violation=max_rel_violation, raw_ev=raw_ev))

        stable_count = stable_count + 1 if max_rel_violation < tol else 0

        is_checkpoint = (t % repair_every == 0) or (t == max_iters - 1) or (stable_count >= stable_patience)
        if is_checkpoint:
            prev_best = best_ev
            _checkpoint(selected)
            plateau_count = plateau_count + 1 if best_ev <= prev_best + 1e-6 else 0

        if stable_count >= stable_patience or plateau_count >= plateau_patience:
            break

    converged = stable_count >= stable_patience
    return SolveResult(selection=best_selection_np, total_ev=best_ev, iterations=t + 1,
                        converged=converged, repair_log=best_repair_log, history=history)
