"""Generic capacitated per-client selection.

Each client's offer choice is a small independent subproblem: given a
per-offer (penalized) value and a handful of `per_client` constraints (each
scoped to some subset of the offers, e.g. "<=3 EMAIL offers", "<=1 OSAGO
offer", or low's synthetic "<=1 offer total"), pick the value-maximizing
subset satisfying all of them.

This degenerates to a plain masked argmax when the only active local rule is
a single scope={} cap of 1 (low's case), and generalizes to med/hard's
"<=1 per product, <=K per channel" case via a bounded demote-until-feasible
fixed point -- all vectorized with scatter_reduce_/scatter_add_, so it never
loops over clients or offers, only over the handful of local constraint rows
and a small bounded number of rounds.
"""

from __future__ import annotations

import torch

from offer_opt.schema import ConstraintSpec
from offer_opt.scope import ScopeIndex

MAX_OUTER_ROUNDS = 15
MAX_DEMOTE_ROUNDS = 15
MAX_FILL_ROUNDS = 15
NEG_INF = float("-inf")
POS_INF = float("inf")


def _mask_tensor(scope_index: ScopeIndex, scope: dict, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(scope_index.mask(scope)).to(device=device)


def _demote_to_max(selected: torch.Tensor, mask_c: torch.Tensor, max_c: float,
                    client_idx: torch.Tensor, value: torch.Tensor, num_clients: int) -> torch.Tensor:
    """Remove the lowest-value selected offer(s) within `mask_c`, one worst
    offer per still-overfull client per round, until every client's usage in
    this constraint's scope is <= max_c. Monotone: never touches clients that
    aren't overfull, never creates a new violation elsewhere."""
    for _ in range(MAX_DEMOTE_ROUNDS):
        subset = selected & mask_c
        usage = torch.zeros(num_clients, device=value.device, dtype=value.dtype)
        usage.scatter_add_(0, client_idx, subset.to(value.dtype))
        overfull = usage > max_c + 1e-9
        if not overfull.any():
            break
        overfull_here = subset & overfull[client_idx]
        masked_value = torch.where(overfull_here, value, torch.full_like(value, POS_INF))
        group_min = torch.full((num_clients,), POS_INF, device=value.device, dtype=value.dtype)
        group_min.scatter_reduce_(0, client_idx, masked_value, reduce="amin", include_self=True)
        is_min = overfull_here & (value == group_min[client_idx])
        selected = selected & ~is_min
    return selected


def _fill_to_min(selected: torch.Tensor, mask_c: torch.Tensor, min_c: float,
                  client_idx: torch.Tensor, value: torch.Tensor, active: torch.Tensor,
                  num_clients: int) -> torch.Tensor:
    """Add the highest-value not-yet-selected active offer(s) within
    `mask_c`, one per still-shortfall client per round, until usage >= min_c
    or no eligible candidates remain (an unattainable min bound is left for
    the verifier/repair stage to report honestly, not silently forced)."""
    for _ in range(MAX_FILL_ROUNDS):
        subset = selected & mask_c
        usage = torch.zeros(num_clients, device=value.device, dtype=value.dtype)
        usage.scatter_add_(0, client_idx, subset.to(value.dtype))
        short = usage < min_c - 1e-9
        if not short.any():
            break
        candidates = active & mask_c & ~selected & short[client_idx]
        if not candidates.any():
            break
        masked_value = torch.where(candidates, value, torch.full_like(value, NEG_INF))
        group_max = torch.full((num_clients,), NEG_INF, device=value.device, dtype=value.dtype)
        group_max.scatter_reduce_(0, client_idx, masked_value, reduce="amax", include_self=True)
        is_max = candidates & (value == group_max[client_idx])
        if not is_max.any():
            break
        selected = selected | is_max
    return selected


def local_select(value: torch.Tensor, client_idx: torch.Tensor, num_clients: int,
                  local_constraints: list[ConstraintSpec], scope_index: ScopeIndex,
                  device: torch.device) -> torch.Tensor:
    active = value > 0
    selected = active.clone()

    mask_cache = {id(c): _mask_tensor(scope_index, c.scope, device) for c in local_constraints}

    for _ in range(MAX_OUTER_ROUNDS):
        changed = False
        for c in local_constraints:
            mask_c = mask_cache[id(c)]
            if c.max is not None:
                before = selected
                selected = _demote_to_max(selected, mask_c, c.max, client_idx, value, num_clients)
                if not torch.equal(before, selected):
                    changed = True
            if c.min is not None and c.min > 0:
                before = selected
                selected = _fill_to_min(selected, mask_c, c.min, client_idx, value, active, num_clients)
                if not torch.equal(before, selected):
                    changed = True
        if not changed:
            break

    return selected
