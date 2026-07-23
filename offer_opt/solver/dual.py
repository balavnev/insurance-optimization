"""Global (cross-client) constraint handling: precompute scope masks and
usage vectors once, then each outer iteration cheaply (a) totals usage under
the current selection, (b) subgradient-updates the multipliers, (c) folds
the multiplier-weighted usage into a per-offer penalty added to `base_ev`.

Only ~60-90 constraints total, so the per-constraint Python loop here is
negligible next to the O(M) tensor ops it triggers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from offer_opt.schema import ConstraintSpec
from offer_opt.scope import ScopeIndex


@dataclass
class GlobalConstraintState:
    spec: ConstraintSpec
    usage_vec: torch.Tensor  # per-offer usage if selected: 1.0 (count) or cost (cost), 0 outside scope
    scale: float             # bound-relative normalization for the step size
    lam_max: float = 0.0
    lam_min: float = 0.0


def build_global_states(global_constraints: list[ConstraintSpec], offer_table_cost: torch.Tensor,
                         scope_index: ScopeIndex, device: torch.device) -> list[GlobalConstraintState]:
    states = []
    for c in global_constraints:
        mask = torch.from_numpy(scope_index.mask(c.scope)).to(device=device)
        if c.measure == "count":
            usage_vec = mask.to(offer_table_cost.dtype)
        else:  # "cost"
            usage_vec = torch.where(mask, offer_table_cost, torch.zeros_like(offer_table_cost))
        bound_ref = c.max if c.max is not None else c.min
        scale = max(1.0, abs(bound_ref)) if bound_ref is not None else 1.0
        states.append(GlobalConstraintState(spec=c, usage_vec=usage_vec, scale=scale))
    return states


def compute_penalty(states: list[GlobalConstraintState], num_offers: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    penalty = torch.zeros(num_offers, device=device, dtype=dtype)
    for s in states:
        coeff = s.lam_min - s.lam_max
        if coeff != 0.0:
            penalty += coeff * s.usage_vec
    return penalty


def update_multipliers(states: list[GlobalConstraintState], selected: torch.Tensor, step_t: float) -> float:
    """Subgradient ascent step. The raw (possibly negative) slack is what
    must drive the update -- clamping it to >=0 before applying would let
    a multiplier only ever grow, never relax back down once its constraint
    goes slack, which stalls the whole dual iteration into permanent
    over-suppression. Only the multiplier itself is floored at 0 (standard
    for inequality-constraint Lagrange multipliers). Returns the max
    relative *violation* (positive part only) as the convergence signal."""
    max_rel_violation = 0.0
    sel = selected.to(dtype=states[0].usage_vec.dtype) if states else None
    for s in states:
        total = float(torch.dot(s.usage_vec, sel).item())
        c = s.spec
        if c.max is not None:
            slack = total - c.max  # positive = violation, negative = room to spare
            s.lam_max = max(0.0, s.lam_max + step_t * slack / s.scale)
            max_rel_violation = max(max_rel_violation, max(0.0, slack) / s.scale)
        if c.min is not None:
            slack = c.min - total
            s.lam_min = max(0.0, s.lam_min + step_t * slack / s.scale)
            max_rel_violation = max(max_rel_violation, max(0.0, slack) / s.scale)
    return max_rel_violation
