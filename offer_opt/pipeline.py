"""Convenience wrapper tying I/O, the solver, and the verifier together --
what the notebook and CLI actually call."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from offer_opt import features as _features
from offer_opt import metrics as _metrics
from offer_opt import verify as _verify
from offer_opt.schema import ConstraintSet
from offer_opt.solver.lagrangian import SolveResult, solve


@dataclass
class CaseResult:
    case: str
    offer_table: pd.DataFrame
    constraint_set: ConstraintSet
    solve_result: SolveResult
    verification: _verify.VerificationReport
    reference_ev: float


def run_case(case: str, device: torch.device, **solve_kwargs) -> CaseResult:
    offer_table, constraint_set = _features.load_case(case)
    offer_table, _n_clients = _features.encode_dims(offer_table)

    result = solve(offer_table, constraint_set, device, **solve_kwargs)
    report = _verify.verify(offer_table, constraint_set, result.selection)

    ref_selection = _metrics.load_reference(case, offer_table)
    reference_ev = _verify.verify(offer_table, constraint_set, ref_selection).total_ev

    return CaseResult(case=case, offer_table=offer_table, constraint_set=constraint_set,
                       solve_result=result, verification=report, reference_ev=reference_ev)
