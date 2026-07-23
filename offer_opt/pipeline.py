"""Convenience wrapper tying I/O, the solver, and the verifier together --
what the notebook and CLI actually call."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

from offer_opt import features as _features
from offer_opt import metrics as _metrics
from offer_opt import verify as _verify
from offer_opt.codegen.generate import GeneratedCheck, cross_check, generate_all
from offer_opt.discovery.conflict import ConstraintConflict, find_conflicts
from offer_opt.discovery.hierarchy import build_tree
from offer_opt.schema import ConstraintSet, DimensionTree
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


@dataclass
class DatasetResult:
    offer_table: pd.DataFrame
    constraint_set: ConstraintSet          # the full, originally-parsed set (used for the honest final report)
    dims: tuple[str, ...]
    trees: dict[str, DimensionTree]
    conflicts: list[ConstraintConflict]
    solve_result: SolveResult
    verification: _verify.VerificationReport
    generated_checks: dict[str, GeneratedCheck] = field(default_factory=dict)
    codegen_agrees: bool = True


def run_dataset(offers_path, constraints_path, device: torch.device,
                 llm_client=None, **solve_kwargs) -> DatasetResult:
    """The arbitrary-new-dataset entrypoint: given raw offers/constraints
    file *paths* (any dataset, any domain -- not just the 3 known cases),
    runs the full pipeline end to end:

      1. generic ingestion (`features.load_from_paths`)
      2. per-dimension hierarchy inference (`discovery.hierarchy.build_tree`)
      3. pre-solve conflict detection (`discovery.conflict.find_conflicts`)
      4/5. the tree-aware solve
      6. verifier code generation + cross-check against the real result
      7. tree-aware verification

    A constraint flagged as the *ancestor* side of a detected conflict is
    excluded from what the solver is actually asked to satisfy (the
    documented "deeper constraint wins" precedence, system_design_overview.md
    Section 3) -- the final `verification` report is still run against the
    FULL original constraint set, so an ancestor bound traded away this way
    shows up there as an honestly-reported violation, explained by
    `conflicts`, not silently hidden."""
    offer_table, constraint_set, dim_names = _features.load_from_paths(
        offers_path, constraints_path, llm_client=llm_client)
    offer_table, _n_clients = _features.encode_dims(offer_table)

    dims = tuple(dim_names)
    trees = {dim: build_tree(dim, offer_table[dim].unique().tolist(), llm_client=llm_client) for dim in dims}

    conflicts = find_conflicts(constraint_set, trees)
    demoted_ids = {c.ancestor.id for c in conflicts}
    solve_constraints = [c for c in constraint_set.constraints if c.id not in demoted_ids]
    solve_constraint_set = ConstraintSet(constraints=solve_constraints, parameters=constraint_set.parameters)

    result = solve(offer_table, solve_constraint_set, device, trees=trees, dims=dims, **solve_kwargs)
    report = _verify.verify(offer_table, constraint_set, result.selection, trees=trees, dims=dims)

    generated = generate_all(constraint_set)
    codegen_agrees = all(cross_check(gc, offer_table, result.selection) for gc in generated.values())

    return DatasetResult(offer_table=offer_table, constraint_set=constraint_set, dims=dims, trees=trees,
                          conflicts=conflicts, solve_result=result, verification=report,
                          generated_checks=generated, codegen_agrees=codegen_agrees)
