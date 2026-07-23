import pytest

from offer_opt import features as F
from offer_opt import metrics as M
from offer_opt import verify as V
from offer_opt.device import get_device
from offer_opt.solver.lagrangian import solve

CASES = ["low", "med", "hard"]

# Loose, machine-independent floors for the fast (default) tier -- set below
# what a 60-iteration solve actually achieves (measured: med ~0.65, hard
# ~0.92 of reference EV) so this stays fast and non-flaky while still
# catching a real regression (e.g. the "-40% of reference" incumbent-
# tracking bug found during development, which this would have caught).
FAST_EV_FLOOR = {"low": 0.90, "med": 0.45, "hard": 0.80}

# Tight floors for the slow tier, set just below what the full iteration
# budget actually achieves (measured via notebooks/solution.ipynb: low
# 100.00%, med 100.02%, hard 94.89% of reference EV).
FULL_EV_FLOOR = {"low": 0.999, "med": 0.98, "hard": 0.90}
FULL_SOLVE_KWARGS = {"low": {}, "med": {}, "hard": dict(max_iters=400, repair_every=20)}


def _solve_case(case, device, **kwargs):
    df, cs = F.load_case(case)
    df, _n_clients = F.encode_dims(df)
    result = solve(df, cs, device, **kwargs)
    ref_ev = V.verify(df, cs, M.load_reference(case, df)).total_ev
    return df, cs, result, ref_ev


@pytest.mark.parametrize("case", CASES)
def test_reference_solution_passes_our_verifier(case):
    """Independent of whether our solver is any good: the vendor's own
    reference solution must satisfy our parsed constraint set. This is the
    strongest cross-check on the parser -- it's ground truth we didn't
    generate ourselves."""
    report = M.verify_reference(case)
    assert report.ok, f"{case} reference failed verification: {report}"


@pytest.mark.parametrize("case", CASES)
def test_solver_produces_a_feasible_and_reasonable_solution(case):
    """Fast tier (default `pytest` run, ~1 min total): feasibility is
    checked exactly, and EV quality is checked against a loose floor -- not
    tight enough to prove near-optimality, but tight enough to catch a
    solver that's badly broken, not just slow to converge."""
    device = get_device(prefer_gpu=False)  # CPU for deterministic, portable test runs
    df, cs, result, ref_ev = _solve_case(case, device, max_iters=60, plateau_patience=10000, repair_every=10)
    report = V.verify(df, cs, result.selection)
    assert report.ok, f"{case} solver output failed verification: {report}"
    assert report.total_ev == result.total_ev
    assert result.total_ev >= ref_ev * FAST_EV_FLOOR[case], (
        f"{case}: only {result.total_ev / ref_ev:.1%} of reference EV at 60 iterations "
        f"(floor {FAST_EV_FLOOR[case]:.0%}) -- solver quality regressed"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES)
def test_solver_matches_reference_ev_at_full_iteration_budget(case):
    """Opt-in tier (`pytest -m slow`, several minutes total, hard alone
    ~2-3 min): the actual "does this really work" proof -- same iteration
    budget as notebooks/solution.ipynb, asserted against tight floors set
    just below the numbers that notebook run actually produced."""
    device = get_device(prefer_gpu=True)
    df, cs, result, ref_ev = _solve_case(case, device, **FULL_SOLVE_KWARGS[case])
    report = V.verify(df, cs, result.selection)
    assert report.ok, f"{case} solver output failed verification: {report}"
    assert result.total_ev >= ref_ev * FULL_EV_FLOOR[case], (
        f"{case}: only {result.total_ev / ref_ev:.1%} of reference EV "
        f"(floor {FULL_EV_FLOOR[case]:.0%}) -- solution quality regressed"
    )
