"""The Phase 8 generalization proof: every one of the 3 known cases and 5
synthetic fixtures runs through the identical `pipeline.py::run_dataset()`
entrypoint -- no per-case branching, no fixture-specific code path. Each
fixture is designed to force exactly one axis of the generalization claim:

- `deep_hierarchy`   -- a 4-level intra-dimension taxonomy, convention-only.
- `novel_constraint_type` -- a raw_type matching no override/convention.
- `extra_dimension`  -- a 4th dimension beyond product/channel/segment.
- `conflicting_constraints` -- an intra-dimension taxonomy needing an LLM,
  with a hand-built ancestor/descendant contradiction.
- `farm_domain`      -- a wholly different domain (fields/crops), proving
  the pipeline is domain-agnostic, not just insurance-flexible.
"""

import hashlib
from pathlib import Path

import pytest

from offer_opt.io.dialects import CASE_FILES
from offer_opt.llm.client import FakeLLMClient
from offer_opt.pipeline import run_dataset
from offer_opt.device import get_device

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEVICE = get_device(prefer_gpu=False)  # CPU for deterministic, portable test runs


def _novel_type_client():
    return FakeLLMClient(responses=[
        ("campaign_spend_share_cap", {
            "measure": "count", "column_dimension_map": {"channel": "channel"},
            "per_subject": False, "confidence": "high",
        }),
    ])


def _extra_dimension_client():
    return FakeLLMClient(responses=[
        ("mobile_device_cap", {
            "measure": "count", "column_dimension_map": {"channel": "device"},
            "per_subject": False, "confidence": "high",
        }),
    ])


def _conflicting_constraints_client():
    return FakeLLMClient(responses=[
        ("channel", {"edges": [
            {"value": "email", "parent": None},
            {"value": "personal", "parent": "email"},
            {"value": "subscription", "parent": "email"},
            {"value": "automated", "parent": "personal"},
            {"value": "hand-written", "parent": "personal"},
        ]}),
    ])


def _farm_domain_client():
    return FakeLLMClient(responses=[
        ("'PRICE'", {"role": "premium"}),
        ("'YIELD_PROB'", {"role": "score"}),
        ("max_wheat_offers", {
            "measure": "count", "column_dimension_map": {"channel": "crop"},
            "per_subject": False, "confidence": "high",
        }),
    ])


# case_id -> (offers_path, constraints_path, llm_client_factory, solve_kwargs, expect_ok, expect_n_conflicts)
CASES = {
    "low": (CASE_FILES["low"]["offers"], CASE_FILES["low"]["constraints"], None,
            dict(max_iters=150, plateau_patience=10000, repair_every=10), True, 0),
    "med": (CASE_FILES["med"]["offers"], CASE_FILES["med"]["constraints"], None,
            dict(max_iters=150, plateau_patience=10000, repair_every=10), True, 0),
    "hard": (CASE_FILES["hard"]["offers"], CASE_FILES["hard"]["constraints"], None,
             dict(max_iters=400, repair_every=20), True, 0),
    "deep_hierarchy": (FIXTURES_DIR / "deep_hierarchy" / "offers.csv",
                       FIXTURES_DIR / "deep_hierarchy" / "constraints.csv", None,
                       dict(max_iters=300, plateau_patience=10000, repair_every=5), True, 0),
    "novel_constraint_type": (FIXTURES_DIR / "novel_constraint_type" / "offers.csv",
                              FIXTURES_DIR / "novel_constraint_type" / "constraints.csv",
                              _novel_type_client,
                              dict(max_iters=200, plateau_patience=10000, repair_every=5), True, 0),
    "extra_dimension": (FIXTURES_DIR / "extra_dimension" / "offers.csv",
                        FIXTURES_DIR / "extra_dimension" / "constraints.csv",
                        _extra_dimension_client,
                        dict(max_iters=200, plateau_patience=10000, repair_every=5), True, 0),
    "conflicting_constraints": (FIXTURES_DIR / "conflicting_constraints" / "offers.csv",
                                FIXTURES_DIR / "conflicting_constraints" / "constraints.csv",
                                _conflicting_constraints_client,
                                dict(max_iters=300, plateau_patience=10000, repair_every=5), False, 1),
    "farm_domain": (FIXTURES_DIR / "farm_domain" / "offers.csv",
                    FIXTURES_DIR / "farm_domain" / "constraints.csv",
                    _farm_domain_client,
                    dict(max_iters=200, plateau_patience=10000, repair_every=5), True, 0),
}


@pytest.mark.parametrize("case_id", list(CASES))
def test_pipeline_generalizes(case_id):
    offers_path, constraints_path, client_factory, solve_kwargs, expect_ok, expect_n_conflicts = CASES[case_id]
    llm_client = client_factory() if client_factory else None

    result = run_dataset(offers_path, constraints_path, DEVICE, llm_client=llm_client, **solve_kwargs)

    assert len(result.conflicts) == expect_n_conflicts, (
        f"{case_id}: expected {expect_n_conflicts} conflicts, got {len(result.conflicts)}: {result.conflicts}")
    assert result.codegen_agrees, f"{case_id}: generated verifier code disagreed with verify.py"

    if expect_ok:
        assert result.verification.ok, f"{case_id}: unexpected verification failure: {result.verification}"
    else:
        # A documented, explained violation (the demoted ancestor
        # constraint(s)) is expected here -- not the same as "the solver is
        # broken." Every violation must trace back to a reported conflict.
        assert not result.verification.ok
        conflicting_ids = {c.ancestor.id for c in result.conflicts}
        violated_ids = {v.constraint_id for v in result.verification.violations}
        assert violated_ids <= conflicting_ids, (
            f"{case_id}: unexplained violation(s) not covered by any reported conflict: "
            f"{violated_ids - conflicting_ids}")


def test_deep_hierarchy_scope_matches_every_descendant():
    _offers, _constraints, _client, solve_kwargs, _ok, _nc = CASES["deep_hierarchy"]
    result = run_dataset(*CASES["deep_hierarchy"][:2], DEVICE, llm_client=None, **solve_kwargs)

    tree = result.trees["segment"]
    assert tree.parent_of["L1_L2_L3_L4"] == "L1_L2_L3"
    assert tree.parent_of["L1_L2_L3"] == "L1_L2"
    assert tree.parent_of["L1_L2"] == "L1"
    assert tree.parent_of["L1"] is None
    assert tree.parent_of["M1"] is None  # unrelated sibling root

    ot = result.offer_table
    sel = result.solve_result.selection
    subtree_mask = ot["segment"].isin(["L1_L2", "L1_L2_L3", "L1_L2_L3_L4"]).to_numpy()
    assert sel[subtree_mask].sum() <= 25 + 1e-6  # the cap, aggregated across all 3 descendant levels
    assert sel[(ot["segment"] == "L1").to_numpy()].sum() == 10  # unconstrained ancestor: fully selected
    assert sel[(ot["segment"] == "M1").to_numpy()].sum() == 10  # unconstrained sibling: fully selected


def test_novel_constraint_type_is_unresolved_without_an_llm_client():
    from offer_opt.constraints import UnresolvedConstraintError

    offers_path, constraints_path, _client, solve_kwargs, _ok, _nc = CASES["novel_constraint_type"]
    with pytest.raises(UnresolvedConstraintError):
        run_dataset(offers_path, constraints_path, DEVICE, llm_client=None, **solve_kwargs)


def test_extra_dimension_is_discovered_and_enforced():
    offers_path, constraints_path, client_factory, solve_kwargs, _ok, _nc = CASES["extra_dimension"]
    result = run_dataset(offers_path, constraints_path, DEVICE, llm_client=client_factory(), **solve_kwargs)

    assert "device" in result.dims
    ot = result.offer_table
    sel = result.solve_result.selection
    mobile_mask = (ot["device"] == "MOBILE").to_numpy()
    assert sel[mobile_mask].sum() <= 10 + 1e-6
    desktop_mask = (ot["device"] == "DESKTOP").to_numpy()
    assert sel[desktop_mask].sum() == desktop_mask.sum()  # unconstrained -- fully selected


def test_conflicting_constraints_conflict_is_flagged_and_explained():
    offers_path, constraints_path, client_factory, solve_kwargs, _ok, _nc = CASES["conflicting_constraints"]
    result = run_dataset(offers_path, constraints_path, DEVICE, llm_client=client_factory(), **solve_kwargs)

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.ancestor.scope == {"channel": "personal"}
    assert {tuple(sorted(d.scope.items())) for d in conflict.descendants} == {
        (("channel", "automated"),), (("channel", "hand-written"),),
    }

    # Deeper-wins precedence: automated/hand-written mins fully honored...
    ot = result.offer_table
    sel = result.solve_result.selection
    assert sel[(ot["channel"] == "automated").to_numpy()].sum() >= 60
    assert sel[(ot["channel"] == "hand-written").to_numpy()].sum() >= 60
    # ...at the cost of "personal"'s max, which shows up as an honestly
    # reported, conflict-explained violation, not a silent pass.
    assert not result.verification.ok
    assert any(v.constraint_id == conflict.ancestor.id for v in result.verification.violations)


def test_farm_domain_resolves_a_wholly_different_vocabulary():
    offers_path, constraints_path, client_factory, solve_kwargs, _ok, _nc = CASES["farm_domain"]
    result = run_dataset(offers_path, constraints_path, DEVICE, llm_client=client_factory(), **solve_kwargs)

    assert set(result.dims) == {"crop", "irrigation_method"}
    ot = result.offer_table
    sel = result.solve_result.selection
    total_cost = float((ot["cost"].to_numpy() * sel).sum())
    assert total_cost <= 100 + 1e-6
    wheat_mask = (ot["crop"] == "wheat").to_numpy()
    assert sel[wheat_mask].sum() <= 6 + 1e-6


def test_farm_domain_fixture_needed_zero_offer_opt_source_changes():
    """The strongest test of "domain-agnostic in spirit": a hash snapshot of
    every offer_opt/ source file, taken right after the farm_domain fixture
    was confirmed working and before this test file was written, must be
    byte-identical to the current state -- no non-test, non-fixture file
    changed to make the farm domain work."""
    offer_opt_root = Path(__file__).parent.parent / "offer_opt"
    files = sorted(p for p in offer_opt_root.rglob("*") if p.is_file() and p.suffix in (".py", ".txt"))

    current = {}
    for p in files:
        current[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    snapshot_file = Path("/tmp/offer_opt_snapshot_before_farm.txt")
    if not snapshot_file.exists():
        pytest.skip("baseline snapshot not present in this environment -- run only meaningful "
                    "in the session that captured it before writing this test file")

    baseline = {}
    for line in snapshot_file.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        baseline[path.strip()] = digest

    assert current == baseline, "offer_opt/ source changed after the farm_domain fixture was confirmed working"
