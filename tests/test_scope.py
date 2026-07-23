import numpy as np
import pandas as pd
import pytest

from offer_opt import features as F
from offer_opt.schema import DimensionTree
from offer_opt.scope import ScopeIndex, scope_mask

CASES = ["low", "med", "hard"]


def _channel_taxonomy() -> DimensionTree:
    """email -> personal/subscription -> (under personal) automated/hand-written
    -> (under automated) expensive/cheap -- the taxonomy from
    system_design_overview.md Section 3, entirely within one dimension."""
    parent_of = {
        "email": None,
        "personal": "email",
        "subscription": "email",
        "automated": "personal",
        "hand-written": "personal",
        "expensive": "automated",
        "cheap": "automated",
    }
    return DimensionTree(dim="channel", parent_of=parent_of)


def _synthetic_table() -> pd.DataFrame:
    # One row per value, including two rows whose own value is an internal
    # (non-leaf) tree node -- a constraint scoped to a node must also match
    # a row whose own value *is* that node, not only its descendants.
    return pd.DataFrame({
        "channel": ["expensive", "cheap", "hand-written", "subscription", "personal", "email"],
    })


def test_build_intervals_gives_each_value_a_unique_non_overlapping_range():
    tree = _channel_taxonomy()
    tree.build_intervals()
    assert set(tree.tin) == set(tree.parent_of)
    for v in tree.parent_of:
        assert tree.tin[v] <= tree.tout[v]
    # ancestor-or-self intervals must nest: a child's range sits strictly
    # inside its parent's.
    for v, p in tree.parent_of.items():
        if p is None:
            continue
        assert tree.tin[p] <= tree.tin[v] and tree.tout[v] <= tree.tout[p]


@pytest.mark.parametrize(
    "scope_value,expected_channels",
    [
        ("email", {"expensive", "cheap", "hand-written", "subscription", "personal", "email"}),
        ("personal", {"expensive", "cheap", "hand-written", "personal"}),
        ("automated", {"expensive", "cheap"}),
        ("subscription", {"subscription"}),
        ("hand-written", {"hand-written"}),
    ],
)
def test_scope_index_matches_every_descendant_and_nothing_outside(scope_value, expected_channels):
    table = _synthetic_table()
    tree = _channel_taxonomy()
    index = ScopeIndex(table, trees={"channel": tree}, dims=("channel",))

    mask = index.mask({"channel": scope_value})
    matched = set(table["channel"].to_numpy()[mask])
    assert matched == expected_channels


def test_scope_index_reports_unknown_value_as_empty_mask():
    table = _synthetic_table()
    tree = _channel_taxonomy()
    index = ScopeIndex(table, trees={"channel": tree}, dims=("channel",))
    mask = index.mask({"channel": "not-a-real-value"})
    assert not mask.any()


@pytest.mark.parametrize("case", CASES)
def test_trivial_tree_masking_matches_old_flat_equality_on_real_cases(case):
    """No tree supplied (the default) must behave exactly like the pre-tree
    flat-equality scope_mask() on every real case -- the non-regression
    guarantee for Phase 1: nothing about low/med/hard's behavior changes
    until hierarchy inference actually produces a non-trivial tree."""
    offer_table, constraint_set = F.load_case(case)
    index = ScopeIndex(offer_table)  # trees=None -> trivial per dimension

    for c in constraint_set.constraints:
        if not c.scope:
            continue
        got = index.mask(c.scope)
        want = scope_mask(offer_table, c.scope)
        assert np.array_equal(got, want), f"{case}: mask mismatch for scope {c.scope}"
