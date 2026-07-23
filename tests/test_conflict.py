from offer_opt.discovery.conflict import find_conflicts
from offer_opt.schema import ConstraintSet, ConstraintSpec, DimensionTree


def _channel_tree() -> DimensionTree:
    tree = DimensionTree(dim="channel", parent_of={
        "email": None,
        "personal": "email",
        "subscription": "email",
        "automated": "personal",
        "hand-written": "personal",
        "expensive": "automated",
        "cheap": "automated",
    })
    tree.build_intervals()
    return tree


def _spec(scope, measure="count", min=None, max=None, per_client=False) -> ConstraintSpec:
    return ConstraintSpec(id=f"test{scope}", raw_type="test", scope=scope,
                           measure=measure, min=min, max=max, per_client=per_client)


def test_finds_a_hand_built_ancestor_descendant_contradiction():
    """max=100 on "personal" cannot coexist with mins of 60+60 on its
    *direct* children "automated"/"hand-written" -- the exact scenario from
    system_design_overview.md Section 3 (there, phrased as "email" instead
    of "personal" -- corrected here to actually be the direct parent, since
    "automated"/"hand-written" nest under "personal", not "email" itself,
    per `_channel_tree()`'s edges)."""
    constraints = ConstraintSet(constraints=[
        _spec({"channel": "personal"}, max=100),
        _spec({"channel": "automated"}, min=60),
        _spec({"channel": "hand-written"}, min=60),
    ], parameters=[])

    conflicts = find_conflicts(constraints, trees={"channel": _channel_tree()})
    assert len(conflicts) == 1
    assert conflicts[0].ancestor.scope == {"channel": "personal"}
    assert {tuple(sorted(d.scope.items())) for d in conflicts[0].descendants} == {
        (("channel", "automated"),), (("channel", "hand-written"),),
    }
    assert "120" in conflicts[0].reason and "100" in conflicts[0].reason


def test_no_false_positive_when_children_mins_fit_under_the_parent_cap():
    constraints = ConstraintSet(constraints=[
        _spec({"channel": "personal"}, max=200),
        _spec({"channel": "automated"}, min=60),
        _spec({"channel": "hand-written"}, min=60),
    ], parameters=[])
    assert find_conflicts(constraints, trees={"channel": _channel_tree()}) == []


def test_grandchild_min_is_not_counted_against_a_grandparent_cap():
    """"expensive" nests two levels under "personal" (via "automated") --
    its min must NOT be summed against "personal"'s cap; if it were, this
    huge min would falsely trigger a conflict that doesn't actually exist
    at the "personal" level (nothing directly under "personal" has a min
    here at all, so there is nothing to sum)."""
    constraints = ConstraintSet(constraints=[
        _spec({"channel": "personal"}, max=65),
        _spec({"channel": "expensive"}, min=1000),
    ], parameters=[])
    assert find_conflicts(constraints, trees={"channel": _channel_tree()}) == []


def test_direct_child_one_level_down_is_still_correctly_checked():
    """Same "expensive" min, but now checked one level down against its
    actual direct parent "automated" -- this SHOULD trigger a conflict,
    confirming "direct children only" isn't just suppressing detection
    everywhere, only at the wrong (grandparent) level."""
    constraints = ConstraintSet(constraints=[
        _spec({"channel": "automated"}, max=5),
        _spec({"channel": "expensive"}, min=1000),
    ], parameters=[])
    conflicts = find_conflicts(constraints, trees={"channel": _channel_tree()})
    assert len(conflicts) == 1
    assert conflicts[0].ancestor.scope == {"channel": "automated"}


def test_ignores_compound_scoped_and_unrelated_dimension_constraints():
    constraints = ConstraintSet(constraints=[
        _spec({"channel": "email"}, max=10),
        _spec({"channel": "automated", "product": "KSK"}, min=60),  # compound scope -- skipped
        _spec({"product": "KSK"}, min=60),                           # different dimension -- skipped
    ], parameters=[])
    assert find_conflicts(constraints, trees={"channel": _channel_tree()}) == []


def test_measure_and_per_client_mismatches_are_not_compared():
    constraints = ConstraintSet(constraints=[
        _spec({"channel": "email"}, measure="count", max=10),
        _spec({"channel": "automated"}, measure="cost", min=1000),      # different measure
        _spec({"channel": "hand-written"}, measure="count", min=5, per_client=True),  # per-client, not campaign-level
    ], parameters=[])
    assert find_conflicts(constraints, trees={"channel": _channel_tree()}) == []
