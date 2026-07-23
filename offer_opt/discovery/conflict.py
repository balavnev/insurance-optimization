"""Detects when constraints scoped at different depths of the same
dimension tree are jointly infeasible, before any solve happens -- e.g. a
max=100 on "email" and mins of 60+60 on its children "automated"/
"hand-written" can never all hold at once. See system_design_overview.md
Section 3 for the full reasoning and the documented (default, not certain)
precedence rule: a more specific (deeper-scoped) constraint should win over
a broader one it conflicts with.

Scope: single-dimension, single-value scoped constraints only (the large
majority in practice -- offers_per_product scoped to one product, etc.). A
constraint with a compound scope (more than one dimension at once) isn't
analyzed here, since "ancestor of" isn't a well-defined relationship across
dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

from offer_opt.schema import ConstraintSet, ConstraintSpec, DimensionTree


@dataclass(frozen=True)
class ConstraintConflict:
    ancestor: ConstraintSpec
    descendants: list[ConstraintSpec]
    reason: str


def _single_dim_scope(c: ConstraintSpec) -> tuple[str, str] | None:
    if len(c.scope) != 1:
        return None
    (dim, value), = c.scope.items()
    return dim, value


def find_conflicts(constraint_set: ConstraintSet, trees: dict[str, DimensionTree]) -> list[ConstraintConflict]:
    """For every (dimension, measure, per_client) group of single-dim-scoped
    constraints, flag any node whose own `max` is smaller than the sum of
    its *direct* children's `min` bounds (comparing only constraints of the
    same measure and per_client-ness -- summing counts against a cost cap,
    or client-level against campaign-level, would be comparing different
    units). Direct children only, not all transitive descendants: children
    of one parent never overlap each other in a tree, so their mins sum
    without double-counting; summing every descendant at every depth would."""
    conflicts: list[ConstraintConflict] = []

    groups: dict[tuple[str, str, bool], dict[str, ConstraintSpec]] = {}
    for c in constraint_set.constraints:
        scoped = _single_dim_scope(c)
        if scoped is None:
            continue
        dim, value = scoped
        if dim not in trees:
            continue
        groups.setdefault((dim, c.measure, c.per_client), {}).setdefault(value, c)

    for (dim, _measure, _per_client), by_value in groups.items():
        tree = trees[dim]
        children_of: dict[str, list[str]] = {}
        for v, p in tree.parent_of.items():
            if p is not None:
                children_of.setdefault(p, []).append(v)

        for value, ancestor_c in by_value.items():
            if ancestor_c.max is None:
                continue
            child_mins = [by_value[child] for child in children_of.get(value, [])
                          if child in by_value and by_value[child].min is not None]
            if not child_mins:
                continue
            total_min = sum(cc.min for cc in child_mins)
            if total_min > ancestor_c.max:
                conflicts.append(ConstraintConflict(
                    ancestor=ancestor_c, descendants=child_mins,
                    reason=(f"{dim}={value!r} max={ancestor_c.max:g} < sum of child mins={total_min:g} "
                            f"over {[c.scope for c in child_mins]}"),
                ))
    return conflicts
