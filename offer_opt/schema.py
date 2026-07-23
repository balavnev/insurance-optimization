"""Generic internal representation shared by every case (low/med/hard).

The parser (constraints.py) is the only place that ever looks at a raw
constraint-type string. Everything downstream (features, solver, verifier)
operates purely on the fields defined here: scope (which dimensions a
constraint/parameter is pinned to), measure (count vs cost), bounds, and the
per_client flag. That separation is what lets a brand-new constraint type
following the same naming convention work with zero code changes anywhere
except the resolver's fallback rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Dimensions a constraint or parameter can be scoped to. Segment is included
# because `offers_per_segment` scopes to (channel, segment).
SCOPE_DIMS = ("channel", "product", "segment")


@dataclass(frozen=True)
class RawConstraintRow:
    """One row of a constraint table, after dialect parsing but before
    semantic resolution. `raw_type` is the first-column string verbatim
    (lower/stripped happens in the resolver, not here, so this stays a
    faithful record of the source)."""

    raw_type: str
    channel: str | None
    product: str | None
    min: float | None
    max: float | None


@dataclass(frozen=True)
class ConstraintSpec:
    """A resolved, generic linear constraint on the campaign."""

    id: str
    raw_type: str
    scope: dict[str, str] = field(default_factory=dict)
    measure: str = "count"  # "count" | "cost"
    min: float | None = None
    max: float | None = None
    per_client: bool = False


@dataclass(frozen=True)
class ParameterSpec:
    """A resolved lookup parameter (feeds the EV formula, not a bound)."""

    kind: str  # "cost" | "margin"
    scope: dict[str, str] = field(default_factory=dict)
    value: float = 0.0


@dataclass
class ConstraintSet:
    constraints: list[ConstraintSpec]
    parameters: list[ParameterSpec]

    def local(self) -> list[ConstraintSpec]:
        return [c for c in self.constraints if c.per_client]

    def global_(self) -> list[ConstraintSpec]:
        return [c for c in self.constraints if not c.per_client]


@dataclass(frozen=True)
class Dimension:
    """A discovered decision dimension -- eventually replaces the hardcoded
    SCOPE_DIMS list once discovery/schema_resolver.py exists (see the
    generalization plan). Unused by name anywhere yet; this is the scaffold."""

    name: str
    source_column: str


@dataclass
class DimensionTree:
    """Parent-pointer forest over one dimension's distinct values, e.g.
    segment "KSK_OSG_AA_A" nested under product "KSK_OSG", or entirely
    *within* one dimension (channel "email" -> "personal"/"subscription" ->
    "automated"/"hand-written" -> "expensive"/"cheap" -- see
    system_design_overview.md Section 3). A value absent from `parent_of`
    (or one for which every value maps to `None`) is a root -- this is what
    makes `trivial()` below just a degenerate case of the same structure,
    not a separate code path.

    `tin`/`tout` (filled by `build_intervals`) are an Euler-tour preorder
    numbering such that `tin[v] <= tin[x] <= tout[v]` iff `v` is an ancestor
    of (or equal to) `x`. That turns "does this row fall under this scope
    node, at any depth" into two vectorized integer comparisons instead of a
    per-row tree walk -- see scope.py's ScopeIndex.
    """

    dim: str
    parent_of: dict[str, str | None] = field(default_factory=dict)
    tin: dict[str, int] = field(default_factory=dict)
    tout: dict[str, int] = field(default_factory=dict)

    @classmethod
    def trivial(cls, dim: str, values) -> "DimensionTree":
        """Every value is its own root -- no hierarchy known/inferred yet.
        Intervals degenerate to one integer per value (tin==tout), which is
        exactly the old flat exact-match behavior once ScopeIndex compares
        against it. This is the default when no tree has been built for a
        dimension, guaranteeing zero behavior change until hierarchy
        inference (a later phase) actually produces a non-trivial tree."""
        return cls(dim=dim, parent_of={v: None for v in values})

    def build_intervals(self) -> None:
        children: dict[str | None, list[str]] = {}
        for v, p in self.parent_of.items():
            children.setdefault(p, []).append(v)

        self.tin = {}
        self.tout = {}
        counter = 0

        def visit(v: str) -> None:
            nonlocal counter
            self.tin[v] = counter
            counter += 1
            for c in children.get(v, []):
                if c not in self.tin:
                    visit(c)
            self.tout[v] = counter - 1

        # Roots: parent is None, or points outside parent_of entirely
        # (defensive against a dangling/unresolved parent reference).
        roots = [v for v, p in self.parent_of.items() if p is None or p not in self.parent_of]
        for r in roots:
            if r not in self.tin:
                visit(r)


# Canonical OfferTable columns. `segment` is always present (synthesized for
# low as f"{product}_{channel}" if the source has no explicit segment dim).
OFFER_TABLE_COLUMNS = [
    "offer_uid",
    "client_id",
    "product",
    "channel",
    "segment",
    "score",     # response probability (proba) or SCORE, whichever the case provides
    "premium",   # premium / avg_check, whichever the case provides
    "margin",    # resolved: from source column or joined ParameterSpec
    "cost",      # resolved: from source column or joined ParameterSpec
    "base_ev",   # margin*premium*score - cost, computed once, formula-agnostic downstream
]
