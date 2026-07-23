"""Infers a `DimensionTree` for one dimension's distinct values -- naming
convention first (arbitrary depth, for free), an LLM fallback only for
values convention can't place (semantic nesting with no shared substring,
e.g. "durum wheat" under "wheat", or "hand-written" under "personal" -- see
system_design_overview.md Section 3).

Scope: intra-dimension only. A constraint scoped to `{"product": X}` already
matches every row whose *product column* is X regardless of what its
*segment column* says -- that cross-column nesting works today for free
(system_design_overview.md Section 4) and needs no tree at all. What a tree
is for is nesting *within* one dimension's own values (e.g. product
"KSK_OSG" under product "KSK" -- both are values of the SAME `product`
column in the real med/hard data, verified in tests below; or channel
"personal" under channel "email", entirely within `channel`).

`build_alias_index` is a separate, smaller mechanism: a reverse value->
dimension lookup, for the case where a raw constraint row's column position
implies one dimension but the value it actually holds belongs to a
different one (the vendor's `offers_per_segment` rows put a SEGMENT value
in the "Product" column position) -- generalizes that one hardcoded
override into a systematic lookup, consumed at constraint-parsing time.
"""

from __future__ import annotations

from offer_opt.llm.client import LLMClient, LLMUnavailable, NullClient
from offer_opt.schema import DimensionTree

_SEPARATORS = ("_", "-", ".")

_HIERARCHY_EDGES_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "parent": {"type": ["string", "null"]},
                },
                "required": ["value", "parent"],
            },
        }
    },
    "required": ["edges"],
}


def _truncation_candidates(value: str) -> list[str]:
    """Every way to drop N>=1 trailing '<sep>token' groups from `value`,
    across all known separators, ordered by fewest characters dropped first
    -- so the *closest* matching known value wins as the immediate parent,
    not a more distant ancestor (this is what recovers arbitrary depth: a
    4-level value tries its 1-token, 2-token, 3-token truncations in that
    order and stops at the first one that's an actually-known value)."""
    candidates: dict[str, int] = {}
    for sep in _SEPARATORS:
        tokens = value.split(sep)
        if len(tokens) < 2:
            continue
        for cut in range(1, len(tokens)):
            candidate = sep.join(tokens[: len(tokens) - cut])
            dropped = len(value) - len(candidate)
            candidates[candidate] = min(candidates.get(candidate, dropped), dropped)
    return sorted(candidates, key=candidates.get)


def build_tree(dim: str, values: list[str], llm_client: LLMClient | None = None) -> DimensionTree:
    llm_client = llm_client or NullClient()
    values = list(dict.fromkeys(values))  # de-dup, preserve encounter order
    known = set(values)

    parent_of: dict[str, str | None] = {}
    residual: list[str] = []
    for v in values:
        parent = next((c for c in _truncation_candidates(v) if c != v and c in known), None)
        parent_of[v] = parent
        if parent is None:
            residual.append(v)

    if residual:
        parent_of.update(_resolve_residual_via_llm(dim, residual, parent_of, llm_client))

    tree = DimensionTree(dim=dim, parent_of=parent_of)
    tree.build_intervals()
    return tree


def _resolve_residual_via_llm(dim: str, residual: list[str], parent_of_so_far: dict[str, str | None],
                               llm_client: LLMClient, max_retries: int = 2) -> dict[str, str | None]:
    resolved_so_far = {v: p for v, p in parent_of_so_far.items() if p is not None}
    known_universe = set(residual) | set(parent_of_so_far)
    prompt = (f"Dimension {dim!r}. These values have no naming-convention-detectable parent: "
              f"{residual!r}. Already-resolved parent edges in this dimension: {resolved_so_far!r}. "
              f"For each value in the unresolved list, decide if it nests under another value already "
              f"known to this dimension (parent) or is a top-level root (parent: null). Do not invent "
              f"values that aren't already known to this dimension.")
    for _ in range(max_retries + 1):
        try:
            response = llm_client.complete_json(
                system="Infer a category hierarchy from a list of values with no shared naming pattern.",
                user=prompt, json_schema=_HIERARCHY_EDGES_SCHEMA,
            )
            edges = response.get("edges", [])
            out: dict[str, str | None] = {}
            valid = True
            for edge in edges:
                value, parent = edge.get("value"), edge.get("parent")
                if value not in residual or (parent is not None and parent not in known_universe):
                    valid = False
                    break
                out[value] = parent
            if valid and set(out) == set(residual):
                return out
            prompt += "\nThat response was invalid (missing a value, or an unknown parent) -- try again."
        except LLMUnavailable:
            break
    # No usable LLM response (unavailable, exhausted retries, kept invalid)
    # -- every residual value becomes its own root rather than blocking the
    # pipeline. A flat dimension is always a valid (if less useful) tree.
    return {v: None for v in residual}


def build_alias_index(values_by_dim: dict[str, set[str]]) -> dict[str, str]:
    """Reverse lookup: bare value -> "<dim>:<value>" for every value that
    belongs unambiguously to exactly one dimension's namespace. A value
    present in more than one dimension's set is genuinely ambiguous and
    deliberately excluded -- callers need more context than this index can
    provide to resolve those."""
    owners: dict[str, set[str]] = {}
    for dim, values in values_by_dim.items():
        for v in values:
            owners.setdefault(v, set()).add(dim)
    return {v: f"{next(iter(dims))}:{v}" for v, dims in owners.items() if len(dims) == 1}
