"""Resolve RawConstraintRow -> ConstraintSpec | ParameterSpec.

This module is the ONLY place a raw constraint-type string is ever
inspected. Two vendor irregularities need explicit, documented overrides
(a parameter disguised as a constraint row, and a column whose header lies
about its content); everything else is resolved by a naming convention so a
brand-new type of the same shape ("cost_per_segment", say) needs zero code
changes here, and none at all in solver/ or verify.py.
"""

from __future__ import annotations

from offer_opt.schema import ConstraintSpec, ConstraintSet, ParameterSpec, RawConstraintRow


def _first_non_null(*values: float | None) -> float:
    for v in values:
        if v is not None:
            return v
    raise ValueError("parameter row has neither min nor max populated")


def _scope_from(row: RawConstraintRow, dims: tuple[str, ...]) -> dict[str, str]:
    scope = {}
    if "channel" in dims and row.channel is not None:
        scope["channel"] = row.channel
    if "product" in dims and row.product is not None:
        scope["product"] = row.product
    return scope


def _handle_cost_of_communication(row: RawConstraintRow) -> ParameterSpec:
    # Verified against hard row 1: EV uses the channel's cost -- only `min`
    # is ever populated for this type, but we take max-if-present-else-min
    # so the same rule also covers margin_per_product below.
    return ParameterSpec(kind="cost", scope=_scope_from(row, ("channel",)),
                          value=_first_non_null(row.max, row.min))


def _handle_margin_per_product(row: RawConstraintRow) -> ParameterSpec:
    # Verified against hard row 1 (INGOLAB/EMAIL): EV = 0.2643*10216.26*0.12-0.02
    # matches the reference exactly, where 0.12 is this type's MAX column.
    return ParameterSpec(kind="margin", scope=_scope_from(row, ("product",)),
                          value=_first_non_null(row.max, row.min))


def _handle_offers_per_segment(row: RawConstraintRow) -> ConstraintSpec:
    # Vendor quirk: the "Product" column actually holds the SEGMENT value for
    # this one constraint type (e.g. "offers_per_segment;EMAIL;IFL_AA;200;").
    scope = {}
    if row.channel is not None:
        scope["channel"] = row.channel
    if row.product is not None:
        scope["segment"] = row.product
    return ConstraintSpec(
        id=f"offers_per_segment{scope}", raw_type=row.raw_type, scope=scope,
        measure="count", min=row.min, max=row.max, per_client=False,
    )


# Explicit overrides for genuinely irregular rows. Deliberately small: this
# is the honest boundary of "generic" -- real vendor-data quirks that no
# naming convention could resolve automatically, not a place to special-case
# ordinary constraint types by name.
_OVERRIDES = {
    "cost_of_communication": _handle_cost_of_communication,
    "margin_per_product": _handle_margin_per_product,
    "offers_per_segment": _handle_offers_per_segment,
}


def resolve_one(row: RawConstraintRow) -> ConstraintSpec | ParameterSpec:
    key = row.raw_type.strip().lower()
    if key in _OVERRIDES:
        return _OVERRIDES[key](row)

    # Convention-driven fallback -- covers total_cost, cost_per_channel,
    # cost_per_product, offers_per_channel, offers_per_product,
    # offers_per_channel_per_client, offers_per_product_per_client, and any
    # future type following the same "<measure>_per_<scope...>[_per_client]"
    # shape, with zero changes needed here.
    measure = "cost" if "cost" in key else "count"
    per_client = key.endswith("_per_client")
    scope = _scope_from(row, ("channel", "product"))
    return ConstraintSpec(
        id=f"{key}{scope}", raw_type=key, scope=scope,
        measure=measure, min=row.min, max=row.max, per_client=per_client,
    )


def resolve_all(rows: list[RawConstraintRow]) -> ConstraintSet:
    constraints: list[ConstraintSpec] = []
    parameters: list[ParameterSpec] = []
    for row in rows:
        resolved = resolve_one(row)
        if isinstance(resolved, ParameterSpec):
            parameters.append(resolved)
        else:
            constraints.append(resolved)
    return ConstraintSet(constraints=constraints, parameters=parameters)
