"""Resolve RawConstraintRow -> ConstraintSpec | ParameterSpec.

This module is the ONLY place a raw constraint-type string is ever
inspected. Two vendor irregularities need explicit, documented overrides
(a parameter disguised as a constraint row, and a column whose header lies
about its content); most others are resolved by a naming convention so a
brand-new type of the same shape ("cost_per_segment", say) needs zero code
changes here, and none at all in solver/ or verify.py. A raw_type that
matches neither an override nor the convention falls back to an LLM that
classifies its *shape* only (never a number) -- see `_resolve_via_llm` below.

Note this LLM fallback only ever produces a `ConstraintSpec`, never a
`ParameterSpec` -- exactly like the naming-convention fallback above it.
Recognizing that a row is actually a *parameter* (an EV-formula input
disguised as a constraint row, like `cost_of_communication`/
`margin_per_product`) is `_OVERRIDES`' job specifically, both today and via
this fallback; that split isn't a new gap this fallback introduces.
"""

from __future__ import annotations

from offer_opt.schema import ConstraintSpec, ConstraintSet, ParameterSpec, RawConstraintRow

_DEFAULT_DIMS = ("channel", "product", "segment")


class UnresolvedConstraintError(Exception):
    """Raised when a raw constraint-type string matches no override, no
    naming convention, and can't be confidently classified by the LLM
    fallback either (unavailable, or still invalid/low-confidence after
    retries). Deliberately not caught and silently guessed: misclassifying a
    constraint's measure or scope can make the solver silently enforce the
    wrong bound, a much higher-consequence mistake than a dropped/ignored
    column elsewhere in the pipeline."""


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


def _matches_naming_convention(key: str, row: RawConstraintRow) -> bool:
    """"<measure>_per_<scope...>[_per_client]"-shaped strings are always
    safe for the convention fallback below. A fully unscoped (global) row
    (no channel, no product populated at all) is *also* safe regardless of
    wording -- there's no scope-dimension ambiguity to get wrong there, only
    a binary count-vs-cost guess ("cost" in key) that's a defensible default
    either way (this is what correctly keeps e.g. "total_cost" -- which
    contains no "_per_" -- on the fast path). Anything else (an unfamiliar
    word *and* a populated channel/product whose relevance isn't certain)
    needs the LLM."""
    if "_per_" in key:
        return True
    return row.channel is None and row.product is None


def _resolve_by_convention(row: RawConstraintRow, key: str) -> ConstraintSpec:
    # Covers total_cost, cost_per_channel, cost_per_product,
    # offers_per_channel, offers_per_product, offers_per_channel_per_client,
    # offers_per_product_per_client, and any future type following the same
    # shape, with zero changes needed here.
    measure = "cost" if "cost" in key else "count"
    per_client = key.endswith("_per_client")
    scope = _scope_from(row, ("channel", "product"))
    return ConstraintSpec(
        id=f"{key}{scope}", raw_type=key, scope=scope,
        measure=measure, min=row.min, max=row.max, per_client=per_client,
    )


def _populated_columns(row: RawConstraintRow) -> list[str]:
    return [c for c in ("channel", "product") if getattr(row, c) is not None]


def _domain_errors(response: dict, populated_columns: list[str], dims: tuple[str, ...]) -> list[str]:
    errors = []
    mapping = response.get("column_dimension_map", {})
    if isinstance(mapping, dict) and set(mapping) != set(populated_columns):
        errors.append(f"column_dimension_map keys {sorted(mapping)} != populated columns {sorted(populated_columns)}")
    if isinstance(mapping, dict):
        bad_values = set(mapping.values()) - set(dims)
        if bad_values:
            errors.append(f"column_dimension_map values {sorted(bad_values)} not in known dims {sorted(dims)}")
    return errors


def _classify_via_llm(row: RawConstraintRow, key: str, dims: tuple[str, ...], llm_client,
                       max_retries: int = 2) -> dict:
    # Lazy import: constraints.py sits on features.py's import path, which
    # metrics.py (the benchmarked hot path) also imports -- a module-scope
    # `import offer_opt.llm` here would defeat the point of keeping that
    # path LLM-free (see system_design_overview.md Section 6, plan Section 10).
    from offer_opt.llm import prompts as _prompts
    from offer_opt.llm.client import LLMUnavailable
    from offer_opt.llm.parsing import validate_against_schema

    populated_columns = _populated_columns(row)
    example = {"raw_type": row.raw_type, "channel": row.channel, "product": row.product,
               "min": row.min, "max": row.max}
    system, user, schema = _prompts.constraint_type_prompt(key, populated_columns, dims, example)

    errors: list[str] = []
    for _ in range(max_retries + 1):
        try:
            response = llm_client.complete_json(system=system, user=user, json_schema=schema)
        except LLMUnavailable as exc:
            raise UnresolvedConstraintError(
                f"constraint type {key!r} matches no override or naming convention, "
                f"and no LLM client is available to classify it"
            ) from exc
        errors = validate_against_schema(response, schema)
        errors += _domain_errors(response, populated_columns, dims)
        if not errors and response.get("confidence") == "high":
            return response
        if not errors:
            errors = [f"low confidence: {response.get('confidence')!r}"]
        user = user + "\n\nA previous attempt was invalid: " + "; ".join(errors) + ". Try again."

    raise UnresolvedConstraintError(
        f"constraint type {key!r} could not be confidently classified after {max_retries} retries: {errors}"
    )


def _resolve_via_llm(row: RawConstraintRow, key: str, dims: tuple[str, ...], llm_client, cache: dict) -> ConstraintSpec:
    from offer_opt.llm.client import NullClient

    llm_client = llm_client or NullClient()
    if key in cache:
        result = cache[key]
    else:
        result = _classify_via_llm(row, key, dims, llm_client)
        cache[key] = result

    mapping = result["column_dimension_map"]
    scope: dict[str, str] = {}
    if "channel" in mapping and row.channel is not None:
        scope[mapping["channel"]] = row.channel
    if "product" in mapping and row.product is not None:
        scope[mapping["product"]] = row.product

    return ConstraintSpec(
        id=f"{key}{scope}", raw_type=key, scope=scope,
        measure=result["measure"], min=row.min, max=row.max,
        per_client=bool(result.get("per_subject", False)),
    )


def resolve_one(row: RawConstraintRow, *, dims: tuple[str, ...] = _DEFAULT_DIMS,
                 trees: dict | None = None, llm_client=None,
                 cache: dict | None = None) -> ConstraintSpec | ParameterSpec:
    key = row.raw_type.strip().lower()
    if key in _OVERRIDES:
        return _OVERRIDES[key](row)
    if _matches_naming_convention(key, row):
        return _resolve_by_convention(row, key)
    return _resolve_via_llm(row, key, dims, llm_client, cache if cache is not None else {})


def resolve_all(rows: list[RawConstraintRow], *, dims: tuple[str, ...] = _DEFAULT_DIMS,
                 trees: dict | None = None, llm_client=None) -> ConstraintSet:
    constraints: list[ConstraintSpec] = []
    parameters: list[ParameterSpec] = []
    cache: dict = {}
    for row in rows:
        resolved = resolve_one(row, dims=dims, trees=trees, llm_client=llm_client, cache=cache)
        if isinstance(resolved, ParameterSpec):
            parameters.append(resolved)
        else:
            constraints.append(resolved)
    return ConstraintSet(constraints=constraints, parameters=parameters)
