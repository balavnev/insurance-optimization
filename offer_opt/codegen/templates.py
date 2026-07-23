"""Python source templates for generated per-constraint verifier functions.

Mirrors `verify.py`'s own `_usage()`/groupby pattern -- deterministic
template-fill, no LLM, for the overwhelming majority of constraints (every
one produced by `constraints.py` today is already fully structured:
scope/measure/min/max/per_client). The generated function's fixed signature
is `check(option_table, selection) -> bool`; `option_table` is the same
canonical offer table `verify.py` consumes, `selection` a 0/1 array. Only
`np`, `pd`, and `scope_mask` are available in the execution namespace the
sandbox provides (see codegen/sandbox.py) -- generated source never contains
its own `import` statements.
"""

from __future__ import annotations

_GLOBAL_TEMPLATE = '''\
def {function_name}(option_table, selection):
    """{description}"""
    mask = scope_mask(option_table, {scope!r})
    usage = np.where(mask, {measure_expr}, 0.0) * np.asarray(selection, dtype="float64")
    total = float(usage.sum())
    ok = True
{bound_checks}
    return ok
'''

_PER_CLIENT_TEMPLATE = '''\
def {function_name}(option_table, selection):
    """{description}"""
    mask = scope_mask(option_table, {scope!r})
    usage = np.where(mask, {measure_expr}, 0.0) * np.asarray(selection, dtype="float64")
    agg = pd.Series(usage).groupby(option_table["client_id"].to_numpy(), sort=False).sum()
    ok = True
{bound_checks}
    return ok
'''


def measure_expr(measure: str) -> str:
    if measure == "count":
        return "1.0"
    if measure == "cost":
        return 'option_table["cost"].to_numpy()'
    raise ValueError(f"unknown measure {measure!r}")


def _bound_checks(min_: float | None, max_: float | None, per_client: bool) -> str:
    # Bound presence is known at generation time -- omit a whole branch
    # entirely rather than emitting a runtime `if <literal> is not None`
    # check against a value that's always the same constant either way.
    lhs = "agg" if per_client else "total"
    suffix = ".any()" if per_client else ""
    lines = []
    if max_ is not None:
        lines.append(f"    if ({lhs} > {max_!r} + 1e-6){suffix}:")
        lines.append("        ok = False")
    if min_ is not None:
        lines.append(f"    if ({lhs} < {min_!r} - 1e-6){suffix}:")
        lines.append("        ok = False")
    return "\n".join(lines) if lines else "    pass"


def render(function_name: str, description: str, scope: dict, measure: str,
           min_: float | None, max_: float | None, per_client: bool) -> str:
    template = _PER_CLIENT_TEMPLATE if per_client else _GLOBAL_TEMPLATE
    return template.format(
        function_name=function_name, description=description, scope=scope,
        measure_expr=measure_expr(measure), bound_checks=_bound_checks(min_, max_, per_client),
    )
