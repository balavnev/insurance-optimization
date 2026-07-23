"""Restricted execution of generated verifier code -- applied to every
generated function regardless of origin (template-fill or LLM):

1. `ast.parse` + allow-list: no imports, no calls to exec/eval/open/
   __import__/compile, no dunder-attribute access.
2. Execute in a restricted namespace (`np`/`pd`/`scope_mask` only, a minimal
   safe builtins subset) with a wall-clock timeout.
3. Run against tiny synthetic golden fixtures derived from the constraint's
   own bounds, with known pass/violate outcomes.

Step 4 (cross-check against `verify.py::verify` on the real dataset) lives
in `codegen/generate.py`, since it needs the real dataset + full
`ConstraintSet`, not just the generated source in isolation.
"""

from __future__ import annotations

import ast
import signal

import numpy as np
import pandas as pd

from offer_opt.schema import ConstraintSpec
from offer_opt.scope import scope_mask

_FORBIDDEN_CALL_NAMES = {"exec", "eval", "open", "__import__", "compile", "getattr", "setattr", "delattr"}

_SAFE_BUILTINS = {
    "len": len, "float": float, "int": int, "bool": bool, "str": str,
    "True": True, "False": False, "None": None, "abs": abs, "min": min, "max": max,
}


class UnsafeGeneratedCodeError(Exception):
    """A generated verifier function violated the sandbox's allow-list."""


class GeneratedCodeTimeoutError(Exception):
    """A generated verifier function exceeded its wall-clock budget."""


class GeneratedCodeFailedGoldenFixture(Exception):
    """A generated verifier function disagreed with a known synthetic
    outcome before it was ever run against real data."""


def check_ast_safety(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeGeneratedCodeError("generated verifier code may not import anything")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            raise UnsafeGeneratedCodeError(f"generated verifier code may not call {node.func.id}()")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeGeneratedCodeError("generated verifier code may not access dunder attributes")


def compile_check_function(source: str, function_name: str):
    """Parse, safety-check, and exec `source` in a restricted namespace;
    returns the callable bound to `function_name`."""
    check_ast_safety(source)
    namespace = {"__builtins__": _SAFE_BUILTINS, "np": np, "pd": pd, "scope_mask": scope_mask}
    exec(compile(source, f"<generated:{function_name}>", "exec"), namespace)
    if function_name not in namespace or not callable(namespace[function_name]):
        raise UnsafeGeneratedCodeError(f"generated code did not define a callable {function_name!r}")
    return namespace[function_name]


def run_with_timeout(fn, args, timeout_s: float = 5.0):
    if not hasattr(signal, "SIGALRM"):
        return fn(*args)  # best-effort on platforms without SIGALRM (e.g. Windows)

    def _handler(signum, frame):
        raise GeneratedCodeTimeoutError(f"generated verifier code exceeded {timeout_s}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def golden_fixtures(constraint: ConstraintSpec, n_rows: int = 6) -> list[tuple[pd.DataFrame, np.ndarray, bool]]:
    """Tiny synthetic (option_table, selection, expected_ok) triples derived
    from the constraint's own scope/measure/bounds -- every row matches the
    constraint's scope exactly, all sharing one client_id if `per_client`
    (so aggregation is meaningful), independent client ids otherwise."""
    data: dict[str, object] = {
        "client_id": np.zeros(n_rows, dtype="int64") if constraint.per_client else np.arange(n_rows),
        "cost": np.ones(n_rows),
    }
    for dim, val in constraint.scope.items():
        data[dim] = [val] * n_rows
    option_table = pd.DataFrame(data)

    fixtures: list[tuple[pd.DataFrame, np.ndarray, bool]] = []
    is_count = constraint.measure == "count"

    if constraint.max is not None and is_count and 0 <= constraint.max < n_rows:
        bound = int(constraint.max)
        sel_ok = np.zeros(n_rows); sel_ok[:bound] = 1
        fixtures.append((option_table, sel_ok, True))
        sel_bad = np.zeros(n_rows); sel_bad[: bound + 1] = 1
        fixtures.append((option_table, sel_bad, False))

    if constraint.min is not None and is_count and 0 < constraint.min <= n_rows:
        bound = int(constraint.min)
        sel_ok = np.zeros(n_rows); sel_ok[:bound] = 1
        fixtures.append((option_table, sel_ok, True))
        if bound > 1:
            sel_bad = np.zeros(n_rows); sel_bad[: bound - 1] = 1
            fixtures.append((option_table, sel_bad, False))

    return fixtures


def check_golden_fixtures(fn, constraint: ConstraintSpec) -> None:
    for option_table, selection, expected_ok in golden_fixtures(constraint):
        actual_ok = run_with_timeout(fn, (option_table, selection))
        if bool(actual_ok) != expected_ok:
            raise GeneratedCodeFailedGoldenFixture(
                f"{constraint.id}: expected ok={expected_ok} on a synthetic fixture, got {actual_ok}"
            )
