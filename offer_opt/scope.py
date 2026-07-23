"""Shared scope-matching helper used by both the verifier and the solver's
global-constraint step, so "what offers does this constraint apply to" is
computed identically everywhere.

Plain `offer_table[dim].to_numpy() == val` on the object-dtype string
columns is ~150ms per constraint on the 5M-row hard case (450x slower than
necessary) because it falls back to elementwise Python string comparison.
`ScopeIndex` factorizes each scoped dimension once and compares integer
codes instead -- same result, ~0.03s total for all ~90 constraints.

Ancestor-aware: each dimension's distinct values are backed by a
`DimensionTree` (schema.py) and encoded via an Euler-tour preorder (tin/tout)
per value, so a constraint scoped to a non-leaf node (e.g. "product" when
segments nest under it, or a mid-level node *within* one dimension's own
taxonomy -- see system_design_overview.md Section 3) matches every descendant
row through one interval comparison instead of a per-row tree walk. When no
tree is supplied for a dimension, every value is its own root
(`DimensionTree.trivial`) and the interval comparison degenerates to exactly
the old flat equality check -- this is what keeps every existing case's
behavior unchanged until hierarchy inference (a later phase) produces a real
tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from offer_opt.schema import DimensionTree

SCOPE_DIMS = ("channel", "product", "segment")


class ScopeIndex:
    def __init__(self, offer_table: pd.DataFrame, trees: dict[str, DimensionTree] | None = None,
                 dims: tuple[str, ...] = SCOPE_DIMS):
        self._n = len(offer_table)
        self._trees: dict[str, DimensionTree] = {}
        self._tin_of_row: dict[str, np.ndarray] = {}
        trees = trees or {}

        for dim in dims:
            if dim not in offer_table.columns:
                continue
            cat = offer_table[dim].astype("category")
            categories = cat.cat.categories
            codes = cat.cat.codes.to_numpy()

            tree = trees.get(dim) or DimensionTree.trivial(dim, categories)
            tree.build_intervals()
            missing = [v for v in categories if v not in tree.tin]
            if missing:
                raise ValueError(f"DimensionTree for {dim!r} is missing {len(missing)} observed value(s), "
                                  f"e.g. {missing[:5]!r} -- every value present in the offer table must appear "
                                  f"in its dimension's tree")

            tin_by_code = np.fromiter((tree.tin[v] for v in categories), dtype=np.int64, count=len(categories))
            # `codes == -1` marks NaN (pandas' categorical-code convention).
            # Indexing tin_by_code[-1] would silently alias the *last*
            # category's tin (numpy negative-index semantics), so route NaN
            # rows to a sentinel below every real tin (>= 0) instead.
            safe_codes = np.where(codes >= 0, codes, 0)
            self._trees[dim] = tree
            self._tin_of_row[dim] = np.where(codes >= 0, tin_by_code[safe_codes], -1)

    def mask(self, scope: dict[str, str]) -> np.ndarray:
        mask = np.ones(self._n, dtype=bool)
        for dim, val in scope.items():
            tree = self._trees.get(dim)
            if tree is None or val not in tree.tin:
                return np.zeros(self._n, dtype=bool)  # constraint scoped to a dim/value absent from this table
            lo, hi = tree.tin[val], tree.tout[val]
            row_tin = self._tin_of_row[dim]
            mask &= (row_tin >= lo) & (row_tin <= hi)
        return mask


def scope_mask(offer_table: pd.DataFrame, scope: dict[str, str]) -> np.ndarray:
    """Convenience one-off path (small tables / tests) -- for repeated calls
    over the same offer_table, build a `ScopeIndex` once instead. Flat
    exact-match only (no tree) -- matches ScopeIndex's default/trivial-tree
    behavior for dimensions with no known hierarchy."""
    mask = np.ones(len(offer_table), dtype=bool)
    for dim, val in scope.items():
        mask &= offer_table[dim].to_numpy() == val
    return mask
