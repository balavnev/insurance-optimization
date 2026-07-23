"""Per-run LLM call budget: a hard ceiling on how many calls a single step
may make and how long the whole run may spend waiting on the LLM, so a
misbehaving or slow endpoint can't stall the pipeline indefinitely. Every
degrade path raises `BudgetExceeded` (an `LLMUnavailable` subclass), so the
existing "catch LLMUnavailable, fall back to a heuristic guess or hard-fail"
logic already in `discovery/schema_resolver.py`, `discovery/hierarchy.py`,
and `constraints.py` handles it with zero changes -- swapping in a budgeted
client is pure dependency injection, not a new code path for callers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from offer_opt.llm.client import LLMClient, LLMUnavailable


class BudgetExceeded(LLMUnavailable):
    """Raised instead of making a call once a budget limit is hit."""


@dataclass
class LLMBudget:
    max_calls_per_step: int | None = None
    total_wallclock_budget_s: float | None = None
    per_call_timeout_s: float = 30.0
    _calls_used: dict[str, int] = field(default_factory=dict, repr=False)
    _start_time: float | None = field(default=None, repr=False)

    def start(self) -> None:
        self._start_time = time.monotonic()

    def _elapsed(self) -> float:
        return 0.0 if self._start_time is None else time.monotonic() - self._start_time

    def check(self, step: str) -> None:
        """Raises BudgetExceeded if this call must not proceed."""
        if self.max_calls_per_step is not None and self._calls_used.get(step, 0) >= self.max_calls_per_step:
            raise BudgetExceeded(
                f"step {step!r} already used its budget of {self.max_calls_per_step} calls")
        if self.total_wallclock_budget_s is not None and self._elapsed() > self.total_wallclock_budget_s:
            raise BudgetExceeded(
                f"total LLM wallclock budget of {self.total_wallclock_budget_s}s exceeded")

    def record(self, step: str) -> None:
        self._calls_used[step] = self._calls_used.get(step, 0) + 1


class BudgetedLLMClient:
    """Wraps any `LLMClient` with an `LLMBudget` -- transparent to callers,
    which only ever see `complete_json` and `LLMUnavailable`/`BudgetExceeded`
    (itself an `LLMUnavailable`), regardless of which concrete client or
    budget policy is behind it."""

    def __init__(self, inner: LLMClient, budget: LLMBudget, step: str):
        self._inner = inner
        self._budget = budget
        self._step = step

    def complete_json(self, *, system: str, user: str, json_schema: dict,
                       temperature: float = 0.0, max_tokens: int = 1024,
                       timeout_s: float | None = None) -> dict:
        self._budget.check(self._step)
        effective_timeout = timeout_s if timeout_s is not None else self._budget.per_call_timeout_s
        try:
            return self._inner.complete_json(system=system, user=user, json_schema=json_schema,
                                               temperature=temperature, max_tokens=max_tokens,
                                               timeout_s=effective_timeout)
        finally:
            self._budget.record(self._step)
