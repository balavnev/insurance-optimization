"""Minimal swappable LLM client interface.

This is a deliberately small slice of the full `llm/` package the
generalization plan describes (Section 7: schemas.py, prompts.py, cache.py,
budget.py, a real vLLM-backed client) -- introduced now, in Phase 2, only
because discovery's LLM-fallback branches need something to call. Everything
above this layer depends on the `LLMClient` protocol, never a concrete
backend, so wiring in a real served model later (Phase 7) is pure dependency
injection, not a rewrite of call sites.
"""

from __future__ import annotations

from typing import Protocol


class LLMUnavailable(Exception):
    """Raised when no usable LLM backend is configured, or a call fails.
    Callers must catch this and degrade to their best heuristic guess --
    never let it propagate and stall the pipeline."""


class LLMClient(Protocol):
    def complete_json(self, *, system: str, user: str, json_schema: dict,
                       temperature: float = 0.0, max_tokens: int = 1024,
                       timeout_s: float = 30.0) -> dict:
        ...


class NullClient:
    """Forces the symbolic/heuristic-only code path. The safe default until
    a real backend (Phase 7) is wired in -- every call raises immediately
    rather than hanging or guessing."""

    def complete_json(self, **kwargs) -> dict:
        raise LLMUnavailable("no LLM client configured")


class FakeLLMClient:
    """Scripted test double. `responses` is an ordered list of
    (substring_of_user_prompt, response_dict) pairs, checked in order, so a
    test can script several distinct calls without a real backend. Records
    every call in `self.calls` for assertions on what was actually asked."""

    def __init__(self, responses: list[tuple[str, dict]]):
        self._responses = responses
        self.calls: list[dict] = []

    def complete_json(self, *, system: str, user: str, json_schema: dict,
                       temperature: float = 0.0, max_tokens: int = 1024,
                       timeout_s: float = 30.0) -> dict:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        for needle, response in self._responses:
            if needle in user:
                return response
        raise LLMUnavailable(f"FakeLLMClient has no scripted response matching prompt: {user!r}")
