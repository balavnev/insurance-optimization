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


class VLLMOpenAIClient:
    """Real backend: a Qwen model served behind vLLM's OpenAI-compatible
    API, per vendor-examples/runbooks/01-first-run.md -- `LLM_BASE_URL`/
    `LLM_API_KEY` env vars, health-checkable via GET /v1/models, chat
    completions via POST /v1/chat/completions. `complete_json` extracts and
    returns a parsed dict exactly like `NullClient`/`FakeLLMClient` do
    (never raw text) -- the `<think>...</think>` reasoning this deployment's
    model emits is stripped by `llm.parsing.extract_json` internally, so
    that detail never leaks to any caller.

    Any request-level failure (missing base URL, connection error, non-2xx
    status, unparseable content) becomes `LLMUnavailable` -- the one
    exception every caller already knows how to degrade from."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "Qwen2.5-32B-Instruct", timeout_s: float = 30.0):
        import os

        self.base_url = (base_url if base_url is not None else os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        self.model = model
        self.default_timeout_s = timeout_s
        if not self.base_url:
            raise LLMUnavailable("LLM_BASE_URL is not configured")

    def complete_json(self, *, system: str, user: str, json_schema: dict,
                       temperature: float = 0.0, max_tokens: int = 1024,
                       timeout_s: float | None = None) -> dict:
        import requests

        from offer_opt.llm.parsing import extract_json

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Grammar-constrained decoding where the served model supports
            # it (vLLM's guided-decoding extension) -- schema validation on
            # the caller side is the backstop regardless, so an endpoint
            # that ignores this field unexpectedly still gets checked.
            "extra_body": {"guided_json": json_schema},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload,
                                      headers=headers, timeout=timeout_s or self.default_timeout_s)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return extract_json(content)
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"vLLM request to {self.base_url!r} failed: {exc}") from exc

    def health_check(self) -> bool:
        import requests

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.get(f"{self.base_url}/v1/models", headers=headers, timeout=self.default_timeout_s)
            return response.ok
        except Exception:
            return False
