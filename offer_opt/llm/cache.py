"""Persistent, content-addressed cache for LLM decisions, plus the
`llm_decisions.jsonl` audit trail non-determinism needs. Keyed on
`sha256(step + schema_version + canonicalized_prompt)` -- per unique
symbolic input (a distinct constraint-type string, a distinct residual
dimension-value batch, a distinct ambiguous-column description), never per
subject-level row -- matching the same in-memory `cache: dict` shape
`constraints.py::resolve_all` already threads through `resolve_one` since
Phase 4; this is that same idea made persistent across runs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _digest(step: str, schema_version: str, prompt: str) -> str:
    return hashlib.sha256(f"{step}\x00{schema_version}\x00{prompt}".encode("utf-8")).hexdigest()


@dataclass
class PersistentLLMCache:
    path: Path
    schema_version: str = "v1"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._store_path = self.path / "cache.json"
        self._log_path = self.path / "llm_decisions.jsonl"
        self._store: dict[str, Any] = json.loads(self._store_path.read_text()) if self._store_path.exists() else {}

    def get(self, step: str, prompt: str) -> Any | None:
        return self._store.get(_digest(step, self.schema_version, prompt))

    def put(self, step: str, prompt: str, response: Any, *, cache_hit: bool,
            latency_s: float, validation_ok: bool) -> None:
        key = _digest(step, self.schema_version, prompt)
        self._store[key] = response
        self._store_path.write_text(json.dumps(self._store, ensure_ascii=False, default=str))
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(), "step": step, "input_digest": key,
                "output": response, "cache_hit": cache_hit,
                "latency_s": latency_s, "validation_ok": validation_ok,
            }, ensure_ascii=False, default=str) + "\n")


class CachingLLMClient:
    """Wraps any `LLMClient` with a `PersistentLLMCache` -- a cache hit never
    calls the inner client at all (temperature is meaningless if the
    response never varies across runs, since a repeat run on the same
    dataset makes zero new LLM calls once warm)."""

    def __init__(self, inner, cache: PersistentLLMCache, step: str):
        self._inner = inner
        self._cache = cache
        self._step = step

    def complete_json(self, *, system: str, user: str, json_schema: dict,
                       temperature: float = 0.0, max_tokens: int = 1024,
                       timeout_s: float = 30.0) -> dict:
        prompt_key = system + "\x00" + user
        cached = self._cache.get(self._step, prompt_key)
        if cached is not None:
            self._cache.put(self._step, prompt_key, cached, cache_hit=True, latency_s=0.0, validation_ok=True)
            return cached

        t0 = time.monotonic()
        response = self._inner.complete_json(system=system, user=user, json_schema=json_schema,
                                               temperature=temperature, max_tokens=max_tokens,
                                               timeout_s=timeout_s)
        self._cache.put(self._step, prompt_key, response, cache_hit=False,
                         latency_s=time.monotonic() - t0, validation_ok=True)
        return response
