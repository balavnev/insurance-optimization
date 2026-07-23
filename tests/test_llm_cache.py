import json

from offer_opt.llm.cache import CachingLLMClient, PersistentLLMCache
from offer_opt.llm.client import FakeLLMClient


def test_cache_put_then_get_roundtrips(tmp_path):
    cache = PersistentLLMCache(tmp_path / "cache_dir")
    assert cache.get("step_a", "prompt_1") is None

    cache.put("step_a", "prompt_1", {"result": 42}, cache_hit=False, latency_s=0.1, validation_ok=True)
    assert cache.get("step_a", "prompt_1") == {"result": 42}


def test_cache_key_is_scoped_to_step_and_schema_version(tmp_path):
    cache = PersistentLLMCache(tmp_path / "cache_dir", schema_version="v1")
    cache.put("step_a", "same_prompt", {"result": "a"}, cache_hit=False, latency_s=0.0, validation_ok=True)

    # Same prompt text, different step -- must not collide.
    assert cache.get("step_b", "same_prompt") is None

    # Same step+prompt, different schema_version -- must not collide either
    # (a prompt/schema change should invalidate old cached decisions).
    cache_v2 = PersistentLLMCache(tmp_path / "cache_dir", schema_version="v2")
    assert cache_v2.get("step_a", "same_prompt") is None


def test_cache_persists_across_instances_pointed_at_the_same_path(tmp_path):
    path = tmp_path / "cache_dir"
    first = PersistentLLMCache(path)
    first.put("step_a", "prompt_1", {"result": "persisted"}, cache_hit=False, latency_s=0.0, validation_ok=True)

    second = PersistentLLMCache(path)  # fresh instance, same directory
    assert second.get("step_a", "prompt_1") == {"result": "persisted"}


def test_cache_writes_one_audit_log_line_per_call(tmp_path):
    cache = PersistentLLMCache(tmp_path / "cache_dir")
    cache.put("step_a", "p1", {"r": 1}, cache_hit=False, latency_s=0.2, validation_ok=True)
    cache.put("step_a", "p1", {"r": 1}, cache_hit=True, latency_s=0.0, validation_ok=True)

    lines = (tmp_path / "cache_dir" / "llm_decisions.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first_entry = json.loads(lines[0])
    assert first_entry["step"] == "step_a"
    assert first_entry["cache_hit"] is False
    second_entry = json.loads(lines[1])
    assert second_entry["cache_hit"] is True


def test_caching_client_only_calls_inner_client_once_for_repeated_prompts(tmp_path):
    fake = FakeLLMClient(responses=[("hello", {"answer": 1})])
    cache = PersistentLLMCache(tmp_path / "cache_dir")
    client = CachingLLMClient(fake, cache, step="test_step")

    r1 = client.complete_json(system="s", user="hello", json_schema={})
    r2 = client.complete_json(system="s", user="hello", json_schema={})

    assert r1 == r2 == {"answer": 1}
    assert len(fake.calls) == 1  # second call was a cache hit, never reached FakeLLMClient


def test_caching_client_calls_inner_client_again_for_a_different_prompt(tmp_path):
    fake = FakeLLMClient(responses=[("hello", {"answer": 1}), ("goodbye", {"answer": 2})])
    cache = PersistentLLMCache(tmp_path / "cache_dir")
    client = CachingLLMClient(fake, cache, step="test_step")

    client.complete_json(system="s", user="hello", json_schema={})
    client.complete_json(system="s", user="goodbye", json_schema={})

    assert len(fake.calls) == 2
