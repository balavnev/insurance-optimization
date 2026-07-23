"""Opt-in smoke test against a REAL configured LLM endpoint (e.g. the
vendor's shared Qwen-behind-vLLM service). Run explicitly with
`pytest -m llm_integration`; skips itself whenever LLM_BASE_URL isn't set,
so it never breaks a normal test run in an environment with no live
endpoint (this dev machine included) -- confirms the real VLLMOpenAIClient
satisfies the exact same LLMClient protocol the fakes exercised throughout
Phases 2-6, with zero code changes at any call site.
"""

import os

import pytest

from offer_opt import constraints as C
from offer_opt.llm.client import VLLMOpenAIClient
from offer_opt.schema import RawConstraintRow

pytestmark = pytest.mark.llm_integration

_SKIP_REASON = "LLM_BASE_URL not configured -- opt-in test, run with a real endpoint to exercise it"


@pytest.mark.skipif(not os.environ.get("LLM_BASE_URL"), reason=_SKIP_REASON)
def test_real_endpoint_health_check_ok():
    client = VLLMOpenAIClient()
    assert client.health_check() is True


@pytest.mark.skipif(not os.environ.get("LLM_BASE_URL"), reason=_SKIP_REASON)
def test_real_endpoint_resolves_a_novel_constraint_type():
    client = VLLMOpenAIClient()
    row = RawConstraintRow(raw_type="campaign_spend_share_cap", channel="SMS", product=None, min=0.0, max=0.3)
    resolved = C.resolve_one(row, llm_client=client, cache={})
    assert resolved.measure in ("count", "cost")
