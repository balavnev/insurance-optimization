import time

import pytest

from offer_opt import constraints as C
from offer_opt.llm.budget import BudgetedLLMClient, BudgetExceeded, LLMBudget
from offer_opt.llm.client import FakeLLMClient, LLMUnavailable
from offer_opt.schema import RawConstraintRow


def test_budget_allows_calls_under_the_max():
    budget = LLMBudget(max_calls_per_step=2)
    budget.check("step")
    budget.record("step")
    budget.check("step")
    budget.record("step")


def test_budget_exceeded_after_max_calls_per_step():
    budget = LLMBudget(max_calls_per_step=2)
    budget.check("step"); budget.record("step")
    budget.check("step"); budget.record("step")
    with pytest.raises(BudgetExceeded):
        budget.check("step")


def test_budget_is_scoped_per_step_independently():
    budget = LLMBudget(max_calls_per_step=1)
    budget.check("step_a"); budget.record("step_a")
    with pytest.raises(BudgetExceeded):
        budget.check("step_a")
    budget.check("step_b")  # different step, its own quota -- unaffected


def test_budget_exceeded_after_total_wallclock_budget():
    budget = LLMBudget(total_wallclock_budget_s=0.05)
    budget.start()
    time.sleep(0.1)
    with pytest.raises(BudgetExceeded):
        budget.check("step")


def test_budget_exceeded_is_an_llm_unavailable_subclass():
    assert issubclass(BudgetExceeded, LLMUnavailable)


def test_budgeted_client_wraps_and_records_calls_on_the_underlying_client():
    fake = FakeLLMClient(responses=[("hello", {"ok": True})])
    budget = LLMBudget(max_calls_per_step=1)
    budgeted = BudgetedLLMClient(fake, budget, step="test_step")

    result = budgeted.complete_json(system="s", user="hello", json_schema={})
    assert result == {"ok": True}
    assert len(fake.calls) == 1

    with pytest.raises(BudgetExceeded):
        budgeted.complete_json(system="s", user="hello", json_schema={})
    assert len(fake.calls) == 1  # the second call never reached the inner client


def test_budget_exhaustion_degrades_gracefully_through_existing_call_sites():
    """The whole point of BudgetExceeded being an LLMUnavailable subclass:
    a real caller (constraints.resolve_one) needs zero code changes to
    degrade correctly once a budgeted client's quota runs out."""
    fake = FakeLLMClient(responses=[("novel_type_x", {
        "measure": "count", "column_dimension_map": {"channel": "channel"},
        "per_subject": False, "confidence": "high",
    })])
    budget = LLMBudget(max_calls_per_step=0)  # exhausted from the start
    budgeted = BudgetedLLMClient(fake, budget, step="constraint_type")

    row = RawConstraintRow(raw_type="novel_type_x", channel="SMS", product=None, min=None, max=1.0)
    with pytest.raises(C.UnresolvedConstraintError):
        C.resolve_one(row, llm_client=budgeted, cache={})
