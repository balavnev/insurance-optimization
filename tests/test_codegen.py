import numpy as np
import pandas as pd
import pytest

from offer_opt import features as F
from offer_opt import metrics as M
from offer_opt import verify as V
from offer_opt.codegen import sandbox
from offer_opt.codegen.generate import CodegenError, cross_check, generate_all, generate_one, generate_via_llm
from offer_opt.llm.client import FakeLLMClient
from offer_opt.schema import ConstraintSpec

FAST_CASES = ["low", "med"]


def _global_constraint(max_=4000.0, min_=None):
    return ConstraintSpec(id="test_global", raw_type="offers_per_product",
                           scope={"channel": "SMS", "product": "KSK"},
                           measure="count", min=min_, max=max_, per_client=False)


def _per_client_constraint(max_=3.0, min_=1.0):
    return ConstraintSpec(id="test_per_client", raw_type="offers_per_channel_per_client",
                           scope={"channel": "EMAIL"}, measure="count", min=min_, max=max_, per_client=True)


# ---------------------------------------------------------------------------
# Template rendering / basic correctness
# ---------------------------------------------------------------------------

def test_generated_global_function_matches_expected_semantics():
    c = _global_constraint(max_=3.0)
    gc = generate_one(c)
    table = pd.DataFrame({"channel": ["SMS"] * 5, "product": ["KSK"] * 5, "client_id": range(5), "cost": [1.0] * 5})

    assert gc.fn(table, np.array([1, 1, 1, 0, 0])) is True   # 3 selected, at the cap
    assert gc.fn(table, np.array([1, 1, 1, 1, 0])) is False  # 4 selected, over the cap


def test_generated_per_client_function_matches_expected_semantics():
    c = _per_client_constraint(max_=2.0, min_=None)
    gc = generate_one(c)
    table = pd.DataFrame({
        "channel": ["EMAIL"] * 6,
        "client_id": [1, 1, 1, 2, 2, 2],
        "cost": [1.0] * 6,
    })
    assert gc.fn(table, np.array([1, 1, 0, 1, 0, 0])) is True   # client 1: 2 (ok), client 2: 1 (ok)
    assert gc.fn(table, np.array([1, 1, 1, 1, 0, 0])) is False  # client 1: 3 (over cap of 2)


def test_generated_source_contains_no_runtime_none_literal_comparisons():
    """A bound that's genuinely absent should omit the branch entirely, not
    embed a runtime `if <literal> is not None` check against a constant --
    confirms the earlier SyntaxWarning-producing bug stays fixed."""
    gc = generate_one(_global_constraint(max_=100.0, min_=None))
    assert "is not None" not in gc.source
    assert " None " not in gc.source


# ---------------------------------------------------------------------------
# Sandbox safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_source", [
    "import os\ndef check(option_table, selection):\n    return True\n",
    "def check(option_table, selection):\n    return exec('1')\n",
    "def check(option_table, selection):\n    return open('/etc/passwd')\n",
    "def check(option_table, selection):\n    return option_table.__class__\n",
])
def test_sandbox_rejects_unsafe_generated_code(bad_source):
    with pytest.raises(sandbox.UnsafeGeneratedCodeError):
        sandbox.compile_check_function(bad_source, "check")


def test_sandbox_accepts_safe_generated_code():
    source = 'def check(option_table, selection):\n    return bool(np.asarray(selection).sum() >= 0)\n'
    fn = sandbox.compile_check_function(source, "check")
    assert fn(pd.DataFrame({"x": [1]}), np.array([1])) is True


# ---------------------------------------------------------------------------
# Golden fixtures + fault injection
# ---------------------------------------------------------------------------

def test_golden_fixtures_pass_for_a_correct_generated_function():
    c = _global_constraint(max_=4000.0)
    gc = generate_one(c)
    sandbox.check_golden_fixtures(gc.fn, c)  # must not raise


def test_golden_fixtures_catch_a_deliberately_wrong_function():
    """Fault injection: a function that always returns True regardless of
    the actual usage must be caught by the golden-fixture self-test before
    it's ever trusted against real data."""
    c = _global_constraint(max_=4.0)  # small enough that golden_fixtures() actually probes it
    always_true_source = "def check_always_true(option_table, selection):\n    return True\n"
    fn = sandbox.compile_check_function(always_true_source, "check_always_true")
    with pytest.raises(sandbox.GeneratedCodeFailedGoldenFixture):
        sandbox.check_golden_fixtures(fn, c)


def test_cross_check_catches_a_deliberately_wrong_function_against_real_data():
    """The other fault-injection path: a function that's wrong specifically
    on real data (not just synthetic fixtures) must be caught by
    cross_check(), the step that runs it against the actual dataset and
    compares its verdict to verify.py's."""
    offer_table, cs = F.load_case("low")
    offer_table, _n = F.encode_dims(offer_table)
    real_constraint = cs.constraints[0]

    from offer_opt.codegen.generate import GeneratedCheck
    always_true_source = "def check_wrong(option_table, selection):\n    return True\n"
    fn = sandbox.compile_check_function(always_true_source, "check_wrong")
    wrong = GeneratedCheck(constraint=real_constraint, function_name="check_wrong",
                            source=always_true_source, fn=fn, origin="template")

    # A selection that selects everything in scope is very likely to blow
    # past a "max" cap on a real dataset -- verify.py will say FAIL, the
    # always-true stub will say True -> disagreement, caught.
    all_selected = np.ones(len(offer_table))
    assert cross_check(wrong, offer_table, all_selected) is False


# ---------------------------------------------------------------------------
# Full generate + cross-check against real cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", FAST_CASES)
def test_generated_code_agrees_with_verify_on_every_constraint(case):
    offer_table, cs = F.load_case(case)
    offer_table, _n = F.encode_dims(offer_table)
    selection = M.load_reference(case, offer_table)

    generated = generate_all(cs)
    assert len(generated) == len(cs.constraints)
    for gc in generated.values():
        assert cross_check(gc, offer_table, selection), f"{case}: disagreement on {gc.constraint.id}"


@pytest.mark.slow
def test_generated_code_agrees_with_verify_on_hard_case():
    """Same as the fast-tier check above, gated behind -m slow: hard's
    scope-index rebuild per isolated constraint (88 of them, 5M rows) is the
    only genuinely slow part of this suite, not the codegen itself."""
    offer_table, cs = F.load_case("hard")
    offer_table, _n = F.encode_dims(offer_table)
    selection = M.load_reference("hard", offer_table)

    generated = generate_all(cs)
    for gc in generated.values():
        assert cross_check(gc, offer_table, selection), f"hard: disagreement on {gc.constraint.id}"


# ---------------------------------------------------------------------------
# LLM-invoked codegen path
# ---------------------------------------------------------------------------

def test_generate_via_llm_produces_a_working_function():
    c = ConstraintSpec(id="llm_test", raw_type="offers_per_product",
                        scope={"channel": "SMS", "product": "KSK"},
                        measure="count", min=None, max=4000.0, per_client=False)
    correct_source = (
        "def check_llm_test(option_table, selection):\n"
        f"    mask = scope_mask(option_table, {c.scope!r})\n"
        "    usage = np.where(mask, 1.0, 0.0) * np.asarray(selection, dtype='float64')\n"
        f"    return bool(usage.sum() <= {c.max!r} + 1e-6)\n"
    )
    fake = FakeLLMClient(responses=[("llm_test", {"source": correct_source})])

    gc = generate_via_llm(c, fake)
    assert gc.origin == "llm"
    table = pd.DataFrame({"channel": ["SMS"] * 5, "product": ["KSK"] * 5, "client_id": range(5), "cost": [1.0] * 5})
    assert gc.fn(table, np.array([1, 1, 1, 0, 0])) is True


def test_generate_via_llm_fault_injection_is_caught_before_being_trusted():
    """A deliberately-wrong LLM-authored function (always returns True) must
    be rejected by the golden-fixture check inside generate_via_llm itself
    -- it never even gets returned as a usable GeneratedCheck."""
    c = ConstraintSpec(id="llm_bad_test", raw_type="offers_per_product",
                        scope={"channel": "SMS", "product": "KSK"},
                        measure="count", min=None, max=2.0, per_client=False)
    always_true_source = "def check_llm_bad_test(option_table, selection):\n    return True\n"
    fake = FakeLLMClient(responses=[("llm_bad_test", {"source": always_true_source})])

    with pytest.raises(CodegenError):
        generate_via_llm(c, fake, max_retries=0)
