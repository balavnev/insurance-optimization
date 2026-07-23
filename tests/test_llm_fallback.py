import json
from pathlib import Path

import pytest

from offer_opt import constraints as C
from offer_opt.constraints import UnresolvedConstraintError, resolve_one
from offer_opt.llm.client import FakeLLMClient, NullClient
from offer_opt.schema import ConstraintSpec, RawConstraintRow

FIXTURES = Path(__file__).parent / "fixtures" / "constraint_classification_cases.json"
CASES = json.loads(FIXTURES.read_text())


def _row_for(case: dict) -> RawConstraintRow:
    return RawConstraintRow(raw_type=case["raw_type"], channel=case["channel"], product=case["product"],
                             min=case["min"], max=case["max"])


def _fake_client_for(case: dict) -> FakeLLMClient:
    response = {
        "measure": case["expected_measure"],
        "column_dimension_map": case["expected_column_dimension_map"],
        "per_subject": case["expected_per_subject"],
        "confidence": "high",
    }
    return FakeLLMClient(responses=[(case["raw_type"], response)])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_novel_constraint_types_resolve_correctly_via_fake_llm_client(case):
    """None of these raw_type strings match any override or the
    "<measure>_per_<scope>[_per_client]" naming convention -- they can only
    resolve through the LLM fallback."""
    row = _row_for(case)
    fake = _fake_client_for(case)

    resolved = resolve_one(row, llm_client=fake, cache={})

    assert isinstance(resolved, ConstraintSpec)
    assert resolved.measure == case["expected_measure"]
    assert resolved.per_client == case["expected_per_subject"]

    expected_scope = {}
    mapping = case["expected_column_dimension_map"]
    if "channel" in mapping and row.channel is not None:
        expected_scope[mapping["channel"]] = row.channel
    if "product" in mapping and row.product is not None:
        expected_scope[mapping["product"]] = row.product
    assert resolved.scope == expected_scope


def test_llm_fallback_caches_on_raw_type_not_per_row():
    """Two different rows sharing the same raw_type must hit the LLM once,
    not once per row -- the cache is keyed on the constraint-type string,
    never on subject-level data."""
    case = CASES[0]
    fake = _fake_client_for(case)
    cache: dict = {}

    row_a = _row_for(case)
    row_b = RawConstraintRow(raw_type=case["raw_type"], channel=case["channel"],
                              product=case["product"], min=0.0, max=999.0)  # different bounds, same type

    resolve_one(row_a, llm_client=fake, cache=cache)
    resolve_one(row_b, llm_client=fake, cache=cache)

    assert len(fake.calls) == 1


def test_no_llm_client_configured_raises_unresolved_immediately():
    """The default (no llm_client passed) is NullClient -- a novel type with
    no client configured must fail loudly, not silently guess."""
    case = CASES[0]
    row = _row_for(case)
    with pytest.raises(UnresolvedConstraintError):
        resolve_one(row, cache={})


def test_always_invalid_response_raises_after_retries_not_a_silent_guess():
    row = _row_for(CASES[0])
    always_invalid = FakeLLMClient(responses=[
        (CASES[0]["raw_type"], {"measure": "count", "column_dimension_map": {}, "per_subject": False,
                                  "confidence": "low"}),
    ])
    with pytest.raises(UnresolvedConstraintError):
        resolve_one(row, llm_client=always_invalid, cache={})
    # 1 initial attempt + 2 retries = 3 calls (max_retries=2 in _classify_via_llm)
    assert len(always_invalid.calls) == 3


def test_resolve_by_convention_is_never_reached_for_these_novel_types():
    """Sanity check on the fixture itself: none of these raw_type strings
    would accidentally be handled by the fast convention path (which would
    make this test suite vacuous)."""
    for case in CASES:
        row = _row_for(case)
        key = row.raw_type.strip().lower()
        assert key not in C._OVERRIDES
        assert not C._matches_naming_convention(key, row)


def test_offers_per_segment_is_re_derivable_via_the_llm_classification_path():
    """`offers_per_segment` is normally handled by the `_OVERRIDES` table
    (it produces a genuine ConstraintSpec with a column-remapping quirk: the
    vendor's "Product" column position actually holds a segment value). This
    calls the LLM classification path directly (bypassing the automatic
    override/convention gate, which would otherwise route this raw_type to
    `_OVERRIDES` or the convention fallback before ever reaching the LLM) to
    prove the LLM path *could* correctly re-derive it if it were ever the
    only available path -- e.g. on a held-out dataset with a differently-
    worded but structurally identical vendor quirk."""
    row = RawConstraintRow(raw_type="offers_per_segment", channel="EMAIL", product="IFL_AA",
                            min=200.0, max=None)
    fake = FakeLLMClient(responses=[
        ("offers_per_segment", {
            "measure": "count",
            "column_dimension_map": {"channel": "channel", "product": "segment"},
            "per_subject": False,
            "confidence": "high",
        }),
    ])

    via_llm = C._resolve_via_llm(row, "offers_per_segment", C._DEFAULT_DIMS, fake, {})
    via_override = C._handle_offers_per_segment(row)

    assert via_llm.scope == via_override.scope == {"channel": "EMAIL", "segment": "IFL_AA"}
    assert via_llm.measure == via_override.measure == "count"
    assert via_llm.per_client == via_override.per_client == False


def test_cost_of_communication_and_margin_per_product_remain_override_only_by_design():
    """These two resolve to ParameterSpec (EV-formula inputs disguised as
    constraint rows), not ConstraintSpec -- a fundamentally different output
    kind the shape-classifying LLM schema (measure/scope/per_subject) isn't
    designed to produce. This is not a gap introduced by the LLM fallback:
    the naming-convention fallback already only ever produces ConstraintSpec
    too. Documented here as an explicit, intentional scope boundary rather
    than an oversight."""
    from offer_opt.schema import ParameterSpec

    cost_row = RawConstraintRow(raw_type="cost_of_communication", channel="SMS", product=None, min=5.2, max=None)
    margin_row = RawConstraintRow(raw_type="margin_per_product", channel=None, product="OSAGO", min=0.0, max=0.231)

    assert isinstance(C._handle_cost_of_communication(cost_row), ParameterSpec)
    assert isinstance(C._handle_margin_per_product(margin_row), ParameterSpec)
