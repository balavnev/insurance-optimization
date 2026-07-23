from offer_opt.constraints import resolve_all, resolve_one
from offer_opt.schema import ConstraintSpec, ParameterSpec, RawConstraintRow


def test_cost_of_communication_becomes_parameter():
    row = RawConstraintRow(raw_type="cost_of_communication", channel="SMS", product=None, min=5.2, max=None)
    resolved = resolve_one(row)
    assert isinstance(resolved, ParameterSpec)
    assert resolved.kind == "cost"
    assert resolved.scope == {"channel": "SMS"}
    assert resolved.value == 5.2


def test_margin_per_product_uses_max_over_min():
    row = RawConstraintRow(raw_type="margin_per_product", channel=None, product="OSAGO", min=0.0, max=0.231)
    resolved = resolve_one(row)
    assert isinstance(resolved, ParameterSpec)
    assert resolved.kind == "margin"
    assert resolved.scope == {"product": "OSAGO"}
    assert resolved.value == 0.231  # max wins over min when both present


def test_offers_per_segment_repurposes_product_column():
    row = RawConstraintRow(raw_type="offers_per_segment", channel="EMAIL", product="IFL_AA", min=200.0, max=None)
    resolved = resolve_one(row)
    assert isinstance(resolved, ConstraintSpec)
    assert resolved.scope == {"channel": "EMAIL", "segment": "IFL_AA"}
    assert resolved.measure == "count"
    assert resolved.min == 200.0


def test_fallback_classifies_measure_and_per_client_by_convention():
    total_cost = resolve_one(RawConstraintRow("total_cost", None, None, 0.0, 63000.0))
    assert isinstance(total_cost, ConstraintSpec)
    assert total_cost.measure == "cost"
    assert total_cost.scope == {}
    assert not total_cost.per_client

    per_client = resolve_one(RawConstraintRow("offers_per_channel_per_client", "EMAIL", None, 0.0, 3.0))
    assert isinstance(per_client, ConstraintSpec)
    assert per_client.measure == "count"
    assert per_client.per_client

    cost_per_channel = resolve_one(RawConstraintRow("cost_per_channel", "SMS", None, None, 25000.0))
    assert isinstance(cost_per_channel, ConstraintSpec)
    assert cost_per_channel.measure == "cost"
    assert cost_per_channel.scope == {"channel": "SMS"}

    offers_per_product = resolve_one(RawConstraintRow("offers_per_product", "OCRM", "IFL", 100.0, 120.0))
    assert isinstance(offers_per_product, ConstraintSpec)
    assert offers_per_product.measure == "count"
    assert offers_per_product.scope == {"channel": "OCRM", "product": "IFL"}


def test_a_brand_new_convention_following_type_needs_no_code_change():
    """A constraint type that never existed in any of the 3 given files, but
    follows the same "<measure>_per_<scope>[_per_client]" naming shape,
    should resolve correctly through the fallback alone."""
    row = RawConstraintRow("cost_per_segment_per_client", "MOBILE", "NEWPRODUCT", 0.0, 500.0)
    resolved = resolve_one(row)
    assert isinstance(resolved, ConstraintSpec)
    assert resolved.measure == "cost"
    assert resolved.per_client
    assert resolved.scope == {"channel": "MOBILE", "product": "NEWPRODUCT"}


def test_resolve_all_splits_constraints_and_parameters():
    rows = [
        RawConstraintRow("cost_of_communication", "EMAIL", None, 0.02, None),
        RawConstraintRow("total_cost", None, None, 0.0, 63000.0),
        RawConstraintRow("margin_per_product", None, "IFL", 0.0, 0.15),
    ]
    cs = resolve_all(rows)
    assert len(cs.parameters) == 2
    assert len(cs.constraints) == 1
    assert cs.constraints[0].raw_type == "total_cost"
