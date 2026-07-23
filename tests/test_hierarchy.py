import pytest

from offer_opt.discovery.hierarchy import build_alias_index, build_tree
from offer_opt.io.dialects import read_offers
from offer_opt.llm.client import FakeLLMClient, NullClient


def test_build_tree_recovers_a_4_level_chain_via_convention_alone():
    """Arbitrary depth "for free": each value's parent is found by peeling
    trailing tokens until a known value matches -- no assumption the tree
    stops at 2 or 3 levels."""
    values = ["biz", "biz_line", "biz_line_prod", "biz_line_prod_seg"]
    tree = build_tree("dim", values, llm_client=NullClient())

    assert tree.parent_of["biz"] is None
    assert tree.parent_of["biz_line"] == "biz"
    assert tree.parent_of["biz_line_prod"] == "biz_line"
    assert tree.parent_of["biz_line_prod_seg"] == "biz_line_prod"

    # Ancestor-or-self interval check: the root's range must cover every
    # descendant's range (same invariant test_scope.py checks on ScopeIndex).
    assert tree.tin["biz"] <= tree.tin["biz_line_prod_seg"] <= tree.tout["biz"]


def test_build_tree_finds_closest_ancestor_not_a_more_distant_one():
    """If both a 1-token and a 2-token truncation are known values, the
    1-token (closer) one wins as the immediate parent."""
    values = ["a", "a_b", "a_b_c"]
    tree = build_tree("dim", values, llm_client=NullClient())
    assert tree.parent_of["a_b_c"] == "a_b"  # not "a"


def test_build_tree_recovers_real_intra_dimension_nesting_in_med_product_data():
    """Verified against real data: med's PRODUCT column has both "KSK" and
    "KSK_OSG" as distinct values -- "KSK_OSG" is a genuine intra-dimension
    child of "KSK", discoverable via naming convention with no LLM at all."""
    products = read_offers("med")["PRODUCT"].unique().tolist()
    assert {"KSK", "KSK_OSG", "IFL"} <= set(products)

    tree = build_tree("product", products, llm_client=NullClient())
    assert tree.parent_of["KSK_OSG"] == "KSK"
    assert tree.parent_of["KSK"] is None
    assert tree.parent_of["IFL"] is None


def test_channel_taxonomy_is_unresolved_by_convention_alone():
    """The system_design_overview.md Section 3 example (email -> personal/
    subscription -> automated/hand-written -> expensive/cheap) shares no
    substrings across levels -- naming convention alone can't place any of
    it, which is exactly why an LLM fallback is load-bearing here, not
    decorative."""
    values = ["email", "personal", "subscription", "automated", "hand-written", "expensive", "cheap"]
    tree = build_tree("channel", values, llm_client=NullClient())
    assert all(tree.parent_of[v] is None for v in values if v != "email")


def test_channel_taxonomy_resolves_correctly_via_fake_llm_client():
    values = ["email", "personal", "subscription", "automated", "hand-written", "expensive", "cheap"]
    fake = FakeLLMClient(responses=[
        ("channel", {"edges": [
            {"value": "email", "parent": None},
            {"value": "personal", "parent": "email"},
            {"value": "subscription", "parent": "email"},
            {"value": "automated", "parent": "personal"},
            {"value": "hand-written", "parent": "personal"},
            {"value": "expensive", "parent": "automated"},
            {"value": "cheap", "parent": "automated"},
        ]}),
    ])
    tree = build_tree("channel", values, llm_client=fake)

    assert tree.parent_of["personal"] == "email"
    assert tree.parent_of["automated"] == "personal"
    assert tree.parent_of["expensive"] == "automated"
    # Ancestor-or-self: email (root) must cover every other value's interval.
    for v in values:
        assert tree.tin["email"] <= tree.tin[v] <= tree.tout["email"]
    # automated's interval must NOT cover subscription (a sibling branch).
    assert not (tree.tin["automated"] <= tree.tin["subscription"] <= tree.tout["automated"])
    assert len(fake.calls) == 1


def test_build_alias_index_maps_unambiguous_values_to_their_owning_dimension():
    index = build_alias_index({
        "product": {"IFL", "KSK", "KSK_OSG"},
        "segment": {"IFL_AA", "IFL_AnA", "KSK_DnA"},
        "channel": {"EMAIL", "SMS"},
    })
    assert index["IFL_AA"] == "segment:IFL_AA"
    assert index["KSK"] == "product:KSK"
    assert index["SMS"] == "channel:SMS"


def test_build_alias_index_excludes_genuinely_ambiguous_values():
    index = build_alias_index({
        "product": {"IFL", "SHARED"},
        "segment": {"IFL_AA", "SHARED"},
    })
    assert "SHARED" not in index
    assert index["IFL"] == "product:IFL"
