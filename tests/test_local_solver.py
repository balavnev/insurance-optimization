import pandas as pd
import torch

from offer_opt.schema import ConstraintSpec
from offer_opt.scope import ScopeIndex
from offer_opt.solver.local import local_select

DEVICE = torch.device("cpu")

# 3 clients x 2 products x 1 channel = 6 candidate offers.
OFFER_TABLE = pd.DataFrame({
    "client_id": [1, 1, 2, 2, 3, 3],
    "product": ["A", "B", "A", "B", "A", "B"],
    "channel": ["SMS"] * 6,
    "segment": ["A_SMS", "B_SMS"] * 3,
})
CLIENT_IDX = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
NUM_CLIENTS = 3


def test_degenerates_to_plain_argmax_with_total_cap_one():
    value = torch.tensor([10.0, 5.0, 8.0, 12.0, 3.0, 20.0])
    scope_index = ScopeIndex(OFFER_TABLE)
    cap = ConstraintSpec(id="total", raw_type="_synthetic", scope={}, measure="count", max=1, per_client=True)
    selected = local_select(value, CLIENT_IDX, NUM_CLIENTS, [cap], scope_index, DEVICE)
    # client 0: best is offer 0 (10 > 5); client 1: best is offer 3 (12 > 8); client 2: best is offer 5 (20 > 3)
    assert selected.tolist() == [True, False, False, True, False, True]


def test_never_selects_negative_value_offers():
    value = torch.tensor([-1.0, -2.0, 8.0, 12.0, 3.0, 20.0])
    scope_index = ScopeIndex(OFFER_TABLE)
    cap = ConstraintSpec(id="total", raw_type="_synthetic", scope={}, measure="count", max=1, per_client=True)
    selected = local_select(value, CLIENT_IDX, NUM_CLIENTS, [cap], scope_index, DEVICE)
    assert selected[0].item() is False and selected[1].item() is False  # client 0 gets nothing


def test_allows_one_per_product_up_to_two_total():
    """med/hard's real shape: <=1 per product (not <=1 total), which permits
    multiple simultaneous offers per client."""
    value = torch.tensor([10.0, 5.0, 8.0, 12.0, 3.0, 20.0])
    scope_index = ScopeIndex(OFFER_TABLE)
    cap_a = ConstraintSpec(id="capA", raw_type="offers_per_product_per_client", scope={"product": "A"},
                            measure="count", max=1, per_client=True)
    cap_b = ConstraintSpec(id="capB", raw_type="offers_per_product_per_client", scope={"product": "B"},
                            measure="count", max=1, per_client=True)
    selected = local_select(value, CLIENT_IDX, NUM_CLIENTS, [cap_a, cap_b], scope_index, DEVICE)
    # every offer is the only candidate for its (client, product) pair, so ALL should be selected
    assert selected.tolist() == [True, True, True, True, True, True]
