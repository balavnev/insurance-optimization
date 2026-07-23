import numpy as np
import pandas as pd

from offer_opt.schema import ConstraintSet, ConstraintSpec
from offer_opt.verify import verify

# 3 clients x 2 products x 1 channel = 6 candidate offers.
OFFER_TABLE = pd.DataFrame({
    "offer_uid": range(6),
    "client_id": [1, 1, 2, 2, 3, 3],
    "product": ["A", "B", "A", "B", "A", "B"],
    "channel": ["SMS"] * 6,
    "segment": ["A_SMS", "B_SMS"] * 3,
    "cost": [1.0] * 6,
    "base_ev": [10.0, 5.0, 8.0, 12.0, 3.0, 20.0],
})


def _cs(constraints):
    return ConstraintSet(constraints=constraints, parameters=[])


def test_passes_when_within_bounds():
    cs = _cs([ConstraintSpec(id="cap", raw_type="total_x", scope={}, measure="count", max=3, per_client=False)])
    selection = np.array([1, 0, 1, 0, 1, 0])  # one per client, product A each -- 3 selected
    report = verify(OFFER_TABLE, cs, selection)
    assert report.ok
    assert report.total_ev == 21.0
    assert report.n_selected == 3


def test_flags_global_max_violation():
    cs = _cs([ConstraintSpec(id="cap", raw_type="total_x", scope={}, measure="count", max=2, per_client=False)])
    selection = np.array([1, 0, 1, 0, 1, 0])  # 3 selected > cap of 2
    report = verify(OFFER_TABLE, cs, selection)
    assert not report.ok
    assert report.violations[0].bound == "max"
    assert report.violations[0].observed == 3


def test_flags_per_client_max_violation():
    cs = _cs([ConstraintSpec(id="one_per_client", raw_type="x_per_client", scope={}, measure="count",
                              max=1, per_client=True)])
    selection = np.array([1, 1, 1, 0, 0, 0])  # client 1 gets BOTH A and B
    report = verify(OFFER_TABLE, cs, selection)
    assert not report.ok
    assert report.violations[0].n_offending_clients == 1


def test_scope_restricts_which_offers_count():
    cs = _cs([ConstraintSpec(id="cap_A", raw_type="offers_per_product", scope={"product": "A"},
                              measure="count", max=1, per_client=False)])
    # 2 offers of product A selected (clients 1 and 2), should violate cap of 1
    selection = np.array([1, 0, 1, 0, 0, 1])
    report = verify(OFFER_TABLE, cs, selection)
    assert not report.ok
    assert report.violations[0].constraint_id == cs.constraints[0].id


def test_min_bound_checked():
    cs = _cs([ConstraintSpec(id="min3", raw_type="total_x", scope={}, measure="count", min=4, per_client=False)])
    selection = np.array([1, 0, 1, 0, 0, 0])  # only 2 selected, below min of 4
    report = verify(OFFER_TABLE, cs, selection)
    assert not report.ok
    assert report.violations[0].bound == "min"
