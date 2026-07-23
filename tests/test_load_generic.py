import numpy as np
import pytest

from offer_opt import features as F
from offer_opt import metrics as M
from offer_opt import verify as V

CASES = ["low", "med", "hard"]


@pytest.mark.parametrize("case", CASES)
def test_load_generic_matches_load_case_offer_table_shape_and_ev(case):
    """load_generic() -- built by sniffing + discovery instead of
    hand-written per-case column names -- must produce an offer table
    verify()-equivalent to the hand-written load_case(): same row count,
    same base_ev values once rows are aligned on (client_id, product,
    channel, segment)."""
    hand, _cs_hand = F.load_case(case)
    generic, _cs_generic = F.load_generic(case)

    assert len(hand) == len(generic)

    key_cols = ["client_id", "product", "channel", "segment"]
    hand_sorted = hand.sort_values(key_cols).reset_index(drop=True)
    generic_sorted = generic.sort_values(key_cols).reset_index(drop=True)

    assert (hand_sorted["client_id"].to_numpy() == generic_sorted["client_id"].to_numpy()).all()
    assert np.allclose(hand_sorted["base_ev"].to_numpy(), generic_sorted["base_ev"].to_numpy(), atol=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_load_generic_constraint_set_matches_load_case_counts(case):
    _hand, cs_hand = F.load_case(case)
    _generic, cs_generic = F.load_generic(case)
    assert len(cs_hand.constraints) == len(cs_generic.constraints)
    assert len(cs_hand.parameters) == len(cs_generic.parameters)


@pytest.mark.parametrize("case", CASES)
def test_vendor_reference_solution_still_passes_verify_under_generic_constraint_set(case):
    """The strongest cross-check on the generic pipeline: the vendor's own
    reference solution (ground truth we didn't generate ourselves) must
    still satisfy every constraint parsed via load_generic()'s discovery
    path, exactly as it already does for the hand-written load_case()
    (tests/test_end_to_end.py::test_reference_solution_passes_our_verifier)."""
    offer_table, constraint_set = F.load_generic(case)
    offer_table, _n = F.encode_dims(offer_table)

    selection = M.load_reference(case, offer_table)
    report = V.verify(offer_table, constraint_set, selection)
    assert report.ok, f"{case}: reference selection failed verification under load_generic()'s constraints: {report}"


@pytest.mark.parametrize("case", CASES)
def test_load_generic_total_ev_under_reference_matches_load_case(case):
    """Not just "both pass verify()" -- the actual EV number under the same
    reference selection must match too, proving the EV arithmetic (the
    generic margin*premium*score-cost formula) is truly equivalent, not
    just coincidentally both non-negative/feasible."""
    hand_offer_table, hand_cs = F.load_case(case)
    hand_offer_table, _n = F.encode_dims(hand_offer_table)
    hand_selection = M.load_reference(case, hand_offer_table)
    hand_ev = V.verify(hand_offer_table, hand_cs, hand_selection).total_ev

    generic_offer_table, generic_cs = F.load_generic(case)
    generic_offer_table, _n = F.encode_dims(generic_offer_table)
    generic_selection = M.load_reference(case, generic_offer_table)
    generic_ev = V.verify(generic_offer_table, generic_cs, generic_selection).total_ev

    assert generic_ev == pytest.approx(hand_ev, rel=1e-6)
