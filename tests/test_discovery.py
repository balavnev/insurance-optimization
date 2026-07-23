import numpy as np
import pandas as pd
import pytest

from offer_opt.discovery.column_roles import classify_columns
from offer_opt.discovery.schema_resolver import resolve_schema
from offer_opt.io import sniff
from offer_opt.io.dialects import CASE_FILES, OFFER_DIALECTS, read_offers
from offer_opt.llm.client import FakeLLMClient, LLMUnavailable, NullClient


# ---------------------------------------------------------------------------
# io/sniff.py: dialect + shape detection against the real, unlabeled files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["low", "med", "hard"])
def test_sniff_dialect_reproduces_hardcoded_registry(case):
    """The generic sniffer, run on the raw file with no knowledge of which
    case it is, should rediscover the same (sep, decimal) the hand-written
    `io/dialects.py` registry hardcodes for it."""
    path = CASE_FILES[case]["offers"]
    sniffed = sniff.sniff_dialect(path)
    expected = OFFER_DIALECTS[case]
    assert sniffed.sep == expected.sep
    assert sniffed.decimal == expected.decimal


def test_detect_shape_distinguishes_wide_low_from_long_med_hard():
    low = read_offers("low")
    assert sniff.detect_shape(low, "subjisn") == "wide"

    med = read_offers("med")
    assert sniff.detect_shape(med, "SUBJISN") == "long"

    hard = read_offers("hard")
    assert sniff.detect_shape(hard, "SUBJISN") == "long"


def test_is_key_value_format_detects_low_constraints_only():
    assert sniff.is_key_value_format(CASE_FILES["low"]["constraints"])
    assert not sniff.is_key_value_format(CASE_FILES["med"]["constraints"])
    assert not sniff.is_key_value_format(CASE_FILES["hard"]["constraints"])


# ---------------------------------------------------------------------------
# discover_metric_clusters / reshape_wide_to_long: the generic version of
# reshape_low.py's hardcoded metric alternation
# ---------------------------------------------------------------------------

def test_discover_metric_clusters_finds_the_four_low_metrics_without_naming_them():
    columns = list(read_offers("low", nrows=10).columns)
    clusters = sniff.discover_metric_clusters(columns)
    assert set(clusters) == {"PREMIUM", "PROBA", "MARGIN", "COST"}
    assert set(clusters["COST"]) == {"SMS", "EMAIL"}
    assert set(clusters["PREMIUM"]) == {"OSG_SMS", "OSG_EMAIL", "KSK_SMS", "KSK_EMAIL"}


def test_reshape_wide_to_long_recovers_the_pivoted_low_table():
    wide = read_offers("low")
    long_df = sniff.reshape_wide_to_long(wide, subject_id_column="subjisn")

    assert set(long_df.columns) == {"subject_id", "dim_anchor", "dim_residual",
                                     "PREMIUM", "PROBA", "MARGIN", "COST", "_offer_slot"}
    # 4 combos (2 products x 2 channels) per client, same client count as the source.
    assert long_df["subject_id"].nunique() == wide["subjisn"].nunique()
    assert len(long_df) == wide["subjisn"].nunique() * 4
    assert set(long_df["dim_anchor"].unique()) == {"SMS", "EMAIL"}
    assert set(long_df["dim_residual"].unique()) == {"OSG", "KSK"}

    # Spot-check client 1's PREMIUM_OSG_SMS value round-trips through the
    # generic reshape unchanged.
    row = long_df[(long_df["subject_id"] == 1) & (long_df["dim_residual"] == "OSG") & (long_df["dim_anchor"] == "SMS")]
    assert len(row) == 1
    assert row["PREMIUM"].iloc[0] == pytest.approx(wide.loc[wide["subjisn"] == 1, "PREMIUM_OSG_SMS"].iloc[0])


# ---------------------------------------------------------------------------
# resolve_schema: heuristic-only reproduction of features.py's hardcoded
# per-case column mapping
# ---------------------------------------------------------------------------

def test_resolve_schema_matches_load_med_mapping():
    df = read_offers("med")
    mapping = resolve_schema(df)  # NullClient by default
    assert mapping.subject_id_column == "SUBJISN"
    assert set(mapping.dimension_columns) == {"PRODUCT", "CHANNEL", "SEGMENT"}
    assert mapping.value_component_columns == {"score": "SCORE"}
    assert "Unnamed: 0" in mapping.unresolved_columns  # ignored bookkeeping index, not a value component


def test_resolve_schema_matches_load_hard_mapping():
    df = read_offers("hard")
    mapping = resolve_schema(df)
    assert mapping.subject_id_column == "SUBJISN"
    assert set(mapping.dimension_columns) == {"PRODUCT", "CHANNEL", "SEGMENT"}
    assert mapping.value_component_columns == {"score": "SCORE", "premium": "AVG_CHECK"}


def test_resolve_schema_matches_load_low_mapping_after_generic_reshape():
    wide = read_offers("low")
    long_df = sniff.reshape_wide_to_long(wide, subject_id_column="subjisn")
    mapping = resolve_schema(long_df)

    assert mapping.subject_id_column == "subject_id"
    assert set(mapping.dimension_columns) == {"dim_anchor", "dim_residual"}
    assert mapping.value_component_columns == {
        "premium": "PREMIUM", "score": "PROBA", "margin": "MARGIN", "cost": "COST",
    }


# ---------------------------------------------------------------------------
# LLM fallback: only exercised for genuinely ambiguous columns
# ---------------------------------------------------------------------------

def test_ambiguous_numeric_column_falls_back_to_null_client_heuristic_guess():
    """No LLM configured -- an unrecognized numeric column degrades to
    "ignore" rather than blocking or crashing."""
    df = pd.DataFrame({
        "client": [1, 1, 2, 2, 3, 3],
        "channel": ["sms", "email"] * 3,
        "score": np.random.rand(6),
        "xyzzy_metric": np.random.rand(6),  # no name-hint match
    })
    mapping = resolve_schema(df, llm_client=NullClient())
    assert mapping.subject_id_column == "client"
    assert "xyzzy_metric" in mapping.unresolved_columns
    assert "xyzzy_metric" not in mapping.value_component_columns


def test_ambiguous_numeric_column_resolved_via_fake_llm_client():
    df = pd.DataFrame({
        "client": [1, 1, 2, 2, 3, 3],
        "channel": ["sms", "email"] * 3,
        "score": np.random.rand(6),
        "xyzzy_metric": np.random.rand(6),
    })
    fake = FakeLLMClient(responses=[("xyzzy_metric", {"role": "cost"})])
    mapping = resolve_schema(df, llm_client=fake)
    assert mapping.value_component_columns["cost"] == "xyzzy_metric"
    assert len(fake.calls) == 1


def test_fake_llm_client_raises_on_unscripted_prompt():
    fake = FakeLLMClient(responses=[("something else", {"role": "cost"})])
    with pytest.raises(LLMUnavailable):
        fake.complete_json(system="s", user="an unrelated prompt", json_schema={})
