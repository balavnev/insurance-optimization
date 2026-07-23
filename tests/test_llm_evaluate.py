import json
from pathlib import Path

import pandas as pd

from offer_opt.llm.client import FakeLLMClient
from offer_opt.llm.evaluate import evaluate_constraint_classification

FIXTURES = Path(__file__).parent / "fixtures" / "constraint_classification_cases.json"
CASES = json.loads(FIXTURES.read_text())


def _perfect_fake_client() -> FakeLLMClient:
    responses = []
    for case in CASES:
        responses.append((case["raw_type"], {
            "measure": case["expected_measure"],
            "column_dimension_map": case["expected_column_dimension_map"],
            "per_subject": case["expected_per_subject"],
            "confidence": "high",
        }))
    return FakeLLMClient(responses=responses)


def test_evaluate_reports_perfect_accuracy_for_a_correctly_scripted_client(tmp_path):
    details, summary = evaluate_constraint_classification(CASES, _perfect_fake_client(), tmp_path)

    assert len(details) == len(CASES)
    assert details["all_correct"].all()
    assert summary["accuracy"].iloc[0] == 1.0
    assert summary["error_rate"].iloc[0] == 0.0
    assert summary["cases"].iloc[0] == len(CASES)


def test_evaluate_reports_errors_when_no_llm_client_is_usable(tmp_path):
    from offer_opt.llm.client import NullClient
    details, summary = evaluate_constraint_classification(CASES, NullClient(), tmp_path)
    assert (details["error"].notna()).all()
    assert summary["error_rate"].iloc[0] == 1.0
    assert summary["accuracy"].iloc[0] == 0.0


def test_evaluate_persists_details_jsonl_and_summary_csv_to_disk(tmp_path):
    evaluate_constraint_classification(CASES, _perfect_fake_client(), tmp_path)

    jsonl_files = list(tmp_path.glob("constraint-classification-evaluation-*.jsonl"))
    csv_files = list(tmp_path.glob("constraint-classification-summary-*.csv"))
    assert len(jsonl_files) == 1
    assert len(csv_files) == 1

    lines = jsonl_files[0].read_text().splitlines()
    assert len(lines) == len(CASES)
    first = json.loads(lines[0])
    assert "case_id" in first and "all_correct" in first

    summary_df = pd.read_csv(csv_files[0])
    assert summary_df["accuracy"].iloc[0] == 1.0


def test_evaluate_detects_a_wrong_measure_as_incorrect(tmp_path):
    case = CASES[0]
    wrong_response = {
        "measure": "cost" if case["expected_measure"] == "count" else "count",  # deliberately wrong
        "column_dimension_map": case["expected_column_dimension_map"],
        "per_subject": case["expected_per_subject"],
        "confidence": "high",
    }
    fake = FakeLLMClient(responses=[(case["raw_type"], wrong_response)])
    details, summary = evaluate_constraint_classification([case], fake, tmp_path)

    assert details["measure_correct"].iloc[0] == False
    assert details["all_correct"].iloc[0] == False
    assert summary["accuracy"].iloc[0] == 0.0
