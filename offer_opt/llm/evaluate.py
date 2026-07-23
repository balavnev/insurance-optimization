"""Prompt-evaluation harness -- house style from
vendor-examples/examples/prompt_lab.py: run a fixed case set through a given
LLM client, score each case's correctness/schema-validity/latency, aggregate
into a summary, and persist both to disk as timestamped JSONL + CSV.

Deferred from Phase 4 to Phase 7 deliberately: with only `FakeLLMClient`
available, "accuracy" is either 100% (scripted correctly) or a bug, and
"latency" is ~0 -- there's no real signal to aggregate until there's an
actual model (`VLLMOpenAIClient`) with genuine accuracy/latency variance to
measure and compare, which is exactly what Phase 7 adds.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from offer_opt.constraints import UnresolvedConstraintError, _classify_via_llm
from offer_opt.llm.client import LLMUnavailable
from offer_opt.schema import RawConstraintRow


def evaluate_constraint_classification(cases: list[dict[str, Any]], llm_client, result_dir: str | Path,
                                        dims: tuple[str, ...] = ("channel", "product", "segment"),
                                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs `constraints._classify_via_llm` -- the actual production call
    path, not a re-implementation -- against every case, so this evaluates
    exactly what the pipeline would really do, not a simplified stand-in."""
    rows: list[dict[str, Any]] = []

    for case in cases:
        row = RawConstraintRow(raw_type=case["raw_type"], channel=case["channel"], product=case["product"],
                                min=case["min"], max=case["max"])
        error: str | None = None
        response: dict | None = None
        t0 = time.perf_counter()
        try:
            response = _classify_via_llm(row, case["raw_type"], dims, llm_client)
        except (LLMUnavailable, UnresolvedConstraintError) as exc:
            error = repr(exc)
        elapsed = time.perf_counter() - t0

        measure_correct = response is not None and response.get("measure") == case["expected_measure"]
        scope_correct = (response is not None
                          and response.get("column_dimension_map") == case["expected_column_dimension_map"])
        per_subject_correct = response is not None and response.get("per_subject") == case["expected_per_subject"]

        rows.append({
            "case_id": case["id"],
            "raw_type": case["raw_type"],
            "measure_correct": measure_correct,
            "scope_correct": scope_correct,
            "per_subject_correct": per_subject_correct,
            "all_correct": bool(measure_correct and scope_correct and per_subject_correct),
            "error": error,
            "elapsed_seconds": elapsed,
            "response": response,
        })

    details = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "cases": len(details),
        "accuracy": float(details["all_correct"].mean()) if len(details) else 0.0,
        "error_rate": float(details["error"].notna().mean()) if len(details) else 0.0,
        "average_seconds": float(details["elapsed_seconds"].mean()) if len(details) else 0.0,
    }])

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    details_file = result_dir / f"constraint-classification-evaluation-{timestamp}.jsonl"
    summary_file = result_dir / f"constraint-classification-summary-{timestamp}.csv"

    with details_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary.to_csv(summary_file, index=False)

    return details, summary
