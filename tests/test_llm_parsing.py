import pytest

from offer_opt.llm.parsing import extract_json, validate_against_schema
from offer_opt.llm.schemas import CONSTRAINT_TYPE_SCHEMA


def test_extract_json_strips_think_blocks():
    text = "<think>let me reason about this...</think>\n{\"measure\": \"count\"}"
    assert extract_json(text) == {"measure": "count"}


def test_extract_json_strips_think_blocks_case_insensitive_and_multiline():
    text = "<THINK>\nline one\nline two\n</THINK>\n{\"measure\": \"cost\"}"
    assert extract_json(text) == {"measure": "cost"}


def test_extract_json_strips_markdown_fences():
    text = "```json\n{\"measure\": \"count\"}\n```"
    assert extract_json(text) == {"measure": "count"}


def test_extract_json_strips_fences_without_json_hint():
    text = "```\n{\"measure\": \"count\"}\n```"
    assert extract_json(text) == {"measure": "count"}


def test_extract_json_falls_back_to_outermost_braces_on_malformed_json():
    text = "Sure, here is the answer: {\"measure\": \"count\"} -- hope that helps!"
    assert extract_json(text) == {"measure": "count"}


def test_extract_json_combines_think_stripping_fences_and_slicing():
    text = (
        "<think>reasoning...</think>\n"
        "```json\n"
        "Here you go: {\"measure\": \"cost\", \"confidence\": \"high\"}\n"
        "```"
    )
    assert extract_json(text) == {"measure": "cost", "confidence": "high"}


def test_extract_json_raises_on_truly_unparseable_text():
    with pytest.raises(Exception):
        extract_json("no json anywhere in this response")


def test_extract_json_rejects_non_object_root():
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")


def test_validate_against_schema_accepts_a_valid_response():
    response = {
        "measure": "count",
        "column_dimension_map": {"channel": "channel"},
        "per_subject": False,
        "confidence": "high",
    }
    assert validate_against_schema(response, CONSTRAINT_TYPE_SCHEMA) == []


def test_validate_against_schema_reports_missing_required_key():
    response = {"measure": "count", "per_subject": False, "confidence": "high"}
    errors = validate_against_schema(response, CONSTRAINT_TYPE_SCHEMA)
    assert errors  # non-empty -- missing column_dimension_map


def test_validate_against_schema_reports_bad_enum_value():
    response = {
        "measure": "not-a-real-measure",
        "column_dimension_map": {},
        "per_subject": False,
        "confidence": "high",
    }
    errors = validate_against_schema(response, CONSTRAINT_TYPE_SCHEMA)
    assert errors
