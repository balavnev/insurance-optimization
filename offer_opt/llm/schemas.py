"""JSON Schemas for LLM step outputs. Populated in Phase 4 with the
constraint-classification step's schema, Phase 6 adds the verifier-codegen
step's; Phase 7 adds the remaining steps' (schema resolution, hierarchy
edges) here too.
"""

from __future__ import annotations

CONSTRAINT_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "measure": {"type": "string", "enum": ["count", "cost"]},
        "column_dimension_map": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "per_subject": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "low"]},
    },
    "required": ["measure", "column_dimension_map", "per_subject", "confidence"],
}

VERIFIER_CODEGEN_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
    },
    "required": ["source"],
}
