"""Provider output normalization and label mappings."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_decision(value: Any, mapping: Mapping[str, int] | None = None) -> tuple[str | None, str]:
    if isinstance(value, str):
        key = value.strip().lower()
        if mapping and key in mapping:
            return ("applied" if int(mapping[key]) == 1 else "not_applied"), "valid"
        if key in {"applied", "not_applied"}:
            return key, "valid"
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return ("applied" if value else "not_applied"), "valid"
    return None, "invalid"


def normalize_prediction(row: Mapping[str, Any], mapping: Mapping[str, int] | None = None) -> dict[str, Any]:
    query_id = row.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("prediction query_id must be a non-empty string")
    decision, status = normalize_decision(row.get("decision", row.get("prediction_label")), mapping)
    result = dict(row)
    result["query_id"] = query_id
    result["decision"] = decision
    result["parse_status"] = status
    return result
