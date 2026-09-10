"""Strict unknown-aware project annotation contracts."""

from typing import Any

STATUSES = ("draft", "reviewed", "needs_review")


def validate_metadata(payload: dict, schema: dict, concept_names: list[str]) -> dict[str, Any]:
    """Reject unknown classes and preserve unknown separately from a negative label."""
    allowed = {
        "binary_review",
        "subtype_label",
        "risk_label",
        "concepts",
        "reviewer",
        "review_status",
        "notes",
    }
    if set(payload) - allowed:
        raise ValueError("Unexpected annotation fields")
    result = {key: payload.get(key) for key in allowed}
    for key in ("binary_review", "subtype_label", "risk_label"):
        if result[key] in ("unknown", "", "not sure"):
            result[key] = None
    binary = result["binary_review"]
    if binary is not None and (type(binary) is not int or binary not in (0, 1)):
        raise ValueError("binary_review must be 0, 1 or unknown")
    for field, section in (("subtype_label", "subtype"), ("risk_label", "risk")):
        if result[field] is not None and result[field] not in schema.get(section, {}).get(
            "classes", []
        ):
            raise ValueError(f"Invalid {field}: {result[field]}")
    concepts = result["concepts"] or {}
    if not isinstance(concepts, dict) or set(concepts) - set(concept_names):
        raise ValueError("Invalid concept names")
    normalized = {}
    for name in concept_names:
        value = concepts.get(name)
        if value is not None and (type(value) not in (int, bool) or value not in (0, 1)):
            raise ValueError(f"Concept {name} must be present, absent or unknown")
        normalized[name] = None if value is None else int(value)
    result["concepts"] = normalized
    reviewer = result["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("A reviewer name is required to save annotations")
    result["reviewer"] = reviewer.strip()
    if result["review_status"] not in STATUSES:
        raise ValueError("Invalid review_status")
    result["notes"] = result["notes"] or ""
    if not isinstance(result["notes"], str):
        raise ValueError("notes must be text")
    return result
