from __future__ import annotations

from pathlib import Path

from bone_cancer_ai.evaluation.btxrd_review import review_btxrd_run


def test_btxrd_review_verifies_completed_registry_bound_run() -> None:
    root = Path(__file__).resolve().parents[1]
    review = review_btxrd_run(root)

    assert review["status"] == "pass"
    assert review["alias"] == "hosp_bonenet.btxrd.subtype.selected"
    assert review["checkpoint_sha256"]
    assert review["binary_test_samples"] == 527
    assert review["subtype_test_samples"] == 262
    assert review["subtype_classes"] == 9
    assert review["fresh_evaluation"] is True

