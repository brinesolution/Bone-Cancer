"""Annotation metadata validation and transactional revision behavior."""

import pytest


def schema():
    return {
        "binary": {"classes": {0: "normal", 1: "bone_tumour"}},
        "subtype": {"classes": ["osteosarcoma"]},
        "risk": {"classes": ["benign", "malignant"]},
    }


def draft():
    return {
        "binary_review": None,
        "subtype_label": None,
        "risk_label": None,
        "concepts": {"lytic": None},
        "reviewer": "test reviewer",
        "review_status": "draft",
        "notes": "synthetic fixture",
    }


def test_unknown_is_not_negative_and_invalid_classes_rejected():
    from bone_cancer_ai.annotations.schema import validate_metadata

    assert validate_metadata(draft(), schema(), ["lytic"])["concepts"] == {"lytic": None}
    for field, value in [
        ("subtype_label", "filename_diagnosis"),
        ("binary_review", 2),
        ("reviewer", ""),
        ("review_status", "radiologist_verified"),
    ]:
        with pytest.raises(ValueError):
            validate_metadata({**draft(), field: value}, schema(), ["lytic"])


def test_create_edit_reload_revision_and_conflict(tmp_path):
    from bone_cancer_ai.annotations.storage import AnnotationStore

    db = tmp_path / "reviews.sqlite3"
    store = AnnotationStore(db, schema(), ["lytic"])
    first = store.save("sample_a", "image.png", "a" * 64, draft(), expected_revision=0)
    assert first["revision"] == 1
    second = store.save(
        "sample_a",
        "image.png",
        "a" * 64,
        {**draft(), "subtype_label": "osteosarcoma"},
        expected_revision=1,
    )
    assert second["revision"] == 2
    reloaded = AnnotationStore(db, schema(), ["lytic"])
    assert reloaded.latest("sample_a")["subtype_label"] == "osteosarcoma"
    assert len(reloaded.history("sample_a")) == 2
    assert reloaded.history("sample_a")[0]["subtype_label"] is None
    with pytest.raises(ValueError, match="revision"):
        store.save("sample_a", "image.png", "a" * 64, draft(), expected_revision=1)
    with pytest.raises(ValueError, match="identity"):
        store.save("sample_a", "image.png", "b" * 64, draft(), expected_revision=2)
    assert reloaded.progress(3)["reviewed"] == 0
    assert "test reviewer" in reloaded.export_frame().to_csv(index=False)


def test_store_rejects_dataset_destination(tmp_path):
    from bone_cancer_ai.annotations.storage import AnnotationStore

    with pytest.raises(ValueError, match="source"):
        AnnotationStore(tmp_path / "dataset/Main/reviews.sqlite3", schema(), [])


@pytest.mark.parametrize(
    "field,value",
    [
        ("provenance", "clinical_ground_truth"),
        ("clinical_verification", "confirmed"),
        ("timestamp", "yesterday"),
    ],
)
def test_validation_rejects_tampered_audit_fields(tmp_path, field, value):
    import json
    from types import SimpleNamespace

    import pandas as pd

    from bone_cancer_ai.annotations.storage import AnnotationStore
    from bone_cancer_ai.annotations.validation import validate_project_annotations

    store = AnnotationStore(tmp_path / "reviews.sqlite3", schema(), ["lytic"])
    record = store.save("sample_a", "image.png", "a" * 64, draft(), expected_revision=0)
    record[field] = value
    with store._connect() as connection:
        connection.execute("UPDATE revisions SET payload=?", (json.dumps(record),))
    ctx = SimpleNamespace(
        root=tmp_path,
        store=store,
        schema=schema(),
        config={"concept_names": ["lytic"]},
        image_path=lambda row: None,
        manifest=pd.DataFrame(
            [{"sample_id": "sample_a", "image_id": "image.png", "sha256": "a" * 64}]
        ),
    )
    with pytest.raises(ValueError):
        validate_project_annotations(ctx)
