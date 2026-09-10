"""Exercise the end-to-end staged spatial review on synthetic images."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from bone_cancer_ai.annotations.labelme_bridge import stage_labelme


def test_import_keeps_spatial_revisions_and_rejects_other_identity(tmp_path):
    from bone_cancer_ai.annotations.storage import AnnotationStore
    from bone_cancer_ai.annotations.validation import validate_project_annotations
    from bone_cancer_ai.annotations.workflow import import_spatial_review, spatial_frame

    source = tmp_path / "source.png"
    Image.new("RGB", (12, 10), "gray").save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    row = {"sample_id": "sample_a", "image_id": source.name, "sha256": digest}
    ctx = SimpleNamespace(
        root=tmp_path / "annotations",
        config={"spatial_labels": ["lesion"]},
        image_path=lambda record: source,
        manifest=pd.DataFrame([row]),
        schema={},
        store=AnnotationStore(tmp_path / "reviews.sqlite3", {}, []),
    )
    ctx.config["concept_names"] = []
    staged = stage_labelme(source, "sample_a", ctx.root, digest)
    payload = json.loads(staged["json_path"].read_text())
    payload["shapes"] = [
        {"label": "lesion", "shape_type": "polygon", "points": [[1, 1], [7, 1], [1, 7]]}
    ]
    staged["json_path"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="reviewer"):
        import_spatial_review(ctx, row, "", "")
    first = import_spatial_review(ctx, row, "Synthetic reviewer", "fixture")
    payload["shapes"] = [{"label": "lesion", "shape_type": "rectangle", "points": [[1, 1], [5, 6]]}]
    staged["json_path"].write_text(json.dumps(payload))
    second = import_spatial_review(ctx, row, "Synthetic reviewer", "box only")
    assert first["revision_id"] != second["revision_id"]
    assert Path(first["mask_path"]).is_file()
    assert second["mask_path"] == ""
    assert spatial_frame(ctx.root).iloc[0].notes == "box only"
    assert validate_project_annotations(ctx)["spatial_revisions"] == 2
    overlay = Path(first["overlay_path"])
    original_overlay = overlay.read_bytes()
    overlay.write_bytes(b"wrong overlay")
    with pytest.raises((ValueError, OSError)):
        validate_project_annotations(ctx)
    overlay.write_bytes(original_overlay)
    box = Path(second["box_path"])
    original_box = box.read_bytes()
    box_payload = json.loads(original_box)
    box_payload["labels"] = ["unverified_class"]
    box.write_text(json.dumps(box_payload))
    with pytest.raises(ValueError):
        validate_project_annotations(ctx)
    box.write_bytes(original_box)
    Path(first["mask_path"]).write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError)):
        validate_project_annotations(ctx)
    binding = json.loads(staged["binding_path"].read_text())
    binding["source_sha256"] = "b" * 64
    staged["binding_path"].write_text(json.dumps(binding))
    with pytest.raises(ValueError):
        import_spatial_review(ctx, row, "Synthetic reviewer", "")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
