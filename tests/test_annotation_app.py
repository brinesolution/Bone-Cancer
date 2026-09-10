"""Exercise real Streamlit review widgets against synthetic local records."""

import hashlib
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from streamlit.testing.v1 import AppTest


def test_autosave_edit_reload_navigation_and_unknown(tmp_path, monkeypatch):
    source = tmp_path / "dataset/Main"
    source.mkdir(parents=True)
    annotations = tmp_path / "workspace/annotations"
    annotations.mkdir(parents=True)
    rows = []
    for index in range(2):
        image = source / f"fixture{index}.png"
        Image.new("RGB", (24, 32), (30 + index, 40, 50)).save(image)
        rows.append(
            {
                "sample_id": f"sample_{index}",
                "image_id": image.name,
                "absolute_path": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "binary_label": 1,
                "is_eligible": True,
                "official_split": "train",
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    schema = {
        "subtype": {"classes": ["osteosarcoma"]},
        "risk": {"classes": ["benign", "malignant"]},
    }
    (annotations / "label_schema.yaml").write_text(yaml.safe_dump(schema))
    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "dataset_root": str(source),
                "schema_path": str(annotations / "label_schema.yaml"),
            }
        )
    )
    config = tmp_path / "annotation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data_config": str(data),
                "manifest": str(manifest),
                "annotation_root": str(annotations),
                "database": str(annotations / "reviews.sqlite3"),
                "concept_names": ["lytic"],
                "spatial_labels": ["lesion"],
            }
        )
    )
    monkeypatch.setenv("BONE_ANNOTATION_CONFIG", str(config))
    app = Path(__file__).resolve().parents[1] / "tools/annotation_app/app.py"
    at = AppTest.from_file(str(app)).run(timeout=30)
    assert not at.exception
    at.text_input(key="reviewer").set_value("Synthetic UI test").run()
    at.selectbox(key="subtype_label").select("osteosarcoma").run()
    at.text_area(key="notes").set_value("fixture only").run()
    assert not at.exception
    at.button(key="next").click().run()
    assert at.selectbox(key="subtype_label").value == "unknown"
    at.button(key="previous").click().run()
    assert at.selectbox(key="subtype_label").value == "osteosarcoma"
    assert at.text_area(key="notes").value == "fixture only"
    at.selectbox(key="subtype_label").select("unknown").run()
    reloaded = AppTest.from_file(str(app)).run(timeout=30)
    assert not reloaded.exception
    assert reloaded.selectbox(key="subtype_label").value == "unknown"
    from bone_cancer_ai.annotations.storage import AnnotationStore

    store = AnnotationStore(annotations / "reviews.sqlite3", schema, ["lytic"])
    assert store.latest("sample_0")["concepts"]["lytic"] is None
    assert store.latest("sample_0")["subtype_label"] is None
    assert len(store.history("sample_0")) >= 3
