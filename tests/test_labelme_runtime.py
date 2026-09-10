"""Smoke-test the pinned LabelMe Qt editor without touching desktop preferences."""

import hashlib
import json

from PIL import Image


def test_labelme_native_load_edit_save_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # LabelMe initializes ONNX Runtime before Qt to satisfy Windows DLL load order.
    from labelme._shape import Shape
    from labelme.app import MainWindow
    from PyQt5 import QtCore, QtWidgets

    original_settings = QtCore.QSettings
    monkeypatch.setattr(
        QtCore,
        "QSettings",
        lambda *args: original_settings(str(tmp_path / "qt_test.ini"), original_settings.IniFormat),
    )

    from bone_cancer_ai.annotations.labelme_bridge import import_staged, stage_labelme

    source = tmp_path / "synthetic.png"
    Image.new("RGB", (48, 40), "gray").save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    staged = stage_labelme(source, "sample_qt", tmp_path / "annotations", digest)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(
        config_overrides={
            "labels": ["lesion"],
            "validate_label": "exact",
            "keep_prev": False,
            "with_image_data": False,
        },
        file_or_dir=str(staged["image_path"]),
        output_dir=str(staged["json_path"].parent),
    )
    assert window._image.width() == 48
    shape = Shape(label="lesion", shape_type="polygon")
    shape.points = [QtCore.QPointF(2, 2), QtCore.QPointF(20, 2), QtCore.QPointF(2, 20)]
    shape.close()
    window._load_shapes([shape])
    assert window.save_labels(str(staged["json_path"]))
    window._load_file(str(staged["json_path"]))
    assert len(window._canvas_widgets.canvas.shapes) == 1
    assert import_staged(staged["binding_path"], ["lesion"])["mask"].sum() > 0
    assert json.loads(staged["json_path"].read_text())["imageData"] is None
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    window.deleteLater()
    app.processEvents()
