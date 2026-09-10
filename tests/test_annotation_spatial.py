from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bone_cancer_ai.annotations.labelme_bridge import (
    import_staged,
    stage_labelme,
    verify_equivalent,
)
from bone_cancer_ai.annotations.spatial import (
    convert_labelme,
    render_overlay,
    save_spatial,
)


def _image(path: Path, *, size: tuple[int, int] = (12, 10), value: int = 20) -> Path:
    Image.new("L", size, color=value).save(path)
    return path


def _payload(image_path: Path, shapes: list[dict]) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "imagePath": image_path.name,
        "imageWidth": width,
        "imageHeight": height,
        "shapes": shapes,
    }


def test_polygon_conversion_aligns_mask_with_source_coordinates(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    payload = _payload(
        image_path,
        [{"label": "lesion", "shape_type": "polygon", "points": [[2, 2], [8, 2], [2, 7]]}],
    )

    result = convert_labelme(payload, image_path, ["lesion"])

    assert result["width"] == 12
    assert result["height"] == 10
    assert result["labels"] == ["lesion"]
    assert result["boxes"] == []
    assert result["mask"].dtype == np.uint8
    assert result["mask"].shape == (10, 12)
    assert result["mask"][3, 3] == 1
    assert result["mask"][8, 8] == 0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda p: p["shapes"][0].update(label="unknown"), "label"),
        (lambda p: p.update(imageWidth=13), "dimensions"),
        (lambda p: p.update(imagePath="another.png"), "imagePath"),
        (lambda p: p["shapes"][0].update(shape_type="circle"), "shape_type"),
        (lambda p: p["shapes"][0].update(points=[[2, 2], [2, 2], [2, 2]]), "degenerate"),
        (lambda p: p["shapes"][0].update(points=[[-0.01, 2], [8, 2], [2, 7]]), "bounds"),
        (lambda p: p["shapes"][0].update(points=[[float("nan"), 2], [8, 2], [2, 7]]), "finite"),
    ],
)
def test_conversion_rejects_invalid_identity_or_geometry(
    tmp_path: Path, change, match: str
) -> None:
    image_path = _image(tmp_path / "scan.png")
    payload = _payload(
        image_path,
        [{"label": "lesion", "shape_type": "polygon", "points": [[2, 2], [8, 2], [2, 7]]}],
    )
    change(payload)

    with pytest.raises(ValueError, match=match):
        convert_labelme(payload, image_path, ["lesion"])


def test_boxes_only_returns_boxes_without_fabricated_mask(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    payload = _payload(
        image_path,
        [{"label": "lesion", "shape_type": "rectangle", "points": [[8, 7], [2, 1]]}],
    )

    result = convert_labelme(payload, image_path, ["lesion"])

    assert result["boxes"] == [[2.0, 1.0, 8.0, 7.0]]
    assert result["mask"] is None


def test_non_identity_exif_orientation_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("L", (12, 10), 20).save(image_path, exif=exif)
    payload = _payload(image_path, [])

    with pytest.raises(ValueError, match="EXIF orientation"):
        convert_labelme(payload, image_path, ["lesion"])


def test_save_and_overlay_write_derived_artifacts(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    result = convert_labelme(
        _payload(
            image_path,
            [
                {"label": "lesion", "shape_type": "rectangle", "points": [[1, 1], [5, 6]]},
                {"label": "lesion", "shape_type": "polygon", "points": [[2, 2], [8, 2], [2, 7]]},
            ],
        ),
        image_path,
        ["lesion"],
    )

    paths = save_spatial(result, tmp_path / "derived", "opaque-1")
    overlay = render_overlay(image_path, result, tmp_path / "derived" / "overlays" / "opaque-1.png")

    assert paths["mask_path"].is_file()
    assert paths["box_path"].is_file()
    assert paths["overlay_path"] is None
    assert overlay.is_file()
    with Image.open(paths["mask_path"]) as mask:
        assert set(np.unique(np.asarray(mask))) <= {0, 1}
    assert json.loads(paths["box_path"].read_text(encoding="utf-8"))["boxes"] == result["boxes"]


def test_boxes_only_save_has_no_mask_file(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    result = convert_labelme(
        _payload(
            image_path,
            [{"label": "lesion", "shape_type": "rectangle", "points": [[1, 1], [5, 6]]}],
        ),
        image_path,
        ["lesion"],
    )

    paths = save_spatial(result, tmp_path / "derived", "opaque-1")

    assert paths["mask_path"] is None
    assert not (tmp_path / "derived" / "masks" / "opaque-1.png").exists()


def test_equivalence_requires_exact_bytes_not_matching_pixels(tmp_path: Path) -> None:
    left = _image(tmp_path / "left.bmp", value=33)
    right = tmp_path / "right.bmp"
    right.write_bytes(left.read_bytes())
    same_pixels_different_encoding = _image(tmp_path / "same.png", value=33)

    assert verify_equivalent(left, right)
    assert not verify_equivalent(left, same_pixels_different_encoding)
    changed = bytearray(right.read_bytes())
    changed[-1] ^= 1
    right.write_bytes(changed)
    assert not verify_equivalent(left, right)


def test_stage_and_import_are_hash_bound_without_touching_source(tmp_path: Path) -> None:
    source = _image(tmp_path / "source.png")
    original_bytes = source.read_bytes()
    expected = hashlib.sha256(original_bytes).hexdigest()

    staged = stage_labelme(source, "opaque-1", tmp_path / "annotations", expected)

    assert staged["image_path"] != source
    assert staged["image_path"].parent == tmp_path / "annotations" / "labelme" / "opaque-1"
    assert staged["json_path"].is_file()
    assert staged["binding_path"].is_file()
    assert source.read_bytes() == original_bytes
    payload = json.loads(staged["json_path"].read_text(encoding="utf-8"))
    payload["shapes"] = [
        {"label": "lesion", "shape_type": "rectangle", "points": [[1, 1], [5, 6]]}
    ]
    staged["json_path"].write_text(json.dumps(payload), encoding="utf-8")

    imported = import_staged(staged["binding_path"], ["lesion"])

    assert imported["boxes"] == [[1.0, 1.0, 5.0, 6.0]]
    assert imported["source_image_path"] == source.resolve()
    assert imported["source_sha256"] == expected


def test_stage_rejects_wrong_hash_and_import_rejects_changed_source(tmp_path: Path) -> None:
    source = _image(tmp_path / "source.png")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="hash"):
        stage_labelme(source, "opaque-1", tmp_path / "wrong", "0" * 64)

    staged = stage_labelme(source, "opaque-1", tmp_path / "annotations", expected)
    _image(source, value=99)

    with pytest.raises(ValueError, match="source"):
        import_staged(staged["binding_path"], ["lesion"])


def test_overlapping_polygons_are_combined_as_a_union(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    first = {"label": "lesion", "shape_type": "polygon", "points": [[1, 1], [7, 1], [1, 7]]}
    second = {"label": "lesion", "shape_type": "polygon", "points": [[3, 1], [9, 1], [9, 7]]}

    combined = convert_labelme(_payload(image_path, [first, second]), image_path, ["lesion"])
    first_mask = convert_labelme(_payload(image_path, [first]), image_path, ["lesion"])["mask"]
    second_mask = convert_labelme(_payload(image_path, [second]), image_path, ["lesion"])["mask"]

    assert np.array_equal(combined["mask"], np.logical_or(first_mask, second_mask).astype(np.uint8))


def test_derived_outputs_reject_dataset_and_source_overwrite(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    result = convert_labelme(_payload(image_path, []), image_path, ["lesion"])

    with pytest.raises(ValueError, match="dataset"):
        save_spatial(result, tmp_path / "dataset" / "derived", "opaque-1")
    with pytest.raises(ValueError, match="source"):
        render_overlay(image_path, result, image_path)
    with pytest.raises(ValueError, match="dataset"):
        stage_labelme(
            image_path,
            "opaque-1",
            tmp_path / "dataset" / "annotations",
            hashlib.sha256(image_path.read_bytes()).hexdigest(),
        )


def test_boxes_only_rewrite_removes_stale_derived_mask(tmp_path: Path) -> None:
    image_path = _image(tmp_path / "scan.png")
    polygon_result = convert_labelme(
        _payload(
            image_path,
            [{"label": "lesion", "shape_type": "polygon", "points": [[2, 2], [8, 2], [2, 7]]}],
        ),
        image_path,
        ["lesion"],
    )
    root = tmp_path / "derived"
    assert save_spatial(polygon_result, root, "opaque-1")["mask_path"].is_file()
    boxes_result = convert_labelme(
        _payload(
            image_path,
            [{"label": "lesion", "shape_type": "rectangle", "points": [[1, 1], [5, 6]]}],
        ),
        image_path,
        ["lesion"],
    )

    save_spatial(boxes_result, root, "opaque-1")

    assert not (root / "masks" / "opaque-1.png").exists()
