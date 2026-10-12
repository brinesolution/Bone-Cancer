"""Hash-bound validation and restoration for the original combined Main tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FIELDS = ("split", "row_index", "image_id", "binary_label", "source", "source_path", "sha256")
CLASS_NAMES = {"0": "normal", "1": "cancer"}
# Trust anchors for the public portable contract. Update only with a reviewed
# source-contract change; metadata alone cannot authenticate its own digests.
PINNED_MANIFEST_SHA256 = "06aecf842b3af537f483e871a4731fc7e0c477ad33ddde571a94e7b76847eea8"
PINNED_METADATA_SHA256 = "b0f479700ed02cf125c4ba085a294e356346a8bc8e5a2c026afa6d670ce651e6"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"Unsafe contract path: {value}")
    if ":" in value or "\\" in value:
        raise ValueError(f"Unsafe contract path: {value}")
    return Path(*pure.parts)


def _reject_reparse(path: Path, description: str) -> None:
    for candidate in (path, *path.parents):
        if not candidate.exists():
            continue
        info = candidate.lstat()
        attributes = getattr(info, "st_file_attributes", 0)
        if candidate.is_symlink() or attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise ValueError(f"{description} crosses a link or reparse point: {candidate}")


def _checked(root: Path, relative: str) -> Path:
    _reject_reparse(root, "Source/destination root")
    path = root / _safe_relative(relative)
    resolved_root = root.resolve()
    resolved = path.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Contract path escapes its source root: {relative}")
    current = root
    for part in _safe_relative(relative).parts:
        current = current / part
        _reject_reparse(current, "Source/destination path")
    return path


def _load_rows(manifest: Path, metadata_path: Path) -> tuple[list[dict[str, str]], dict]:
    metadata_bytes = metadata_path.read_bytes()
    manifest_bytes = manifest.read_bytes()
    if (
        manifest.name == "main_dataset_manifest.csv"
        and metadata_path.name == "main_dataset_manifest.json"
    ):
        if hashlib.sha256(metadata_bytes).hexdigest() != PINNED_METADATA_SHA256:
            raise ValueError("Pinned Main dataset metadata checksum mismatch")
        if hashlib.sha256(manifest_bytes).hexdigest() != PINNED_MANIFEST_SHA256:
            raise ValueError("Pinned Main dataset manifest checksum mismatch")
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    if metadata.get("schema") != 1 or metadata.get("class_names") != CLASS_NAMES:
        raise ValueError("Unsupported Main dataset manifest metadata or class mapping")
    raw = manifest_bytes
    if hashlib.sha256(raw).hexdigest() != metadata.get("manifest_sha256"):
        raise ValueError("Main dataset manifest checksum mismatch")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if not rows or tuple(rows[0]) != FIELDS:
        raise ValueError("Main dataset manifest columns are invalid")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        split = row["split"]
        name = row["image_id"]
        if (
            split not in SPLITS
            or Path(name).name != name
            or Path(name).suffix.lower() not in IMAGE_SUFFIXES
        ):
            raise ValueError(f"Invalid split or image name in manifest: {split}/{name}")
        if row["binary_label"] not in CLASS_NAMES:
            raise ValueError(f"Invalid class label for {split}/{name}")
        if row["source"] not in ("btxrd", "roboflow_candidate"):
            raise ValueError(f"Invalid source classification for {split}/{name}")
        source_path = _safe_relative(row["source_path"]).as_posix()
        if row["source"] == "btxrd" and not source_path.startswith("BTXRD/images/"):
            raise ValueError(f"BTXRD source path is invalid for {split}/{name}")
        if row["source"] == "roboflow_candidate" and not source_path.startswith("Roboflow/"):
            raise ValueError(f"Roboflow candidate source path is invalid for {split}/{name}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid image checksum for {split}/{name}")
        key = (split, name)
        if key in seen:
            raise ValueError(f"Duplicate manifest image: {split}/{name}")
        seen.add(key)
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        if [row["row_index"] for row in split_rows] != [
            str(index) for index in range(len(split_rows))
        ]:
            raise ValueError(f"Main label row order is invalid for split {split}")
    counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    if dict(counts) != metadata.get("split_counts") or dict(source_counts) != metadata.get(
        "source_counts"
    ):
        raise ValueError("Main dataset manifest counts disagree with metadata")
    return rows, metadata


def build_manifest(
    main_root: Path, btxrd_root: Path, manifest_path: Path, metadata_path: Path
) -> None:
    """Create a portable contract from a currently complete, observed Main tree."""
    btx_by_hash: dict[str, list[str]] = defaultdict(list)
    btx_by_name: dict[str, Path] = {}
    images_root = btxrd_root / "BTXRD" / "images"
    for image in sorted(images_root.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            btx_by_hash[_hash(image)].append(image.name)
            btx_by_name[image.name] = image
    rows: list[dict[str, str]] = []
    split_csv_hashes: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    source_counts: Counter[str] = Counter()
    for split in SPLITS:
        csv_path = main_root / split / f"_{split}_labels.csv"
        csv_bytes = csv_path.read_bytes()
        split_csv_hashes[f"{split}/{csv_path.name}"] = hashlib.sha256(csv_bytes).hexdigest()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_id", "binary_label"]:
                raise ValueError(f"Unexpected label CSV header: {csv_path}")
            source_rows = list(reader)
        split_counts[split] = len(source_rows)
        for index, label_row in enumerate(source_rows):
            name, label = label_row["image_id"], label_row["binary_label"]
            image = _checked(main_root, f"{split}/{name}")
            if not image.is_file() or label not in CLASS_NAMES:
                raise ValueError(f"Missing image or unsupported label: {split}/{name}")
            digest = _hash(image)
            candidates = btx_by_hash.get(digest, [])
            if candidates:
                source = "btxrd"
                source_name = name if name in candidates else candidates[0]
                source_path = f"BTXRD/images/{source_name}"
            else:
                source = "roboflow_candidate"
                # Roboflow's export split differs from this project's original
                # combined split; restoration matches by name and hash only.
                source_path = f"Roboflow/{name}"
            source_counts[source] += 1
            rows.append(
                {
                    "split": split,
                    "row_index": str(index),
                    "image_id": name,
                    "binary_label": label,
                    "source": source,
                    "source_path": source_path,
                    "sha256": digest,
                }
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema": 1,
        "roboflow_project": "https://universe.roboflow.com/normal-bones/bone-cancer-detection-xa7ru/dataset/1",
        "roboflow_export_verification": (
            "Candidate provenance only; current local bytes are not independently "
            "verified against an official export."
        ),
        "class_names": CLASS_NAMES,
        "split_counts": split_counts,
        "source_counts": dict(source_counts),
        "split_csv_sha256": split_csv_hashes,
        "manifest_sha256": _hash(manifest_path),
        "csv_encoding": "UTF-8",
        "csv_line_ending": "LF",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_file(row: dict[str, str], roots: Iterable[Path]) -> Path:
    attempted = []
    for root in roots:
        if row["source"] == "roboflow_candidate":
            filename = _safe_relative(row["source_path"]).name
            for candidate in root.rglob(filename):
                relative = candidate.relative_to(root).as_posix()
                checked = _checked(root, relative)
                attempted.append(checked)
                if checked.is_file() and _hash(checked) == row["sha256"]:
                    return checked
            continue
        path = _checked(root, row["source_path"])
        attempted.append(path)
        if path.is_file() and _hash(path) == row["sha256"]:
            return path
    raise ValueError(
        f"Missing or wrong-source image hash for {row['source_path']}; checked {attempted}"
    )


def _atomic_copy(source: Path, destination: Path, expected: str) -> None:
    _reject_reparse(destination.parent, "Destination")
    if destination.exists():
        _reject_reparse(destination, "Existing destination")
        if _hash(destination) == expected:
            return
        raise ValueError(f"Different existing file was preserved: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError(f"Destination crosses a link: {destination.parent}")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".restore-", delete=False
    ) as temporary:
        temp_path = Path(temporary.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, temporary, length=1024 * 1024)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        if _hash(temp_path) != expected:
            raise ValueError(f"Source changed during restore: {source}")
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            if _hash(destination) != expected:
                raise ValueError(f"Different existing file was preserved: {destination}") from None
    finally:
        temp_path.unlink(missing_ok=True)


def validate_main(main_root: Path, manifest_path: Path, metadata_path: Path) -> dict:
    rows, metadata = _load_rows(manifest_path, metadata_path)
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for row in rows:
        path = _checked(main_root, f"{row['split']}/{row['image_id']}")
        if not path.is_file() or _hash(path) != row["sha256"]:
            raise ValueError(f"Main image missing or changed: {row['split']}/{row['image_id']}")
        counts[row["split"]] += 1
        sources[row["source"]] += 1
    for split in SPLITS:
        expected = [row for row in rows if row["split"] == split]
        csv_path = _checked(main_root, f"{split}/_{split}_labels.csv")
        if not csv_path.is_file() or _hash(csv_path) != metadata["split_csv_sha256"].get(
            f"{split}/{csv_path.name}"
        ):
            raise ValueError(f"Main label CSV missing or changed: {split}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            actual = list(csv.DictReader(stream))
        projected = [
            {"image_id": row["image_id"], "binary_label": row["binary_label"]} for row in expected
        ]
        if actual != projected:
            raise ValueError(f"Main labels or split assignment changed: {split}")
        expected_names = {row["image_id"] for row in expected}
        actual_names = {
            path.name
            for path in (main_root / split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if actual_names != expected_names:
            raise ValueError(f"Main split is incomplete or has extra images: {split}")
    return {
        "valid": True,
        "counts": dict(counts),
        "source_counts": dict(sources),
        "total_images": sum(counts.values()),
    }


def restore_main(
    source_roots: Path | Iterable[Path], destination: Path, manifest_path: Path, metadata_path: Path
) -> dict:
    roots = [source_roots] if isinstance(source_roots, Path) else list(source_roots)
    rows, metadata = _load_rows(manifest_path, metadata_path)
    targets: dict[tuple[str, str], Path] = {}
    missing_sources: dict[tuple[str, str], Path] = {}
    rows_by_key = {(row["split"], row["image_id"]): row for row in rows}
    for row in rows:
        target = _checked(destination, f"{row['split']}/{row['image_id']}")
        key = (row["split"], row["image_id"])
        targets[key] = target
        if target.exists():
            _reject_reparse(target, "Existing destination")
            if not target.is_file() or _hash(target) != row["sha256"]:
                raise ValueError(f"Different existing file was preserved: {target}")
        else:
            missing_sources[key] = _source_file(row, roots)
    label_outputs: dict[str, tuple[Path, bytes]] = {}
    for split in SPLITS:
        lines = ["image_id,binary_label"]
        lines.extend(
            f"{row['image_id']},{row['binary_label']}" for row in rows if row["split"] == split
        )
        label_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        expected = metadata["split_csv_sha256"][f"{split}/_{split}_labels.csv"]
        if hashlib.sha256(label_bytes).hexdigest() != expected:
            raise ValueError(f"Manifest cannot reproduce original {split} label CSV bytes")
        path = _checked(destination, f"{split}/_{split}_labels.csv")
        if path.exists():
            _reject_reparse(path, "Existing destination")
            if not path.is_file() or _hash(path) != expected:
                raise ValueError(f"Different existing file was preserved: {path}")
        label_outputs[split] = (path, label_bytes)
    # Resolve and validate every missing source and existing output before the
    # first new file is installed, so a partial valid Main tree needs no access
    # to source bytes it already preserves.
    for key, source in missing_sources.items():
        row = rows_by_key[key]
        _atomic_copy(source, targets[key], row["sha256"])
    for split, (path, label_bytes) in label_outputs.items():
        expected = metadata["split_csv_sha256"][f"{split}/_{split}_labels.csv"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            _reject_reparse(path, "Existing destination")
            if not path.is_file() or _hash(path) != expected:
                raise ValueError(f"Different existing file was preserved: {path}")
        else:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".restore-", delete=False
            ) as temporary:
                temporary.write(label_bytes)
                temporary.flush()
                os.fsync(temporary.fileno())
                staged = Path(temporary.name)
            try:
                try:
                    os.link(staged, path)
                except FileExistsError:
                    if _hash(path) != expected:
                        raise ValueError(f"Different existing file was preserved: {path}") from None
            finally:
                staged.unlink(missing_ok=True)
    return validate_main(destination, manifest_path, metadata_path)


def _main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifest")
    build.add_argument("--main", type=Path, required=True)
    build.add_argument("--btxrd", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--metadata", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--main", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--metadata", type=Path, required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--source-root", type=Path, action="append", required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build-manifest":
            build_manifest(args.main, args.btxrd, args.manifest, args.metadata)
            result = {"manifest": str(args.manifest), "metadata": str(args.metadata)}
        elif args.command == "validate":
            result = validate_main(args.main, args.manifest, args.metadata)
        else:
            result = restore_main(args.source_root, args.destination, args.manifest, args.metadata)
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"valid": False, "error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
