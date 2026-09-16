"""Strict validation for deterministic pre-augmentation image caches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from bone_cancer_ai.data.preprocessing import CACHE_FORMAT_VERSION, cache_preprocess_fingerprint

_REQUIRED_COLUMNS = {
    "sample_id",
    "assigned_split",
    "cache_path",
    "source_sha256",
    "cache_sha256",
    "preprocess_fingerprint",
    "cache_format_version",
    "image_size",
    "force_grayscale",
    "pad_value",
    "original_height",
    "original_width",
    "status",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def attach_verified_cache(
    frames: dict[str, pd.DataFrame],
    *,
    project_root: Path,
    cache_root: Path,
    index_path: Path,
    required_splits: tuple[str, ...],
    image_size: int,
    force_grayscale: bool,
    pad_value: int,
) -> None:
    """Attach verified cache paths to required split frames or fail closed."""
    root = Path(project_root).resolve()
    cache_directory = Path(cache_root).resolve()
    index_file = Path(index_path).resolve()
    if not cache_directory.is_relative_to(root) or not index_file.is_relative_to(
        cache_directory
    ):
        raise ValueError("cache paths must remain inside the project cache root")
    if not index_file.is_file():
        raise FileNotFoundError(f"Missing required cache index: {index_file}")
    index = pd.read_csv(index_file, keep_default_na=False)
    missing = _REQUIRED_COLUMNS - set(index)
    if missing:
        raise ValueError(f"Cache index missing columns: {sorted(missing)}")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("Cache index contains duplicate sample_id values")
    indexed = index.set_index(index["sample_id"].astype(str), drop=False)

    for split in required_splits:
        if split not in frames:
            continue
        frame = frames[split]
        if "sha256" not in frame:
            raise ValueError(f"Split {split} lacks audited source SHA-256 values")
        expected_ids = set(frame["sample_id"].astype(str))
        split_index = index.loc[index["assigned_split"].astype(str).eq(split)]
        actual_ids = set(split_index["sample_id"].astype(str))
        if actual_ids != expected_ids:
            missing_ids = len(expected_ids - actual_ids)
            extra_ids = len(actual_ids - expected_ids)
            raise ValueError(
                f"Cache index coverage mismatch for {split}: "
                f"missing={missing_ids}, extra={extra_ids}"
            )
        cache_paths: list[str] = []
        cache_hashes: list[str] = []
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            cached = indexed.loc[sample_id]
            if isinstance(cached, pd.DataFrame):
                raise ValueError(f"Cache index is ambiguous for sample {sample_id}")
            expected_path = (cache_directory / f"{sample_id}.png").resolve()
            actual_path = Path(str(cached["cache_path"])).resolve()
            if actual_path != expected_path or not actual_path.is_relative_to(cache_directory):
                raise ValueError(f"Cache path is unsafe for sample {sample_id}")
            if str(cached["status"]) != "cached" or not actual_path.is_file():
                raise ValueError(f"Cache entry is unavailable for sample {sample_id}")
            source_path = Path(str(row.absolute_path)).resolve()
            source_sha = str(row.sha256)
            if str(cached["source_sha256"]) != source_sha or _sha256(source_path) != source_sha:
                raise ValueError(f"Cache source checksum mismatch for sample {sample_id}")
            fingerprint = cache_preprocess_fingerprint(
                source_sha,
                image_size=image_size,
                force_grayscale=force_grayscale,
                pad_value=pad_value,
            )
            if str(cached["preprocess_fingerprint"]) != fingerprint:
                raise ValueError(f"Cache preprocessing is stale for sample {sample_id}")
            if str(cached["cache_format_version"]) != CACHE_FORMAT_VERSION:
                raise ValueError(f"Cache format is stale for sample {sample_id}")
            if (
                int(cached["image_size"]) != image_size
                or _as_bool(cached["force_grayscale"]) != force_grayscale
                or int(cached["pad_value"]) != pad_value
            ):
                raise ValueError(f"Cache configuration mismatch for sample {sample_id}")
            cache_sha = str(cached["cache_sha256"])
            if _sha256(actual_path) != cache_sha:
                raise ValueError(f"Cache checksum mismatch for sample {sample_id}")
            if int(cached["original_height"]) < 1 or int(cached["original_width"]) < 1:
                raise ValueError(f"Cache geometry is missing for sample {sample_id}")
            cache_paths.append(str(actual_path))
            cache_hashes.append(cache_sha)
        frame["cache_path"] = cache_paths
        frame["cache_sha256"] = cache_hashes


def resolve_cache_settings(
    config: dict[str, Any], *, project_root: Path
) -> tuple[Path, Path, tuple[str, ...]] | None:
    """Resolve the optional cache section without permitting paths outside the project."""
    raw = config.get("cache")
    if raw is None:
        return None
    if not isinstance(raw, dict) or type(raw.get("enabled")) is not bool:
        raise ValueError("cache must be a mapping with boolean enabled")
    if not raw["enabled"]:
        return None
    required = raw.get("required_splits")
    if (
        not isinstance(required, list)
        or not required
        or len(set(required)) != len(required)
        or any(value not in {"fit_train", "val"} for value in required)
    ):
        raise ValueError("cache.required_splits must be a unique subset of fit_train and val")
    root = Path(project_root).resolve()
    cache_root = _project_path(root, raw.get("root"), name="cache.root")
    index_path = _project_path(root, raw.get("index_path"), name="cache.index_path")
    if not index_path.is_relative_to(cache_root):
        raise ValueError("cache.index_path must be inside cache.root")
    return cache_root, index_path, tuple(str(value) for value in required)


def _project_path(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes the project root")
    return resolved


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError("force_grayscale must be boolean")
