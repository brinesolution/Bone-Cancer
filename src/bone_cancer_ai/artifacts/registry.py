"""Promote and resolve SHA-256-verified checkpoints without mutable aliases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_ALIAS_PATTERN = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_REGISTRY_RELATIVE_PATH = Path("artifacts/checkpoints/registry.json")
_CHECKPOINT_DIRECTORY = _REGISTRY_RELATIVE_PATH.parent
_REGISTRY_LOCK_RELATIVE_PATH = _CHECKPOINT_DIRECTORY / ".registry.lock"
_REGISTRY_LOCK_TIMEOUT_SECONDS = 5.0
_REGISTRY_LOCK_RETRY_SECONDS = 0.01
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_RECORD_FIELDS = {
    "alias",
    "model_family",
    "phase",
    "source_run",
    "selected_metric",
    "checkpoint_path",
    "checkpoint_sha256",
    "configuration_fingerprint",
    "test_status",
}


@dataclass(frozen=True)
class CheckpointRecord:
    """Immutable registry metadata for one content-addressed checkpoint."""

    alias: str
    model_family: str
    phase: str
    source_run: str
    selected_metric: dict[str, Any]
    checkpoint_path: str
    checkpoint_sha256: str
    configuration_fingerprint: str
    test_status: dict[str, Any]

    def absolute_checkpoint_path(self, project_root: Path) -> Path:
        """Resolve the stored project-relative checkpoint path safely."""
        root = Path(project_root).resolve()
        return _project_relative_path(root, self.checkpoint_path, name="checkpoint_path")

    def as_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe representation for registry persistence."""
        return {
            "alias": self.alias,
            "model_family": self.model_family,
            "phase": self.phase,
            "source_run": self.source_run,
            "selected_metric": self.selected_metric,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "configuration_fingerprint": self.configuration_fingerprint,
            "test_status": self.test_status,
        }


def promote_checkpoint(
    project_root: Path,
    *,
    alias: str,
    checkpoint_path: Path,
    replace_alias: bool = False,
    model_family: str,
    phase: str,
    source_run: str | Path,
    selected_metric: Mapping[str, Any],
    configuration_fingerprint: str,
    test_status: Mapping[str, Any],
) -> CheckpointRecord:
    """Promote a checkpoint; replace a matching selected alias only when explicitly requested."""
    root = Path(project_root).resolve()
    if type(replace_alias) is not bool:
        raise ValueError("replace_alias must be a boolean")
    source = _project_relative_path(root, checkpoint_path, name="checkpoint_path")
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {source}")
    source_run_path = _project_relative_path(root, source_run, name="source_run")
    if source_run_path == root:
        raise ValueError("source_run must identify a directory inside the project root")
    dataset_root = root / "dataset" / "Main"
    if source.is_relative_to(dataset_root):
        raise ValueError("checkpoint source must not come from immutable dataset/Main")
    if not source.is_relative_to(source_run_path):
        raise ValueError("checkpoint source must be inside source_run")

    digest = _sha256(source)
    record = _make_record(
        root=root,
        alias=alias,
        model_family=model_family,
        phase=phase,
        source_run=source_run_path,
        selected_metric=selected_metric,
        checkpoint_path=_CHECKPOINT_DIRECTORY / f"{digest}{source.suffix}",
        checkpoint_sha256=digest,
        configuration_fingerprint=configuration_fingerprint,
        test_status=test_status,
    )
    registry_path = root / _REGISTRY_RELATIVE_PATH
    with _registry_lock(root):
        registry = _load_registry(root, registry_path)
        existing = registry.get(alias)
        if existing is not None:
            existing_record = _record_from_mapping(root, existing)
            _verify_checkpoint(root, existing_record)
            if existing_record == record:
                return existing_record
            if not replace_alias:
                raise ValueError(
                    f"checkpoint alias {alias!r} already identifies a different artifact"
                )
            if not alias.endswith(".selected"):
                raise ValueError(
                    "alias replacement is only permitted for aliases ending in .selected"
                )
            if (
                existing_record.model_family != record.model_family
                or existing_record.phase != record.phase
            ):
                raise ValueError(
                    "alias replacement requires the existing model_family and phase to match"
                )

        destination = record.absolute_checkpoint_path(root)
        _promote_file(source, destination, digest)
        registry[alias] = record.as_dict()
        _write_registry(registry_path, registry)
        return record


def resolve_checkpoint(project_root: Path, alias: str) -> CheckpointRecord:
    """Return a registered checkpoint only after its path and digest verify."""
    root = Path(project_root).resolve()
    _validate_alias(alias)
    registry = _load_registry(root, root / _REGISTRY_RELATIVE_PATH)
    try:
        record_raw = registry[alias]
    except KeyError as error:
        raise ValueError(f"checkpoint alias {alias!r} is not registered") from error
    record = _record_from_mapping(root, record_raw)
    if record.alias != alias:
        raise ValueError("registry alias does not match its record")
    _verify_checkpoint(root, record)
    return record


def restore_checkpoint_alias(
    project_root: Path,
    *,
    alias: str,
    expected_current: CheckpointRecord,
    previous: CheckpointRecord | None,
) -> None:
    """Rollback one alias only if it still identifies the expected promotion."""

    root = Path(project_root).resolve()
    _validate_alias(alias)
    if expected_current.alias != alias or (previous is not None and previous.alias != alias):
        raise ValueError("rollback records must match the requested alias")
    registry_path = root / _REGISTRY_RELATIVE_PATH
    with _registry_lock(root):
        registry = _load_registry(root, registry_path)
        current_raw = registry.get(alias)
        if current_raw is None:
            raise ValueError("checkpoint alias changed before rollback")
        current = _record_from_mapping(root, current_raw)
        _verify_checkpoint(root, current)
        if current != expected_current:
            raise ValueError("checkpoint alias changed before rollback")
        if previous is None:
            del registry[alias]
        else:
            _verify_checkpoint(root, previous)
            registry[alias] = previous.as_dict()
        if registry:
            _write_registry(registry_path, registry)
        else:
            registry_path.unlink(missing_ok=True)


def _make_record(
    *,
    root: Path,
    alias: str,
    model_family: str,
    phase: str,
    source_run: Path,
    selected_metric: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    configuration_fingerprint: str,
    test_status: Mapping[str, Any],
) -> CheckpointRecord:
    _validate_alias(alias)
    for name, value in (("model_family", model_family), ("phase", phase)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    for name, value in (("selected_metric", selected_metric), ("test_status", test_status)):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
    checkpoint_relative = _relative_to_root(root, root / checkpoint_path, name="checkpoint_path")
    if not Path(checkpoint_relative).is_relative_to(_CHECKPOINT_DIRECTORY):
        raise ValueError("checkpoint path must be inside the checkpoint store")
    return CheckpointRecord(
        alias=alias,
        model_family=model_family,
        phase=phase,
        source_run=_relative_to_root(root, source_run, name="source_run"),
        selected_metric=_json_copy(dict(selected_metric), name="selected_metric"),
        checkpoint_path=checkpoint_relative,
        checkpoint_sha256=_validate_sha256(checkpoint_sha256, name="checkpoint_sha256"),
        configuration_fingerprint=_validate_sha256(
            configuration_fingerprint, name="configuration_fingerprint"
        ),
        test_status=_json_copy(dict(test_status), name="test_status"),
    )


def _record_from_mapping(root: Path, raw: object) -> CheckpointRecord:
    if not isinstance(raw, Mapping) or set(raw) != _RECORD_FIELDS:
        raise ValueError("registry entry is incomplete or ambiguous")
    source_run = _project_relative_path(root, raw["source_run"], name="source_run")
    return _make_record(
        root=root,
        alias=raw["alias"],
        model_family=raw["model_family"],
        phase=raw["phase"],
        source_run=source_run,
        selected_metric=raw["selected_metric"],
        checkpoint_path=Path(raw["checkpoint_path"]),
        checkpoint_sha256=raw["checkpoint_sha256"],
        configuration_fingerprint=raw["configuration_fingerprint"],
        test_status=raw["test_status"],
    )


def _load_registry(root: Path, registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {}
    if not registry_path.is_file():
        raise ValueError("checkpoint registry is not a file")
    try:
        raw = json.loads(
            registry_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("checkpoint registry is malformed") from error
    if not isinstance(raw, dict):
        raise ValueError("checkpoint registry must map aliases to records")
    for alias, record in raw.items():
        _validate_alias(alias)
        parsed = _record_from_mapping(root, record)
        if parsed.alias != alias:
            raise ValueError("registry alias does not match its record")
    return raw


def _write_registry(registry_path: Path, registry: Mapping[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_name(f".{registry_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(registry, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, registry_path)
    finally:
        temporary.unlink(missing_ok=True)


def _promote_file(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha256:
            raise ValueError("existing promoted checkpoint does not match its checksum")
        return
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        if _sha256(temporary) != expected_sha256:
            raise ValueError("promoted checkpoint checksum mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_checkpoint(root: Path, record: CheckpointRecord) -> None:
    checkpoint = record.absolute_checkpoint_path(root)
    if not checkpoint.is_file():
        raise ValueError(f"registered checkpoint is missing: {record.checkpoint_path}")
    if _sha256(checkpoint) != record.checkpoint_sha256:
        raise ValueError("registered checkpoint checksum mismatch")


def _project_relative_path(root: Path, value: str | Path, *, name: str) -> Path:
    if not isinstance(value, str | Path):
        raise ValueError(f"{name} must be a path")
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes the project root")
    return resolved


def _relative_to_root(root: Path, path: Path, *, name: str) -> str:
    resolved = _project_relative_path(root, path, name=name)
    return resolved.relative_to(root).as_posix()


def _validate_alias(alias: object) -> str:
    if not isinstance(alias, str) or not _ALIAS_PATTERN.fullmatch(alias):
        raise ValueError(
            "checkpoint alias must use dot-separated lowercase letters, numbers, "
            "underscores, or hyphens"
        )
    return alias


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _json_copy(value: object, *, name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict JSON-safe data") from error
    if not isinstance(copied, dict):
        raise ValueError(f"{name} must be a mapping")
    return copied


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


@contextmanager
def _registry_lock(project_root: Path) -> Iterator[None]:
    """Serialize one project registry transaction across threads and processes."""
    lock_path = project_root / _REGISTRY_LOCK_RELATIVE_PATH
    deadline = time.monotonic() + _REGISTRY_LOCK_TIMEOUT_SECONDS
    thread_lock = _thread_lock(lock_path)
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not thread_lock.acquire(timeout=remaining):
        raise TimeoutError("timed out waiting for checkpoint registry lock")
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as stream:
            try:
                stream.seek(0, os.SEEK_END)
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                while not acquired:
                    try:
                        _try_file_lock(stream)
                        acquired = True
                    except OSError as error:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "timed out waiting for checkpoint registry lock"
                            ) from error
                        time.sleep(_REGISTRY_LOCK_RETRY_SECONDS)
                yield
            finally:
                if acquired:
                    _unlock_file(stream)
    finally:
        thread_lock.release()


def _thread_lock(lock_path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(lock_path, threading.Lock())


def _try_file_lock(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(stream: Any) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
