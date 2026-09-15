"""Behavioral contracts for verified checkpoint promotion and reuse."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from bone_cancer_ai.artifacts import registry as registry_module
from bone_cancer_ai.artifacts.registry import promote_checkpoint, resolve_checkpoint


def _promotion_details(
    *, source_run: str = "experiments/hosp_bonenet/run-001"
) -> dict[str, object]:
    return {
        "model_family": "hosp_bonenet",
        "phase": "phase_7",
        "source_run": source_run,
        "selected_metric": {"split": "val", "name": "AUROC", "value": 0.91},
        "configuration_fingerprint": "a" * 64,
        "test_status": {"status": "not_run", "planned_phase": 9},
    }


def _checkpoint(project_root: Path, payload: bytes = b"verified checkpoint") -> Path:
    checkpoint = project_root / "experiments" / "hosp_bonenet" / "run-001" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    return checkpoint


def _second_checkpoint(project_root: Path, payload: bytes) -> Path:
    checkpoint = project_root / "experiments" / "hosp_bonenet" / "run-002" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    return checkpoint


def _synchronize_initial_registry_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = registry_module._load_registry
    barrier = threading.Barrier(2)

    def synchronized_load(root: Path, registry_path: Path) -> dict[str, object]:
        if not registry_path.exists():
            try:
                barrier.wait(timeout=0.25)
            except threading.BrokenBarrierError:
                pass
        return original_load(root, registry_path)

    monkeypatch.setattr(registry_module, "_load_registry", synchronized_load)


def _start_parallel_promotions(
    promotions: list[tuple[str, Path, dict[str, object]]], project_root: Path
) -> tuple[list[object], list[Exception]]:
    start = threading.Event()
    results: list[object] = []
    errors: list[Exception] = []

    def run(alias: str, checkpoint: Path, details: dict[str, object]) -> None:
        start.wait()
        try:
            results.append(
                promote_checkpoint(
                    project_root, alias=alias, checkpoint_path=checkpoint, **details
                )
            )
        except Exception as error:
            errors.append(error)

    workers = [
        threading.Thread(target=run, args=(alias, checkpoint, details))
        for alias, checkpoint, details in promotions
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    return results, errors


def test_promotion_stores_a_project_relative_sha256_verified_checkpoint(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)

    record = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    promoted = project_root / record.checkpoint_path
    assert record.alias == "hosp-selected"
    assert record.source_run == "experiments/hosp_bonenet/run-001"
    assert record.checkpoint_sha256 == hashlib.sha256(b"verified checkpoint").hexdigest()
    assert promoted.read_bytes() == b"verified checkpoint"
    assert promoted.is_relative_to(project_root)
    registry = json.loads(
        (project_root / "artifacts" / "checkpoints" / "registry.json").read_text(encoding="utf-8")
    )
    assert registry["hosp-selected"]["checkpoint_path"] == record.checkpoint_path


def test_alias_resolution_returns_the_verified_promoted_checkpoint(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    promoted = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    resolved = resolve_checkpoint(project_root, "hosp-selected")

    assert resolved == promoted
    assert resolved.absolute_checkpoint_path(project_root) == (
        project_root / promoted.checkpoint_path
    )


def test_repeating_the_same_promotion_is_idempotent(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    first = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    registry_path = project_root / "artifacts" / "checkpoints" / "registry.json"
    registry_before = registry_path.read_bytes()

    second = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    assert second == first
    assert registry_path.read_bytes() == registry_before


def test_alias_resolution_rejects_a_tampered_promoted_checkpoint(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    record = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    (project_root / record.checkpoint_path).write_bytes(b"tampered checkpoint")

    with pytest.raises(ValueError, match="checksum"):
        resolve_checkpoint(project_root, "hosp-selected")


def test_alias_resolution_rejects_a_registry_path_outside_the_checkpoint_store(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)
    promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    registry_path = project_root / "artifacts" / "checkpoints" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["hosp-selected"]["checkpoint_path"] = "experiments/hosp_bonenet/run-001/best.pt"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint path"):
        resolve_checkpoint(project_root, "hosp-selected")


def test_promotion_rejects_a_checkpoint_from_the_immutable_dataset(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "dataset" / "Main" / "best.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source data must not be linked")
    details = _promotion_details()
    details["source_run"] = "dataset/Main"

    with pytest.raises(ValueError, match="dataset/Main"):
        promote_checkpoint(
            project_root,
            alias="hosp-selected",
            checkpoint_path=source,
            **details,
        )


def test_promotion_accepts_the_approved_dotted_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)

    record = promote_checkpoint(
        project_root,
        alias="hosp_bonenet.phase7.selected",
        checkpoint_path=source,
        **_promotion_details(),
    )

    assert resolve_checkpoint(project_root, "hosp_bonenet.phase7.selected") == record


def test_concurrent_promotions_preserve_distinct_aliases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    first = _checkpoint(project_root, b"first checkpoint")
    second = _second_checkpoint(project_root, b"second checkpoint")
    _synchronize_initial_registry_reads(monkeypatch)

    results, errors = _start_parallel_promotions(
        [
            ("hosp-one", first, _promotion_details()),
            ("hosp-two", second, _promotion_details(source_run="experiments/hosp_bonenet/run-002")),
        ],
        project_root,
    )

    assert len(results) == 2
    assert errors == []
    assert resolve_checkpoint(project_root, "hosp-one").checkpoint_sha256 == hashlib.sha256(
        b"first checkpoint"
    ).hexdigest()
    assert resolve_checkpoint(project_root, "hosp-two").checkpoint_sha256 == hashlib.sha256(
        b"second checkpoint"
    ).hexdigest()


def test_concurrent_promotions_reject_a_conflicting_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    first = _checkpoint(project_root, b"first checkpoint")
    second = _second_checkpoint(project_root, b"second checkpoint")
    _synchronize_initial_registry_reads(monkeypatch)

    results, errors = _start_parallel_promotions(
        [
            ("hosp-selected", first, _promotion_details()),
            (
                "hosp-selected",
                second,
                _promotion_details(source_run="experiments/hosp_bonenet/run-002"),
            ),
        ],
        project_root,
    )

    assert len(results) == 1
    assert len(errors) == 1
    assert "already identifies a different artifact" in str(errors[0])
    assert resolve_checkpoint(project_root, "hosp-selected") == results[0]


def test_promotion_falls_back_to_an_atomic_verified_copy_when_hard_links_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    source = _checkpoint(project_root)

    def hard_link_unavailable(*args: object) -> None:
        del args
        raise OSError("hard links unavailable")

    monkeypatch.setattr(registry_module.os, "link", hard_link_unavailable)
    record = promote_checkpoint(
        project_root,
        alias="hosp-selected",
        checkpoint_path=source,
        **_promotion_details(),
    )
    source.write_bytes(b"changed source checkpoint")

    resolved = resolve_checkpoint(project_root, "hosp-selected")

    assert resolved == record
    assert (project_root / record.checkpoint_path).read_bytes() == b"verified checkpoint"


def test_promotion_requires_explicit_replacement_for_a_different_artifact(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    first = _checkpoint(project_root, b"first checkpoint")
    second = _second_checkpoint(project_root, b"second checkpoint")
    promote_checkpoint(
        project_root,
        alias="hosp_bonenet.phase7.selected",
        checkpoint_path=first,
        **_promotion_details(),
    )

    with pytest.raises(ValueError, match="already identifies a different artifact"):
        promote_checkpoint(
            project_root,
            alias="hosp_bonenet.phase7.selected",
            checkpoint_path=second,
            **_promotion_details(source_run="experiments/hosp_bonenet/run-002"),
        )


def test_explicit_alias_replacement_preserves_content_addressed_checkpoints(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    first = _checkpoint(project_root, b"first checkpoint")
    second = _second_checkpoint(project_root, b"second checkpoint")
    original = promote_checkpoint(
        project_root,
        alias="hosp_bonenet.phase7.selected",
        checkpoint_path=first,
        **_promotion_details(),
    )

    replacement = promote_checkpoint(
        project_root,
        alias="hosp_bonenet.phase7.selected",
        checkpoint_path=second,
        replace_alias=True,
        **_promotion_details(source_run="experiments/hosp_bonenet/run-002"),
    )

    assert resolve_checkpoint(project_root, "hosp_bonenet.phase7.selected") == replacement
    assert (project_root / original.checkpoint_path).read_bytes() == b"first checkpoint"
    assert (project_root / replacement.checkpoint_path).read_bytes() == b"second checkpoint"


@pytest.mark.parametrize("failure", ["promotion", "registry_write"])
def test_failed_alias_replacement_leaves_the_previous_alias_resolvable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    project_root = tmp_path / "project"
    first = _checkpoint(project_root, b"first checkpoint")
    second = _second_checkpoint(project_root, b"second checkpoint")
    original = promote_checkpoint(
        project_root,
        alias="hosp_bonenet.phase7.selected",
        checkpoint_path=first,
        **_promotion_details(),
    )

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected replacement failure")

    target = "_promote_file" if failure == "promotion" else "_write_registry"
    monkeypatch.setattr(registry_module, target, fail)

    with pytest.raises(OSError, match="injected replacement failure"):
        promote_checkpoint(
            project_root,
            alias="hosp_bonenet.phase7.selected",
            checkpoint_path=second,
            replace_alias=True,
            **_promotion_details(source_run="experiments/hosp_bonenet/run-002"),
        )

    assert resolve_checkpoint(project_root, "hosp_bonenet.phase7.selected") == original
