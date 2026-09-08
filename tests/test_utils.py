"""Protect configuration, filesystem, logging, and reproducibility contracts."""

import importlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch


def test_config_loads_mapping(tmp_path):
    from bone_cancer_ai.utils.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text("nested:\n  enabled: false\nseed: 42\n", encoding="utf-8")
    assert load_config(path) == {"nested": {"enabled": False}, "seed": 42}


@pytest.mark.parametrize("content", ["", "[1, 2]", "key: [", "key: 1\nkey: 2"])
def test_config_rejects_ambiguous_or_invalid_yaml(tmp_path, content):
    from bone_cancer_ai.utils.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_config_missing_file_is_explicit(tmp_path):
    from bone_cancer_ai.utils.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_paths_resolve_relative_roots_and_allow_derived_output(tmp_path):
    from bone_cancer_ai.utils.paths import ProjectPaths

    paths = ProjectPaths.from_config(
        {"project_root": str(tmp_path), "dataset_root": "dataset/Main"}
    )
    assert paths.dataset_root == tmp_path / "dataset" / "Main"
    assert paths.output_path("workspace/cache/image.png") == (
        tmp_path / "workspace" / "cache" / "image.png"
    )
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize(
    "target", ["dataset/Main/a.jpg", "dataset/Main", "dataset", ".", "../outside"]
)
def test_output_paths_reject_source_overlap_and_escape(tmp_path, target):
    from bone_cancer_ai.utils.paths import ProjectPaths

    paths = ProjectPaths.from_config(
        {"project_root": str(tmp_path), "dataset_root": "dataset/Main"}
    )
    with pytest.raises(ValueError):
        paths.output_path(target)


def test_relative_project_root_is_rejected():
    from bone_cancer_ai.utils.paths import ProjectPaths

    with pytest.raises(ValueError):
        ProjectPaths.from_config({"project_root": ".", "dataset_root": "dataset/Main"})


def test_json_logging_reconfiguration_does_not_duplicate_records(tmp_path):
    from bone_cancer_ai.utils.logging import configure_logging

    path = tmp_path / "events.jsonl"
    configure_logging(path)
    logger = configure_logging(path)
    logger.info("value %s", 4, extra={"event": "probe"})
    try:
        raise ValueError("sample error")
    except ValueError:
        logger.exception("failure")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["message"] == "value 4"
    assert rows[0]["event"] == "probe"
    assert rows[0]["timestamp"].endswith("+00:00")
    assert "ValueError: sample error" in rows[1]["exception"]
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_seed_repeats_python_numpy_and_torch_sequences():
    from bone_cancer_ai.utils.seed import seed_everything

    def sample():
        return random.random(), np.random.rand(), torch.rand(3)

    seed_everything(42)
    first = sample()
    seed_everything(42)
    second = sample()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    seed_everything(123)
    assert not torch.equal(first[2], torch.rand(3))


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_seed_rejects_invalid_range(seed):
    from bone_cancer_ai.utils.seed import seed_everything

    with pytest.raises(ValueError):
        seed_everything(seed)


def test_explicit_cuda_request_never_silently_uses_cpu(monkeypatch):
    from bone_cancer_ai.utils.device import get_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        get_device("cuda")
    assert get_device("cpu") == torch.device("cpu")


def test_invalid_cuda_index_rejected():
    from bone_cancer_ai.utils.device import get_device

    with pytest.raises((RuntimeError, ValueError)):
        get_device("cuda:999")


def test_source_snapshot_detects_content_change_without_size_change(tmp_path):
    from bone_cancer_ai.utils.reproducibility import snapshot_source

    source = tmp_path / "source"
    source.mkdir()
    path = source / "sample.bin"
    path.write_bytes(b"abc")
    first = snapshot_source(source)
    assert first["file_count"] == 1
    assert first["files"][0]["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert first == snapshot_source(source)
    path.write_bytes(b"abd")
    assert first["files"][0]["sha256"] != snapshot_source(source)["files"][0]["sha256"]


def test_snapshot_rejects_missing_source(tmp_path):
    from bone_cancer_ai.utils.reproducibility import snapshot_source

    with pytest.raises(FileNotFoundError):
        snapshot_source(tmp_path / "absent")


def test_installed_package_import_smoke():
    package = importlib.import_module("bone_cancer_ai")
    assert Path(package.__file__).is_file()
