"""Exercise the installed environment through its public CLI and GPU boundary."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_environment_cli_reports_cuda_and_dependencies(tmp_path):
    script = ROOT / "scripts" / "00_verify_environment.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["python"].startswith("3.11.")
    assert report["cuda"]["available"] is torch.cuda.is_available()
    assert report["packages"]["torch"]["import_ok"] is True


def test_environment_cli_rejects_source_output_path():
    result = subprocess.run(
        [sys.executable, "scripts/00_verify_environment.py", "--output", "dataset/Main/probe.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode != 0
    assert "read-only source" in result.stderr
    assert not (ROOT / "dataset/Main/probe.json").exists()


@pytest.mark.cuda
def test_cuda_synthetic_forward_backward():
    from bone_cancer_ai.utils.device import verify_tensor_device

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit status is recorded by environment verification")
    result = verify_tensor_device("cuda")
    assert result["finite_gradient"] is True
    assert result["device"].startswith("cuda")
    assert result["peak_memory_bytes"] > 0


def test_later_training_script_fails_without_creating_experiment():
    before = sorted((ROOT / "experiments").rglob("*"))
    result = subprocess.run(
        [sys.executable, "scripts/09_train_baseline.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "Phase 5" in result.stderr
    assert sorted((ROOT / "experiments").rglob("*")) == before


def test_report_fails_when_a_required_import_is_missing(monkeypatch):
    from bone_cancer_ai.utils import environment

    monkeypatch.setattr(environment, "PACKAGE_IMPORTS", {"missing": "nonexistent_bone_dependency"})
    report = environment.environment_report()
    assert report["ok"] is False
    assert report["packages"]["missing"]["import_ok"] is False
    assert "ModuleNotFoundError" in report["packages"]["missing"]["error"]


def test_report_fails_when_cuda_required_but_unavailable(monkeypatch):
    from bone_cancer_ai.utils import environment

    monkeypatch.setattr(environment, "PACKAGE_IMPORTS", {"numpy": "numpy"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = environment.environment_report(require_cuda=True)
    assert report["ok"] is False
    assert report["cuda"]["available"] is False
    assert report["tensor_check"]["device"] == "cpu"
    assert report["tensor_check"]["finite_gradient"] is True


def test_report_fails_when_tensor_computation_fails(monkeypatch):
    from bone_cancer_ai.utils import environment

    def failed_compute(request):
        raise RuntimeError("synthetic device failure")

    monkeypatch.setattr(environment, "PACKAGE_IMPORTS", {"numpy": "numpy"})
    monkeypatch.setattr(environment, "verify_tensor_device", failed_compute)
    report = environment.environment_report()
    assert report["ok"] is False
    assert report["tensor_check"]["finite_gradient"] is False
    assert "synthetic device failure" in report["tensor_check"]["error"]
