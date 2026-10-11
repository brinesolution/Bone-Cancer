"""Exercise the Windows launchers against real files, HTTP downloads and ports."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(os.name != "nt" or POWERSHELL is None, reason="Windows launcher")


def _quote(path: str | Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _powershell(code: str, *, launcher: str = "setup.ps1") -> subprocess.CompletedProcess:
    command = (
        "$ErrorActionPreference = 'Stop'; "
        f". {_quote(ROOT / launcher)}; "
        + code
    )
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        cwd=ROOT,
    )


def _json(result: subprocess.CompletedProcess) -> object:
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


@contextmanager
def _http_payload(payload: bytes):
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/asset", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_python_path_tracks_a_clone_with_spaces(tmp_path: Path):
    root = tmp_path / "cloned project"
    python = root / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"placeholder")
    result = _powershell(f"Get-ProjectPython -Root {_quote(root)} | ConvertTo-Json -Compress")
    assert _json(result) == str(python)


def test_dataset_scan_detects_missing_images_even_when_metadata_exists(tmp_path: Path):
    dataset = tmp_path / "dataset" / "BTXRD"
    dataset.mkdir(parents=True)
    (dataset / "dataset.xlsx").write_bytes(b"not a validated workbook")
    result = _powershell(
        f"Get-BtxrdDatasetState -Root {_quote(tmp_path)} | ConvertTo-Json -Compress"
    )
    state = _json(result)
    assert state["present"] is False
    assert state["image_count"] == 0
    assert state["metadata_exists"] is True


def test_dataset_scan_reports_the_original_archive_structure(tmp_path: Path):
    dataset = tmp_path / "dataset" / "BTXRD"
    (dataset / "images").mkdir(parents=True)
    (dataset / "dataset.xlsx").write_bytes(b"workbook is validated by the Python check")
    (dataset / "images" / "IMG0000001.jpeg").write_bytes(b"pixels")
    result = _powershell(
        f"Get-BtxrdDatasetState -Root {_quote(tmp_path)} | ConvertTo-Json -Compress"
    )
    state = _json(result)
    assert state["present"] is True
    assert state["image_count"] == 1
    assert state["metadata_exists"] is True


def test_verified_download_reuses_good_bytes_without_another_request(tmp_path: Path):
    payload = b"verified public research data\x00\xff"
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "downloaded file.zip"
    with _http_payload(payload) as (url, requests):
        invocation = (
            f"Save-VerifiedDownload -Url {_quote(url)} -Destination {_quote(destination)} "
            f"-Checksum '{digest}' -Algorithm SHA256 | Out-Null; "
        )
        result = _powershell(invocation + invocation + "'done' | ConvertTo-Json")
    assert _json(result) == "done"
    assert destination.read_bytes() == payload
    assert requests == ["/asset"]
    assert not destination.with_name(destination.name + ".partial").exists()


def test_corrupt_download_is_rejected_and_does_not_replace_existing_data(tmp_path: Path):
    destination = tmp_path / "asset.zip"
    destination.write_bytes(b"existing data")
    with _http_payload(b"corrupt response") as (url, _requests):
        result = _powershell(
            f"Save-VerifiedDownload -Url {_quote(url)} -Destination {_quote(destination)} "
            f"-Checksum '{hashlib.sha256(b'expected new data').hexdigest()}' -Algorithm SHA256"
        )
    assert result.returncode != 0
    assert "checksum" in result.stderr.lower()
    assert destination.read_bytes() == b"existing data"
    assert not destination.with_name(destination.name + ".partial").exists()


def test_zip_extraction_preserves_contents_and_accepts_space_paths(tmp_path: Path):
    archive = tmp_path / "official archive.zip"
    destination = tmp_path / "safe destination"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("BTXRD/images/IMG0000001.jpeg", b"image bytes")
        writer.writestr("BTXRD/dataset.xlsx", b"metadata bytes")
    result = _powershell(
        f"Expand-SafeArchive -Archive {_quote(archive)} -Destination {_quote(destination)} "
        "-AllowedPrefixes @('BTXRD/') | Out-Null; 'done' | ConvertTo-Json"
    )
    assert _json(result) == "done"
    assert (destination / "BTXRD/images/IMG0000001.jpeg").read_bytes() == b"image bytes"
    assert (destination / "BTXRD/dataset.xlsx").read_bytes() == b"metadata bytes"


@pytest.mark.parametrize("unsafe", ["../outside.txt", "C:/outside.txt", "BTXRD/../../outside.txt"])
def test_zip_extraction_rejects_escape_before_writing_any_entry(tmp_path: Path, unsafe: str):
    archive = tmp_path / "bad.zip"
    destination = tmp_path / "destination"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("BTXRD/dataset.xlsx", b"good entry before bad entry")
        writer.writestr(unsafe, b"must never be written")
    result = _powershell(
        f"Expand-SafeArchive -Archive {_quote(archive)} -Destination {_quote(destination)} "
        "-AllowedPrefixes @('BTXRD/')"
    )
    assert result.returncode != 0
    assert "archive" in result.stderr.lower()
    assert not (tmp_path / "outside.txt").exists()
    assert not (destination / "BTXRD/dataset.xlsx").exists()


def test_zip_extraction_rejects_symbolic_links(tmp_path: Path):
    archive = tmp_path / "link.zip"
    entry = zipfile.ZipInfo("BTXRD/images/link")
    entry.create_system = 3
    entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(entry, "../../../outside")
    result = _powershell(
        f"Expand-SafeArchive -Archive {_quote(archive)} -Destination {_quote(tmp_path / 'out')} "
        "-AllowedPrefixes @('BTXRD/')"
    )
    assert result.returncode != 0
    assert "archive" in result.stderr.lower()
    assert not (tmp_path / "out/BTXRD/images/link").exists()


def test_missing_runtime_assets_produce_an_explicit_not_ready_state(tmp_path: Path):
    registry = tmp_path / "artifacts/checkpoints/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "hosp_bonenet.btxrd.subtype.selected": {
                    "checkpoint_path": "artifacts/checkpoints/missing.pt",
                    "checkpoint_sha256": "a" * 64,
                    "source_run": "experiments/btxrd_subtype/run-fixture",
                }
            }
        )
    )
    result = _powershell(
        f"Get-RuntimeAssetState -Root {_quote(tmp_path)} | ConvertTo-Json -Depth 5"
    )
    state = _json(result)
    assert state["present"] is False
    assert "artifacts/checkpoints/missing.pt" in state["missing"]
    assert "experiments/btxrd_subtype/run-fixture/metrics.json" in state["missing"]


def test_runtime_scan_rejects_a_changed_checkpoint(tmp_path: Path):
    registry = tmp_path / "artifacts/checkpoints/registry.json"
    registry.parent.mkdir(parents=True)
    checkpoint = registry.parent / "model.pt"
    checkpoint.write_bytes(b"wrong checkpoint")
    registry.write_text(
        json.dumps(
            {
                "hosp_bonenet.btxrd.subtype.selected": {
                    "checkpoint_path": "artifacts/checkpoints/model.pt",
                    "checkpoint_sha256": "a" * 64,
                    "source_run": "experiments/btxrd_subtype/run-fixture",
                }
            }
        )
    )
    result = _powershell(
        f"Get-RuntimeAssetState -Root {_quote(tmp_path)} | ConvertTo-Json -Depth 5"
    )
    state = _json(result)
    assert state["present"] is False
    assert state["checkpoint_valid"] is False


def test_runtime_scan_rejects_registry_paths_outside_the_clone(tmp_path: Path):
    registry = tmp_path / "artifacts/checkpoints/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "hosp_bonenet.btxrd.subtype.selected": {
                    "checkpoint_path": "../outside.pt",
                    "checkpoint_sha256": "a" * 64,
                    "source_run": "experiments/btxrd_subtype/run-fixture",
                }
            }
        )
    )
    result = _powershell(f"Get-RuntimeAssetState -Root {_quote(tmp_path)}")
    assert result.returncode != 0
    assert "project" in result.stderr.lower()


def test_run_chooses_a_free_port_without_stopping_an_existing_listener():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = _powershell(
            f"Get-AvailablePort -Preferred {port} | ConvertTo-Json", launcher="run.ps1"
        )
        selected = _json(result)
        assert selected != port
        assert busy.getsockname()[1] == port
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", selected))


def test_run_health_check_waits_for_an_actual_http_response():
    with _http_payload(b"ok") as (url, requests):
        result = _powershell(
            f"Wait-WebHealth -Url {_quote(url)} -TimeoutSeconds 5 | ConvertTo-Json",
            launcher="run.ps1",
        )
    assert _json(result) is True
    assert requests


def test_run_command_keeps_the_app_path_as_a_single_quoted_argument(tmp_path: Path):
    root = tmp_path / "clone with spaces"
    result = _powershell(
        f"Get-WebArguments -Root {_quote(root)} -Port 8765 | ConvertTo-Json -Compress",
        launcher="run.ps1",
    )
    arguments = _json(result)
    assert arguments[:3] == ["-m", "streamlit", "run"]
    assert arguments[3] == '"' + str(root / "app/streamlit_app.py") + '"'
    assert arguments[arguments.index("--server.port") + 1] == "8765"


def test_run_check_only_preserves_the_flag_after_loading_setup(tmp_path: Path):
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy2(ROOT / "run.ps1", clone / "run.ps1")
    # Replace only the external environment/model check. Run's argument scope,
    # test selection, lifecycle, and check-only branch remain real.
    (clone / "setup.ps1").write_text(
        "param([string]$Device='auto',[switch]$CheckOnly,[switch]$SkipDataset,"
        "[string]$ReportPath='')\n"
        f"function Get-ProjectPython {{ param($Root) return {_quote(sys.executable)} }}\n"
        "function Get-InstallProfile { param($Requested,$Root) return 'cpu' }\n"
        "if ($MyInvocation.InvocationName -ne '.') { exit 0 }\n"
    )
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(clone / "run.ps1"),
            "-CheckOnly",
            "-Tests",
            "none",
            "-NoBrowser",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert not (clone / "logs/web_app_process.json").exists()


def test_project_command_does_not_turn_native_output_into_a_python_path():
    result = _powershell(
        f"$value = Invoke-ProjectCommand -Executable {_quote(sys.executable)} "
        f"-Arguments @('-c', 'print(123)') -Root {_quote(ROOT)}; "
        "if ($null -ne $value) { throw 'Native command output leaked into a return value' }"
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "123" in result.stdout


def test_runtime_export_binds_hashes_and_excludes_dataset_and_environment(tmp_path: Path):
    checkpoint = b"selected checkpoint bytes"
    digest = hashlib.sha256(checkpoint).hexdigest()
    files = {
        "artifacts/checkpoints/model.pt": checkpoint,
        "experiments/btxrd_subtype/run-test/metrics.json": b"{}",
        "experiments/btxrd_subtype/run-test/status.json": b"{}",
        "experiments/btxrd_subtype/run-test/contract.json": b"{}",
        "artifacts/metrics/btxrd_subtype_review.json": b"{}",
        "artifacts/metrics/btxrd_subtype_temperature.json": b"{}",
        "artifacts/metrics/btxrd_robustness.json": b"{}",
        "artifacts/metrics/final_model_comparison.csv": b"model\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "artifacts/checkpoints/registry.json").write_text(
        json.dumps(
            {
                "hosp_bonenet.btxrd.subtype.selected": {
                    "checkpoint_path": "artifacts/checkpoints/model.pt",
                    "checkpoint_sha256": digest,
                    "source_run": "experiments/btxrd_subtype/run-test",
                }
            }
        )
    )
    for relative in ("dataset/private.txt", ".venv/private.txt", ".env"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded")
    result = _powershell(f"Export-RuntimeArchive -Root {_quote(tmp_path)} | Out-Null")
    assert result.returncode == 0, result.stderr + result.stdout
    with zipfile.ZipFile(tmp_path / "dist/bone-cancer-runtime.zip") as archive:
        assert set(archive.namelist()) == set(files) | {"asset-manifest.json"}
        manifest = json.loads(archive.read("asset-manifest.json"))
        assert manifest["checkpoint_sha256"] == digest
        for entry in manifest["files"]:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]


def test_auto_setup_uses_cpu_when_the_installed_cuda_stack_cannot_execute(tmp_path: Path):
    clone = tmp_path / "clone"
    (clone / "app").mkdir(parents=True)
    (clone / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (clone / "app/streamlit_app.py").write_text("")
    shutil.copy2(ROOT / "setup.ps1", clone / "setup.ps1")
    code = (
        "$ErrorActionPreference='Stop'; "
        f". {_quote(clone / 'setup.ps1')} -SkipDataset; "
        "function Get-InstallProfile { param($Requested,$Root) return 'cuda' }; "
        "function Get-RuntimeAssetState { param($Root) return @{present=$true} }; "
        "function Install-ProjectEnvironment { param($Root,$Profile) "
        "[IO.File]::AppendAllText((Join-Path $Root 'profiles.txt'),"
        "$Profile+[Environment]::NewLine); "
        "$p=Get-ProjectPython -Root $Root; "
        "[IO.Directory]::CreateDirectory((Split-Path -Parent $p)) | Out-Null; "
        "[IO.File]::WriteAllText($p,'fixture'); return $p }; "
        "function Invoke-PythonReadiness { param($Root,$Python,$Profile,[switch]$RepairPaths) "
        "$ok=$Profile -eq 'cpu'; return @{python_ok=$true;requirements_ok=$ok;"
        "dataset_valid=$false;model_ready=$ok;errors=@();environment=@{"
        "cuda=@{available=$false};tensor_check=@{finite_gradient=$ok}}} }; "
        "$exitCode=Invoke-SetupMain; if ($exitCode -ne 0) { throw 'CPU fallback did not complete' }"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (clone / "profiles.txt").read_text().splitlines() == ["cuda", "cpu"]
    report = json.loads((clone / "logs/setup_report.json").read_text())
    assert report["profile"] == "cpu"
    assert report["app_ready"] is True


def test_runtime_scan_requires_the_selected_temperature_calibration(tmp_path: Path):
    record = json.loads((ROOT / "artifacts/checkpoints/registry.json").read_text())[
        "hosp_bonenet.btxrd.subtype.selected"
    ]
    registry = tmp_path / "artifacts/checkpoints/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"hosp_bonenet.btxrd.subtype.selected": record}))
    checkpoint = tmp_path / record["checkpoint_path"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    os.link(ROOT / record["checkpoint_path"], checkpoint)
    for relative in (
        f"{record['source_run']}/metrics.json",
        f"{record['source_run']}/status.json",
        f"{record['source_run']}/contract.json",
        "artifacts/metrics/btxrd_subtype_review.json",
        "artifacts/metrics/btxrd_robustness.json",
        "artifacts/metrics/final_model_comparison.csv",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    result = _powershell(f"Get-RuntimeAssetState -Root {_quote(tmp_path)} | ConvertTo-Json")
    state = _json(result)
    assert state["checkpoint_valid"] is True
    assert state["present"] is False
    assert state["missing"] == ["artifacts/metrics/btxrd_subtype_temperature.json"]


def test_installation_rejects_a_junction_as_the_destination_root(tmp_path: Path):
    source = tmp_path / "source"
    (source / "BTXRD").mkdir(parents=True)
    (source / "BTXRD/dataset.xlsx").write_bytes(b"source workbook")
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "clone/dataset"
    destination.parent.mkdir()
    result = _powershell(
        f"New-Item -ItemType Junction -Path {_quote(destination)} -Target {_quote(outside)} "
        "| Out-Null; "
        f"Install-ArchiveTree -Source {_quote(source)} -Root {_quote(destination)} "
        "-AllowedPrefixes @('BTXRD/') -PreserveDifferent"
    )
    assert result.returncode != 0
    assert "link" in result.stderr.lower()
    assert not (outside / "BTXRD/dataset.xlsx").exists()


def test_a_corrupt_optional_dataset_does_not_block_model_readiness(tmp_path: Path):
    for folder in ("app", "configs", "requirements", "artifacts/metrics"):
        shutil.copytree(
            ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__")
        )
    registry = tmp_path / "artifacts/checkpoints/registry.json"
    registry.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "artifacts/checkpoints/registry.json", registry)
    record = json.loads(registry.read_text())["hosp_bonenet.btxrd.subtype.selected"]
    checkpoint = tmp_path / record["checkpoint_path"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    os.link(ROOT / record["checkpoint_path"], checkpoint)
    for namespace in ("btxrd_subtype", "btxrd_subtype_balanced"):
        for path in (ROOT / "experiments" / namespace).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv"}:
                target = tmp_path / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    source = tmp_path / "dataset/BTXRD"
    source.mkdir(parents=True)
    (source / "dataset.xlsx").write_bytes(b"damaged workbook")
    result = _powershell(
        f"Invoke-PythonReadiness -Root {_quote(tmp_path)} -Python {_quote(sys.executable)} "
        "-Profile cpu -RepairPaths | ConvertTo-Json -Depth 30"
    )
    report = _json(result)
    assert report["dataset_valid"] is False
    assert report["dataset_errors"]
    assert report["requirements_ok"] is True
    assert report["model_ready"] is True


def test_run_refreshes_its_report_after_automatic_setup(tmp_path: Path):
    shutil.copy2(ROOT / "run.ps1", tmp_path / "run.ps1")
    (tmp_path / "setup.ps1").write_text(
        "param([string]$Device='auto',[switch]$CheckOnly,[switch]$SkipDataset,"
        "[string]$ReportPath='')\n"
        f"function Get-ProjectPython {{ param($Root) return {_quote(sys.executable)} }}\n"
        "function Get-InstallProfile { param($Requested,$Root) return 'cpu' }\n"
        "if ($MyInvocation.InvocationName -ne '.') {\n"
        " $marker=Join-Path $PSScriptRoot 'installed';\n"
        " if (-not $CheckOnly) { [IO.File]::WriteAllText($marker,'ready'); exit 0 };\n"
        " [IO.Directory]::CreateDirectory((Split-Path -Parent $ReportPath)) | Out-Null;\n"
        " $ready=Test-Path -LiteralPath $marker;\n"
        " [IO.File]::WriteAllText($ReportPath,(@{app_ready=$ready}|ConvertTo-Json));\n"
        " if ($ready) { exit 0 } else { exit 1 }\n"
        "}\n"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(tmp_path / "run.ps1"), "-AutoSetup", "-CheckOnly", "-Tests", "none"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads((tmp_path / "logs/run_readiness.json").read_text())["app_ready"] is True


def test_run_stops_only_its_new_server_when_startup_fails(tmp_path: Path):
    shutil.copy2(ROOT / "run.ps1", tmp_path / "run.ps1")
    (tmp_path / "setup.ps1").write_text(
        "param([string]$Device='auto',[switch]$CheckOnly,[switch]$SkipDataset,"
        "[string]$ReportPath='')\n"
        f"function Get-ProjectPython {{ param($Root) return {_quote(sys.executable)} }}\n"
        "function Get-InstallProfile { param($Requested,$Root) return 'cpu' }\n"
        "if ($MyInvocation.InvocationName -ne '.') { exit 0 }\n"
    )
    code = (
        "$ErrorActionPreference='Stop'; "
        f". {_quote(tmp_path / 'run.ps1')} -Tests none -NoBrowser; "
        "function Get-WebArguments { param($Root,$Port) return @('-c',"
        "'\"import time; time.sleep(60)\"') }; "
        "function Wait-WebHealth { param($Url,$ServerProcess) "
        f"[IO.File]::WriteAllText({_quote(tmp_path / 'spawned.pid')},[string]$ServerProcess.Id); "
        "throw 'Forced startup failure' }; "
        "try { Invoke-RunMain | Out-Null; throw 'Failure was swallowed' } "
        "catch { if ($_.Exception.Message -ne 'Forced startup failure') { throw } }; "
        f"$spawned=[int](Get-Content -LiteralPath {_quote(tmp_path / 'spawned.pid')}); "
        "if (Get-Process -Id $spawned -ErrorAction SilentlyContinue) { "
        "Stop-Process -Id $spawned; throw 'Failed server was left running' }; "
        "'clean' | ConvertTo-Json"
    )
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    assert _json(result) == "clean"
    assert not (tmp_path / "logs/web_app_process.json").exists()
