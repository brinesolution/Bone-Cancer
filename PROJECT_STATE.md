# Project state

Updated: 2026-09-09 (Asia/Calcutta).

**PHASE 1 COMPLETE — Environment, Filesystem and Project Scaffold.**
Completed phases: 1 only. Phases 2–10: Pending. No model training has run.

Authoritative specification: [master directive](BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md), read in full before changes. Engineering loop and review: [PHASE_1_WORK_NOTES.md](PHASE_1_WORK_NOTES.md). Latest-only transition: [PHASE_HANDOFF.md](PHASE_HANDOFF.md).

## Environment

| Component | Verified value |
|---|---|
| OS | Windows 11 Pro, 10.0.22631 |
| Python | 3.11.15, project-local .venv from installed uv-managed interpreter |
| PyTorch / torchvision | 2.14.0+cu130 / 0.29.0+cu130 |
| CUDA available | True |
| Wheel CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| GPU memory | 8,585,216,000 bytes (nvidia-smi: 8,188 MiB) |
| Compute capability | 8.9 |
| NVIDIA driver | 591.86 (advertises CUDA 13.1 capability) |
| Synthetic tensor verification | cuda:0; AMP enabled; finite gradient; peak allocated 67,112,448 bytes |
| Git | 2.45.2.windows.1; local branch phase1-scaffold; existing configured identity |
| VS Code | 1.136.2; Python and Pylance extensions present |
| Dependencies | 40 declared imports verified; 237 distributions frozen; pip check passed |

The tensor check is matrix multiplication and backward only: no network, dataset training, optimizer step, checkpoint or performance metric. Peak allocation is the synthetic verification process, not a model-training memory budget.

Use the existing `.venv`. `py -3.11` cannot resolve the installed uv-managed runtime on this machine; initial setup via uv or its direct interpreter is documented in requirements/README.md. `grad-cam` is the maintained distribution name. OpenCV uses only `opencv-python-headless` because Albumentations and Grad-CAM require it; the competing cv2 distributions were removed before reinstalling headless. See requirements/README.md for sources and exact commands.

## Source dataset and label status

Source: `E:\Bone Cancer\dataset\Main`, read-only.

| Official folder | JPG/JPEG files | CSV files | Total bytes including CSV |
|---|---:|---:|---:|
| train | 8,789 | 1 | 777,734,767 |
| val | 1,884 | 1 | 181,993,079 |
| test | 1,884 | 1 | 160,690,161 |
| Total | 12,557 | 3 | 1,120,418,007 |

These are filesystem counts only. Images have not been decoded/audited; CSV rows, class counts, binary semantics, label integrity, duplicate/leakage groups and patient/study identifiers remain **Pending Phase 2**. The shared sample is not used as a substitute for the local audit. All optional task heads and schema labels remain disabled/unverified. CSV scaffolds contain headers only, not audited records.

Other existing dataset material includes `archive`, `archive(2)`, `archive(3)`, `bone cancer detection.v1i.folder`, `BTXRD`, and ZIP archives. Their provenance and relationship to Main are unverified; they were not merged or modified.

**Source preservation:** all 12,560 relative file paths, SHA-256 hashes, sizes and modification times match before/after Phase 1. Evidence: `logs/source_before_phase1.json`, `logs/source_after_phase1.json`, `logs/scaffold_verification.json`. Initial snapshot file SHA-256: `ec6bee8b6fed7a315e634bdf3b7f9580351f23a5a53652543141fc7c63c88ad4`. These snapshots can inform later hash indexing but are not a duplicate/leakage analysis.

## Implemented and verified

- All 201 Section 5 paths exist, with READMEs in authored meaningful directories; no documentation was added inside Main.
- Installable src package; safe YAML config loading, guarded output paths, UTC JSON logging, seed/device utilities, source hashing and reusable environment report.
- Environment CLI works from another working directory and returns explicit failures for unsafe destinations or unavailable requirements. Reports use exclusive creation to avoid overwriting existing files.
- VS Code interpreter, debugger and Phase 1 tasks; PowerShell verify/test/lint/compile commands.
- Future modules are docstring scaffolds, future scripts fail explicitly, and notebooks are valid markdown-only shells.
- Root guide contains all 31 required sections, with later implementation clearly Pending.
- Independent read-only review: no Critical/Important findings; both minor test suggestions addressed.

## Latest successful commands

From the root, with `.venv` activated:

```powershell
python scripts\00_verify_environment.py --require-cuda --output logs\environment_phase1.json
pytest -q
ruff check .
python -m compileall -q src scripts app tools
python -m pip check
.\scripts\run_phase.ps1 -Action verify
python -m pip freeze --all --exclude-editable
python -m pip install --dry-run -r requirements\lock_windows_gpu.txt
```

Results: environment exit 0, **31 tests passed**, Ruff exit 0, compile exit 0, no broken requirements, PowerShell exit 0, lock dry-run exit 0. All VS Code JSON and both notebook schemas validated. Source and directive-path verification passed. Logs: `logs/environment_phase1.json`, `logs/phase1_tests_final.txt`, `logs/phase1_ruff.txt`, `logs/phase1_compile.txt`, `logs/pip_check.txt`, `logs/phase1_powershell_verify.txt`, `logs/lock_dry_run.txt`.

The `--output` destination above already exists; omit that option or choose a new filename on subsequent verification. Dependency import emits an upstream torch.jit deprecation warning and MLflow informational hint on stderr; neither failed the checks. Installation is verified in the existing environment, not by a second full clean-room reinstall.

## Results and checkpoints

| Model | Training/evaluation | Checkpoint |
|---|---|---|
| DenseNet121 | Not run | Unavailable |
| EfficientNet-B3 | Not run | Unavailable |
| ConvNeXt-Tiny | Not run | Unavailable |
| Swin-Tiny | Not run | Unavailable |
| HOSP-BoneNet | Not run | Unavailable |

No accuracy, AUROC, segmentation, annotation, calibration or clinical verification results exist.

## Outstanding issues and next actions

No Phase 1 acceptance blockers. Dataset semantics, leakage, quality and annotation availability require Phase 2. Later training must verify actual model memory use and CUDA again. Import verification alone does not validate future library APIs.

1. Manually select **SOL**; automatic switching of this session is unavailable. Do not begin Phase 2 until switched.
2. Read this state, latest handoff, full directive, data/config/workspace READMEs and existing utilities.
3. Execute **Phase 2 — Dataset Audit and Leakage Analysis** only; maintain the same engineering loop.

Do not rerun environment creation, dependency installation, scaffold generation or notebook creation during ordinary resume. Preserve source snapshots and existing evidence. Never modify Main; never train before later phase gates. Git ignores original data, local tables/annotations, caches, large checkpoints, logs and secrets. The freeze excludes the editable project, which is installed separately.
