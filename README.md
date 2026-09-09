# HOSP-BoneNet bone-tumour radiograph project

Research and decision-support prototype. Phases 1 and 2 establish the verified environment and full local dataset audit. No trained model or diagnostic performance is available.

Start with [PROJECT_STATE.md](PROJECT_STATE.md), [PROJECT_GUIDE.md](PROJECT_GUIDE.md), and the authoritative [master directive](BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md). The latest transition is in [PHASE_HANDOFF.md](PHASE_HANDOFF.md).

## Quick start

From the project root in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\.venv\Scripts\Activate.ps1
python scripts\00_verify_environment.py
python scripts\01_index_dataset.py
python scripts\02_audit_dataset.py
python scripts\03_find_duplicates.py
pytest -q
ruff check .
python -m compileall -q src scripts app tools
```

Fresh setup and wheel sources: [requirements/README.md](requirements/README.md). The src-layout package requires `python -m pip install -e .` after dependency installation.

## Safety and status

`dataset/Main` is immutable. Derived manifests, caches and annotations belong in `workspace`; outputs in `artifacts`; runs in `experiments`. Source filenames never become predictive features. The audit verifies binary label 0 as `normal` and 1 as `bone_tumour`; it also reports major duplicate leakage that Phase 3 must resolve through manifests. Subtype, risk and spatial metadata exist for only part of the dataset and remain disabled until Phase 4 validation.

No source data, checkpoints, private annotations, secrets or environment files belong in Git. Folder READMEs describe ownership. No notebooks contain core implementation.
