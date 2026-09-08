# HOSP-BoneNet bone-tumour radiograph project

Research and decision-support prototype. Phase 1 is complete: 31 tests, CUDA verification and source-integrity checks passed. No trained model or diagnostic performance is available.

Start with [PROJECT_STATE.md](PROJECT_STATE.md), [PROJECT_GUIDE.md](PROJECT_GUIDE.md), and the authoritative [master directive](BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md). The latest transition is in [PHASE_HANDOFF.md](PHASE_HANDOFF.md).

## Quick start

From the project root in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\.venv\Scripts\Activate.ps1
python scripts\00_verify_environment.py
pytest -q
ruff check .
python -m compileall -q src scripts app tools
```

Fresh setup and wheel sources: [requirements/README.md](requirements/README.md). The src-layout package requires `python -m pip install -e .` after dependency installation.

## Safety and status

`dataset/Main` is immutable. Derived manifests, caches and annotations belong in `workspace`; outputs in `artifacts`; runs in `experiments`. Source filenames never become predictive features. Binary meanings remain unverified until Phase 2; optional label heads are disabled. CSV scaffolds contain headers only. Later-phase commands exit with an explicit pending-phase error.

No source data, checkpoints, private annotations, secrets or environment files belong in Git. Folder READMEs describe ownership. No notebooks contain core implementation.
