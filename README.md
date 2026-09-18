# HOSP-BoneNet bone-tumour radiograph project

Research-use prototype. Phases 1–10 establish the verified environment, audited data, leakage-controlled manifests, trained models, frozen evaluation, label-free inference and a Streamlit demo. Outputs are not a diagnosis.

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
python scripts\04_build_clean_manifest.py
python scripts\06_make_internal_splits.py
pytest -q
ruff check .
python -m compileall -q src scripts app tools
```

Run one image through the frozen checkpoint:

```powershell
python scripts\15_infer_image.py --image "PATH_TO_XRAY" --device cuda
python launch_web_app.py
```

The web app has an image-analysis tab and a read-only model-results tab backed by the sealed comparison table and figures. Use `python launch_web_app.py --port 8502` to choose another local port.

Fresh setup and wheel sources: [requirements/README.md](requirements/README.md). The src-layout package requires `python -m pip install -e .` after dependency installation.

## Safety and status

`dataset/Main` is immutable. Derived manifests, caches and annotations belong in `workspace`; outputs in `artifacts`; runs in `experiments`. Source filenames never become predictive features. The audit verifies binary label 0 as `normal` and 1 as `bone_tumour`. Phase 3 excludes lower-priority cross-split duplicates and exact-byte conflict groups without moving source files, then splits only clean official training data into fit and calibration partitions. Subtype, risk and spatial metadata exist for only part of the dataset and remain disabled pending an explicit validated supervision release.

Phase 4 validated 1,825 source mask/box records, quarantined 16 conflicting source drawings and left 26 sources unmatched. Optional supervision is not automatically released into training manifests. Human project reviews remain pending. Launch `.\.venv\Scripts\python.exe scripts/07_launch_annotation_tool.py`; follow [the annotation workflow](workspace/annotations/README.md).

No source data, checkpoints, private annotations, secrets or environment files belong in Git. Folder READMEs describe ownership. No notebooks contain core implementation.
