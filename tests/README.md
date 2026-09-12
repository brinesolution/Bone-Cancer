# tests

## Purpose
Phase-scoped behavioral tests.

## Files
`test_utils.py`: configuration and provenance utilities; data/annotation tests cover Phases 2–4; model/training tests cover baselines and HOSP-BoneNet. `test_hosp_bonenet.py` and `test_hosp_losses.py` verify Phase 6 gates, shapes, AMP, gradients and masked losses.

## Inputs
Synthetic temporary files/tensors, subprocess CLIs, and the installed environment. Tests do not alter the real source dataset.

## Outputs
pytest results; disposable outputs under `logs/pytest_tmp`.

## Main Libraries
pytest, NumPy, Pandas, Pillow, OpenCV, Albumentations, scikit-learn, torch and subprocess.

## Important Functions / Classes
Tests cover deterministic data contracts, leakage controls, annotation validity, baseline behavior, HOSP output gates, missing-label losses, and 384×384 CUDA AMP. Pytest does not run full experiments.

## Theory
Synthetic geometric supports check transform alignment; invariant tests establish that connected groups occupy one final split and held-out records never enter fit/calibration.

## Execution
`python -m pytest -q`; focused: `python -m pytest tests/test_hosp_bonenet.py tests/test_hosp_losses.py -q`.

## Notes
Phase 4 adds metadata revision/stale-write/unknown validation, Streamlit AppTest navigation/autosave, LabelMe geometry/source safety, immutable spatial history, CLI failure status and source-import exact-duplicate/conflict regressions. The pinned LabelMe native Qt smoke loads/edits/saves/reloads a synthetic image offscreen with isolated settings; it imports LabelMe before Qt for Windows DLL order. No clinical labels are produced by tests.

Keep tests independent of private source filenames and never lower leakage/label safety assertions to pass a run.
