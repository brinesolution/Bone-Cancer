# tests

## Purpose
Phase-scoped behavioral tests.

## Files
`test_utils.py`: YAML, paths, RNG, logging and hashes; `test_environment.py`: installed CLI/CUDA/inactive later entry points; `test_dataset.py` and `test_dedupe.py`: Phase 2 audit; `test_splits.py`: Phase 3 cleaning/group partition rules; `test_transforms.py`: preprocessing, partial-label Dataset, DataModule, cache and visual-grid contracts.

## Inputs
Synthetic temporary files/tensors, subprocess CLIs, and the installed environment. Tests do not alter the real source dataset.

## Outputs
pytest results; disposable outputs under `logs/pytest_tmp`.

## Main Libraries
pytest, NumPy, Pandas, Pillow, OpenCV, Albumentations, scikit-learn, torch and subprocess.

## Important Functions / Classes
Phase 3 tests cover deterministic validation, seeded training transforms, no flips/crops, image/mask alignment, malformed/missing labels, opaque batches, corrupt decode containment, group overlap and deterministic optional caching. Tests never train models.

## Theory
Synthetic geometric supports check transform alignment; invariant tests establish that connected groups occupy one final split and held-out records never enter fit/calibration.

## Execution
`python -m pytest -q`; focused: `python -m pytest tests/test_splits.py tests/test_transforms.py -q`.

## Notes
Phase 4 adds metadata revision/stale-write/unknown validation, Streamlit AppTest navigation/autosave, LabelMe geometry/source safety, immutable spatial history, CLI failure status and source-import exact-duplicate/conflict regressions. The pinned LabelMe native Qt smoke loads/edits/saves/reloads a synthetic image offscreen with isolated settings; it imports LabelMe before Qt for Windows DLL order. No clinical labels are produced by tests.

Keep tests independent of private source filenames and never lower leakage/label safety assertions to pass a run.
