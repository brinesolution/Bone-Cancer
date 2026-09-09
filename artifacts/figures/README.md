# artifacts/figures

## Purpose
Audit/evaluation figures.

## Files
`dataset_class_distribution.png`, `dimension_distribution.png` and `sample_grid.png` are Phase 2 audit figures. `preprocessing_batch_grid.png` is the Phase 3 real-loader QA grid. Later model/evaluation figures remain pending.

## Inputs
Audited manifests and one batch from each Phase 3 split.

## Outputs
Executed class-count, stored-dimension, source-preview and preprocessed-batch figures.

## Main Libraries
Matplotlib, Pillow and NumPy.

## Important Functions / Classes
`save_audit_figures` owns Phase 2 plots; `smoke_test_loaders` owns the Phase 3 batch grid.

## Theory
The Phase 3 grid denormalizes tensors for display and uses opaque titles. Visual review confirmed recognizable anatomy, grayscale consistency and expected aspect padding across all four splits. Burned-in markers/capture artifacts remain a documented shortcut risk.

## Execution
`python scripts/02_audit_dataset.py` regenerates Phase 2 figures; `python scripts/06_make_internal_splits.py` regenerates the Phase 3 grid.

## Notes
Do not edit generated figures manually or present visual QA as clinical/radiologist review. No model result is shown here.
